"""Strict real-token accounting, cumulative token-bucket control, and replayable ledger.

Every LLM/VLM call is logged with actual prompt_tokens, completion_tokens,
request_hash, latency, and raw response so all budget comparisons are exact
and auditable.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


def canonical_request_hash(payload: Dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 hash for a canonicalized request payload."""
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass
class TokenRecord:
    request_id: str
    block_id: int
    strategy: str
    query_type: str
    model: str
    request_hash: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_cny: float
    latency_sec: float
    status: str
    from_cache: bool
    timestamp: float


class CumulativeTokenBucket:
    """Streaming token-bucket budget controller with cumulative balance.

    Implements Proposal A §3.3:
        R_t = R_{t-1} + b_bar_t - c_{t-1}
        B_t = min(max(0, R_t), clip(b_bar_t * (U_t / (U_ema + eps)), b_min, b_max))

    Guarantees that cumulative actual tokens spent across blocks 0..t never exceed
    the cumulative baseline allowance plus any explicitly permitted initial reserve.

    Note: when `block_weight` is passed as `|M_t| / N_eval` using a future total
    stream length `N_eval`, this controller acts as a non-causal reference oracle;
    for strictly causal online deployment (zero future lookahead), use
    `CausalUniformBucket` or `CausalSlidingBucket`.
    """

    def __init__(
        self,
        total_token_budget: int,
        num_blocks: int,
        ema_alpha: float = 0.3,
        min_scale: float = 0.25,
        max_scale: float = 2.0,
        eps: float = 1e-8,
    ) -> None:
        if total_token_budget < 0 or num_blocks <= 0:
            raise ValueError("total_token_budget must be >= 0 and num_blocks > 0.")
        self.total_token_budget = int(total_token_budget)
        self.num_blocks = int(num_blocks)
        self.ema_alpha = float(ema_alpha)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.eps = float(eps)

        self.cumulative_allowance = 0.0
        self.cumulative_spent = 0
        self.u_ema: Optional[float] = None
        self.current_block_cap = 0
        self.current_block_spent = 0

    def start_block(
        self,
        block_weight: float,
        mean_uncertainty: float,
    ) -> int:
        """Compute the strict token cap for a new streaming block.

        Args:
            block_weight: Fraction of total stream weight assigned to this block
                (e.g., |M_t| / N_total, or 1.0 / num_blocks for equal blocks).
            mean_uncertainty: Mean routing uncertainty U_t of the current block.

        Returns:
            Maximum real tokens allowed for this block.
        """
        base_quota = self.total_token_budget * float(block_weight)
        self.cumulative_allowance = min(
            float(self.total_token_budget),
            self.cumulative_allowance + base_quota,
        )
        balance = max(0.0, self.cumulative_allowance - self.cumulative_spent)

        if self.u_ema is None:
            self.u_ema = max(float(mean_uncertainty), self.eps)
            scale = 1.0
        else:
            scale = float(mean_uncertainty) / (self.u_ema + self.eps)
            self.u_ema = (1.0 - self.ema_alpha) * self.u_ema + self.ema_alpha * float(mean_uncertainty)

        b_min = base_quota * self.min_scale
        b_max = base_quota * self.max_scale
        adaptive_target = min(max(base_quota * scale, b_min), b_max)

        remaining_global = max(0, self.total_token_budget - self.cumulative_spent)
        self.current_block_cap = int(min(balance, adaptive_target, remaining_global))
        self.current_block_spent = 0
        return self.current_block_cap

    def can_spend(self, estimated_tokens: int) -> bool:
        """Check whether an upcoming query fits in both block cap and global cap."""
        est = int(estimated_tokens)
        return (
            self.current_block_spent + est <= self.current_block_cap
            and self.cumulative_spent + est <= self.total_token_budget
        )

    def record_spend(self, prompt_tokens: int, completion_tokens: int) -> None:
        total = int(prompt_tokens) + int(completion_tokens)
        self.current_block_spent += total
        self.cumulative_spent += total

    @property
    def remaining_block_tokens(self) -> int:
        return max(0, self.current_block_cap - self.current_block_spent)

    @property
    def remaining_total_tokens(self) -> int:
        return max(0, self.total_token_budget - self.cumulative_spent)


class CausalUniformBucket:
    """Strictly causal streaming budget controller with fixed per-block base quota.

    Never inspects future block counts or future stream length N_eval.
    """

    def __init__(
        self,
        tokens_per_block: int,
        total_token_budget: Optional[int] = None,
        allow_carryover: bool = False,
        max_carryover_mult: float = 1.5,
        ema_alpha: float = 0.3,
        min_scale: float = 0.5,
        max_scale: float = 1.5,
        adaptive_uncertainty: bool = False,
        eps: float = 1e-8,
    ) -> None:
        if tokens_per_block < 0 or (total_token_budget is not None and total_token_budget < 0):
            raise ValueError("tokens_per_block must be >= 0 and total_token_budget (if set) >= 0.")
        self.tokens_per_block = int(tokens_per_block)
        self.total_token_budget = int(total_token_budget) if total_token_budget is not None else None
        self.allow_carryover = bool(allow_carryover)
        self.max_carryover_mult = float(max_carryover_mult)
        self.ema_alpha = float(ema_alpha)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.adaptive_uncertainty = bool(adaptive_uncertainty)
        self.eps = float(eps)

        self.cumulative_allowance = 0.0
        self.cumulative_spent = 0
        self.u_ema: Optional[float] = None
        self.current_block_cap = 0
        self.current_block_spent = 0

    def start_block(
        self,
        block_weight: float = 1.0,
        mean_uncertainty: float = 1.0,
    ) -> int:
        """Compute strictly causal block token cap (ignores future-dependent weights)."""
        _ = block_weight
        base_quota = float(self.tokens_per_block)
        self.cumulative_allowance += base_quota
        if self.total_token_budget is not None:
            self.cumulative_allowance = min(float(self.total_token_budget), self.cumulative_allowance)

        if self.allow_carryover:
            balance = max(0.0, self.cumulative_allowance - self.cumulative_spent)
        else:
            balance = base_quota

        if self.adaptive_uncertainty:
            if self.u_ema is None:
                self.u_ema = max(float(mean_uncertainty), self.eps)
                scale = 1.0
            else:
                raw_scale = float(mean_uncertainty) / (self.u_ema + self.eps)
                scale = min(max(raw_scale, self.min_scale), self.max_scale)
                self.u_ema = (1.0 - self.ema_alpha) * self.u_ema + self.ema_alpha * float(mean_uncertainty)
            target = min(base_quota * self.max_carryover_mult, base_quota * scale)
        else:
            target = base_quota * (self.max_carryover_mult if self.allow_carryover else 1.0)

        cap = min(balance, target)
        if self.total_token_budget is not None:
            remaining_global = max(0, self.total_token_budget - self.cumulative_spent)
            cap = min(cap, float(remaining_global))

        self.current_block_cap = int(cap)
        self.current_block_spent = 0
        return self.current_block_cap

    def can_spend(self, estimated_tokens: int) -> bool:
        est = int(estimated_tokens)
        if self.current_block_spent + est > self.current_block_cap:
            return False
        if self.total_token_budget is not None and (self.cumulative_spent + est > self.total_token_budget):
            return False
        return True

    def record_spend(self, prompt_tokens: int, completion_tokens: int) -> None:
        total = int(prompt_tokens) + int(completion_tokens)
        self.current_block_spent += total
        self.cumulative_spent += total

    @property
    def remaining_block_tokens(self) -> int:
        return max(0, self.current_block_cap - self.current_block_spent)

    @property
    def remaining_total_tokens(self) -> int:
        if self.total_token_budget is None:
            return self.remaining_block_tokens
        return max(0, self.total_token_budget - self.cumulative_spent)


class CausalSlidingBucket:
    """Strictly causal online budget controller scaling quota by past block sizes only."""

    def __init__(
        self,
        tokens_per_block: int,
        total_token_budget: Optional[int] = None,
        init_expected_posts: float = 2000.0,
        ema_alpha: float = 0.3,
        min_scale: float = 0.5,
        max_scale: float = 1.5,
        max_carryover_mult: float = 2.0,
        eps: float = 1e-8,
    ) -> None:
        if tokens_per_block < 0 or (total_token_budget is not None and total_token_budget < 0):
            raise ValueError("tokens_per_block must be >= 0 and total_token_budget (if set) >= 0.")
        self.tokens_per_block = int(tokens_per_block)
        self.total_token_budget = int(total_token_budget) if total_token_budget is not None else None
        self.init_expected_posts = float(init_expected_posts)
        self.ema_alpha = float(ema_alpha)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.max_carryover_mult = float(max_carryover_mult)
        self.eps = float(eps)

        self.cumulative_allowance = 0.0
        self.cumulative_spent = 0
        self.n_ema: Optional[float] = None
        self.current_block_cap = 0
        self.current_block_spent = 0

    def observe_warmup_block(self, n_posts: int) -> None:
        """Update running block-size estimate from historical/warmup blocks only."""
        val = max(float(n_posts), 1.0)
        if self.n_ema is None:
            self.n_ema = val
        else:
            self.n_ema = (1.0 - self.ema_alpha) * self.n_ema + self.ema_alpha * val

    def start_block(
        self,
        n_posts: int,
        mean_uncertainty: float = 1.0,
    ) -> int:
        """Compute strictly causal block cap from current block size and past EMA only."""
        _ = mean_uncertainty
        ref_n = self.n_ema if self.n_ema is not None else self.init_expected_posts
        raw_scale = float(n_posts) / (ref_n + self.eps)
        scale = min(max(raw_scale, self.min_scale), self.max_scale)

        base_quota = float(self.tokens_per_block)
        self.cumulative_allowance += base_quota
        if self.total_token_budget is not None:
            self.cumulative_allowance = min(float(self.total_token_budget), self.cumulative_allowance)

        balance = max(0.0, self.cumulative_allowance - self.cumulative_spent)
        target = min(base_quota * self.max_carryover_mult, base_quota * scale)
        cap = min(balance, target)
        if self.total_token_budget is not None:
            remaining_global = max(0, self.total_token_budget - self.cumulative_spent)
            cap = min(cap, float(remaining_global))

        self.current_block_cap = int(cap)
        self.current_block_spent = 0

        # Update n_ema AFTER computing current block cap to avoid any lookahead/self-circularity
        self.observe_warmup_block(n_posts)
        return self.current_block_cap

    def can_spend(self, estimated_tokens: int) -> bool:
        est = int(estimated_tokens)
        if self.current_block_spent + est > self.current_block_cap:
            return False
        if self.total_token_budget is not None and (self.cumulative_spent + est > self.total_token_budget):
            return False
        return True

    def record_spend(self, prompt_tokens: int, completion_tokens: int) -> None:
        total = int(prompt_tokens) + int(completion_tokens)
        self.current_block_spent += total
        self.cumulative_spent += total

    @property
    def remaining_block_tokens(self) -> int:
        return max(0, self.current_block_cap - self.current_block_spent)

    @property
    def remaining_total_tokens(self) -> int:
        if self.total_token_budget is None:
            return self.remaining_block_tokens
        return max(0, self.total_token_budget - self.cumulative_spent)


class TokenLedger:
    """Append-only JSONL ledger with optional deterministic response replay.

    Important audit rule: even when a response is replayed from cache during
    Branch 1 (Direct Assignment) vs. Branch 2 (Topology Repair) comparisons,
    its real prompt_tokens and completion_tokens are STILL counted in total
    deployment token cost so cached API hits are never misrepresented as free.
    """

    def __init__(
        self,
        log_dir: Path | str,
        price_per_1m_prompt_cny: float = 0.5,
        price_per_1m_completion_cny: float = 2.0,
        extra_cache_dirs: Optional[Sequence[Path | str]] = None,
        shared_cache_file: Optional[Path | str] = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.journal_path = self.log_dir / "journal.jsonl"
        self.responses_path = self.log_dir / "responses.jsonl"
        self.shared_cache_file = Path(shared_cache_file) if shared_cache_file is not None else None
        if self.shared_cache_file is not None:
            self.shared_cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.price_prompt = float(price_per_1m_prompt_cny)
        self.price_completion = float(price_per_1m_completion_cny)
        self._lock = threading.Lock()

        self._cache: Dict[str, Dict[str, Any]] = {}
        if extra_cache_dirs:
            for d in extra_cache_dirs:
                p = Path(d)
                rp = p if p.is_file() else (p / "responses.jsonl")
                self._load_responses_from_file(rp)
        if self.shared_cache_file is not None:
            self._load_responses_from_file(self.shared_cache_file)
        self._load_responses_from_file(self.responses_path)

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0
        self.cached_calls = 0

    @staticmethod
    def _is_transient_http_error(obj: Dict[str, Any]) -> bool:
        resp = obj.get("response", {})
        pt = int(obj.get("prompt_tokens", 0))
        ct = int(obj.get("completion_tokens", 0))
        return isinstance(resp, dict) and ("error" in resp) and pt == 0 and ct == 0

    def _load_responses_from_file(self, path: Path) -> None:
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if self._is_transient_http_error(obj):
                    continue
                h = obj.get("request_hash")
                if h:
                    self._cache[h] = obj

    def lookup_cache(self, request_hash: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._cache.get(request_hash)

    def estimate_cost_cny(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.price_prompt + completion_tokens * self.price_completion
        ) / 1_000_000.0

    def log_call(
        self,
        *,
        request_id: str,
        block_id: int,
        strategy: str,
        query_type: str,
        model: str,
        request_hash: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_sec: float,
        status: str,
        response_payload: Dict[str, Any],
        from_cache: bool = False,
    ) -> TokenRecord:
        total = int(prompt_tokens) + int(completion_tokens)
        cost = self.estimate_cost_cny(prompt_tokens, completion_tokens)
        rec = TokenRecord(
            request_id=request_id,
            block_id=int(block_id),
            strategy=strategy,
            query_type=query_type,
            model=model,
            request_hash=request_hash,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            total_tokens=total,
            estimated_cost_cny=cost,
            latency_sec=float(latency_sec),
            status=status,
            from_cache=bool(from_cache),
            timestamp=time.time(),
        )

        with self._lock:
            self.total_prompt_tokens += rec.prompt_tokens
            self.total_completion_tokens += rec.completion_tokens
            self.total_calls += 1
            if from_cache:
                self.cached_calls += 1

            with self.journal_path.open("a", encoding="utf-8") as jf:
                jf.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

            if not from_cache and not (rec.prompt_tokens == 0 and rec.completion_tokens == 0 and status.startswith("HTTP_ERROR")):
                resp_entry = {
                    "request_hash": request_hash,
                    "request_id": request_id,
                    "model": model,
                    "prompt_tokens": rec.prompt_tokens,
                    "completion_tokens": rec.completion_tokens,
                    "latency_sec": round(rec.latency_sec, 4),
                    "response": response_payload,
                }
                self._cache[request_hash] = resp_entry
                line_str = json.dumps(resp_entry, ensure_ascii=False) + "\n"
                with self.responses_path.open("a", encoding="utf-8") as rf:
                    rf.write(line_str)
                if self.shared_cache_file is not None:
                    with self.shared_cache_file.open("a", encoding="utf-8") as sf:
                        sf.write(line_str)

        return rec

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            total = self.total_prompt_tokens + self.total_completion_tokens
            return {
                "total_calls": self.total_calls,
                "cached_calls": self.cached_calls,
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
                "total_tokens": total,
                "estimated_cost_cny": round(
                    self.estimate_cost_cny(self.total_prompt_tokens, self.total_completion_tokens), 6
                ),
            }
