"""OpenAI-compatible budgeted LLM client & prompt builders for SUG-SED and baselines.

Tracks real billed prompt + completion tokens under `TokenLedger` and zero-lookahead
`CausalUniformBucket`, with deterministic request hashing (`canonical_request_hash`) and
wall-clock API latency accounting.
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core_shared.data_loader import SocialDataset
from core_shared.token_ledger import TokenLedger, canonical_request_hash


_PACE_LOCK = threading.Lock()
_PACE_NEXT_TS: Dict[str, float] = {}
_PACE_INTERVAL: Dict[str, float] = {}
_PACE_OK_STREAK: Dict[str, int] = {}

EST_CHARS_PER_TOKEN: float = 3.4


def estimate_tokens(sys_msg: str, user_msg: str, est_completion_tokens: int = 90) -> int:
    """Unified character-based preflight token estimator matching `CausalUniformBucket` gating."""
    return int((len(sys_msg) + len(user_msg)) / EST_CHARS_PER_TOKEN) + int(est_completion_tokens)


def _pace_acquire(pace_key: str, default_interval: float = 1.5) -> None:
    """Serialize request dispatches across worker threads to avoid RPM bursts."""
    with _PACE_LOCK:
        now = time.monotonic()
        interval = _PACE_INTERVAL.get(pace_key, default_interval)
        target = max(now, _PACE_NEXT_TS.get(pace_key, 0.0))
        wait = target - now
        _PACE_NEXT_TS[pace_key] = target + interval
    if wait > 0.0:
        time.sleep(wait)


def _pace_on_rate_limit(pace_key: str, cooldown_sec: float) -> None:
    """Slow down shared dispatch cadence cooperatively when HTTP 429 occurs."""
    with _PACE_LOCK:
        _PACE_OK_STREAK[pace_key] = 0
        cur = _PACE_INTERVAL.get(pace_key, 1.5)
        _PACE_INTERVAL[pace_key] = min(8.0, max(cur * 1.4, 2.5))
        _PACE_NEXT_TS[pace_key] = max(_PACE_NEXT_TS.get(pace_key, 0.0), time.monotonic() + cooldown_sec)


def _pace_on_success(pace_key: str, min_interval: float = 0.8) -> None:
    """Gradually relax dispatch interval after consecutive healthy responses."""
    with _PACE_LOCK:
        streak = _PACE_OK_STREAK.get(pace_key, 0) + 1
        if streak >= 8:
            cur = _PACE_INTERVAL.get(pace_key, 1.5)
            _PACE_INTERVAL[pace_key] = max(min_interval, cur * 0.90)
            streak = 0
        _PACE_OK_STREAK[pace_key] = streak


def _clean_snippet(text: str, max_chars: int = 140) -> str:
    return text.replace("\n", " ").strip()[:max_chars]


def _extract_json(raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"No JSON object found in response: {text[:120]!r}")


class BudgetedLLMClient:
    """OpenAI-compatible JSON chat client guarded by TokenLedger & causal token buckets."""

    def __init__(
        self,
        model: str,
        ledger: TokenLedger,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 384,
        timeout_sec: float = 45.0,
        use_json_mode: bool = True,
        enable_thinking: Optional[bool] = None,
        min_pace_sec: float = 0.25,
    ) -> None:
        self.model = model
        self.ledger = ledger
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout_sec = float(timeout_sec)
        self.use_json_mode = bool(use_json_mode)
        self.enable_thinking = (False if "qwen3" in self.model.lower() else None) if enable_thinking is None else enable_thinking
        self.min_pace_sec = max(0.0, float(min_pace_sec))

    def _call_http(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 10,
        use_json_mode: Optional[bool] = None,
        max_tokens_override: Optional[int] = None,
        temperature_override: Optional[float] = None,
        top_p_override: Optional[float] = None,
    ) -> Tuple[str, int, int, float, str, bool]:
        url = f"{self.base_url}/chat/completions"
        eff_json = self.use_json_mode if use_json_mode is None else bool(use_json_mode)
        eff_max_tokens = self.max_tokens if max_tokens_override is None else int(max_tokens_override)
        eff_temp = self.temperature if temperature_override is None else float(temperature_override)
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": eff_temp,
            "max_tokens": eff_max_tokens,
        }
        if top_p_override is not None:
            payload["top_p"] = float(top_p_override)
        if eff_json:
            payload["response_format"] = {"type": "json_object"}
        if self.enable_thinking is not None:
            payload["enable_thinking"] = self.enable_thinking

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        pace_key = f"{self.base_url}|{self.model}"
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            _pace_acquire(pace_key, default_interval=self.min_pace_sec)
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                latency = time.time() - t0
                _pace_on_success(pace_key, min_interval=max(0.05, self.min_pace_sec))
                # Cloudflare OpenAI compat vs native envelope tolerance
                if "choices" in body:
                    choice = body["choices"][0]
                    content = choice["message"].get("content") or ""
                    finish_reason = str(choice.get("finish_reason", ""))
                    usage = body.get("usage", {})
                elif "result" in body and isinstance(body["result"], dict):
                    res = body["result"]
                    content = str(res.get("response", ""))
                    finish_reason = "stop"
                    usage = res.get("usage", {})
                else:
                    raise ValueError(f"Unexpected response schema keys: {list(body.keys())}")
                pt = int(usage.get("prompt_tokens", 0) or 0)
                ct = int(usage.get("completion_tokens", 0) or 0)
                usage_missing = (pt == 0) or (bool(content) and ct == 0)
                return content, pt, ct, latency, finish_reason, usage_missing
            except urllib.error.HTTPError as he:
                last_exc = he
                if he.code == 400 and "response_format" in payload:
                    # Some endpoints (e.g. Cloudflare fast workers) reject response_format
                    payload.pop("response_format", None)
                    continue
                if he.code in {429, 500, 502, 503, 504} and attempt < max_retries:
                    retry_after = he.headers.get("Retry-After") if he.headers else None
                    base_wait = (
                        float(retry_after)
                        if (retry_after and retry_after.replace(".", "", 1).isdigit())
                        else min(45.0, 3.0 * (1.8**attempt))
                    )
                    wait_s = max(1.0, base_wait + random.uniform(0.5, 2.5))
                    if he.code == 429:
                        _pace_on_rate_limit(pace_key, min(12.0, wait_s * 0.5))
                    time.sleep(wait_s)
                    continue
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    time.sleep(min(20.0, 2.0 * (attempt + 1)) + random.uniform(0.5, 2.0))
                    continue
                raise
        raise last_exc  # type: ignore[misc]

    def _call_http_multi(
        self,
        messages: List[Dict[str, str]],
        n_completions: int = 1,
        max_retries: int = 10,
        max_tokens_override: Optional[int] = None,
        temperature_override: Optional[float] = None,
        top_p_override: Optional[float] = None,
    ) -> Tuple[List[str], int, int, float, str, bool]:
        url = f"{self.base_url}/chat/completions"
        eff_max_tokens = self.max_tokens if max_tokens_override is None else int(max_tokens_override)
        eff_temp = self.temperature if temperature_override is None else float(temperature_override)
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": eff_temp,
            "max_tokens": eff_max_tokens,
            "n": max(1, int(n_completions)),
        }
        if top_p_override is not None:
            payload["top_p"] = float(top_p_override)
        if self.enable_thinking is not None:
            payload["enable_thinking"] = self.enable_thinking

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        pace_key = f"{self.base_url}|{self.model}"
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            _pace_acquire(pace_key, default_interval=self.min_pace_sec)
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                latency = time.time() - t0
                _pace_on_success(pace_key, min_interval=max(0.05, self.min_pace_sec))
                if "choices" not in body:
                    raise ValueError(f"Unexpected multi-choice schema keys: {list(body.keys())}")
                choices = body.get("choices") or []
                texts = [(c.get("message", {}).get("content") or "").strip() for c in choices]
                finish_reason = str(choices[0].get("finish_reason", "")) if choices else ""
                usage = body.get("usage", {})
                pt = int(usage.get("prompt_tokens", 0) or 0)
                ct = int(usage.get("completion_tokens", 0) or 0)
                usage_missing = (pt == 0) or (any(bool(t) for t in texts) and ct == 0)
                return texts, pt, ct, latency, finish_reason, usage_missing
            except urllib.error.HTTPError as he:
                last_exc = he
                if he.code in {429, 500, 502, 503, 504} and attempt < max_retries:
                    retry_after = he.headers.get("Retry-After") if he.headers else None
                    base_wait = (
                        float(retry_after)
                        if (retry_after and retry_after.replace(".", "", 1).isdigit())
                        else min(45.0, 3.0 * (1.8**attempt))
                    )
                    wait_s = max(1.0, base_wait + random.uniform(0.5, 2.5))
                    if he.code == 429:
                        _pace_on_rate_limit(pace_key, min(12.0, wait_s * 0.5))
                    time.sleep(wait_s)
                    continue
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    time.sleep(min(20.0, 2.0 * (attempt + 1)) + random.uniform(0.5, 2.0))
                    continue
                raise
        raise last_exc  # type: ignore[misc]

    def call_json(
        self,
        request_id: str,
        block_id: int,
        strategy: str,
        query_type: str,
        sys_msg: str,
        user_msg: str,
        bucket: Optional[Any] = None,
        est_completion_tokens: int = 90,
        max_tokens_override: Optional[int] = None,
        temperature_override: Optional[float] = None,
        top_p_override: Optional[float] = None,
        cache_salt: Optional[str] = None,
        use_json_mode_override: Optional[bool] = None,
    ) -> Optional[Tuple[Dict[str, Any], int, int, float, str]]:
        """Execute one cached/live JSON LLM call under optional budget enforcement.

        Returns None if `bucket.can_spend(est_tokens)` refuses before spending;
        otherwise returns `(parsed_dict, prompt_tokens, completion_tokens, latency_sec, status)`.
        """
        est_tokens = estimate_tokens(sys_msg, user_msg, est_completion_tokens)
        if bucket is not None and not bucket.can_spend(est_tokens):
            return None

        eff_max_tokens = self.max_tokens if max_tokens_override is None else int(max_tokens_override)
        eff_temp = self.temperature if temperature_override is None else float(temperature_override)
        eff_json = self.use_json_mode if use_json_mode_override is None else bool(use_json_mode_override)
        req_payload: Dict[str, Any] = {
            "model": self.model,
            "temperature": eff_temp,
            "max_tokens": eff_max_tokens,
            "response_format": {"type": "json_object"} if eff_json else None,
            "enable_thinking": self.enable_thinking,
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
        }
        if top_p_override is not None:
            req_payload["top_p"] = float(top_p_override)
        if cache_salt is not None:
            req_payload["cache_salt"] = str(cache_salt)
        req_hash = canonical_request_hash(req_payload)
        cached = self.ledger.lookup_cache(req_hash)
        if cached is None:
            default_thinking = False if "qwen3" in self.model.lower() else None
            if (
                max_tokens_override is None
                and temperature_override is None
                and top_p_override is None
                and cache_salt is None
                and use_json_mode_override is None
                and self.max_tokens == 384
                and self.use_json_mode
                and self.enable_thinking == default_thinking
            ):
                legacy_hash = canonical_request_hash(
                    {
                        "model": self.model,
                        "temperature": self.temperature,
                        "messages": req_payload["messages"],
                    }
                )
                cached = self.ledger.lookup_cache(legacy_hash)

        if cached is not None:
            parsed = cached["response"]
            pt = int(cached["prompt_tokens"])
            ct = int(cached["completion_tokens"])
            usage_missing = bool(cached.get("usage_missing", False))
            latency = 0.0
            status = "CACHED_REPLAY|USAGE_MISSING" if usage_missing else "CACHED_REPLAY"
            self.ledger.log_call(
                request_id=request_id,
                block_id=block_id,
                strategy=strategy,
                query_type=query_type,
                model=self.model,
                request_hash=req_hash,
                prompt_tokens=pt,
                completion_tokens=ct,
                latency_sec=0.0,
                status=status,
                response_payload=parsed,
                from_cache=True,
                usage_missing=usage_missing,
            )
        else:
            pt, ct, latency = 0, 0, 0.0
            usage_missing = False
            try:
                http_kw: Dict[str, Any] = {
                    "use_json_mode": eff_json,
                    "max_tokens_override": eff_max_tokens,
                }
                if temperature_override is not None:
                    http_kw["temperature_override"] = temperature_override
                if top_p_override is not None:
                    http_kw["top_p_override"] = top_p_override
                raw_text, pt, ct, latency, finish_reason, usage_missing = self._call_http(
                    req_payload["messages"],
                    **http_kw,
                )
                if pt == 0:
                    pt = max(1, int((len(sys_msg) + len(user_msg)) / EST_CHARS_PER_TOKEN))
                if ct == 0 and raw_text:
                    ct = max(1, int(len(raw_text) / EST_CHARS_PER_TOKEN))
                try:
                    parsed = _extract_json(raw_text)
                    status = "OK"
                except Exception as parse_exc:
                    parsed = {
                        "error": str(parse_exc),
                        "finish_reason": finish_reason,
                        "raw_snippet": raw_text[:140],
                    }
                    status = f"PARSE_ERROR:{type(parse_exc).__name__}"
                if usage_missing:
                    status = f"{status}|USAGE_MISSING"
                    if isinstance(parsed, dict):
                        parsed["usage_missing"] = True
            except Exception as http_exc:
                parsed = {"error": str(http_exc)}
                status = f"HTTP_ERROR:{type(http_exc).__name__}"

            self.ledger.log_call(
                request_id=request_id,
                block_id=block_id,
                strategy=strategy,
                query_type=query_type,
                model=self.model,
                request_hash=req_hash,
                prompt_tokens=pt,
                completion_tokens=ct,
                latency_sec=latency,
                status=status,
                response_payload=parsed,
                from_cache=False,
                usage_missing=usage_missing,
            )

        if bucket is not None:
            bucket.record_spend(pt, ct)

        return parsed, pt, ct, latency, status

    def call_text(
        self,
        request_id: str,
        block_id: int,
        strategy: str,
        query_type: str,
        sys_msg: str,
        user_msg: str,
        bucket: Optional[Any] = None,
        est_completion_tokens: int = 64,
        max_tokens_override: Optional[int] = None,
        temperature_override: Optional[float] = None,
        top_p_override: Optional[float] = None,
        cache_salt: Optional[str] = None,
    ) -> Optional[Tuple[str, int, int, float, str]]:
        """Execute one cached/live plain-text LLM call (no response_format=json_object)."""
        est_tokens = estimate_tokens(sys_msg, user_msg, est_completion_tokens)
        if bucket is not None and not bucket.can_spend(est_tokens):
            return None

        eff_max_tokens = self.max_tokens if max_tokens_override is None else int(max_tokens_override)
        eff_temp = self.temperature if temperature_override is None else float(temperature_override)
        messages = (
            [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]
            if sys_msg
            else [{"role": "user", "content": user_msg}]
        )
        req_payload: Dict[str, Any] = {
            "model": self.model,
            "temperature": eff_temp,
            "max_tokens": eff_max_tokens,
            "response_format": None,
            "enable_thinking": self.enable_thinking,
            "messages": messages,
        }
        if top_p_override is not None:
            req_payload["top_p"] = float(top_p_override)
        if cache_salt is not None:
            req_payload["cache_salt"] = str(cache_salt)
        req_hash = canonical_request_hash(req_payload)
        cached = self.ledger.lookup_cache(req_hash)

        if cached is not None:
            resp_obj = cached["response"]
            out_text = str(resp_obj.get("text", "")) if isinstance(resp_obj, dict) else str(resp_obj)
            pt = int(cached["prompt_tokens"])
            ct = int(cached["completion_tokens"])
            usage_missing = bool(cached.get("usage_missing", False))
            latency = 0.0
            status = "CACHED_REPLAY|USAGE_MISSING" if usage_missing else "CACHED_REPLAY"
            self.ledger.log_call(
                request_id=request_id,
                block_id=block_id,
                strategy=strategy,
                query_type=query_type,
                model=self.model,
                request_hash=req_hash,
                prompt_tokens=pt,
                completion_tokens=ct,
                latency_sec=0.0,
                status=status,
                response_payload={"text": out_text},
                from_cache=True,
                usage_missing=usage_missing,
            )
        else:
            pt, ct, latency = 0, 0, 0.0
            usage_missing = False
            out_text = ""
            try:
                http_kw: Dict[str, Any] = {
                    "use_json_mode": False,
                    "max_tokens_override": eff_max_tokens,
                }
                if temperature_override is not None:
                    http_kw["temperature_override"] = temperature_override
                if top_p_override is not None:
                    http_kw["top_p_override"] = top_p_override
                raw_text, pt, ct, latency, _finish_reason, usage_missing = self._call_http(
                    messages,
                    **http_kw,
                )
                if pt == 0:
                    pt = max(1, int((len(sys_msg) + len(user_msg)) / EST_CHARS_PER_TOKEN))
                if ct == 0 and raw_text:
                    ct = max(1, int(len(raw_text) / EST_CHARS_PER_TOKEN))
                out_text = raw_text.strip()
                status = "OK|USAGE_MISSING" if usage_missing else "OK"
                payload_store: Dict[str, Any] = {"text": out_text}
                if usage_missing:
                    payload_store["usage_missing"] = True
            except Exception as http_exc:
                payload_store = {"error": str(http_exc)}
                status = f"HTTP_ERROR:{type(http_exc).__name__}"

            self.ledger.log_call(
                request_id=request_id,
                block_id=block_id,
                strategy=strategy,
                query_type=query_type,
                model=self.model,
                request_hash=req_hash,
                prompt_tokens=pt,
                completion_tokens=ct,
                latency_sec=latency,
                status=status,
                response_payload=payload_store,
                from_cache=False,
                usage_missing=usage_missing,
            )

        if bucket is not None:
            bucket.record_spend(pt, ct)

        return out_text, pt, ct, latency, status

    def call_text_n(
        self,
        request_id: str,
        block_id: int,
        strategy: str,
        query_type: str,
        sys_msg: str,
        user_msg: str,
        n_completions: int = 5,
        bucket: Optional[Any] = None,
        est_completion_tokens: int = 5,
        max_tokens_override: Optional[int] = 1,
        temperature_override: Optional[float] = 1.0,
        top_p_override: Optional[float] = None,
        cache_salt: Optional[str] = None,
    ) -> Optional[Tuple[List[str], int, int, float, str]]:
        """Execute one cached/live multi-sample plain-text LLM call (`n` completions) with sequential fallback."""
        n_req = int(n_completions)
        if n_req <= 1:
            single = self.call_text(
                request_id=request_id,
                block_id=block_id,
                strategy=strategy,
                query_type=query_type,
                sys_msg=sys_msg,
                user_msg=user_msg,
                bucket=bucket,
                est_completion_tokens=est_completion_tokens,
                max_tokens_override=max_tokens_override,
                temperature_override=temperature_override,
                top_p_override=top_p_override,
                cache_salt=cache_salt,
            )
            if single is None:
                return None
            txt_s, pt_s, ct_s, lat_s, st_s = single
            return [txt_s], pt_s, ct_s, lat_s, st_s

        est_tokens = estimate_tokens(sys_msg, user_msg, est_completion_tokens)
        if bucket is not None and not bucket.can_spend(est_tokens):
            return None

        eff_max_tokens = self.max_tokens if max_tokens_override is None else int(max_tokens_override)
        eff_temp = self.temperature if temperature_override is None else float(temperature_override)
        messages = (
            [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]
            if sys_msg
            else [{"role": "user", "content": user_msg}]
        )
        req_payload: Dict[str, Any] = {
            "model": self.model,
            "temperature": eff_temp,
            "max_tokens": eff_max_tokens,
            "n": n_req,
            "response_format": None,
            "enable_thinking": self.enable_thinking,
            "messages": messages,
        }
        if top_p_override is not None:
            req_payload["top_p"] = float(top_p_override)
        if cache_salt is not None:
            req_payload["cache_salt"] = str(cache_salt)
        req_hash = canonical_request_hash(req_payload)
        cached = self.ledger.lookup_cache(req_hash)

        if cached is not None:
            resp_obj = cached["response"]
            if isinstance(resp_obj, dict) and isinstance(resp_obj.get("texts"), list):
                texts = [str(x).strip() for x in resp_obj["texts"]]
            else:
                texts = [str(resp_obj)]
            pt = int(cached["prompt_tokens"])
            ct = int(cached["completion_tokens"])
            usage_missing = bool(cached.get("usage_missing", False))
            latency = 0.0
            status = "CACHED_REPLAY|USAGE_MISSING" if usage_missing else "CACHED_REPLAY"
            self.ledger.log_call(
                request_id=request_id,
                block_id=block_id,
                strategy=strategy,
                query_type=query_type,
                model=self.model,
                request_hash=req_hash,
                prompt_tokens=pt,
                completion_tokens=ct,
                latency_sec=0.0,
                status=status,
                response_payload={"texts": texts},
                from_cache=True,
                usage_missing=usage_missing,
            )
            if bucket is not None:
                bucket.record_spend(pt, ct)
            return texts, pt, ct, latency, status

        pt_http, ct_http, lat_http = 0, 0, 0.0
        pt_sub, ct_sub, lat_sub = 0, 0, 0.0
        usage_missing = False
        texts: List[str] = []
        need_fallback = False
        status = "OK"
        payload_store: Dict[str, Any] = {}

        try:
            http_kw: Dict[str, Any] = {
                "n_completions": n_req,
                "max_tokens_override": eff_max_tokens,
                "temperature_override": eff_temp,
            }
            if top_p_override is not None:
                http_kw["top_p_override"] = top_p_override
            texts, pt_http, ct_http, lat_http, _finish_reason, usage_missing = self._call_http_multi(
                messages,
                **http_kw,
            )
            if pt_http == 0:
                pt_http = max(1, int((len(sys_msg) + len(user_msg)) / EST_CHARS_PER_TOKEN))
            if ct_http == 0 and any(texts):
                ct_http = max(1, int(sum(len(t) for t in texts) / EST_CHARS_PER_TOKEN))
            if bucket is not None:
                bucket.record_spend(pt_http, ct_http)
            if len(texts) < n_req:
                need_fallback = True
            status = "OK|USAGE_MISSING" if usage_missing else "OK"
        except urllib.error.HTTPError as he:
            if he.code == 400:
                need_fallback = True
                status = "OK_FALLBACK_N"
            else:
                payload_store = {"error": str(he)}
                status = f"HTTP_ERROR:{type(he).__name__}"
        except Exception as http_exc:
            payload_store = {"error": str(http_exc)}
            status = f"HTTP_ERROR:{type(http_exc).__name__}"

        if need_fallback:
            salt_base = cache_salt or "n"
            per_est = max(1, est_completion_tokens // n_req)
            for idx in range(len(texts), n_req):
                sub = self.call_text(
                    request_id=f"{request_id}_s{idx}",
                    block_id=block_id,
                    strategy=strategy,
                    query_type=f"{query_type}_sub",
                    sys_msg=sys_msg,
                    user_msg=user_msg,
                    bucket=bucket,
                    est_completion_tokens=per_est,
                    max_tokens_override=eff_max_tokens,
                    temperature_override=eff_temp,
                    top_p_override=top_p_override,
                    cache_salt=f"{salt_base}_s{idx}",
                )
                if sub is None:
                    break
                sub_txt, s_pt, s_ct, s_lat, s_st = sub
                pt_sub += s_pt
                ct_sub += s_ct
                lat_sub += s_lat
                if s_st.startswith("HTTP_ERROR"):
                    continue
                texts.append(sub_txt)

        pt_total = pt_http + pt_sub
        ct_total = ct_http + ct_sub
        lat_total = lat_http + lat_sub
        if not payload_store:
            payload_store = {"texts": texts}
            if usage_missing:
                payload_store["usage_missing"] = True
            if need_fallback and texts:
                status = "OK_FALLBACK_N"

        self.ledger.log_call(
            request_id=request_id,
            block_id=block_id,
            strategy=strategy,
            query_type=query_type,
            model=self.model,
            request_hash=req_hash,
            prompt_tokens=pt_total,
            completion_tokens=ct_total,
            latency_sec=lat_total,
            status=status,
            response_payload=payload_store,
            from_cache=False,
            usage_missing=usage_missing,
        )

        return texts, pt_total, ct_total, lat_total, status


# ---------------------------------------------------------------------------
# Prompt Builders: Core SUG-SED (Star-Merge, Clean-Seam, Cross-Star)
# ---------------------------------------------------------------------------

_EVENT_EQUIV_SYS = (
    "You adjudicate whether social-media clusters describe the EXACT SAME specific real-world event.\n"
    "Rules:\n"
    "- SAME EVENT ONLY when two groups cover the same specific incident, match, announcement, disaster, or developing story "
    "(e.g., match scores and player goal reactions from the same fixture; resignation report and immediate fallout).\n"
    "- DIFFERENT EVENTS when they cover distinct fixtures, distinct locations/victims, recurring generic topics, "
    "or loosely related celebrities/teams on separate occasions.\n"
    "Always summarize the factual event(s) first and output strict JSON."
)


def build_frag_select_prompt(
    anchor_summ: Any,
    candidate_summs: Sequence[Any],
    ds: SocialDataset,
    text_override: Optional[Dict[int, str]] = None,
) -> Tuple[str, str]:
    """Unified k-way selective verification operator Q(a, {c_1..c_m}) -> {"matches": [ids]}."""
    def _txt(idx: int) -> str:
        return text_override.get(idx, ds.texts[idx]) if text_override is not None else ds.texts[idx]

    sys_msg = (
        "You adjudicate whether social-media event fragments describe the EXACT SAME specific real-world event.\n"
        "Rules:\n"
        "- MATCH ONLY when a candidate fragment covers the exact same specific incident, match, announcement, disaster, or developing story as the Anchor Fragment.\n"
        "- Do NOT match fragments that merely share a broad topic, generic hashtag, or the same public figure on separate occasions.\n"
        'Output strictly valid JSON: {"matches": [<int id>, ...]} (empty list [] if none match).'
    )
    a_kw = ", ".join(anchor_summ.top_keywords[:5])
    a_lines = "\n".join(f"  - {_clean_snippet(_txt(m), 140)}" for m in anchor_summ.medoids[:2])
    cand_blocks: List[str] = []
    valid_ids: List[int] = []
    for idx, cs in enumerate(candidate_summs, start=1):
        valid_ids.append(idx)
        c_kw = ", ".join(cs.top_keywords[:5])
        c_lines = "\n".join(f"  - {_clean_snippet(_txt(m), 140)}" for m in cs.medoids[:2])
        cand_blocks.append(f"Candidate [{idx}] (size={cs.size}, keywords: {c_kw}):\n{c_lines}")

    user_msg = (
        f"Anchor Fragment (size={anchor_summ.size}, keywords: {a_kw}):\n{a_lines}\n\n"
        f"Candidate Fragments (IDs {valid_ids}):\n" + "\n\n".join(cand_blocks) + "\n\n"
        "Select ONLY candidate IDs describing the EXACT SAME specific real-world event as the Anchor Fragment.\n"
        'Return strict JSON: {"matches": [<int>, ...]}'
    )
    return sys_msg, user_msg


def build_star_merge_prompt(
    anchor_summ: Any,
    candidate_summs: Sequence[Any],
    ds: SocialDataset,
    text_override: Optional[Dict[int, str]] = None,
) -> Tuple[str, str]:
    """Contrastive multi-candidate Star-Merge prompt (1 anchor + up to 4 ambiguous neighbors)."""
    def _txt(idx: int) -> str:
        return text_override.get(idx, ds.texts[idx]) if text_override is not None else ds.texts[idx]

    a_kw = ", ".join(anchor_summ.top_keywords[:5])
    a_lines = "\n".join(f"  - {_clean_snippet(_txt(m), 140)}" for m in anchor_summ.medoids[:2])

    cand_blocks: List[str] = []
    valid_ids: List[int] = []
    for idx, cs in enumerate(candidate_summs, start=1):
        valid_ids.append(idx)
        c_kw = ", ".join(cs.top_keywords[:5])
        c_lines = "\n".join(f"  - {_clean_snippet(_txt(m), 140)}" for m in cs.medoids[:2])
        cand_blocks.append(f"Candidate [{idx}] (size={cs.size}, keywords: {c_kw}):\n{c_lines}")

    user_msg = (
        f"Anchor Cluster (size={anchor_summ.size}, keywords: {a_kw}):\n{a_lines}\n\n"
        f"Candidate Neighbor Clusters (IDs {valid_ids}):\n" + "\n\n".join(cand_blocks) + "\n\n"
        "Step 1: Summarize the Anchor Cluster's specific event in 5-10 words (`anchor_event`).\n"
        "Step 2: Select ONLY those candidate IDs (if any) that refer to the EXACT SAME specific real-world incident as the Anchor Cluster, "
        "stating a concise 4-8 word `shared_event` evidence string for each match. Do NOT match candidates that merely share a broad theme or entity on different incidents. "
        "Return an empty list `[]` if none match.\n"
        'Return strict JSON: {"anchor_event": "<5-10 words>", "matches": [{"id": <int>, "shared_event": "<4-8 words>"}]}'
    )
    return _EVENT_EQUIV_SYS, user_msg


def build_seam_split_prompt(
    sub1_medoids: Sequence[int],
    sub2_medoids: Sequence[int],
    size1: int,
    size2: int,
    ds: SocialDataset,
    text_override: Optional[Dict[int, str]] = None,
) -> Tuple[str, str]:
    """Balanced contrastive Subgroup-A vs Subgroup-B Seam-Split prompt."""
    def _txt(idx: int) -> str:
        return text_override.get(idx, ds.texts[idx]) if text_override is not None else ds.texts[idx]

    sys_msg = (
        "You examine whether two subgroups partitioned inside a social-media cluster actually belong to ONE single event "
        "or TWO DIFFERENT real-world events.\n"
        "Rules:\n"
        "- Return different_events=false (KEEP TOGETHER) if both subgroups discuss the same incident, match, broadcast, or fallout "
        "from different viewpoints, sub-angles, languages, or user reactions.\n"
        "- Return different_events=true (SPLIT) ONLY when Subgroup A and Subgroup B clearly describe separate specific incidents, "
        "unrelated stories, or distinct events merged by accident.\n"
        "Always summarize Subgroup A and Subgroup B first, then return strict JSON."
    )
    s1 = "\n".join(f"  - {_clean_snippet(_txt(m), 140)}" for m in sub1_medoids[:3])
    s2 = "\n".join(f"  - {_clean_snippet(_txt(m), 140)}" for m in sub2_medoids[:3])
    user_msg = (
        f"Subgroup A (size={size1}):\n{s1}\n\n"
        f"Subgroup B (size={size2}):\n{s2}\n\n"
        "First summarize Subgroup A (`event_a`) and Subgroup B (`event_b`) in 5-10 words each, "
        "then set `different_events` to true ONLY if they are clearly two different specific real-world events.\n"
        'Return strict JSON: {"event_a": "<5-10 words>", "event_b": "<5-10 words>", "different_events": true or false}'
    )
    return sys_msg, user_msg


def build_cross_star_prompt(
    cur_summ: Any,
    cand_hist_events: Sequence[Any],
    cur_ds: SocialDataset,
) -> Tuple[str, str]:
    """Contrastive multi-candidate Cross-Day continuation prompt (daily cluster vs top-k active events)."""
    c_kw = ", ".join(cur_summ.top_keywords[:5])
    c_lines = "\n".join(f"  - {_clean_snippet(cur_ds.texts[m], 140)}" for m in cur_summ.medoids[:2])
    hist_blocks: List[str] = []
    valid_ids: List[int] = []
    for idx, hev in enumerate(cand_hist_events, start=1):
        valid_ids.append(idx)
        h_kw = ", ".join(hev.top_keywords[:5])
        h_lines = "\n".join(f"  - {_clean_snippet(t, 140)}" for t in hev.rep_texts[:2])
        hist_blocks.append(
            f"Historical Event [{idx}] (blocks {hev.first_block}..{hev.last_block}, size={hev.total_count}, keywords: {h_kw}):\n{h_lines}"
        )

    user_msg = (
        f"New Daily Cluster (size={cur_summ.size}, keywords: {c_kw}):\n{c_lines}\n\n"
        f"Candidate Active Historical Events (IDs {valid_ids}):\n" + "\n\n".join(hist_blocks) + "\n\n"
        "Step 1: Summarize the New Daily Cluster's specific event (`daily_event`, 5-10 words).\n"
        "Step 2: Choose `matched_id` (an integer from the list above) ONLY if that historical event is the exact same ongoing real-world incident; "
        "set `matched_id` to `null` if it is a new or separate event.\n"
        'Return strict JSON: {"daily_event": "<5-10 words>", "matched_id": <int or null>, "reason": "<4-8 words>"}'
    )
    return _EVENT_EQUIV_SYS, user_msg


# ---------------------------------------------------------------------------
# Prompt Builders: Published Same-Budget LLM Baselines
# ---------------------------------------------------------------------------

def select_display_medoids(
    summ: Any,
    exclude_idx: Optional[int] = None,
    max_medoids: int = 2,
) -> List[int]:
    """Select up to `max_medoids` non-excluded representative post indices for a cluster."""
    meds: List[int] = []
    seen: Set[int] = set()
    for m in getattr(summ, "medoids", []):
        mi = int(m)
        if exclude_idx is not None and mi == exclude_idx:
            continue
        if mi not in seen:
            seen.add(mi)
            meds.append(mi)
            if len(meds) >= max_medoids:
                return meds
    for m in getattr(summ, "members", []):
        mi = int(m)
        if exclude_idx is not None and mi == exclude_idx:
            continue
        if mi not in seen:
            seen.add(mi)
            meds.append(mi)
            if len(meds) >= max_medoids:
                return meds
    return meds


def build_pairwise_same_event_prompt(
    text_1: str,
    text_2: str,
    date_1: str = "",
    date_2: str = "",
) -> Tuple[str, str]:
    """Pairwise event equivalence prompt (ClusterLLM / Cequel)."""
    sys_msg = (
        "You decide whether two social-media posts refer to the EXACT SAME specific real-world event.\n"
        "Rules:\n"
        "- SAME EVENT ONLY when both posts cover the same specific incident, match, announcement, disaster, or developing story "
        "(e.g., match scores and player goal reactions from the same fixture; resignation report and immediate fallout).\n"
        "- DIFFERENT EVENTS when they cover distinct fixtures, distinct locations/victims, recurring generic topics, "
        "or loosely related celebrities/teams on separate occasions.\n"
        "Summarize each post's event in 4-8 words first, then return same_event (true/false) in strict JSON."
    )
    d1 = f" ({date_1})" if date_1 else ""
    d2 = f" ({date_2})" if date_2 else ""
    user_msg = (
        f"Post 1{d1}: {_clean_snippet(text_1, 180)}\n"
        f"Post 2{d2}: {_clean_snippet(text_2, 180)}\n\n"
        'Return strict JSON: {"event_a": "<4-8 words>", "event_b": "<4-8 words>", "same_event": true or false}'
    )
    return sys_msg, user_msg


def build_post_assign_prompt(
    target_idx: int,
    cand_cids: Sequence[int],
    summaries: Dict[int, Any],
    ds: SocialDataset,
    rag_context: Optional[Sequence[str]] = None,
    max_medoids: int = 2,
    max_rag: int = 2,
) -> Tuple[str, str]:
    """Target post assignment prompt (Few-Shot LLM-Post / RagSEDE-Budgeted / Random)."""
    sys_msg = (
        "You assign a social-media post to one of the candidate event clusters, or mark NEW_EVENT / NOISE.\n"
        "Rules:\n"
        "- ASSIGN ONLY when the post refers to the EXACT SAME specific real-world incident, match, announcement, disaster, or developing story "
        "as that candidate cluster.\n"
        "- Do NOT assign to a candidate that merely shares a broad theme or celebrity/team on a separate occasion.\n"
        "Respond strictly in JSON."
    )
    t_str = _clean_snippet(ds.texts[target_idx], 190)
    c_blocks: List[str] = []
    valid_cids: List[int] = []
    for cid in cand_cids:
        if cid not in summaries:
            continue
        s = summaries[cid]
        meds = select_display_medoids(s, exclude_idx=target_idx, max_medoids=max_medoids)
        if not meds:
            continue
        valid_cids.append(cid)
        kw = ", ".join(s.top_keywords[:5])
        med_lines = "\n".join(f"  - {_clean_snippet(ds.texts[m], 125)}" for m in meds)
        c_blocks.append(f"[Cluster {cid}] (size={s.size}, keywords: {kw}):\n{med_lines}")

    rag_sec = ""
    if rag_context and max_rag > 0:
        rag_lines = "\n".join(f"  * {_clean_snippet(r, 125)}" for r in rag_context[:max_rag])
        rag_sec = f"\nRetrieved Neighbor Context:\n{rag_lines}\n"

    user_msg = (
        f"Target Post ({ds.dates[target_idx]}): {t_str}\n"
        f"{rag_sec}\n"
        f"Candidate Event Clusters:\n" + "\n\n".join(c_blocks) + "\n\n"
        f"Valid candidate cluster IDs: {valid_cids}.\n"
        'Return strict JSON: {"event_cue": "<4-8 words>", "action": "ASSIGN" | "NEW_EVENT" | "NOISE", "target_cluster": <int or null>}'
    )
    return sys_msg, user_msg


def build_lsed_rewrite_prompt(text: str, date: str) -> Tuple[str, str]:
    """Short-text disambiguation & factual rewriting prompt (LSED, Qiu et al., ACL 2025)."""
    sys_msg = (
        "You normalize noisy social-media posts for streaming event clustering.\n"
        "Expand slang, resolve obvious implicit context cleanly WITHOUT inventing entities, "
        "and rewrite the post as one crisp factual event statement (8-18 words) in strict JSON."
    )
    user_msg = (
        f"Post ({date}): {_clean_snippet(text, 220)}\n"
        'Return strict JSON: {"rewritten_event": "<8-18 factual words>"}'
    )
    return sys_msg, user_msg


def build_lhg_pseudolabel_prompt(text: str, neighbor_snippets: Sequence[str], date: str) -> Tuple[str, str]:
    """Canonical event pseudo-label prompt over heterogeneous neighbors (LHG, Qiu et al., WWW 2026)."""
    sys_msg = (
        "You assign a canonical normalized event label (4-8 words) to a social-media post given its local neighbor posts, "
        "so posts from the same specific real-world event share matching canonical phrasing. Return strict JSON."
    )
    nb_txt = "\n".join(f"  - {_clean_snippet(s, 110)}" for s in neighbor_snippets[:3])
    user_msg = (
        f"Target Post ({date}): {_clean_snippet(text, 180)}\n"
        f"Local Graph Neighbors:\n{nb_txt if nb_txt else '  (none)'}\n\n"
        'Return strict JSON: {"event_label": "<4-8 canonical event words>"}'
    )
    return sys_msg, user_msg
