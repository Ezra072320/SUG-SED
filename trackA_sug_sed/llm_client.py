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


def _pace_acquire(pace_key: str) -> None:
    """Serialize request dispatches across worker threads to avoid RPM bursts."""
    with _PACE_LOCK:
        now = time.monotonic()
        interval = _PACE_INTERVAL.get(pace_key, 1.5)
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


def _pace_on_success(pace_key: str) -> None:
    """Gradually relax dispatch interval after consecutive healthy responses."""
    with _PACE_LOCK:
        streak = _PACE_OK_STREAK.get(pace_key, 0) + 1
        if streak >= 8:
            cur = _PACE_INTERVAL.get(pace_key, 1.5)
            _PACE_INTERVAL[pace_key] = max(0.8, cur * 0.90)
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
    ) -> None:
        self.model = model
        self.ledger = ledger
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout_sec = float(timeout_sec)

    def _call_http(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 10,
        use_json_mode: bool = True,
    ) -> Tuple[str, int, int, float, str]:
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if use_json_mode:
            payload["response_format"] = {"type": "json_object"}
        if "qwen3" in self.model.lower():
            payload["enable_thinking"] = False

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        pace_key = f"{self.base_url}|{self.model}"
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            _pace_acquire(pace_key)
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                latency = time.time() - t0
                _pace_on_success(pace_key)
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
                pt = int(usage.get("prompt_tokens", 0))
                ct = int(usage.get("completion_tokens", 0))
                return content, pt, ct, latency, finish_reason
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
    ) -> Optional[Tuple[Dict[str, Any], int, int, float, str]]:
        """Execute one cached/live JSON LLM call under optional budget enforcement.

        Returns None if `bucket.can_spend(est_tokens)` refuses before spending;
        otherwise returns `(parsed_dict, prompt_tokens, completion_tokens, latency_sec, status)`.
        """
        est_tokens = int((len(sys_msg) + len(user_msg)) / 3.4) + est_completion_tokens
        if bucket is not None and not bucket.can_spend(est_tokens):
            return None

        req_payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
        }
        req_hash = canonical_request_hash(req_payload)
        cached = self.ledger.lookup_cache(req_hash)

        if cached is not None:
            parsed = cached["response"]
            pt = int(cached["prompt_tokens"])
            ct = int(cached["completion_tokens"])
            latency = 0.0
            status = "CACHED_REPLAY"
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
            )
        else:
            pt, ct, latency = 0, 0, 0.0
            try:
                raw_text, pt, ct, latency, finish_reason = self._call_http(req_payload["messages"])
                if pt == 0:
                    pt = max(1, int((len(sys_msg) + len(user_msg)) / 3.8))
                if ct == 0 and raw_text:
                    ct = max(1, int(len(raw_text) / 3.8))
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
            )

        if bucket is not None:
            bucket.record_spend(pt, ct)

        return parsed, pt, ct, latency, status


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

def build_pairwise_same_event_prompt(
    text_1: str,
    text_2: str,
    date_1: str = "",
    date_2: str = "",
) -> Tuple[str, str]:
    """Pairwise event equivalence prompt (ClusterLLM / Cequel)."""
    sys_msg = (
        "You decide whether two social-media posts refer to the EXACT SAME specific real-world event.\n"
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
) -> Tuple[str, str]:
    """Target post assignment prompt (Few-Shot LLM-Post / RagSEDE-Budgeted / Random)."""
    sys_msg = (
        "You assign a social-media post to one of the candidate event clusters, or mark NEW_EVENT / NOISE.\n"
        "Assign ONLY if the post refers to the exact same specific real-world event as that candidate cluster.\n"
        "Respond strictly in JSON."
    )
    t_str = _clean_snippet(ds.texts[target_idx], 190)
    c_blocks: List[str] = []
    valid_cids: List[int] = []
    for cid in cand_cids:
        if cid not in summaries:
            continue
        valid_cids.append(cid)
        s = summaries[cid]
        kw = ", ".join(s.top_keywords[:5])
        med_lines = "\n".join(f"  - {_clean_snippet(ds.texts[m], 125)}" for m in s.medoids[:3])
        c_blocks.append(f"[Cluster {cid}] (size={s.size}, keywords: {kw}):\n{med_lines}")

    rag_sec = ""
    if rag_context:
        rag_lines = "\n".join(f"  * {_clean_snippet(r, 125)}" for r in rag_context[:3])
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
