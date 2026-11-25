"""OpenAI-compatible LLM client with strict real-token ledger & JSON decoding.

Works seamlessly with:
  - Local / cloud vLLM servers (Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct)
  - Hosted OpenAI-compatible APIs (DashScope/Qwen, DeepSeek, OpenAI)

Uses Python stdlib `urllib.request` so zero extra client libraries are needed!
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from core_shared.data_loader import SocialDataset
from core_shared.token_ledger import CumulativeTokenBucket, TokenLedger, canonical_request_hash
from trackA_sug_sed.high_leverage_router import ClusterSummary, QueryItem
from trackA_sug_sed.topology_surgery import LLMDecision


def _format_cluster_brief(
    cid: int,
    summ: ClusterSummary,
    ds: SocialDataset,
    max_chars_per_tweet: int = 130,
) -> str:
    kw = ", ".join(summ.top_keywords[:5])
    lines = [f"[Cluster {cid}] (size={summ.size}, keywords: {kw})"]
    for m in summ.medoids[:3]:
        txt = ds.texts[m].replace("\n", " ").strip()[:max_chars_per_tweet]
        lines.append(f"  - {txt}")
    return "\n".join(lines)


def build_query_prompt(
    q: QueryItem,
    ds: SocialDataset,
    summaries: Dict[int, ClusterSummary],
) -> Tuple[str, str]:
    """Build Extract-Then-Compare prompts tailored for lightweight 7B/8B open-source LLMs.

    For non-thinking dense 7B/8B models, forcing a short factual event summary BEFORE
    emitting the boolean/enum decision eliminates autoregressive yes-bias/sycophancy
    and removes ambiguous float confidence semantics (e.g., 0.0 for SPLIT).
    """
    sys_msg = (
        "You decide whether social-media posts or post groups refer to the SAME specific real-world event.\n"
        "Rules:\n"
        "- SAME EVENT (true / ASSIGN) if both cover the same specific incident, match, announcement, or developing story, "
        "even when they highlight different details, updates, players/angles, or reactions "
        "(e.g., 'Team X beats Team Y 4-2' and 'Player Z scores for Team X against Team Y' -> true; "
        "'Official resigns' and 'Official quits over scandal' -> true).\n"
        "- DIFFERENT EVENT (false / NEW_EVENT) if they cover different incidents, different matches, different countries, "
        "or different main subjects even if sharing a broad topic "
        "(e.g., 'Team X vs Team Y' and 'Team A vs Team B' -> false; 'Scandal at Org 1' and 'Scandal at Org 2' -> false).\n"
        "Always write concise 5-10 word factual event summaries FIRST, then output your final boolean or action in strict JSON."
    )

    if q.query_type == "merge_pair":
        s1_lines = "\n".join(
            f"  - {ds.texts[m].replace(chr(10), ' ').strip()[:140]}"
            for m in q.merge_c1_medoids[:3]
        )
        s2_lines = "\n".join(
            f"  - {ds.texts[m].replace(chr(10), ' ').strip()[:140]}"
            for m in q.merge_c2_medoids[:3]
        )
        c1_kw = ", ".join(summaries[q.merge_c1].top_keywords[:5]) if q.merge_c1 in summaries else ""
        c2_kw = ", ".join(summaries[q.merge_c2].top_keywords[:5]) if q.merge_c2 in summaries else ""
        user_msg = (
            f"Group A (size={summaries[q.merge_c1].size}, keywords: {c1_kw}):\n{s1_lines}\n\n"
            f"Group B (size={summaries[q.merge_c2].size}, keywords: {c2_kw}):\n{s2_lines}\n\n"
            "First summarize Group A's specific event (event_a) and Group B's specific event (event_b) in 5-10 words each, "
            "then set same_event to true ONLY if both describe the exact same specific incident.\n"
            'Return strict JSON: {"event_a": "<5-10 words>", "event_b": "<5-10 words>", "same_event": true or false}'
        )
        return sys_msg, user_msg

    if q.query_type == "seam":
        s1_lines = "\n".join(
            f"  - {ds.texts[m].replace(chr(10), ' ').strip()[:140]}"
            for m in q.seam_sub1_medoids[:3]
        )
        s2_lines = "\n".join(
            f"  - {ds.texts[m].replace(chr(10), ' ').strip()[:140]}"
            for m in q.seam_sub2_medoids[:3]
        )
        user_msg = (
            f"Subgroup A (size={len(q.seam_sub1)}):\n{s1_lines}\n\n"
            f"Subgroup B (size={len(q.seam_sub2)}):\n{s2_lines}\n\n"
            "First summarize Subgroup A's specific event (event_a) and Subgroup B's specific event (event_b) in 5-10 words each, "
            "then set same_event to true if they belong to the exact same specific incident, or false if they are two different events.\n"
            'Return strict JSON: {"event_a": "<5-10 words>", "event_b": "<5-10 words>", "same_event": true or false}'
        )
        return sys_msg, user_msg

    if q.query_type == "pair":
        u, v = q.primary_nodes[0], q.primary_nodes[1]
        t_u = ds.texts[u].replace("\n", " ").strip()[:180]
        t_v = ds.texts[v].replace("\n", " ").strip()[:180]
        user_msg = (
            f"Post 1 ({ds.dates[u]}): {t_u}\n"
            f"Post 2 ({ds.dates[v]}): {t_v}\n\n"
            "First summarize Post 1's event (event_a) and Post 2's event (event_b) in 5-10 words each, "
            "then set same_event to true ONLY if both refer to the exact same specific incident.\n"
            'Return strict JSON: {"event_a": "<5-10 words>", "event_b": "<5-10 words>", "same_event": true or false}'
        )
        return sys_msg, user_msg

    # Node / Bridge / Seed query with Extract-Then-Assign + explicit NEW_EVENT escape
    node = q.primary_nodes[0]
    t_node = ds.texts[node].replace("\n", " ").strip()[:200]
    cand_blocks = "\n\n".join(
        _format_cluster_brief(cid, summaries[cid], ds)
        for cid in q.candidate_clusters
        if cid in summaries
    )
    cand_ids = [cid for cid in q.candidate_clusters if cid in summaries]
    user_msg = (
        f"Target Post ({ds.dates[node]}): {t_node}\n\n"
        f"Candidate Event Clusters:\n{cand_blocks}\n\n"
        f"Valid candidate cluster IDs: {cand_ids}.\n"
        "First summarize the Target Post's specific event in 4-8 words (event_cue), then choose:\n"
        "- ASSIGN (with target_cluster in valid IDs) ONLY if the post clearly refers to the exact same incident as that cluster;\n"
        "- NEW_EVENT (target_cluster: null) if it refers to a different specific event not listed;\n"
        "- NOISE (target_cluster: null) if it is personal chatter / non-event.\n"
        'Return strict JSON: {"event_cue": "<4-8 words>", "action": "ASSIGN" | "NEW_EVENT" | "NOISE", "target_cluster": <int or null>}'
    )
    return sys_msg, user_msg


def build_action_first_prompt(
    q: QueryItem,
    ds: SocialDataset,
    summaries: Dict[int, ClusterSummary],
) -> Tuple[str, str]:
    """Build Action-First direct JSON prompts (no event extraction) with identical evidence."""
    sys_msg = (
        "You decide whether social-media posts or post groups refer to the SAME specific real-world event.\n"
        "Rules:\n"
        "- SAME EVENT if both cover the same specific incident, match, announcement, or developing story, "
        "even when they highlight different details, updates, players/angles, or reactions "
        "(e.g., 'Team X beats Team Y 4-2' and 'Player Z scores for Team X against Team Y'; "
        "'Official resigns' and 'Official quits over scandal').\n"
        "- DIFFERENT EVENT if they cover different incidents, different matches, different countries, "
        "or different main subjects even if sharing a broad topic "
        "(e.g., 'Team X vs Team Y' and 'Team A vs Team B'; 'Scandal at Org 1' and 'Scandal at Org 2').\n"
        "Respond ONLY with strict JSON containing action and confidence."
    )
    if q.query_type == "merge_pair":
        s1 = "\n".join(f"  - {ds.texts[m].replace(chr(10), ' ').strip()[:140]}" for m in q.merge_c1_medoids[:3])
        s2 = "\n".join(f"  - {ds.texts[m].replace(chr(10), ' ').strip()[:140]}" for m in q.merge_c2_medoids[:3])
        c1_kw = ", ".join(summaries[q.merge_c1].top_keywords[:5]) if q.merge_c1 in summaries else ""
        c2_kw = ", ".join(summaries[q.merge_c2].top_keywords[:5]) if q.merge_c2 in summaries else ""
        user_msg = (
            f"Group A (size={summaries[q.merge_c1].size}, keywords: {c1_kw}):\n{s1}\n\n"
            f"Group B (size={summaries[q.merge_c2].size}, keywords: {c2_kw}):\n{s2}\n\n"
            "Should Group A and Group B be MERGED into the same event or kept SEPARATE?\n"
            'Return strict JSON: {"action": "MERGE" | "SEPARATE", "confidence": <float 0.0 to 1.0>}'
        )
        return sys_msg, user_msg
    if q.query_type == "seam":
        s1 = "\n".join(f"  - {ds.texts[m].replace(chr(10), ' ').strip()[:140]}" for m in q.seam_sub1_medoids[:3])
        s2 = "\n".join(f"  - {ds.texts[m].replace(chr(10), ' ').strip()[:140]}" for m in q.seam_sub2_medoids[:3])
        user_msg = (
            f"Subgroup A (size={len(q.seam_sub1)}):\n{s1}\n\n"
            f"Subgroup B (size={len(q.seam_sub2)}):\n{s2}\n\n"
            "Should this cluster be SPLIT into two separate events or KEEP_MERGED as one event?\n"
            'Return strict JSON: {"action": "SPLIT" | "KEEP_MERGED", "confidence": <float 0.0 to 1.0>}'
        )
        return sys_msg, user_msg
    return build_query_prompt(q, ds, summaries)


def build_cross_block_prompt(
    hist_ev: Any,
    cur_summ: ClusterSummary,
    cur_ds: SocialDataset,
    prompt_mode: str = "extract_compare",
) -> Tuple[str, str]:
    """Build longitudinal cross-block adjudication prompt (extract_compare or action_first)."""
    h_lines = "\n".join(f"  - {t}" for t in hist_ev.rep_texts[:3])
    c_lines = "\n".join(
        f"  - {cur_ds.texts[m].replace(chr(10), ' ').strip()[:140]}"
        for m in cur_summ.medoids[:3]
    )
    h_kw = ", ".join(hist_ev.top_keywords[:5])
    c_kw = ", ".join(cur_summ.top_keywords[:5])

    if prompt_mode == "action_first":
        sys_msg = (
            "You decide whether social-media posts or post groups refer to the SAME specific real-world event.\n"
            "Rules:\n"
            "- SAME EVENT if both cover the same specific incident, match, announcement, or developing story, "
            "even when they highlight different details, updates, players/angles, or reactions "
            "(e.g., 'Team X beats Team Y 4-2' and 'Player Z scores for Team X against Team Y'; "
            "'Official resigns' and 'Official quits over scandal').\n"
            "- DIFFERENT EVENT if they cover different incidents, different matches, different countries, "
            "or different main subjects even if sharing a broad topic "
            "(e.g., 'Team X vs Team Y' and 'Team A vs Team B'; 'Scandal at Org 1' and 'Scandal at Org 2').\n"
            "Respond ONLY with strict JSON containing action and confidence."
        )
        user_msg = (
            f"Group A (ongoing historical event, size={hist_ev.total_count}, keywords: {h_kw}):\n{h_lines}\n\n"
            f"Group B (new daily cluster, size={cur_summ.size}, keywords: {c_kw}):\n{c_lines}\n\n"
            "Should Group A and Group B be MERGED into the same event or kept SEPARATE?\n"
            'Return strict JSON: {"action": "MERGE" | "SEPARATE", "confidence": <float 0.0 to 1.0>}'
        )
        return sys_msg, user_msg

    if prompt_mode == "extract_compare":
        sys_msg = (
            "You decide whether social-media posts or post groups refer to the SAME specific real-world event.\n"
            "Rules:\n"
            "- SAME EVENT (true / ASSIGN) if both cover the same specific incident, match, announcement, or developing story, "
            "even when they highlight different details, updates, players/angles, or reactions "
            "(e.g., 'Team X beats Team Y 4-2' and 'Player Z scores for Team X against Team Y' -> true; "
            "'Official resigns' and 'Official quits over scandal' -> true).\n"
            "- DIFFERENT EVENT (false / NEW_EVENT) if they cover different incidents, different matches, different countries, "
            "or different main subjects even if sharing a broad topic "
            "(e.g., 'Team X vs Team Y' and 'Team A vs Team B' -> false; 'Scandal at Org 1' and 'Scandal at Org 2' -> false).\n"
            "Always write concise 5-10 word factual event summaries FIRST, then output your final boolean or action in strict JSON."
        )
        user_msg = (
            f"Group A (ongoing historical event, size={hist_ev.total_count}, keywords: {h_kw}):\n{h_lines}\n\n"
            f"Group B (new daily cluster, size={cur_summ.size}, keywords: {c_kw}):\n{c_lines}\n\n"
            "First summarize Group A's specific event (event_a) and Group B's specific event (event_b) in 5-10 words each, "
            "then set same_event to true ONLY if both describe the exact same specific incident.\n"
            'Return strict JSON: {"event_a": "<5-10 words>", "event_b": "<5-10 words>", "same_event": true or false}'
        )
        return sys_msg, user_msg

    raise ValueError(f"Unknown cross-block prompt_mode: {prompt_mode!r}")


def parse_llm_response_to_decision(
    q: QueryItem,
    parsed: Dict[str, Any],
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> LLMDecision:
    """Parse an Extract-Then-Compare or Action-First JSON response into an LLMDecision."""
    action = "NOOP"
    conf = 1.0
    tgt = None
    if not isinstance(parsed, dict) or "error" in parsed:
        action = "NOOP"
        conf = 0.0
    elif q.query_type == "merge_pair":
        if "same_event" in parsed:
            same = bool(parsed["same_event"])
            action = "MERGE" if same else "SEPARATE"
            conf = 1.0
        else:
            action = str(parsed.get("action", "NOOP")).upper()
            conf = float(parsed.get("confidence", 1.0))
    elif q.query_type == "seam":
        if "same_event" in parsed:
            same = bool(parsed["same_event"])
            action = "KEEP_MERGED" if same else "SPLIT"
            conf = 1.0
        else:
            action = str(parsed.get("action", "NOOP")).upper()
            conf = float(parsed.get("confidence", 1.0))
    elif q.query_type == "pair":
        if "same_event" in parsed:
            same = bool(parsed["same_event"])
            action = "MUST_LINK" if same else "CANNOT_LINK"
            conf = 1.0
        else:
            action = str(parsed.get("action", "NOOP")).upper()
            conf = float(parsed.get("confidence", 1.0))
    else:
        action = str(parsed.get("action", "NOOP")).upper()
        conf = float(parsed.get("confidence", 1.0)) if "confidence" in parsed else 1.0
        tgt = parsed.get("target_cluster", None)
        if tgt is not None:
            try:
                tgt = int(tgt)
            except (ValueError, TypeError):
                tgt = None
        if action == "ASSIGN" and tgt not in q.candidate_clusters:
            action = "NOOP"
            tgt = None
            conf = 0.0

    event_cue = str(parsed.get("event_cue", "")) if isinstance(parsed, dict) else ""
    return LLMDecision(
        query_id=q.query_id,
        query_type=q.query_type,
        action=action,
        target_cluster=tgt,
        confidence=conf,
        event_cue=event_cue,
        prompt_tokens=int(prompt_tokens),
        completion_tokens=int(completion_tokens),
    )


def _extract_json(raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


class BudgetedLLMClient:
    """OpenAI-compatible chat completions client guarded by TokenLedger & CumulativeTokenBucket."""

    def __init__(
        self,
        model: str,
        ledger: TokenLedger,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout_sec: float = 45.0,
    ) -> None:
        self.model = model
        self.ledger = ledger
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout_sec = float(timeout_sec)

    def _call_http(self, messages: List[Dict[str, str]], max_retries: int = 6) -> Tuple[str, int, int, float, str]:
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        # Explicitly disable hidden thinking monologue for Qwen3 lightweight dense models
        # so structured JSON adjudication stays fast, low-token, and strictly comparable.
        if "qwen3" in self.model.lower():
            payload["enable_thinking"] = False
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_exc: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                latency = time.time() - t0
                choice = body["choices"][0]
                content = choice["message"].get("content") or ""
                finish_reason = str(choice.get("finish_reason", ""))
                usage = body.get("usage", {})
                pt = int(usage.get("prompt_tokens", 0))
                ct = int(usage.get("completion_tokens", 0))
                return content, pt, ct, latency, finish_reason
            except urllib.error.HTTPError as he:
                last_exc = he
                if he.code in {429, 500, 502, 503, 504} and attempt < max_retries:
                    retry_after = he.headers.get("Retry-After") if he.headers else None
                    wait_s = float(retry_after) if (retry_after and retry_after.replace(".", "", 1).isdigit()) else min(32.0, 2.0 * (2**attempt))
                    time.sleep(max(1.0, wait_s))
                    continue
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    time.sleep(min(15.0, 2.0 * (attempt + 1)))
                    continue
                raise
        raise last_exc  # type: ignore[misc]

    def execute_queries(
        self,
        queries: List[QueryItem],
        ds: SocialDataset,
        summaries: Dict[int, ClusterSummary],
        block_id: int,
        bucket: Optional[Any] = None,
        prompt_fn: Any = None,
        strategy_tag: Optional[str] = None,
    ) -> List[LLMDecision]:
        """Run queries sequentially under strict cumulative real-token budget & ledger audit."""
        decisions: List[LLMDecision] = []
        pfn = prompt_fn if prompt_fn is not None else build_query_prompt

        for q in queries:
            sys_msg, user_msg = pfn(q, ds, summaries)
            strat_label = strategy_tag if strategy_tag is not None else q.strategy
            # Conservative pre-spend check (~1 token / 3.5 chars + expected completion ~100 tok)
            est_tokens = int((len(sys_msg) + len(user_msg)) / 3.5) + 120
            if bucket is not None and not bucket.can_spend(est_tokens):
                break

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
                self.ledger.log_call(
                    request_id=q.query_id,
                    block_id=block_id,
                    strategy=strat_label,
                    query_type=q.query_type,
                    model=self.model,
                    request_hash=req_hash,
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    latency_sec=0.0,
                    status="CACHED_REPLAY",
                    response_payload=parsed,
                    from_cache=True,
                )
            else:
                pt, ct, latency = 0, 0, 0.0
                try:
                    raw_text, pt, ct, latency, finish_reason = self._call_http(req_payload["messages"])
                    try:
                        parsed = _extract_json(raw_text)
                        status = "OK"
                    except Exception as parse_exc:
                        # IMPORTANT: Always retain actual HTTP usage tokens even on parse failure!
                        parsed = {
                            "action": "NOOP",
                            "confidence": 0.0,
                            "finish_reason": finish_reason,
                            "raw_snippet": raw_text[:120],
                            "error": str(parse_exc),
                        }
                        status = f"PARSE_ERROR:{type(parse_exc).__name__}"
                except Exception as http_exc:
                    parsed = {"action": "NOOP", "confidence": 0.0, "error": str(http_exc)}
                    status = f"HTTP_ERROR:{type(http_exc).__name__}"

                self.ledger.log_call(
                    request_id=q.query_id,
                    block_id=block_id,
                    strategy=strat_label,
                    query_type=q.query_type,
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

            decisions.append(
                parse_llm_response_to_decision(
                    q, parsed, prompt_tokens=pt, completion_tokens=ct
                )
            )

        return decisions
