"""Budgeted Multimodal VLM Client (`Qwen/Qwen3-VL-8B-Instruct`) with Real-Token Ledger.

Supports 3-condition paired causal auditing (`text_only`, `orig_img`, `mismatched_img`)
with compact JPEG resizing via `QImage` (zero PIL dependency), SHA-256 image hashing
for deduplicated deterministic request caching, and strict real input+output token accounting.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core_shared.token_ledger import CumulativeTokenBucket, TokenLedger, canonical_request_hash
from trackB_evid_msed.ned_loader import encode_image_resized_jpeg_bytes


@dataclass
class VLDecision:
    query_id: str
    condition: str               # 'text_only' | 'orig_img' | 'mismatched_img'
    chosen_slot: str             # 'A'..'D' | 'NEW' | 'ERROR'
    chosen_event_idx: Optional[int]
    visual_clue: str
    visual_discriminative: Optional[bool]
    visual_conflict_with_text: Optional[bool]
    evidence_source: str
    confidence: float
    prompt_tokens: int
    completion_tokens: int
    status: str


@dataclass
class SeamVLDecision:
    query_id: str
    query_type: str              # 'merge_pair' | 'seam'
    mode: str                    # 'text_only' | 'multimodal'
    event_a: str
    event_b: str
    visual_clue_a: str
    visual_clue_b: str
    visual_discriminative: Optional[bool]
    visual_conflict_with_text: Optional[bool]
    same_event_text_only: Optional[bool]
    same_event_multimodal: Optional[bool]
    evidence_source: str
    confidence: float
    prompt_tokens: int
    completion_tokens: int
    status: str


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


def build_vlm_candidate_prompt(
    post_text: str,
    candidate_briefs: List[Tuple[str, List[str], List[str]]],
    condition: str,
) -> Tuple[str, str]:
    """Build strictly anonymized paired event-assignment prompt.

    Args:
        post_text: Target social post text (raw text, NO event_name/year metadata).
        candidate_briefs: List of (slot_letter, top_keywords, medoid_texts) for A..D.
        condition: 'text_only' | 'orig_img' | 'mismatched_img'.
    """
    cand_lines = []
    valid_slots = []
    for slot, kws, meds in candidate_briefs:
        valid_slots.append(slot)
        kw_str = ", ".join(kws[:5])
        med_str = "\n".join(f"    - {m.replace(chr(10), ' ').strip()[:145]}" for m in meds[:3])
        cand_lines.append(f"Cluster {slot} (keywords: {kw_str}):\n{med_str}")
    cand_block = "\n\n".join(cand_lines)
    slots_str = " | ".join(f'"{s}"' for s in valid_slots + ["NEW"])
    target_txt = post_text.replace("\n", " ").strip()[:240]

    if condition == "text_only":
        sys_msg = (
            "You classify a social-media post into one of the candidate real-world events based ONLY on text evidence.\n"
            "Ignore broad generic topics if the specific incident, match, location, or participants conflict.\n"
            "If none of the candidate clusters match the specific incident, choose \"NEW\".\n"
            "Return strict JSON only."
        )
        user_msg = (
            f"Target Post Text: {target_txt}\n\n"
            f"Candidate Event Clusters:\n{cand_block}\n\n"
            "First briefly summarize the event clue from the post (text_clue, 5-10 words), then select chosen_cluster.\n"
            f'Return strict JSON: {{"text_clue": "<5-10 words>", "chosen_cluster": {slots_str}, "confidence": <float 0.0-1.0>}}'
        )
        return sys_msg, user_msg

    # orig_img or mismatched_img
    sys_msg = (
        "You classify a social-media post with its attached image into one of the candidate real-world events.\n"
        "Carefully evaluate whether the image contains event-specific visual clues (e.g., recognizable teams/uniforms, "
        "landmarks, street signs, damaged structures, banners) that distinguish among these specific candidates, "
        "or whether it is generic / uninformative / mismatched with the post text.\n"
        "Return strict JSON only."
    )
    user_msg = (
        f"Target Post Text: {target_txt}\n"
        "(An image attached to this post is provided above.)\n\n"
        f"Candidate Event Clusters:\n{cand_block}\n\n"
        "Step 1: Describe specific visual evidence visible in the image (visual_clue, 6-12 words).\n"
        "Step 2: Set visual_discriminative to true ONLY if visual details concretely help distinguish which candidate event this is "
        "(false if generic crowd/water/flames/meme/unrelated).\n"
        "Step 3: Set visual_conflict_with_text to true if the image depicts a clearly different incident/subject than the text.\n"
        "Step 4: Output chosen_cluster (" + slots_str + "), evidence_source (\"text\" | \"image\" | \"both\" | \"none\"), and confidence.\n"
        'Return strict JSON: {"visual_clue": "<6-12 words>", "visual_discriminative": true or false, '
        '"visual_conflict_with_text": true or false, "chosen_cluster": <slot>, '
        '"evidence_source": "text" | "image" | "both" | "none", "confidence": <float 0.0-1.0>}'
    )
    return sys_msg, user_msg


def _stitch_side_by_side_jpeg(
    path_a: Optional[Path],
    path_b: Optional[Path],
    max_side: int = 384,
    quality: int = 82,
) -> bytes:
    """Stitch up to two cluster images side-by-side via Qt (fallback if 2 image_urls rejected)."""
    from PySide6.QtCore import QBuffer, QIODevice, Qt
    from PySide6.QtGui import QColor, QImage, QPainter

    def _load_panel(p: Optional[Path]) -> QImage:
        if p is not None and p.exists():
            q = QImage(str(p))
            if not q.isNull():
                q = q.convertToFormat(QImage.Format.Format_RGB888)
                return q.scaled(
                    max_side,
                    max_side,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
        blank = QImage(max_side // 2, max_side // 2, QImage.Format.Format_RGB888)
        blank.fill(QColor(235, 235, 235))
        return blank

    qa = _load_panel(path_a)
    qb = _load_panel(path_b)
    gap = 12
    w_tot = qa.width() + gap + qb.width()
    h_tot = max(qa.height(), qb.height())
    canvas = QImage(w_tot, h_tot, QImage.Format.Format_RGB888)
    canvas.fill(QColor(255, 255, 255))
    painter = QPainter(canvas)
    painter.drawImage(0, (h_tot - qa.height()) // 2, qa)
    painter.drawImage(qa.width() + gap, (h_tot - qb.height()) // 2, qb)
    painter.end()

    qbuf = QBuffer()
    qbuf.open(QIODevice.OpenModeFlag.WriteOnly)
    if not canvas.save(qbuf, "JPEG", quality):
        raise RuntimeError("Failed to save stitched side-by-side JPEG")
    return bytes(qbuf.data())


def build_seam_vlm_prompt(
    brief_a: Tuple[List[str], List[str]],
    brief_b: Tuple[List[str], List[str]],
    query_type: str,
    mode: str,
    has_img_a: bool,
    has_img_b: bool,
) -> Tuple[str, str]:
    """Build strictly anonymized Extract-Then-Compare seam prompt (`Cluster A` vs `Cluster B`)."""

    def _fmt(brief: Tuple[List[str], List[str]]) -> str:
        kws, meds = brief
        kw_str = ", ".join(kws[:5]) if kws else "none"
        lines = [f"    - {m.replace(chr(10), ' ').strip()[:145]}" for m in meds[:3]]
        med_str = "\n".join(lines) if lines else "    - (empty)"
        return f"(keywords: {kw_str})\n{med_str}"

    block_a = _fmt(brief_a)
    block_b = _fmt(brief_b)
    context_note = (
        "Comparing two candidate sibling clusters from an event stream:"
        if query_type == "merge_pair"
        else "Comparing two sub-pockets inside a potentially mixed cluster:"
    )

    if mode == "text_only":
        sys_msg = (
            "You adjudicate whether two social-media message clusters describe the exact SAME real-world event "
            "or two DIFFERENT events based ONLY on text evidence.\n"
            "Ignore broad generic topic similarity (e.g., two different storms, two different games, two different protests) "
            "if the specific incident, teams, location, or participants differ.\n"
            "Return strict JSON only."
        )
        user_msg = (
            f"{context_note}\n\n"
            f"Cluster A:\n{block_a}\n\n"
            f"Cluster B:\n{block_b}\n\n"
            "Step 1: Briefly identify the specific real-world event of Cluster A (event_a, 5-10 words) "
            "and Cluster B (event_b, 5-10 words).\n"
            "Step 2: Set same_event to true ONLY if both clusters refer to the exact same specific incident/event.\n"
            'Return strict JSON: {"event_a": "<5-10 words>", "event_b": "<5-10 words>", '
            '"same_event": true or false, "confidence": <float 0.0-1.0>}'
        )
        return sys_msg, user_msg

    # multimodal mode
    img_status = (
        f"Attached images above: Cluster A image={'PRESENT' if has_img_a else 'MISSING'}, "
        f"Cluster B image={'PRESENT' if has_img_b else 'MISSING'}."
    )
    sys_msg = (
        "You adjudicate whether two social-media message clusters describe the exact SAME real-world event "
        "or two DIFFERENT events using both cluster texts and attached representative image(s).\n"
        "Be strictly vigilant against visual noise and misleading/off-topic attachments: an image might depict an "
        "unrelated meme/reaction or a DIFFERENT competing event that conflicts with a cluster's text.\n"
        "Return strict JSON only."
    )
    user_msg = (
        f"{context_note}\n{img_status}\n\n"
        f"Cluster A:\n{block_a}\n\n"
        f"Cluster B:\n{block_b}\n\n"
        "Step 1: Summarize each cluster's specific event from text: event_a (5-10 words), event_b (5-10 words).\n"
        "Step 2: Summarize concrete visual clues if present: visual_clue_a, visual_clue_b (5-10 words or \"none\").\n"
        "Step 3: Set visual_discriminative to true ONLY if the supplied image(s) provide event-specific visual "
        "clues (landmarks, damaged sites, teams/jerseys, banners, distinct scene) that concretely distinguish or confirm "
        "whether A and B are the same specific event (false if generic crowd/sky/water/meme/missing).\n"
        "Step 4: Set visual_conflict_with_text to true if EITHER representative image depicts a clearly different "
        "incident, sport/match, place, or subject than its own cluster's text (false if images are missing or consistent).\n"
        "Step 5: Output BOTH same_event_text_only (judgment from text alone) AND same_event_multimodal "
        "(judgment combining text + image(s)), plus evidence_source (\"text\"|\"image\"|\"both\"|\"none\") and confidence.\n"
        'Return strict JSON: {"event_a": "<5-10 words>", "event_b": "<5-10 words>", '
        '"visual_clue_a": "<5-10 words or none>", "visual_clue_b": "<5-10 words or none>", '
        '"visual_discriminative": true or false, "visual_conflict_with_text": true or false, '
        '"same_event_text_only": true or false, "same_event_multimodal": true or false, '
        '"evidence_source": "text" | "image" | "both" | "none", "confidence": <float 0.0-1.0>}'
    )
    return sys_msg, user_msg


class BudgetedVLMClient:
    """OpenAI-compatible multimodal VLM client (`Qwen/Qwen3-VL-8B-Instruct`)."""

    def __init__(
        self,
        model: str,
        ledger: TokenLedger,
        base_url: str,
        api_key: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        timeout_sec: float = 60.0,
    ) -> None:
        self.model = model
        self.ledger = ledger
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout_sec = float(timeout_sec)
        self._lock = threading.Lock()

    def _call_http(
        self,
        messages: List[Dict[str, Any]],
        max_retries: int = 6,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, int, int, float, str]:
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": int(max_tokens) if max_tokens is not None else self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

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
                    wait_s = (
                        float(retry_after)
                        if (retry_after and retry_after.replace(".", "", 1).isdigit())
                        else min(32.0, 2.0 * (2**attempt))
                    )
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

    def adjudicate_paired(
        self,
        query_id: str,
        condition: str,
        post_text: str,
        candidate_briefs: List[Tuple[str, int, List[str], List[str]]],  # (slot, event_idx, kws, meds)
        image_path: Optional[Path] = None,
        bucket: Optional[CumulativeTokenBucket] = None,
    ) -> VLDecision:
        slot_to_ev = {slot: ev_idx for slot, ev_idx, _, _ in candidate_briefs}
        prompt_briefs = [(slot, kws, meds) for slot, _, kws, meds in candidate_briefs]
        sys_msg, user_msg = build_vlm_candidate_prompt(post_text, prompt_briefs, condition)

        img_sha = ""
        b64_data = ""
        if condition != "text_only":
            if image_path is None:
                raise ValueError(f"Condition '{condition}' requires image_path.")
            jpeg_bytes = encode_image_resized_jpeg_bytes(image_path, max_side=512, quality=85)
            img_sha = hashlib.sha256(jpeg_bytes).hexdigest()
            b64_data = base64.b64encode(jpeg_bytes).decode("ascii")

        # Hash using text + image_sha so responses.jsonl never embeds huge base64 strings
        hash_payload = {
            "model": self.model,
            "temperature": self.temperature,
            "condition": condition,
            "system": sys_msg,
            "user": user_msg,
            "image_sha256": img_sha,
        }
        req_hash = canonical_request_hash(hash_payload)
        with self._lock:
            cached = self.ledger.lookup_cache(req_hash)

        if cached is not None:
            parsed = cached["response"]
            pt = int(cached["prompt_tokens"])
            ct = int(cached["completion_tokens"])
            status = "CACHED_REPLAY"
            with self._lock:
                self.ledger.log_call(
                    request_id=query_id,
                    block_id=0,
                    strategy=condition,
                    query_type="gateB_vlm",
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
            if condition == "text_only":
                messages: List[Dict[str, Any]] = [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg},
                ]
            else:
                messages = [
                    {"role": "system", "content": sys_msg},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}},
                            {"type": "text", "text": user_msg},
                        ],
                    },
                ]
            pt, ct, latency = 0, 0, 0.0
            try:
                raw_text, pt, ct, latency, finish_reason = self._call_http(messages)
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

            with self._lock:
                self.ledger.log_call(
                    request_id=query_id,
                    block_id=0,
                    strategy=condition,
                    query_type="gateB_vlm",
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
            with self._lock:
                bucket.record_spend(pt, ct)

        if "error" in parsed:
            return VLDecision(
                query_id=query_id,
                condition=condition,
                chosen_slot="ERROR",
                chosen_event_idx=None,
                visual_clue="",
                visual_discriminative=None,
                visual_conflict_with_text=None,
                evidence_source="none",
                confidence=0.0,
                prompt_tokens=pt,
                completion_tokens=ct,
                status=status,
            )

        slot = str(parsed.get("chosen_cluster", "NEW")).strip().upper()
        if slot.startswith("CLUSTER "):
            slot = slot.replace("CLUSTER ", "").strip()
        ev_idx = slot_to_ev.get(slot, None)
        try:
            conf = float(parsed.get("confidence", 0.5))
        except (ValueError, TypeError):
            conf = 0.5

        clue = str(parsed.get("visual_clue") or parsed.get("text_clue") or "")
        vis_disc = parsed.get("visual_discriminative")
        vis_conf = parsed.get("visual_conflict_with_text")
        return VLDecision(
            query_id=query_id,
            condition=condition,
            chosen_slot=slot,
            chosen_event_idx=ev_idx,
            visual_clue=clue,
            visual_discriminative=(None if vis_disc is None else bool(vis_disc)),
            visual_conflict_with_text=(None if vis_conf is None else bool(vis_conf)),
            evidence_source=str(parsed.get("evidence_source", "text" if condition == "text_only" else "unknown")).lower(),
            confidence=conf,
            prompt_tokens=pt,
            completion_tokens=ct,
            status=status,
        )

    @staticmethod
    def _parse_opt_bool(val: Any) -> Optional[bool]:
        if val is None:
            return None
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        s = str(val).strip().lower()
        if s in {"true", "yes", "1", "same", "merge"}:
            return True
        if s in {"false", "no", "0", "different", "separate", "split"}:
            return False
        return None

    def adjudicate_cluster_seam(
        self,
        query_id: str,
        query_type: str,
        mode: str,
        brief_a: Tuple[List[str], List[str]],
        brief_b: Tuple[List[str], List[str]],
        img_path_a: Optional[Path] = None,
        img_path_b: Optional[Path] = None,
        max_side: int = 384,
        quality: int = 82,
        bucket: Optional[CumulativeTokenBucket] = None,
    ) -> SeamVLDecision:
        """Adjudicate a cluster merge_pair or intra-cluster seam (`text_only` or `multimodal`)."""
        sha_a = ""
        sha_b = ""
        b64_a = ""
        b64_b = ""
        if mode == "multimodal":
            if img_path_a is not None:
                try:
                    jb_a = encode_image_resized_jpeg_bytes(img_path_a, max_side=max_side, quality=quality)
                    sha_a = hashlib.sha256(jb_a).hexdigest()
                    b64_a = base64.b64encode(jb_a).decode("ascii")
                except Exception:
                    sha_a, b64_a = "", ""
            if img_path_b is not None:
                try:
                    jb_b = encode_image_resized_jpeg_bytes(img_path_b, max_side=max_side, quality=quality)
                    sha_b = hashlib.sha256(jb_b).hexdigest()
                    b64_b = base64.b64encode(jb_b).decode("ascii")
                except Exception:
                    sha_b, b64_b = "", ""

        has_a = bool(b64_a)
        has_b = bool(b64_b)
        sys_msg, user_msg = build_seam_vlm_prompt(brief_a, brief_b, query_type, mode, has_a, has_b)

        hash_payload = {
            "task": "cluster_seam_v1",
            "model": self.model,
            "temperature": self.temperature,
            "query_type": query_type,
            "mode": mode,
            "system": sys_msg,
            "user": user_msg,
            "img_sha_a": sha_a,
            "img_sha_b": sha_b,
        }
        req_hash = canonical_request_hash(hash_payload)
        with self._lock:
            cached = self.ledger.lookup_cache(req_hash)

        if cached is not None:
            parsed = cached["response"]
            pt = int(cached["prompt_tokens"])
            ct = int(cached["completion_tokens"])
            status = "CACHED_REPLAY"
            with self._lock:
                self.ledger.log_call(
                    request_id=query_id,
                    block_id=0,
                    strategy=mode,
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
            if mode == "text_only" or (not has_a and not has_b):
                messages: List[Dict[str, Any]] = [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg},
                ]
            else:
                content_blocks: List[Dict[str, Any]] = []
                if has_a:
                    content_blocks.append({"type": "text", "text": "Representative Image for Cluster A:"})
                    content_blocks.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_a}"}})
                else:
                    content_blocks.append({"type": "text", "text": "(Cluster A has no representative image.)"})
                if has_b:
                    content_blocks.append({"type": "text", "text": "Representative Image for Cluster B:"})
                    content_blocks.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_b}"}})
                else:
                    content_blocks.append({"type": "text", "text": "(Cluster B has no representative image.)"})
                content_blocks.append({"type": "text", "text": user_msg})
                messages = [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": content_blocks},
                ]

            pt, ct, latency = 0, 0, 0.0
            try:
                raw_text, pt, ct, latency, finish_reason = self._call_http(messages, max_tokens=max(self.max_tokens, 320))
                try:
                    parsed = _extract_json(raw_text)
                    status = "OK"
                except Exception as parse_exc:
                    parsed = {
                        "error": str(parse_exc),
                        "finish_reason": finish_reason,
                        "raw_snippet": raw_text[:180],
                    }
                    status = f"PARSE_ERROR:{type(parse_exc).__name__}"
            except Exception as http_exc:
                parsed = {"error": str(http_exc)}
                status = f"HTTP_ERROR:{type(http_exc).__name__}"

            with self._lock:
                self.ledger.log_call(
                    request_id=query_id,
                    block_id=0,
                    strategy=mode,
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
            with self._lock:
                bucket.record_spend(pt, ct)

        if "error" in parsed:
            return SeamVLDecision(
                query_id=query_id,
                query_type=query_type,
                mode=mode,
                event_a="",
                event_b="",
                visual_clue_a="",
                visual_clue_b="",
                visual_discriminative=None,
                visual_conflict_with_text=None,
                same_event_text_only=None,
                same_event_multimodal=None,
                evidence_source="none",
                confidence=0.0,
                prompt_tokens=pt,
                completion_tokens=ct,
                status=status,
            )

        try:
            conf = float(parsed.get("confidence", 0.5))
        except (ValueError, TypeError):
            conf = 0.5

        if mode == "text_only":
            same_txt = self._parse_opt_bool(parsed.get("same_event"))
            return SeamVLDecision(
                query_id=query_id,
                query_type=query_type,
                mode=mode,
                event_a=str(parsed.get("event_a", "")),
                event_b=str(parsed.get("event_b", "")),
                visual_clue_a="",
                visual_clue_b="",
                visual_discriminative=False,
                visual_conflict_with_text=False,
                same_event_text_only=same_txt,
                same_event_multimodal=None,
                evidence_source="text",
                confidence=conf,
                prompt_tokens=pt,
                completion_tokens=ct,
                status=status,
            )

        vis_disc = self._parse_opt_bool(parsed.get("visual_discriminative"))
        vis_conf = self._parse_opt_bool(parsed.get("visual_conflict_with_text"))
        if not has_a and not has_b:
            vis_disc = False
            vis_conf = False

        same_txt = self._parse_opt_bool(parsed.get("same_event_text_only"))
        same_mm = self._parse_opt_bool(parsed.get("same_event_multimodal"))
        if same_mm is None:
            same_mm = same_txt
        if same_txt is None:
            same_txt = same_mm

        return SeamVLDecision(
            query_id=query_id,
            query_type=query_type,
            mode=mode,
            event_a=str(parsed.get("event_a", "")),
            event_b=str(parsed.get("event_b", "")),
            visual_clue_a=str(parsed.get("visual_clue_a", "")),
            visual_clue_b=str(parsed.get("visual_clue_b", "")),
            visual_discriminative=vis_disc,
            visual_conflict_with_text=vis_conf,
            same_event_text_only=same_txt,
            same_event_multimodal=same_mm,
            evidence_source=str(parsed.get("evidence_source", "both")).lower(),
            confidence=conf,
            prompt_tokens=pt,
            completion_tokens=ct,
            status=status,
        )
