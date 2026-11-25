"""Freeze Track A v2 (SUG-SED Pure-Star-Meso) protocol snapshot with SHA-256 digests.

Writes an immutable protocol specification (`trackA_sug_sed/protocol_v2.json`) capturing
the Dev-locked hyperparameters, dataset splits, causal budget controller specs, and
SHA-256 hashes of all `core_shared/*.py` and `trackA_sug_sed/*.py` modules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trackA_sug_sed.run_heldout_gate2 import (
    BUDGET_PER_BLK,
    DEV_LOCKED_0TOK,
    EMBED_MODELS,
    HELDOUT_BLOCKS,
    STRATIFIED_ANON_BLOCKS,
    WARMUP_BLOCKS,
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def build_protocol_v2() -> Dict[str, object]:
    tracked_paths = sorted(
        list((ROOT / "core_shared").glob("*.py")) + list((ROOT / "trackA_sug_sed").glob("*.py")),
        key=lambda p: p.relative_to(ROOT).as_posix(),
    )
    code_sha256 = {
        p.relative_to(ROOT).as_posix(): _sha256_file(p)
        for p in tracked_paths
    }

    return {
        "protocol_version": "SUG-SED-v2.0-frozen",
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "llm": {
            "provider": "siliconflow",
            "primary_model": "Qwen/Qwen3-8B",
            "dev_cross_model": "Qwen/Qwen2.5-7B-Instruct",
            "enable_thinking": False,
            "temperature": 0.0,
            "max_tokens": 384,
            "response_format": {"type": "json_object"},
        },
        "datasets": {
            "Event2012": {
                "embed_model": EMBED_MODELS["Event2012"],
                "warmup_blocks": list(WARMUP_BLOCKS),
                "dev_blocks": [2, 3, 4, 5],
                "heldout_test_blocks": list(HELDOUT_BLOCKS["Event2012"]),
                "stratified_anon_blocks": list(STRATIFIED_ANON_BLOCKS["Event2012"]),
            },
            "Event2018": {
                "embed_model": EMBED_MODELS["Event2018"],
                "warmup_blocks": list(WARMUP_BLOCKS),
                "dev_blocks": [2, 3, 4, 5],
                "heldout_test_blocks": list(HELDOUT_BLOCKS["Event2018"]),
                "stratified_anon_blocks": list(STRATIFIED_ANON_BLOCKS["Event2018"]),
            },
        },
        "dev_locked_0tok": DEV_LOCKED_0TOK,
        "pure_star_meso_locked": {
            "knn_k": 15,
            "knn_sim_threshold": 0.50,
            "attr_alpha": 0.5,
            "refine_moves": True,
            "tau_auto": 0.85,
            "tau_lo": 0.55,
            "tier2_lo": 0.45,
            "guard_sim": 0.40,
            "min_cl_votes": 1,
            "k_nb": 4,
            "max_star_calls": 24,
            "max_seam_calls": 3,
            "min_cluster_size_seam": 8,
            "min_pocket_size_seam": 3,
            "min_sem_sep": 0.12,
            "use_seam_gate": False,
        },
        "budget_controllers": {
            "within_block": {
                "controller": "CausalUniformBucket(tokens_per_block=10000, allow_carryover=False)",
                "tokens_per_block": BUDGET_PER_BLK,
                "allow_carryover": False,
                "preflight_gate": "int((len(sys_msg) + len(user_msg)) / 3.4) + est_completion_tokens",
                "observed_max_final_call_overrun_pct_at_10k": 0.35,
            },
            "cross_day": {
                "controller": "CausalUniformBucket(tokens_per_block=6000, allow_carryover=True, max_carryover_mult=1.5)",
                "budget_per_block": 6000,
                "allow_carryover": True,
                "max_carryover_mult": 1.5,
                "warmup_blocks": list(WARMUP_BLOCKS),
                "warmup_partition_source": "0-token Pure 2D-SE Base (M_00..M_05)",
                "eval_start_block": 6,
                "window_blocks": 14,
                "tau_lo": 0.65,
                "tau_hi": 0.90,
                "tau_fb": 0.90,
                "min_cross_size": 3,
                "top_k_cand": 3,
            },
        },
        "code_sha256": code_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write immutable Track A v2 protocol snapshot.")
    parser.add_argument("--out", type=Path, default=ROOT / "trackA_sug_sed" / "protocol_v2.json")
    args = parser.parse_args()
    snapshot = build_protocol_v2()
    args.out.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[Frozen v2] Wrote protocol snapshot ({len(snapshot['code_sha256'])} files) -> {args.out}")


if __name__ == "__main__":
    main()
