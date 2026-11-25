"""Freeze immutable Track B protocol (`protocol_trackB_v1.json`) with SHA-256 verification."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent

TRACKED_FILES = [
    "core_shared/graph_builder.py",
    "core_shared/se_solver.py",
    "core_shared/metrics_audit.py",
    "core_shared/token_ledger.py",
    "trackB_evid_msed/ned_loader.py",
    "trackB_evid_msed/mv_2dse_pipeline.py",
    "trackB_evid_msed/vlm_client.py",
    "trackB_evid_msed/mm_seam_adjudicator.py",
    "trackB_evid_msed/run_phase2_ned.py",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze immutable Track B Phase 2 protocol.")
    parser.add_argument("--budget-queries", type=int, default=180)
    parser.add_argument("--merge-ratio", type=float, default=0.70)
    parser.add_argument("--conf-threshold", type=float, default=0.60)
    parser.add_argument("--primary-policy", type=str, default="evid_self_gate_inline")
    args = parser.parse_args()

    checksums: Dict[str, str] = {}
    for rel in TRACKED_FILES:
        fp = ROOT / rel
        if not fp.exists():
            raise FileNotFoundError(f"Tracked file missing: {fp}")
        checksums[rel] = _sha256(fp)

    protocol = {
        "protocol_version": "Evid-MSED-v1.0-frozen",
        "frozen_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "llm": {
            "provider": "siliconflow",
            "model": "Qwen/Qwen3-VL-8B-Instruct",
            "temperature": 0.0,
            "max_tokens": 320,
            "response_format": "json_object",
            "max_side_px": 384,
            "jpeg_quality": 82,
            "prompt_style": "Extract-Then-Compare Multimodal Cluster Seam + Explicit Conflict/Discriminability Gate",
        },
        "stage1": {
            "evidence_scorer": "visual_idf",
            "lam_v": 0.65,
            "tau_imp": 0.55,
            "gamma_imp": 0.70,
            "k_v": 10,
            "vis_sim_thresh": 0.72,
            "block_size": 1024,
            "use_soft_imputation": True,
        },
        "stage2_router": {
            "merge_ratio": float(args.merge_ratio),
            "conf_threshold": float(args.conf_threshold),
            "min_text_sim": 0.42,
            "vis_rescue_sim": 0.68,
            "vis_rescue_min_txt": 0.18,
            "vis_rescue_min_vidf": 0.50,
            "max_pairs_per_cluster": 3,
            "primary_policy": str(args.primary_policy),
        },
        "holdout": {
            "split": "holdout",
            "n_events": 24,
            "n_posts": 11955,
            "budget_queries": int(args.budget_queries),
            "prefix_steps": [60, 120, 180],
        },
        "code_sha256": checksums,
    }

    out_path = ROOT / "trackB_evid_msed" / "protocol_trackB_v1.json"
    out_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Frozen] Track B protocol saved to {out_path} ({len(checksums)} SHA-256 digests).")


if __name__ == "__main__":
    main()
