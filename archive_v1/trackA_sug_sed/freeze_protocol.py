"""Freeze Track A (SUG-SED) hyperparameters, prompts, and code hashes prior to held-out testing.

Writes an immutable protocol snapshot (`trackA_sug_sed/protocol_v1.json`) with SHA-256
digests of core router, surgery, prompt, and solver modules so held-out evaluation on
Event2012 (M_06..M_21) and Event2018 (M_06..M_16) is 100% reproducible and free of
post-hoc test-set tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau-proto", type=float, default=0.68)
    parser.add_argument("--tau-lo", type=float, default=0.50)
    parser.add_argument("--tau-hi", type=float, default=0.78)
    parser.add_argument("--out", type=Path, default=ROOT / "trackA_sug_sed" / "protocol_v1.json")
    args = parser.parse_args()

    tracked_files = [
        "core_shared/data_loader.py",
        "core_shared/graph_builder.py",
        "core_shared/se_solver.py",
        "core_shared/metrics_audit.py",
        "core_shared/token_ledger.py",
        "trackA_sug_sed/high_leverage_router.py",
        "trackA_sug_sed/topology_surgery.py",
        "trackA_sug_sed/llm_client.py",
        "trackA_sug_sed/streaming_sed.py",
    ]

    protocol = {
        "protocol_version": "SUG-SED-v1.0-frozen",
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "llm": {
            "provider": "siliconflow",
            "model": "Qwen/Qwen3-8B",
            "enable_thinking": False,
            "temperature": 0.0,
            "max_tokens": 512,
            "response_format": "json_object",
            "prompt_style": "Balanced Extract-Then-Compare (event_a, event_b, same_event: bool)",
        },
        "datasets": {
            "Event2012": {
                "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
                "warmup_blocks": [0, 1],
                "dev_blocks": [2, 3, 4, 5],
                "heldout_test_blocks": list(range(6, 22)),
            },
            "Event2018": {
                "embed_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "warmup_blocks": [0, 1],
                "dev_blocks": [2, 3, 4, 5],
                "heldout_test_blocks": list(range(6, 17)),
            },
        },
        "graph_and_solver": {
            "knn_k": 15,
            "knn_sim_threshold": 0.35,
            "attr_alpha": 0.5,
            "se_solver": "solve_2d_se(refine_moves=True, max_refine_iters=10, max_outer_passes=3)",
        },
        "router_and_budget": {
            "max_queries_per_block": 120,
            "conf_threshold": 0.6,
            "sug_meso_only_ratios": {"merge": 0.65, "seam": 0.35, "bridge": 0.0, "seed": 0.0},
            "sug_router_ratios": {"merge": 0.45, "seam": 0.25, "bridge": 0.15, "seed": 0.15},
            "within_tokens_per_block": 10000,
            "cross_tokens_per_block": 3000,
            "budget_controller": "CumulativeTokenBucket(ema_alpha=0.3, min_scale=0.25, max_scale=2.0)",
        },
        "cross_block_memory": {
            "memory_window_blocks": 14,
            "min_cross_size": 3,
            "tau_proto": args.tau_proto,
            "tau_lo": args.tau_lo,
            "tau_hi": args.tau_hi,
        },
        "code_sha256": {rel: _sha256_file(ROOT / rel) for rel in tracked_files},
    }

    args.out.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Frozen] Wrote immutable protocol snapshot to {args.out}")


if __name__ == "__main__":
    main()
