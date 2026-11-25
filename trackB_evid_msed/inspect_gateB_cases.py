"""Inspect Gate B Tier 2 ledger health, confusion/gating breakdown, and concrete case studies."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = ROOT / "results" / "trackB_gateB" / "ned_dev_qwen3_vl_8b" / "tier2_vlm_audit.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    print("Statuses:", d["statuses"])
    print("Ledger Summary:", d["ledger_summary"])
    print("\nVLM Comparison:")
    for k, v in d["vlm_comparison"].items():
        print(" ", k, v)

    recs = d["records"]
    # Breakdown of VLM self-diagnosis signals under Orig-Img vs Mismatched-Img
    for cond_name, key in [("Orig-Img", "orig_img_res"), ("Mismatched-Img", "mismatched_img_res")]:
        c_disc = Counter(bool(r[key]["visual_discriminative"]) for r in recs)
        c_conf = Counter(bool(r[key]["visual_conflict_with_text"]) for r in recs)
        c_src = Counter(r[key]["evidence_source"] for r in recs)
        print(f"\nSignal breakdown [{cond_name}]: disc={dict(c_disc)}, conflict={dict(c_conf)}, src={dict(c_src)}")

    cs = d["case_studies"]
    for cat in ["rescues", "misleads", "mismatch_flips"]:
        print(f"\n=== CASE STUDY: {cat} ({len(cs[cat])} samples) ===")
        for r in cs[cat]:
            pid = r["post_id"]
            tev = r["true_event_id"]
            mev = r["mis_event_id"]
            cm = r["comp_margin_score"]
            vi = r["visual_idf_score"]
            print(f"  [{pid} | True={tev} | MisEv={mev} | comp_margin={cm:.3f} vidf={vi:.3f}]")
            print(f"    text: {r['text']!r}")
            t = r["text_only"]
            o = r["orig_img_res"]
            m = r["mismatched_img_res"]
            print(f"    TXT : pred={t['chosen_event_idx']} conf={t['confidence']:.2f} clue={t['visual_clue']!r}")
            print(f"    ORIG: pred={o['chosen_event_idx']} disc={o['visual_discriminative']} conf_txt={o['visual_conflict_with_text']} src={o['evidence_source']} conf={o['confidence']:.2f} clue={o['visual_clue']!r}")
            print(f"    MIS : pred={m['chosen_event_idx']} disc={m['visual_discriminative']} conf_txt={m['visual_conflict_with_text']} src={m['evidence_source']} conf={m['confidence']:.2f} clue={m['visual_clue']!r}")


if __name__ == "__main__":
    main()
