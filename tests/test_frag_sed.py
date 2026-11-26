"""Unit tests for the simplified FRAG-SED framework (Algorithm 1)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_shared.data_loader import SocialDataset
from core_shared.token_ledger import CausalUniformBucket
from trackA_sug_sed.frag_sed import (
    CleanConstraintDSU,
    deterministic_spherical_2means,
    run_frag_sed,
)


def _make_synthetic_dataset() -> Tuple[SocialDataset, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    # 3 true events:
    # Event 0 (posts 0..11) split into 2 initial fragments: 0..5 and 6..11 (high sim ~0.72)
    # Event 1 (posts 12..19) merged with Event 2 (posts 20..23) into 1 mixed fragment (size 12 >= 8)
    d = 32
    b0 = rng.normal(size=d); b0 /= np.linalg.norm(b0)
    b1 = rng.normal(size=d); b1 /= np.linalg.norm(b1)
    b2 = rng.normal(size=d); b2 /= np.linalg.norm(b2)
    # Make b1 & b2 orthogonal-ish and b0 fragments moderately similar
    b1 = b1 - np.dot(b1, b0) * b0; b1 /= np.linalg.norm(b1)
    b2 = b2 - np.dot(b2, b0) * b0 - np.dot(b2, b1) * b1; b2 /= np.linalg.norm(b2)

    emb = np.zeros((24, d), dtype=np.float32)
    for i in range(4):
        v = b0 + 0.04 * rng.normal(size=d)
        emb[i] = v / np.linalg.norm(v)
    for i in range(4, 8):
        v = 0.88 * b0 + 0.30 * b1 + 0.04 * rng.normal(size=d)
        emb[i] = v / np.linalg.norm(v)
    for i in range(8, 12):
        v = 0.84 * b0 + 0.35 * b2 + 0.04 * rng.normal(size=d)
        emb[i] = v / np.linalg.norm(v)
    for i in range(12, 20):
        v = b1 + 0.04 * rng.normal(size=d)
        emb[i] = v / np.linalg.norm(v)
    for i in range(20, 24):
        v = b2 + 0.04 * rng.normal(size=d)
        emb[i] = v / np.linalg.norm(v)

    texts = (
        [f"Alpha earthquake tremor hits town partA {i}" for i in range(4)]
        + [f"Alpha quake rescue teams arrive partB {i}" for i in range(4, 8)]
        + [f"Alpha seismic aftershock damage report partC {i}" for i in range(8, 12)]
        + [f"Beta election debate live results {i}" for i in range(12, 20)]
        + [f"Gamma hurricane flood warning coast {i}" for i in range(20, 24)]
    )
    ds = SocialDataset(
        name="synth",
        tweet_ids=np.array([str(i) for i in range(24)], dtype=object),
        texts=texts,
        labels=np.array([0] * 12 + [1] * 8 + [2] * 4, dtype=np.int64),
        tokens_lemma=[[w.lower() for w in t.split()] for t in texts],
        entities=[["Alpha"] if i < 12 else (["Beta"] if i < 20 else ["Gamma"]) for i in range(24)],
        hashtags=[[] for _ in range(24)],
        user_ids=[f"u{i}" for i in range(24)],
        mentioned_users=[[] for _ in range(24)],
        timestamps=np.arange(24, dtype=np.float64) * 60.0,
        dates=["2012-10-18"] * 24,
    )
    labels_init = np.array([0] * 4 + [1] * 4 + [2] * 4 + [3] * 8 + [3] * 4, dtype=np.int64)
    return ds, emb, labels_init


class _MockClient:
    def __init__(self, fail_first_merge: bool = False) -> None:
        self.fail_first_merge = fail_first_merge
        self.merge_calls_seen = 0
        self.requests: list[Tuple[str, str]] = []

    def call_json(
        self,
        *,
        request_id: str,
        block_id: int,
        strategy: str,
        query_type: str,
        sys_msg: str,
        user_msg: str,
        bucket: Optional[Any] = None,
        est_completion_tokens: int = 90,
    ) -> Optional[Tuple[Dict[str, Any], int, int, float, str]]:
        est = int((len(sys_msg) + len(user_msg)) / 3.4) + est_completion_tokens
        if bucket is not None and not bucket.can_spend(est):
            return None
        self.requests.append((query_type, user_msg))
        pt = int((len(sys_msg) + len(user_msg)) / 3.4)
        ct = 12
        if bucket is not None:
            bucket.record_spend(pt, ct)

        if query_type == "frag_split":
            # Different events (Gamma vs Beta) -> empty matches [] -> split!
            if "Gamma" in user_msg and "Beta" in user_msg:
                return {"matches": []}, pt, ct, 0.01, "OK"
            return {"matches": [1]}, pt, ct, 0.01, "OK"

        # Merge verification
        self.merge_calls_seen += 1
        if self.fail_first_merge and self.merge_calls_seen == 1:
            return {"error": "simulated parse failure"}, pt, ct, 0.01, "PARSE_ERROR"

        # Match candidates that share Alpha with Anchor
        lines = user_msg.split("Candidate Fragments")[1].split("Candidate [")
        matches = []
        for idx, blk in enumerate(lines[1:], start=1):
            if "Alpha" in user_msg.split("Candidate Fragments")[0] and "Alpha" in blk:
                matches.append(idx)
        return {"matches": matches}, pt, ct, 0.01, "OK"


def test_clean_dsu_remapping_and_cl_priority() -> None:
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(5, 8))
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    dsu = CleanConstraintDSU({0: [0, 1], 1: [2], 2: [3], 3: [4]}, emb, initial_cannot_links=[(0, 2)])
    dsu.add_queried(1, 3)
    dsu.add_queried(0, 3)

    # Merge 0 and 1 -> root 0 (size 2 >= 1)
    r_new = dsu.union(0, 1)
    assert r_new == 0
    assert 1 not in dsu.active_roots
    # Symmetric pointers remapped cleanly and retired root stripped
    assert dsu.N_CL[0] == {2} and dsu.N_CL[2] == {0}
    assert dsu.N_Q[0] == {3} and dsu.N_Q[3] == {0}
    # Cannot-link blocks union(0, 2)
    assert not dsu.can_merge(0, 2)
    assert dsu.union(0, 2) is None
    assert dsu.blocked_by_cl == 1
    # Exact vector sum preserved
    np.testing.assert_allclose(dsu.comp_vec[0], emb[[0, 1, 2]].sum(axis=0))


def test_spherical_2means_deterministic() -> None:
    ds, emb, _ = _make_synthetic_dataset()
    r1, p1 = deterministic_spherical_2means(emb[12:24])
    r2, p2 = deterministic_spherical_2means(emb[12:24])
    assert np.array_equal(r1, r2) and np.array_equal(p1, p2)
    assert len(p1) == 4 and len(r1) == 8


def test_frag_sed_end_to_end_and_error_isolation() -> None:
    ds, emb, labels_init = _make_synthetic_dataset()
    client = _MockClient(fail_first_merge=True)
    bucket = CausalUniformBucket(tokens_per_block=20000, allow_carryover=False)
    bucket.start_block()

    y_hat, shown, stats = run_frag_sed(
        block_id=2,
        ds=ds,
        embeddings=emb,
        labels_init=labels_init,
        client=client,
        bucket=bucket,
        tau_lo=0.40,
        k=4,
        m_split=8,
        delta=0.10,
        rho=0.30,
    )
    assert stats["parse_errors"] == 1
    assert stats["split_accepted"] == 1
    assert stats["merge_ops"] >= 1
    # Event 0 (0..11), Event 1 (12..19), Event 2 (20..23) recovered cleanly!
    # Even if first merge failed (e.g. fragment 0+1+2 in a pack), fragment 0..11 merges cleanly on retry or remaining pairs
    assert len(set(y_hat[:12])) <= 2
    assert len(set(y_hat[12:20])) == 1
    assert len(set(y_hat[20:24])) == 1
    assert int(shown.sum()) > 0

    # Without parse error, all 3 Alpha fragments merge into 1 cluster (total 3 clusters)
    y_clean, _, st_clean = run_frag_sed(
        block_id=2,
        ds=ds,
        embeddings=emb,
        labels_init=labels_init,
        client=_MockClient(fail_first_merge=False),
        bucket=None,
        tau_lo=0.40,
    )
    assert st_clean["parse_errors"] == 0 and st_clean["merge_ops"] == 2
    assert len(set(y_clean[:12])) == 1
    assert len(set(y_clean)) == 3


def test_frag_sed_unbounded_vs_rho_cap() -> None:
    ds, emb, labels_init = _make_synthetic_dataset()
    # Unbounded (bucket=None) runs both split and merge cleanly
    y_unb, _, st_unb = run_frag_sed(
        block_id=2, ds=ds, embeddings=emb, labels_init=labels_init, client=_MockClient(), bucket=None, tau_lo=0.40
    )
    assert len(set(y_unb)) == 3 and st_unb["split_calls"] >= 1

    # Tight bucket where rho * cap < 1 prompt prevents split calls
    tight = CausalUniformBucket(tokens_per_block=400, allow_carryover=False)
    tight.start_block()
    _, _, st_tight = run_frag_sed(
        block_id=2, ds=ds, embeddings=emb, labels_init=labels_init, client=_MockClient(), bucket=tight, tau_lo=0.40, rho=0.30
    )
    assert st_tight["split_calls"] == 0


if __name__ == "__main__":
    test_clean_dsu_remapping_and_cl_priority()
    test_spherical_2means_deterministic()
    test_frag_sed_end_to_end_and_error_isolation()
    test_frag_sed_unbounded_vs_rho_cap()
    print("ALL test_frag_sed.py CHECKS PASSED!")
