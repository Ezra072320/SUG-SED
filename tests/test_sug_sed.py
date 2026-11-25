"""Unit test suite for SUG-SED (Pure-Star-Meso, GuardedUF, Hyperbolic Ops, Anonymizer)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_shared.data_loader import SocialDataset
from core_shared.graph_builder import build_message_graph
from core_shared.metrics_audit import full_partition_audit
from core_shared.se_solver import _self_test as se_self_test, solve_2d_se
from core_shared.token_ledger import CausalUniformBucket, TokenLedger
from trackA_sug_sed.anonymizer import (
    AnonymizationAuditStore,
    QueryAnonymizer,
    QueryItem,
    audit_pre_anonymized_query,
    check_summary_reconstruction_leakage,
)
from trackA_sug_sed.baselines_llm import (
    run_automerge_only,
    run_cequel,
    run_clusterllm,
    run_fewshot_llm_post,
    run_lhg_budgeted,
    run_lsed_budgeted,
    run_meso_heur_best,
    run_ragsede_budgeted,
    run_random_llm,
)
from trackA_sug_sed.baselines_sed import run_eventx, run_hypersed, run_sbert_agglom
from trackA_sug_sed.cross_day import evaluate_cross_day_stream
from trackA_sug_sed.star_meso import (
    ClusterSummary,
    GuardedUnionFind,
    build_cluster_summaries,
    run_pure_star_meso,
)


def _make_synthetic_dataset(n_per_ev: int = 12, n_ev: int = 4) -> tuple[SocialDataset, np.ndarray]:
    rng = np.random.default_rng(7)
    n = n_per_ev * n_ev
    dim = 32
    centers = rng.normal(size=(n_ev, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    if n_ev >= 2:
        # Place event 1 at ~0.52 cosine to event 0 so hierarchy cuts [0.45..0.65] & [0.48..0.90] see ambiguity
        c1 = centers[1] - float(np.dot(centers[1], centers[0])) * centers[0]
        c1 /= max(float(np.linalg.norm(c1)), 1e-9)
        centers[1] = 0.52 * centers[0] + np.sqrt(1.0 - 0.52**2) * c1
        centers[1] /= np.linalg.norm(centers[1])
    embs = []
    labels = []
    texts = []
    entities = []
    hashtags = []
    lemmas = []
    for e in range(n_ev):
        for k in range(n_per_ev):
            v = centers[e] + 0.08 * rng.normal(size=dim).astype(np.float32)
            v /= np.linalg.norm(v)
            embs.append(v)
            labels.append(e)
            texts.append(f"Breaking incident City{e} summit #{f'tag{e}'} update {k}")
            entities.append([f"city{e}"])
            hashtags.append([f"tag{e}"])
            lemmas.append(["breaking", "incident", "summit", "update"])
    ds = SocialDataset(
        name="synthetic",
        tweet_ids=np.arange(n, dtype=np.int64),
        texts=texts,
        labels=np.array(labels, dtype=np.int64),
        tokens_lemma=lemmas,
        entities=entities,
        hashtags=hashtags,
        user_ids=[f"u{i%5}" for i in range(n)],
        mentioned_users=[[] for _ in range(n)],
        timestamps=np.arange(n, dtype=np.float64) * 60.0 + 1e9,
        dates=["2026-01-01"] * n,
    )
    return ds, np.vstack(embs)


def test_core_se_and_guarded_uf() -> None:
    se_self_test()
    # 3 unit-norm clusters: c0 near [1,0], c1 near [0.9, 0.43], c2 near [0.0, 1.0]
    cids = [0, 1, 2]
    p = np.array([[1.0, 0.0], [0.9, 0.43589], [0.0, 1.0]], dtype=np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    sim = p @ p.T
    uf = GuardedUnionFind(cids, sim, guard_sim=0.45, enforce_guards=True)
    uf.add_cannot_link(0, 1)
    assert not uf.merge(0, 1), "Cannot-link must block direct merge"
    assert not uf.merge(0, 2), "Geometric diameter guard (0.0 < 0.45) must block merge"
    assert uf.blocked_count == 2

    # Allow transitive check: 0<->1 (>=0.9) merges, then merging with 2 blocked via cannot-link or sim
    uf2 = GuardedUnionFind(cids, sim, guard_sim=0.45, enforce_guards=True)
    assert uf2.merge(0, 1)
    assert not uf2.merge(1, 2), "Diameter check across component must block snowball"

    # Centroid cohesion mode: {0, 1} can merge with {2} when component centroids have cosine >= 0.40
    # even though leaf pair (0, 2) < 0.40, while CANNOT-LINK or unrelated centroid < 0.40 remain blocked.
    p_multi = np.array(
        [
            [0.95, 0.3122, 0.0],
            [0.95, -0.3122, 0.0],
            [0.50, 0.0, 0.8660],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    p_multi /= np.linalg.norm(p_multi, axis=1, keepdims=True)
    sim_multi = p_multi @ p_multi.T
    uf_c = GuardedUnionFind([0, 1, 2, 3], sim_multi, guard_sim=0.40, enforce_guards=True, protos=p_multi, sizes=[5, 5, 4, 4])
    assert uf_c.merge(0, 1)
    assert uf_c.merge(1, 2), "Centroid cohesion (>=0.40) should permit valid multi-facet merge"
    assert not uf_c.merge(0, 3), "Centroid cohesion (<0.40) must block orthogonal/unrelated snowball"
    uf_cl = GuardedUnionFind([0, 1, 2, 3], sim_multi, guard_sim=0.40, enforce_guards=True, protos=p_multi, sizes=[5, 5, 4, 4])
    uf_cl.add_cannot_link(0, 2)
    assert not uf_cl.merge(0, 2), "Direct CANNOT-LINK must block merging (0, 2)"
    assert uf_cl.merge(0, 1)
    assert not uf_cl.merge(1, 2), "Default strict 0-violation GuardedUF must block indirect bridge {0,1} <-> {2}"

    p_cl = np.array(
        [
            [0.95, 0.31, 0.0],
            [0.95, -0.31, 0.0],
            [0.55, 0.20, 0.81],
            [0.55, -0.20, 0.81],
        ],
        dtype=np.float32,
    )
    p_cl /= np.linalg.norm(p_cl, axis=1, keepdims=True)
    sim_cl = p_cl @ p_cl.T
    uf_cl2 = GuardedUnionFind([0, 1, 2, 3], sim_cl, guard_sim=0.40, enforce_guards=True, protos=p_cl, sizes=[5, 5, 4, 4], min_cl_votes=2)
    uf_cl2.add_cannot_link(0, 2)
    assert uf_cl2.merge(0, 1)
    assert uf_cl2.merge(2, 3)
    assert uf_cl2.merge(1, 2), "With explicit min_cl_votes=2, single omission (0, 2) does not block bridge (1, 2)"

    uf_cl3 = GuardedUnionFind([0, 1, 2, 3], sim_cl, guard_sim=0.40, enforce_guards=True, protos=p_cl, sizes=[5, 5, 4, 4], min_cl_votes=2)
    uf_cl3.add_cannot_link(0, 2)
    uf_cl3.add_cannot_link(1, 3)
    assert uf_cl3.merge(0, 1)
    assert uf_cl3.merge(2, 3)
    assert not uf_cl3.merge(0, 3), "Two cross-component CANNOT-LINK rejections must block merging {0,1} and {2,3}"


def test_star_meso_and_zero_llm_baselines() -> None:
    ds, emb = _make_synthetic_dataset(n_per_ev=12, n_ev=4)
    g = build_message_graph(ds, emb, knn_k=6, knn_sim_threshold=0.50, attr_alpha=0.5)
    lab0, _ = solve_2d_se(g.adj, refine_moves=True)
    out, init_out, all_shown, stats = run_pure_star_meso(
        "synthetic",
        block_id=0,
        ds=ds,
        embeddings=emb,
        client=None,
        dry_run_oracle=True,
        graph=g,
        labels_init=lab0,
    )
    assert len(out) == len(ds)
    assert len(all_shown) == len(ds)
    assert stats["parse_errors"] == 0
    aud = full_partition_audit(ds.labels, out, y_before=init_out, queried_mask=all_shown)
    assert aud["ami"] >= 0.85

    # Baselines sanity
    lx, _ = run_eventx(ds, resolution=1.0)
    ls, _ = run_sbert_agglom(emb, distance_threshold=0.55)
    lh, _ = run_hypersed(ds, emb, graph=g, epochs=10)
    la, _ = run_automerge_only(ds, emb, lab0)
    lm, _ = run_meso_heur_best(ds, emb, g, lab0)
    for arr in (lx, ls, lh, la, lm):
        assert len(arr) == len(ds)


def test_causal_bucket_and_anonymizer() -> None:
    b = CausalUniformBucket(tokens_per_block=1000, allow_carryover=True)
    b.start_block()
    assert b.can_spend(500)
    b.record_spend(400, 100)
    cap2 = b.start_block()
    assert cap2 == 1500
    assert b.can_spend(1400)  # carryover 500 + 1000

    with tempfile.TemporaryDirectory() as tmp:
        led = TokenLedger(log_dir=Path(tmp))
        led.log_call(
            request_id="r1",
            block_id=0,
            strategy="s",
            query_type="t",
            model="m",
            request_hash="h1",
            prompt_tokens=10,
            completion_tokens=5,
            latency_sec=0.1,
            status="OK",
            response_payload={"ok": True},
            from_cache=False,
        )
        assert led.lookup_cache("h1")["prompt_tokens"] == 10

    ds, emb = _make_synthetic_dataset(n_per_ev=6, n_ev=2)
    labels = np.array([0] * 6 + [1] * 6, dtype=np.int64)
    summs = build_cluster_summaries(ds, emb, labels)
    q = QueryItem(
        query_id="m01",
        query_type="merge_pair",
        primary_nodes=[summs[0].medoids[0], summs[1].medoids[0]],
        candidate_clusters=[0, 1],
        merge_c1=0,
        merge_c2=1,
        merge_c1_medoids=summs[0].medoids[:2],
        merge_c2_medoids=summs[1].medoids[:2],
    )
    anon = QueryAnonymizer("pseudonym")
    anon_texts, _anon_summs, audit = anon.build_query_anonymized_inputs(q, ds, summs)
    assert audit["entity_mask_rate"] if "entity_mask_rate" in audit else audit["entity_masked"] <= audit["entity_total"]
    for t in anon_texts.values():
        assert "City0" not in t and "City1" not in t
    leaks = check_summary_reconstruction_leakage("Meeting at City0", "Summit", audit["masked_originals"])
    assert "city0" in leaks


def test_all_llm_baselines_and_crossday_mock() -> None:
    ds, emb = _make_synthetic_dataset(n_per_ev=10, n_ev=4)
    g = build_message_graph(ds, emb, knn_k=6, knn_sim_threshold=0.50, attr_alpha=0.5)
    lab0, _ = solve_2d_se(g.adj, refine_moves=True)

    class _MockClient:
        def call_json(self, *, query_type, bucket=None, **_kw):
            if bucket is not None:
                if not bucket.can_spend(120):
                    return None
                bucket.record_spend(100, 20)
            if query_type in {"clusterllm_pair", "pair"}:
                return {"same_event": True}, 100, 20, 0.0, "MOCK"
            if query_type in {"post_correct", "ragsede_node", "node"}:
                return {"action": "NEW_EVENT", "target_cluster": None}, 100, 20, 0.0, "MOCK"
            if query_type == "lsed_rewrite":
                return {"rewritten_event": "City summit incident update"}, 80, 15, 0.0, "MOCK"
            if query_type == "lhg_pseudo":
                return {"event_label": "City summit event"}, 80, 15, 0.0, "MOCK"
            if query_type == "cross_star":
                return {"daily_event": "City summit", "matched_id": 1, "reason": "same"}, 100, 20, 0.0, "MOCK"
            if query_type == "star_merge":
                return {"anchor_event": "x", "matches": []}, 100, 20, 0.0, "MOCK"
            if query_type == "seam":
                return {"different_events": False}, 100, 20, 0.0, "MOCK"
            return {}, 10, 5, 0.0, "MOCK"

    mc = _MockClient()
    for name, fn in [
        ("clusterllm", lambda b: run_clusterllm(0, ds, emb, lab0, mc, b)),
        ("fewshot", lambda b: run_fewshot_llm_post(0, ds, emb, g, lab0, mc, b)),
        ("random", lambda b: run_random_llm(0, ds, emb, g, lab0, mc, b)),
        ("ragsede", lambda b: run_ragsede_budgeted(0, ds, emb, g, lab0, mc, b)),
    ]:
        bk = CausalUniformBucket(tokens_per_block=1200, allow_carryover=False)
        bk.start_block()
        pred, mask, st = fn(bk)
        assert len(pred) == len(ds) and len(mask) == len(ds) and st["total_calls"] >= 1, f"{name} failed: {st}"
        assert "router_cpu_sec" in st and 0.0 <= st["router_cpu_sec"] <= st["total_wall_sec"], f"{name} bad timing: {st}"
        if name == "clusterllm":
            assert st["selected_cut_dist"] > 0.45, f"ClusterLLM collapsed to 0.45: {st}"

    bk_c = CausalUniformBucket(tokens_per_block=1200, allow_carryover=False)
    bk_c.start_block()
    p_e, p_t, m_c, st_c = run_cequel(0, ds, emb, g, lab0, mc, bk_c)
    assert len(p_e) == len(ds) and len(p_t) == len(ds) and len(m_c) == len(ds)
    assert "router_cpu_sec" in st_c and 0.0 <= st_c["router_cpu_sec"] <= st_c["total_wall_sec"]

    # Anonymized Pure-Star-Meso smoke test (pseudonym & blind modes)
    for amode in ["pseudonym", "blind"]:
        a_store = AnonymizationAuditStore()
        bk_a = CausalUniformBucket(tokens_per_block=3000, allow_carryover=False)
        bk_a.start_block()
        p_a, _, m_a, st_a = run_pure_star_meso(
            "synthetic",
            0,
            ds,
            emb,
            mc,
            bk_a,
            graph=g,
            labels_init=lab0,
            anonymizer=QueryAnonymizer(amode),
            audit_store=a_store,
        )
        assert len(p_a) == len(ds) and len(m_a) == len(ds)
        assert "anon_leak_queries" in st_a
        assert 0.0 <= st_a["router_cpu_sec"] <= st_a["total_wall_sec"]
        asum = a_store.summarize_for_dataset("synthetic")
        assert asum["entity_mask_rate"] >= 0.99 or asum["n_queries"] == 0

    # Cross-day smoke test across all modes
    stream = [(bid, ds, emb, lab0) for bid in range(4)]
    for mode in ["no_link", "proto_only", "proto_attr_only", "oracle_band", "sug_cross_star"]:
        res = evaluate_cross_day_stream(stream, mode, eval_start_block=2, client=mc, budget_per_block=1000)
        assert "g_ami" in res and "g_mac_b3" in res


def test_e2e_deid_zero_truncation_and_clusterllm_band() -> None:
    import trackA_sug_sed.run_heldout_gate2 as rg2

    # 1. Verify E2E de-identification handles > 1,500 distinct entities with zero truncation
    short_targets = ["France", "Iran", "New Jersey", "Braun-Pivet", "Pope", "Puigdemont", "Hambourg", "Jarry"]
    n_long = 1520
    texts = [
        f"Reports about {short_targets[0]} and {short_targets[1]} in {short_targets[2]} with #{short_targets[0]}",
        f"{short_targets[3]} meets {short_targets[4]} before {short_targets[5]} flies to {short_targets[6]} vs {short_targets[7]}",
        "BACK home after three meetings",
        "They said TROIS and walked back to the station",
    ]
    entities = [
        ["france", "iran", "new jersey"],
        ["braun-pivet", "pope", "puigdemont", "hambourg", "jarry"],
        [],
        [],
    ]
    hashtags = [["france"], [], [], []]
    lemmas = [
        ["reports", "france", "iran", "new", "jersey"],
        ["braun", "pivet", "meets", "pope", "puigdemont", "hambourg", "jarry"],
        ["back", "home", "three", "meetings"],
        ["said", "trois", "walked", "back", "station"],
    ]
    for k in range(n_long):
        ent_long = f"VeryLongInternationalEntityName{k:04d}"
        texts.append(f"Update from {ent_long} summit today")
        entities.append([ent_long.lower()])
        hashtags.append([])
        lemmas.append(["update", ent_long.lower(), "summit", "today"])

    n_tot = len(texts)
    ds_big = SocialDataset(
        name="synthetic_big",
        tweet_ids=np.arange(n_tot, dtype=np.int64),
        texts=texts,
        labels=np.zeros(n_tot, dtype=np.int64),
        tokens_lemma=lemmas,
        entities=entities,
        hashtags=hashtags,
        user_ids=[f"u{i%10}" for i in range(n_tot)],
        mentioned_users=[[] for _ in range(n_tot)],
        timestamps=np.arange(n_tot, dtype=np.float64) * 10.0 + 1e9,
        dates=["2026-01-01"] * n_tot,
    )

    captured_emb_texts = []
    orig_encode = rg2.encode_texts_hf
    try:
        def _fake_encode(txts, **_kw):
            captured_emb_texts.extend(txts)
            return np.ones((len(txts), 8), dtype=np.float32) / np.sqrt(8.0)

        rg2.encode_texts_hf = _fake_encode
        anon_ds, _anon_emb, meta = rg2._deidentify_block_full(ds_big, 0, "mock-model")
    finally:
        rg2.encode_texts_hf = orig_encode

    assert meta["n_ent_mapped"] >= 1520, f"Expected >= 1520 mapped entities, got {meta}"
    assert meta["block_ent_mask_rate"] == 1.0, f"Block entity leak detected: {meta}"
    for idx in (0, 1):
        low_t = anon_ds.texts[idx].lower()
        for tgt in short_targets:
            assert tgt.lower() not in low_t, f"Entity {tgt!r} leaked in E2E text: {anon_ds.texts[idx]}"
        assert "[ent_[" not in low_t, f"Corrupted nested placeholder: {anon_ds.texts[idx]}"
    assert "france" not in anon_ds.tokens_lemma[0] and "puigdemont" not in anon_ds.tokens_lemma[1]
    # Verify shouted words in one tweet do NOT bleed onto lowercase words in another tweet
    assert "back" in anon_ds.texts[3].lower(), f"Cross-post lowercase bleed on 'back': {anon_ds.texts[3]}"
    # Verify opaque anonymized entity IDs (no raw names) and predicate-only embedding inputs
    assert all(e.startswith("ent_") for e in anon_ds.entities[0]) and len(anon_ds.entities[0]) >= 3
    assert all("[ENT_" not in et for et in captured_emb_texts), "Placeholder brackets leaked into MiniLM input!"

    q_aud = audit_pre_anonymized_query("q_e2e_test", [0, 1], [], ds_big, anon_ds)
    assert q_aud["entity_total"] >= 7 and q_aud["entity_masked"] == q_aud["entity_total"], f"Bad E2E audit: {q_aud}"
    assert q_aud["hashtag_total"] == 1 and q_aud["hashtag_masked"] == 1

    # 2. Verify ClusterLLM picks exact transition granularity 0.55 when true threshold is 0.55
    rng = np.random.default_rng(123)
    dim = 32
    base = rng.normal(size=dim).astype(np.float32)
    base /= np.linalg.norm(base)
    orth1 = rng.normal(size=dim).astype(np.float32)
    orth1 -= float(np.dot(orth1, base)) * base
    orth1 /= np.linalg.norm(orth1)
    orth2 = rng.normal(size=dim).astype(np.float32)
    orth2 -= float(np.dot(orth2, base)) * base + float(np.dot(orth2, orth1)) * orth1
    orth2 /= np.linalg.norm(orth2)
    # c0=base, c1 at cos=0.52 (dist=0.48), c2 at cos=0.38 (dist=0.62)
    centers = [
        base,
        0.52 * base + np.sqrt(1.0 - 0.52**2) * orth1,
        0.38 * base + np.sqrt(1.0 - 0.38**2) * orth2,
    ]
    c_embs, c_labs, c_txts = [], [], []
    for e_id, ctr in enumerate(centers):
        for k in range(5):
            v = ctr + 0.01 * rng.normal(size=dim).astype(np.float32)
            v /= np.linalg.norm(v)
            c_embs.append(v)
            c_labs.append(0 if e_id in (0, 1) else 1)
            c_txts.append(f"Event cluster {e_id} post {k}")
    ds_c = SocialDataset(
        name="syn_cllm",
        tweet_ids=np.arange(15, dtype=np.int64),
        texts=c_txts,
        labels=np.array(c_labs, dtype=np.int64),
        tokens_lemma=[["event", "post"] for _ in range(15)],
        entities=[[] for _ in range(15)],
        hashtags=[[] for _ in range(15)],
        user_ids=[f"u{i}" for i in range(15)],
        mentioned_users=[[] for _ in range(15)],
        timestamps=np.arange(15, dtype=np.float64) + 1e9,
        dates=["2026-01-01"] * 15,
    )
    emb_c = np.vstack(c_embs)
    lab0_c = np.arange(15, dtype=np.int64)

    class _OraclePairClient:
        def call_json(self, *, request_id="", bucket=None, **_kw):
            # request_id format: cllm_M00_u{u}_v{v}
            parts = request_id.split("_")
            u = int(parts[-2][1:])
            v = int(parts[-1][1:])
            same = bool(ds_c.labels[u] == ds_c.labels[v])
            if bucket is not None:
                if not bucket.can_spend(100):
                    return None
                bucket.record_spend(80, 20)
            return {"same_event": same}, 80, 20, 0.01, "MOCK"

    bk_cl = CausalUniformBucket(tokens_per_block=5000, allow_carryover=False)
    bk_cl.start_block()
    _, _, st_cl = run_clusterllm(0, ds_c, emb_c, lab0_c, _OraclePairClient(), bk_cl)
    assert st_cl["selected_cut_dist"] in (0.50, 0.55), f"Expected 0.50 or 0.55, got {st_cl['selected_cut_dist']}"


def test_priority2_clusterllm_imbalance_and_dynamic_cannot_link() -> None:
    rng = np.random.default_rng(2026)
    dim = 32
    # 1. Negative-skewed verdicts (~20% positive) where d=0.45 has higher raw accuracy
    # (0.80) than d=0.55 (0.70), while d=0.55 has strictly higher balanced accuracy.
    base = rng.normal(size=dim).astype(np.float32)
    base /= np.linalg.norm(base)
    c_embs, c_labs, c_txts = [], [], []
    # Pair clusters: event 0 splits at cosine 0.52 (dist 0.48), plus 4 well-separated events
    for eid in range(5):
        if eid == 0:
            orth = rng.normal(size=dim).astype(np.float32)
            orth -= float(np.dot(orth, base)) * base
            orth /= max(float(np.linalg.norm(orth)), 1e-9)
            sub_ctrs = [base, 0.52 * base + np.sqrt(1.0 - 0.52**2) * orth]
        else:
            v = rng.normal(size=dim).astype(np.float32)
            for _ in range(3):
                v -= float(np.dot(v, base)) * base
            v /= max(float(np.linalg.norm(v)), 1e-9)
            sub_ctrs = [v]
        for s_idx, ctr in enumerate(sub_ctrs):
            for k in range(4):
                x = ctr + 0.01 * rng.normal(size=dim).astype(np.float32)
                x /= np.linalg.norm(x)
                c_embs.append(x)
                c_labs.append(eid)
                c_txts.append(f"SkewEvent {eid}_{s_idx} post {k}")

    n_posts = len(c_txts)
    ds_sk = SocialDataset(
        name="syn_skew",
        tweet_ids=np.arange(n_posts, dtype=np.int64),
        texts=c_txts,
        labels=np.array(c_labs, dtype=np.int64),
        tokens_lemma=[["event", "post"] for _ in range(n_posts)],
        entities=[[] for _ in range(n_posts)],
        hashtags=[[] for _ in range(n_posts)],
        user_ids=[f"u{i}" for i in range(n_posts)],
        mentioned_users=[[] for _ in range(n_posts)],
        timestamps=np.arange(n_posts, dtype=np.float64) + 1e9,
        dates=["2026-01-01"] * n_posts,
    )
    emb_sk = np.vstack(c_embs)
    lab0_sk = np.arange(n_posts, dtype=np.int64)

    class _SkewClient:
        # Inject ~20% positives (2 same, 8 different) with both true positives at dist 0.48
        def __init__(self) -> None:
            self.calls = 0

        def call_json(self, *, request_id="", bucket=None, **_kw):
            parts = request_id.split("_")
            u = int(parts[-2][1:])
            v = int(parts[-1][1:])
            same = bool(ds_sk.labels[u] == ds_sk.labels[v])
            if bucket is not None:
                if not bucket.can_spend(100):
                    return None
                bucket.record_spend(80, 20)
            self.calls += 1
            return {"same_event": same}, 80, 20, 0.01, "MOCK"

    bk_sk = CausalUniformBucket(tokens_per_block=5000, allow_carryover=False)
    bk_sk.start_block()
    _, _, st_sk = run_clusterllm(0, ds_sk, emb_sk, lab0_sk, _SkewClient(), bk_sk)
    assert st_sk["selected_cut_dist"] > 0.45, f"ClusterLLM collapsed under negative skew: {st_sk}"

    # 2. Dynamic CANNOT-LINK guard: 3 clusters A, B, C at pairwise cosine ~0.58 (dist ~0.42 < 0.45):
    # Test directly with dists at ~0.52 so cut 0.45 separates all three, then (A,B)=True, (B,C)=True, (A,C)=False.
    # Under static pre-merge cannot_c, {(A,B)} merged to root A, then (B,C) could fail to see (A,C) if B!=root;
    # dynamic component check must keep A and C in distinct final clusters.
    orth_a = rng.normal(size=dim).astype(np.float32)
    orth_a -= float(np.dot(orth_a, base)) * base
    orth_a /= np.linalg.norm(orth_a)
    orth_b = rng.normal(size=dim).astype(np.float32)
    orth_b -= float(np.dot(orth_b, base)) * base + float(np.dot(orth_b, orth_a)) * orth_a
    orth_b /= np.linalg.norm(orth_b)
    orth_c = rng.normal(size=dim).astype(np.float32)
    orth_c -= (
        float(np.dot(orth_c, base)) * base
        + float(np.dot(orth_c, orth_a)) * orth_a
        + float(np.dot(orth_c, orth_b)) * orth_b
    )
    orth_c /= np.linalg.norm(orth_c)
    tri_ctrs = [
        0.70 * base + np.sqrt(1.0 - 0.70**2) * orth_a,
        0.70 * base + np.sqrt(1.0 - 0.70**2) * orth_b,
        0.70 * base + np.sqrt(1.0 - 0.70**2) * orth_c,
    ]  # pairwise cosine = 0.49 (dist = 0.51)
    t_embs, t_txts = [], []
    for cid_t, ctr in enumerate(tri_ctrs):
        for k in range(3):
            x = ctr + 0.01 * rng.normal(size=dim).astype(np.float32)
            x /= np.linalg.norm(x)
            t_embs.append(x)
            t_txts.append(f"Tri {cid_t} post {k}")
    ds_tri = SocialDataset(
        name="syn_tri",
        tweet_ids=np.arange(9, dtype=np.int64),
        texts=t_txts,
        labels=np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64),
        tokens_lemma=[["tri", "post"] for _ in range(9)],
        entities=[[] for _ in range(9)],
        hashtags=[[] for _ in range(9)],
        user_ids=[f"u{i}" for i in range(9)],
        mentioned_users=[[] for _ in range(9)],
        timestamps=np.arange(9, dtype=np.float64) + 1e9,
        dates=["2026-01-01"] * 9,
    )
    emb_tri = np.vstack(t_embs)
    lab0_tri = np.arange(9, dtype=np.int64)

    class _TriClient:
        def call_json(self, *, request_id="", bucket=None, **_kw):
            parts = request_id.split("_")
            u = int(parts[-2][1:])
            v = int(parts[-1][1:])
            cu, cv = int(ds_tri.labels[u]), int(ds_tri.labels[v])
            pair = {cu, cv}
            same = (pair == {0, 1}) or (pair == {1, 2})  # (0,1)=True, (1,2)=True, (0,2)=False
            if bucket is not None:
                if not bucket.can_spend(100):
                    return None
                bucket.record_spend(80, 20)
            return {"same_event": same}, 80, 20, 0.01, "MOCK"

    bk_tri = CausalUniformBucket(tokens_per_block=5000, allow_carryover=False)
    bk_tri.start_block()
    pred_tri, _, _ = run_clusterllm(0, ds_tri, emb_tri, lab0_tri, _TriClient(), bk_tri)
    assert pred_tri[0] != pred_tri[6], f"Dynamic CANNOT-LINK failed to separate A and C: {pred_tri}"


def test_priority1_audit_and_anonymizer_fixes() -> None:
    import trackA_sug_sed.run_heldout_gate2 as rg2

    # 1. star_merge anonymizer must only anonymize & audit shown medoids[:2], NOT medoids[2]
    ds, emb = _make_synthetic_dataset(n_per_ev=4, n_ev=2)
    ds.texts[2] = "HiddenThirdMedoidEntity report about City0 update"
    ds.entities[2] = ["hiddenthirdmedoidentity", "city0"]
    labels = np.array([0] * 4 + [1] * 4, dtype=np.int64)
    summs = build_cluster_summaries(ds, emb, labels, max_medoids=3)
    summs[0].medoids = [0, 1, 2]
    summs[1].medoids = [4, 5, 6]
    q_star = QueryItem(
        query_id="star_test_0",
        query_type="star_merge",
        primary_nodes=[0, 1, 4, 5],
        candidate_clusters=[0, 1],
    )
    anon_texts, _anon_summs, audit = QueryAnonymizer("pseudonym").build_query_anonymized_inputs(q_star, ds, summs)
    assert set(anon_texts.keys()) == {0, 1, 4, 5}, f"Unexpected medoid keys: {anon_texts.keys()}"
    assert "hiddenthirdmedoidentity" not in audit["masked_originals"], f"Unshown medoid[2] leaked into audit: {audit}"

    # 2. Cross-day sug_cross_star must track parse_errors when client returns {"error": ...}
    class _ErrCrossClient:
        def __init__(self) -> None:
            self.called = False

        def call_json(self, *, query_type, bucket=None, **_kw):
            if query_type == "cross_star":
                if not self.called:
                    self.called = True
                    return {"error": "bad json"}, 50, 10, 0.01, "PARSE_ERROR"
                return {"matched_id": None}, 50, 10, 0.01, "OK"
            return {}, 10, 5, 0.0, "MOCK"

    # Rotate block 2..3 prototypes into the ambiguity band [0.65, 0.90)
    rng_cd = np.random.default_rng(99)
    orth_dir = rng_cd.normal(size=(1, emb.shape[1])).astype(np.float32)
    orth_dir /= max(float(np.linalg.norm(orth_dir)), 1e-12)
    emb_rot = 0.75 * emb + np.sqrt(1.0 - 0.75**2) * orth_dir
    emb_rot /= np.maximum(np.linalg.norm(emb_rot, axis=1, keepdims=True), 1e-12)
    stream = [(0, ds, emb, labels), (1, ds, emb, labels), (2, ds, emb_rot, labels), (3, ds, emb_rot, labels)]
    cd_res = evaluate_cross_day_stream(
        stream, "sug_cross_star", eval_start_block=2, tau_lo=0.65, tau_hi=0.95, tau_fb=0.90,
        min_cross_size=2, client=_ErrCrossClient(), budget_per_block=5000,
    )
    assert cd_res["stats"]["parse_errors"] == 1, f"Expected parse_errors=1, got {cd_res['stats']}"

    # 3. compute_wilcoxon_holm must return exact non-zero scientific-notation p-values on 27 blocks
    b_en = list(range(6, 22))
    b_fr = list(range(6, 17))
    def _mk_res(bids, offset):
        return {
            "per_block": {
                f"M_{b:02d}": {
                    "ami": 80.0 + offset + 0.01 * b,
                    "macro_b3": 70.0 + offset + 0.02 * b,
                    "micro_b3": 75.0 + offset + 0.03 * b,
                    "tail_b3": 50.0 + offset + 0.04 * b,
                }
                for b in bids
            }
        }
    ours_ent = [("Event2012", _mk_res(b_en, 5.0), b_en), ("Event2018", _mk_res(b_fr, 5.0), b_fr)]
    base_map = {"DummyBase": [("Event2012", _mk_res(b_en, 0.0), b_en), ("Event2018", _mk_res(b_fr, 0.0), b_fr)]}
    wh = rg2.compute_wilcoxon_holm(ours_ent, base_map)
    for m in ("ami", "macro_b3", "micro_b3", "tail_b3"):
        assert 0.0 < wh["DummyBase"][m]["p_raw"] < 1e-6, f"Expected exact tiny >0 p_raw, got {wh['DummyBase'][m]}"
        assert 0.0 < wh["DummyBase"][m]["p_holm"] < 1e-6, f"Expected exact tiny >0 p_holm, got {wh['DummyBase'][m]}"


def test_trackA_audit_bugfixes_and_refinements() -> None:
    """Regression tests for Stage-4 live refresh, Stage-3 Seam, Stage-5 Cross-Day, caches, and loaders."""
    import difflib
    import re
    import warnings
    from core_shared.data_loader import load_socialed_npy, split_daily_blocks
    from core_shared.partition_cache import load_or_solve_2d_se
    import trackA_sug_sed.embed_cache as ec_mod
    from trackA_sug_sed.llm_client import BudgetedLLMClient
    from trackA_sug_sed.star_meso import _parse_bool, detect_clean_seams

    # 1. _parse_bool must handle string "false"/"0"/"no" safely
    assert _parse_bool("false") is False
    assert _parse_bool("FALSE") is False
    assert _parse_bool("0") is False
    assert _parse_bool("no") is False
    assert _parse_bool("true") is True
    assert _parse_bool("1") is True
    assert _parse_bool(False) is False
    assert _parse_bool(True) is True

    # 2. Stage-4 live component refresh: 6x12 clusters at ~0.72 cosine
    rng = np.random.default_rng(314)
    dim = 32
    shared = rng.normal(size=dim).astype(np.float32)
    shared /= np.linalg.norm(shared)
    embs_6, labs_6, txts_6, ents_6, tags_6, lems_6 = [], [], [], [], [], []
    init_6 = []
    for c in range(6):
        orth = rng.normal(size=dim).astype(np.float32)
        orth -= float(np.dot(orth, shared)) * shared
        orth /= max(float(np.linalg.norm(orth)), 1e-9)
        ctr = np.sqrt(0.72) * shared + np.sqrt(1.0 - 0.72) * orth
        ctr /= np.linalg.norm(ctr)
        for k in range(12):
            v = ctr + 0.01 * rng.normal(size=dim).astype(np.float32)
            v /= np.linalg.norm(v)
            embs_6.append(v)
            labs_6.append(0)
            init_6.append(c)
            txts_6.append(f"LiveRefresh event part {c} post {k}")
            ents_6.append([f"part{c}"])
            tags_6.append([f"tag{c}"])
            lems_6.append(["liverefresh", "event", f"part{c}"])
    ds_6 = SocialDataset(
        name="syn_6x12",
        tweet_ids=np.arange(72, dtype=np.int64),
        texts=txts_6,
        labels=np.array(labs_6, dtype=np.int64),
        tokens_lemma=lems_6,
        entities=ents_6,
        hashtags=tags_6,
        user_ids=[f"u{i}" for i in range(72)],
        mentioned_users=[[] for _ in range(72)],
        timestamps=np.arange(72, dtype=np.float64) + 1e9,
        dates=["2026-01-01"] * 72,
    )
    emb_6 = np.vstack(embs_6)
    lab_6 = np.array(init_6, dtype=np.int64)
    g_6 = build_message_graph(ds_6, emb_6, knn_k=5, knn_sim_threshold=0.50, attr_alpha=0.5, compute_non_entity_tfidf=False)

    anchor_sizes_seen = []
    cand_sizes_seen = []

    class _StepMergeAllClient:
        def call_json(self, *, query_type, user_msg="", bucket=None, **_kw):
            if query_type == "star_merge":
                m_anc = re.search(r"Anchor Cluster \(size=(\d+)", user_msg)
                assert m_anc is not None, user_msg
                anchor_sizes_seen.append(int(m_anc.group(1)))
                c_szs = [int(x) for x in re.findall(r"Candidate \[\d+\] \(size=(\d+)", user_msg)]
                cand_sizes_seen.append(c_szs)
                return {
                    "anchor_event": "merged",
                    "matches": [{"id": i, "shared_event": "same"} for i in range(1, len(c_szs) + 1)],
                }, 100, 20, 0.0, "MOCK"
            return {"different_events": False}, 50, 10, 0.0, "MOCK"

    pred_6, _, _, st_6 = run_pure_star_meso(
        "syn_6x12", 0, ds_6, emb_6, _StepMergeAllClient(), graph=g_6, labels_init=lab_6, max_seam_calls=0
    )
    assert anchor_sizes_seen[0] == 12 and cand_sizes_seen[0] == [12, 12, 12, 12]
    assert sorted([anchor_sizes_seen[1], cand_sizes_seen[1][0]]) == [12, 60], (anchor_sizes_seen, cand_sizes_seen)
    assert len(set(pred_6.tolist())) == 1, f"Expected all 6x12 merged into 1 component, got {len(set(pred_6.tolist()))}"
    assert st_6["star_calls"] == 2, f"Expected 2 star calls (4+1), got {st_6['star_calls']}"

    # Also verify sequential 12 -> 24 -> 36 -> 48 -> 60 live size refresh with k_nb=1
    anchor_sizes_seen.clear()
    cand_sizes_seen.clear()
    pred_6_seq, _, _, st_6_seq = run_pure_star_meso(
        "syn_6x12", 0, ds_6, emb_6, _StepMergeAllClient(), graph=g_6, labels_init=lab_6, max_seam_calls=0, k_nb=1
    )
    seen_all_seq = set(anchor_sizes_seen) | {c[0] for c in cand_sizes_seen}
    assert {12, 24, 36, 48, 60}.issubset(seen_all_seq), f"Expected live sizes {{12,24,36,48,60}} in {seen_all_seq}"
    assert len(set(pred_6_seq.tolist())) == 1 and st_6_seq["star_calls"] == 5

    # 3. w/o Star Packing uses build_star_merge_prompt (preserves size= and keywords:)
    prompts_k1 = []

    class _K1CaptureClient:
        def call_json(self, *, query_type, user_msg="", **_kw):
            if query_type in {"star_merge", "merge_pair"}:
                prompts_k1.append(user_msg)
                return {"anchor_event": "ev", "matches": []}, 80, 15, 0.0, "MOCK"
            return {"different_events": False}, 50, 10, 0.0, "MOCK"

    run_pure_star_meso(
        "syn_6x12", 0, ds_6, emb_6, _K1CaptureClient(), graph=g_6, labels_init=lab_6,
        max_seam_calls=0, use_star_packing=False,
    )
    assert prompts_k1 and "Anchor Cluster (size=" in prompts_k1[0] and "keywords:" in prompts_k1[0], prompts_k1[0]

    # 4. Decoupled use_auto_merge & use_bayes_band: use_bayes_band=False must NOT disable AutoMerge
    ctr_hi = shared.copy()
    orth_hi = rng.normal(size=dim).astype(np.float32)
    orth_hi -= float(np.dot(orth_hi, ctr_hi)) * ctr_hi
    orth_hi /= np.linalg.norm(orth_hi)
    ctr_hi2 = 0.90 * ctr_hi + np.sqrt(1.0 - 0.90**2) * orth_hi
    emb_hi = np.vstack([ctr_hi] * 4 + [ctr_hi2] * 4).astype(np.float32)
    ds_hi = SocialDataset(
        name="syn_decouple",
        tweet_ids=np.arange(8, dtype=np.int64),
        texts=[f"AutoMerge test {i}" for i in range(8)],
        labels=np.zeros(8, dtype=np.int64),
        tokens_lemma=[["automerge", "test"] for _ in range(8)],
        entities=[[] for _ in range(8)],
        hashtags=[[] for _ in range(8)],
        user_ids=[f"u{i}" for i in range(8)],
        mentioned_users=[[] for _ in range(8)],
        timestamps=np.arange(8, dtype=np.float64) + 1e9,
        dates=["2026-01-01"] * 8,
    )
    g_hi = build_message_graph(ds_hi, emb_hi, knn_k=3, knn_sim_threshold=0.50, compute_non_entity_tfidf=False)
    lab_hi = np.array([0] * 4 + [1] * 4, dtype=np.int64)
    _, _, _, st_dec = run_pure_star_meso(
        "syn_decouple", 0, ds_hi, emb_hi, None, dry_run_oracle=True, graph=g_hi, labels_init=lab_hi,
        use_auto_merge=True, use_bayes_band=False, max_seam_calls=0, max_star_calls=0,
    )
    assert st_dec["auto_merge_ops"] == 1, f"use_bayes_band=False wrongly disabled AutoMerge: {st_dec}"

    # 5. use_purity_base mismatch warning when pre-built graph uses 0.50
    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")
        run_pure_star_meso(
            "syn_decouple", 0, ds_hi, emb_hi, None, dry_run_oracle=True, graph=g_hi, labels_init=lab_hi,
            use_purity_base=False, max_seam_calls=0, max_star_calls=0,
        )
        assert any("overrides use_purity_base" in str(w.message) for w in wrec)

    # 6. Stage-3 Seam: minority exact-RT pocket (low cross_sim), high cross_sim RT burst filtered, balanced 4+4 split, B=2k budget cap
    b_main = shared.copy()
    o_min = rng.normal(size=dim).astype(np.float32)
    o_min -= float(np.dot(o_min, b_main)) * b_main
    o_min /= np.linalg.norm(o_min)
    b_min = 0.40 * b_main + np.sqrt(1.0 - 0.40**2) * o_min
    b_rt_same = 0.78 * b_main + np.sqrt(1.0 - 0.78**2) * o_min  # cross_sim > 0.65
    # Case 6a: 20 main + 4 identical minority (cross_sim ~ 0.40 < 0.65) -> MUST detect seam
    emb_s1 = np.vstack([b_main + 0.03 * rng.normal(size=(20, dim)).astype(np.float32), np.tile(b_min, (4, 1))])
    emb_s1 /= np.linalg.norm(emb_s1, axis=1, keepdims=True)
    ds_s1 = SocialDataset(
        name="s1", tweet_ids=np.arange(24, dtype=np.int64), texts=[f"t{i}" for i in range(24)],
        labels=np.zeros(24, dtype=np.int64), tokens_lemma=[["tok"] for _ in range(24)],
        entities=[[] for _ in range(24)], hashtags=[[] for _ in range(24)],
        user_ids=["u"] * 24, mentioned_users=[[] for _ in range(24)],
        timestamps=np.arange(24, dtype=np.float64) + 1e9, dates=["2026-01-01"] * 24,
    )
    g_s1 = build_message_graph(ds_s1, emb_s1, knn_k=5, knn_sim_threshold=0.50, compute_non_entity_tfidf=False)
    seams_s1 = detect_clean_seams(g_s1, build_cluster_summaries(ds_s1, emb_s1, np.zeros(24, dtype=np.int64)))
    assert len(seams_s1) == 1, f"Minority RT pocket with low cross_sim should be detected: {seams_s1}"

    # Case 6b: 20 main + 4 identical same-event RT burst (cross_sim ~ 0.78 > 0.65, intra > 0.96) -> MUST be filtered
    emb_s2 = np.vstack([b_main + 0.03 * rng.normal(size=(20, dim)).astype(np.float32), np.tile(b_rt_same, (4, 1))])
    emb_s2 /= np.linalg.norm(emb_s2, axis=1, keepdims=True)
    g_s2 = build_message_graph(ds_s2 := ds_s1, emb_s2, knn_k=5, knn_sim_threshold=0.50, compute_non_entity_tfidf=False)
    seams_s2 = detect_clean_seams(g_s2, build_cluster_summaries(ds_s2, emb_s2, np.zeros(24, dtype=np.int64)))
    assert len(seams_s2) == 0, f"Same-event RT burst should be filtered: {seams_s2}"

    # Case 6c: balanced 4+4 split in size-8 cluster -> detected via k=2 spherical k-means
    emb_s3 = np.vstack([
        b_main + 0.02 * rng.normal(size=(4, dim)).astype(np.float32),
        b_min + 0.02 * rng.normal(size=(4, dim)).astype(np.float32),
    ])
    emb_s3 /= np.linalg.norm(emb_s3, axis=1, keepdims=True)
    ds_s3 = ds_s1.subset(list(range(8)))
    g_s3 = build_message_graph(ds_s3, emb_s3, knn_k=4, knn_sim_threshold=0.35, compute_non_entity_tfidf=False)
    seams_s3 = detect_clean_seams(g_s3, build_cluster_summaries(ds_s3, emb_s3, np.zeros(8, dtype=np.int64)))
    assert len(seams_s3) == 1 and len(seams_s3[0].sub2) == 4, f"Balanced 4+4 seam missed: {seams_s3}"

    # Case 6d: B=2000 seam cap preserves budget for Stage-4 Star-Merge
    bk_2k = CausalUniformBucket(tokens_per_block=2000, allow_carryover=False)
    bk_2k.start_block()
    # Build 1 block with 3 large seamable clusters + 2 star-mergeable clusters
    class _BudgetTestClient:
        def call_json(self, *, query_type, sys_msg="", user_msg="", bucket=None, est_completion_tokens=90, **_kw):
            est = int((len(sys_msg) + len(user_msg)) / 3.4) + est_completion_tokens
            if bucket is not None:
                if not bucket.can_spend(est):
                    return None
                bucket.record_spend(500, 80)
            if query_type == "seam":
                return {"event_a": "a", "event_b": "b", "different_events": True}, 500, 80, 0.0, "MOCK"
            return {"anchor_event": "a", "matches": []}, 500, 80, 0.0, "MOCK"

    emb_b2k = np.vstack([emb_s1, emb_6[:24]])
    emb_b2k /= np.linalg.norm(emb_b2k, axis=1, keepdims=True)
    ds_b2k = SocialDataset(
        name="b2k", tweet_ids=np.arange(48, dtype=np.int64), texts=[f"Post text {i} " * 20 for i in range(48)],
        labels=np.zeros(48, dtype=np.int64), tokens_lemma=[["post", "text"] for _ in range(48)],
        entities=[[] for _ in range(48)], hashtags=[[] for _ in range(48)],
        user_ids=["u"] * 48, mentioned_users=[[] for _ in range(48)],
        timestamps=np.arange(48, dtype=np.float64) + 1e9, dates=["2026-01-01"] * 48,
    )
    lab_b2k = np.array([0] * 24 + [1] * 12 + [2] * 12, dtype=np.int64)
    g_b2k = build_message_graph(ds_b2k, emb_b2k, knn_k=5, knn_sim_threshold=0.50, compute_non_entity_tfidf=False)
    _, _, _, st_2k = run_pure_star_meso(
        "b2k", 0, ds_b2k, emb_b2k, _BudgetTestClient(), bucket=bk_2k, graph=g_b2k, labels_init=lab_b2k
    )
    assert st_2k["seam_calls"] <= 1 and st_2k["star_calls"] >= 1, f"Seam starved Star-Merge at 2k: {st_2k}"

    # 7. Stage-5 Cross-Day: block uniqueness, CANNOT_LINK block, window eviction, and empty eval guard
    class _AlwaysMatch1Client:
        def call_json(self, *, query_type, **_kw):
            return {"daily_event": "x", "matched_id": 1, "reason": "same"}, 50, 10, 0.0, "MOCK"

    # Block 0 has 1 cluster (ev 0); Block 2 has 2 distinct clusters at ~0.75 sim to ev 0
    orth_cd = rng.normal(size=dim).astype(np.float32)
    orth_cd -= float(np.dot(orth_cd, b_main)) * b_main
    orth_cd /= np.linalg.norm(orth_cd)
    b_cd = 0.75 * b_main + np.sqrt(1.0 - 0.75**2) * orth_cd
    emb_cd0 = np.tile(b_main, (6, 1)).astype(np.float32)
    emb_cd2 = np.vstack([np.tile(b_cd, (4, 1)), np.tile(b_cd, (4, 1))]).astype(np.float32)
    ds_cd0 = ds_s1.subset(list(range(6)))
    ds_cd0.labels = np.zeros(6, dtype=np.int64)
    ds_cd2 = ds_s1.subset(list(range(8)))
    ds_cd2.labels = np.array([0] * 4 + [1] * 4, dtype=np.int64)
    stream_cd = [
        (0, ds_cd0, emb_cd0, np.zeros(6, dtype=np.int64)),
        (2, ds_cd2, emb_cd2, np.array([0] * 4 + [1] * 4, dtype=np.int64)),
    ]
    res_cd = evaluate_cross_day_stream(
        stream_cd, "sug_cross_star", eval_start_block=2, client=_AlwaysMatch1Client(), enforce_block_uniqueness=True
    )
    assert res_cd["n_pred_events"] == 2, f"Cross-day silently merged 2 same-block clusters: {res_cd}"
    assert res_cd["stats"]["blocked_same_block_collisions"] >= 1

    # Empty eval guard
    res_empty = evaluate_cross_day_stream(stream_cd, "no_link", eval_start_block=99)
    assert res_empty["n_pred_events"] == 0 and res_empty.get("warning") == "no_eval_blocks"

    # Window eviction check
    from trackA_sug_sed.cross_day import GlobalEventMemory
    mem_ev = GlobalEventMemory(window_blocks=2)
    s_tmp = build_cluster_summaries(ds_cd0, emb_cd0, np.zeros(6, dtype=np.int64))[0]
    mem_ev.spawn_event(0, s_tmp, ds_cd0)
    assert len(mem_ev.events) == 1
    evicted = mem_ev.evict_expired(5)
    assert evicted == 1 and len(mem_ev.events) == 0

    # 8. llm_client cache key completeness + USAGE_MISSING tag
    with tempfile.TemporaryDirectory() as tmp:
        led_c = TokenLedger(log_dir=Path(tmp))
        calls_http = []

        class _FakeHTTPClient(BudgetedLLMClient):
            def _call_http(self, messages, max_retries=10, use_json_mode=None, max_tokens_override=None):
                calls_http.append((max_tokens_override, use_json_mode))
                return '{"ok": true}', 0, 0, 0.01, "stop", True

        cl1 = _FakeHTTPClient("Qwen/Qwen3-8B", led_c, max_tokens=384, use_json_mode=True)
        r1 = cl1.call_json("r1", 0, "s", "t", "sys", "usr")
        assert r1 is not None and "USAGE_MISSING" in r1[4] and r1[0].get("usage_missing") is True
        cl2 = _FakeHTTPClient("Qwen/Qwen3-8B", led_c, max_tokens=128, use_json_mode=True)
        r2 = cl2.call_json("r2", 0, "s", "t", "sys", "usr")
        assert r2 is not None and len(calls_http) == 2, f"Different max_tokens wrongly hit cache: {calls_http}"

    # 9. embed_cache bit-identical first-run vs cached reload
    with tempfile.TemporaryDirectory() as tmp:
        orig_enc = ec_mod.encode_texts_hf
        try:
            ec_mod.encode_texts_hf = lambda txts, **_kw: np.full((len(txts), 8), 0.12345678, dtype=np.float32)
            e_first = ec_mod.load_or_extract_block_embeddings("syn_ec", 0, ["a", "b"], cache_dir=Path(tmp))
            e_second = ec_mod.load_or_extract_block_embeddings("syn_ec", 0, ["a", "b"], cache_dir=Path(tmp))
            assert np.array_equal(e_first, e_second) and e_first.dtype == np.float32
        finally:
            ec_mod.encode_texts_hf = orig_enc

    # 10. partition_cache isolates different embed_model / modified graphs
    with tempfile.TemporaryDirectory() as tmp:
        adj_a = sparse.csr_matrix(np.array([
            [0.0, 0.9, 0.0, 0.0],
            [0.9, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.9],
            [0.0, 0.0, 0.9, 0.0],
        ]))
        adj_b = sparse.csr_matrix(np.array([
            [0.0, 0.0, 0.9, 0.0],
            [0.0, 0.0, 0.0, 0.9],
            [0.9, 0.0, 0.0, 0.0],
            [0.0, 0.9, 0.0, 0.0],
        ]))
        la, _ = load_or_solve_2d_se("syn_pc", 0, adj_a, cache_dir=Path(tmp), embed_model="modelA")
        lb, _ = load_or_solve_2d_se("syn_pc", 0, adj_b, cache_dir=Path(tmp), embed_model="modelB")
        assert not np.array_equal(la, lb), (la, lb)
        assert len(list(Path(tmp).glob("*.npz"))) == 2

    # 11. data_loader NaT -> np.nan & unknown date isolation + graph_builder large bucket (>max_bucket_size) retention
    with tempfile.TemporaryDirectory() as tmp:
        arr_npy = np.empty((4, 11), dtype=object)
        for i in range(4):
            arr_npy[i] = [i, f"txt {i}", 0, "0", ["tok"], [("ent", "ORG")], f"u{i}",
                          "2026-01-01 00:00:00" if i < 3 else None, "d", ["tag"], []]
        np.save(Path(tmp) / "MockDS.npy", arr_npy, allow_pickle=True)
        ds_nat = load_socialed_npy("MockDS", dataset_dir=Path(tmp), sort_chronologically=True)
        assert np.isnan(ds_nat.timestamps[-1]) and ds_nat.dates[-1] == "unknown"
        blks = split_daily_blocks(ds_nat, merge_initial_days=1, min_block_size=1)
        assert blks[-1][0] == "unknown" and len(blks[-1][1]) == 1

    # graph_builder: bucket size > max_attr_bucket_size must still create edges + compute_non_entity_tfidf=True works
    emb_big = np.tile(b_main, (10, 1)).astype(np.float32)
    ds_big_bkt = ds_s1.subset(list(range(10)))
    ds_big_bkt.entities = [["hubent"] for _ in range(10)]
    g_big_bkt = build_message_graph(ds_big_bkt, emb_big, knn_k=1, knn_sim_threshold=1.1, max_attr_bucket_size=4, compute_non_entity_tfidf=True)
    assert g_big_bkt.attr_adj.nnz > 0, "Large attribute bucket (>max_attr_bucket_size) was wrongly dropped"
    assert g_big_bkt.non_entity_tfidf.shape[0] == 10

    # 12. Verify sug_sed_method is synchronized with core_shared + trackA_sug_sed
    sync_pairs = [
        ("sug_sed_method/core/data_loader.py", "core_shared/data_loader.py"),
        ("sug_sed_method/core/graph_builder.py", "core_shared/graph_builder.py"),
        ("sug_sed_method/core/se_solver.py", "core_shared/se_solver.py"),
        ("sug_sed_method/core/partition_cache.py", "core_shared/partition_cache.py"),
        ("sug_sed_method/core/token_ledger.py", "core_shared/token_ledger.py"),
        ("sug_sed_method/core/metrics_audit.py", "core_shared/metrics_audit.py"),
        ("sug_sed_method/core/key_loader.py", "core_shared/key_loader.py"),
        ("sug_sed_method/method/embed_cache.py", "trackA_sug_sed/embed_cache.py"),
        ("sug_sed_method/method/llm_client.py", "trackA_sug_sed/llm_client.py"),
        ("sug_sed_method/method/anonymizer.py", "trackA_sug_sed/anonymizer.py"),
        ("sug_sed_method/method/star_meso.py", "trackA_sug_sed/star_meso.py"),
        ("sug_sed_method/method/cross_day.py", "trackA_sug_sed/cross_day.py"),
    ]
    for a_p, b_p in sync_pairs:
        la = (ROOT / a_p).read_text(encoding="utf-8").splitlines()
        lb = (ROOT / b_p).read_text(encoding="utf-8").splitlines()
        diff_cnt = sum(1 for ln in difflib.unified_diff(lb, la, n=0) if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---")))
        assert diff_cnt <= 30, f"Unexpected divergence between {a_p} and {b_p}: {diff_cnt} lines"


def test_trackA_round2_audit_fixes() -> None:
    """Regression tests for round-2 audit findings A1-A4, B1-B5, and low-severity fixes."""
    import time as _time
    import warnings
    from core_shared.data_loader import split_daily_blocks
    from core_shared.key_loader import OPEN_SOURCE_PROVIDERS
    import trackA_sug_sed.embed_cache as ec_mod
    from trackA_sug_sed.anonymizer import make_anonymized_prompt_fn
    from trackA_sug_sed.llm_client import BudgetedLLMClient
    from trackA_sug_sed.star_meso import detect_clean_seams

    rng = np.random.default_rng(2027)
    dim = 32
    u0 = rng.normal(size=dim).astype(np.float32); u0 /= np.linalg.norm(u0)
    u1 = rng.normal(size=dim).astype(np.float32); u1 -= float(np.dot(u1, u0)) * u0; u1 /= np.linalg.norm(u1)
    u2 = rng.normal(size=dim).astype(np.float32); u2 -= float(np.dot(u2, u0)) * u0 + float(np.dot(u2, u1)) * u1; u2 /= np.linalg.norm(u2)

    # A1: Seam-Split cannot be silently re-merged by Stage-4 Star-Merge
    v_main = u0
    v_pock = 0.65 * u0 + np.sqrt(1.0 - 0.65**2) * u1
    emb_a1 = np.vstack([
        v_main + 0.02 * rng.normal(size=(12, dim)).astype(np.float32),
        v_pock + 0.02 * rng.normal(size=(4, dim)).astype(np.float32),
    ])
    emb_a1 /= np.linalg.norm(emb_a1, axis=1, keepdims=True)
    ds_a1 = SocialDataset(
        name="a1", tweet_ids=np.arange(16, dtype=np.int64), texts=[f"post {i}" for i in range(16)],
        labels=np.array([0] * 12 + [1] * 4, dtype=np.int64), tokens_lemma=[["post"] for _ in range(16)],
        entities=[[] for _ in range(16)], hashtags=[[] for _ in range(16)],
        user_ids=["u"] * 16, mentioned_users=[[] for _ in range(16)],
        timestamps=np.arange(16, dtype=np.float64) + 1e9, dates=["2026-01-01"] * 16,
    )
    g_a1 = build_message_graph(ds_a1, emb_a1, knn_k=6, knn_sim_threshold=0.50, compute_non_entity_tfidf=False)

    class _A1ConflictClient:
        def call_json(self, *, query_type, **_kw):
            if query_type == "seam":
                return {"event_a": "main", "event_b": "pock", "different_events": True}, 60, 15, 0.0, "MOCK"
            # Even if Star-Merge tries to re-merge them, Seam CANNOT-LINK must block it!
            return {"anchor_event": "x", "matches": [{"id": 1, "shared_event": "re-merge"}]}, 60, 15, 0.0, "MOCK"

    pred_a1, _, _, st_a1 = run_pure_star_meso(
        "a1", 0, ds_a1, emb_a1, _A1ConflictClient(), graph=g_a1, labels_init=np.zeros(16, dtype=np.int64)
    )
    assert st_a1["seam_splits"] == 1 and st_a1["star_merges"] == 0, st_a1
    assert len(set(pred_a1.tolist())) == 2 and len(st_a1["intra_cannot_link_pairs"]) >= 1, st_a1

    # A2: AutoMerge 2-hop chain drift (A.B=0.86, B.C=0.86, A.C=0.479) must block second merge
    vec_b = u0
    vec_a = 0.86 * u0 + np.sqrt(1.0 - 0.86**2) * u1
    # Make vec_c = 0.86 * vec_b + y * u1 + z * u2 such that dot(vec_a, vec_c) = 0.479
    y_c = (0.479 - 0.86 * 0.86) / float(np.sqrt(1.0 - 0.86**2))
    z_c = float(np.sqrt(max(0.0, 1.0 - 0.86**2 - y_c**2)))
    vec_c = 0.86 * u0 + y_c * u1 + z_c * u2
    emb_a2 = np.vstack([np.tile(vec_a, (4, 1)), np.tile(vec_b, (4, 1)), np.tile(vec_c, (4, 1))]).astype(np.float32)
    ds_a2 = ds_a1.subset(list(range(12)))
    ds_a2.labels = np.array([0] * 4 + [1] * 4 + [2] * 4, dtype=np.int64)
    g_a2 = build_message_graph(ds_a2, emb_a2, knn_k=3, knn_sim_threshold=0.50, compute_non_entity_tfidf=False)
    pred_a2, _, _, st_a2 = run_pure_star_meso(
        "a2", 0, ds_a2, emb_a2, None, dry_run_oracle=True, graph=g_a2,
        labels_init=np.array([0] * 4 + [1] * 4 + [2] * 4, dtype=np.int64),
        max_seam_calls=0, max_star_calls=0,
    )
    assert st_a2["auto_merge_ops"] == 1 and st_a2["blocked_by_guard"] >= 1, st_a2
    assert len(set(pred_a2.tolist())) == 2

    # A3: Out-of-range / 0-based / malformed matches must increment parse_errors and NOT add CANNOT_LINK
    emb_a3 = np.vstack([np.tile(u0, (4, 1)), np.tile(0.70 * u0 + np.sqrt(1.0 - 0.70**2) * u1, (4, 1))]).astype(np.float32)
    ds_a3 = ds_a1.subset(list(range(8)))
    ds_a3.labels = np.zeros(8, dtype=np.int64)
    g_a3 = build_message_graph(ds_a3, emb_a3, knn_k=3, knn_sim_threshold=0.50, compute_non_entity_tfidf=False)
    lab_a3 = np.array([0] * 4 + [1] * 4, dtype=np.int64)

    for bad_resp in [
        {"anchor_event": "x", "matches": [{"id": 9}]},
        {"anchor_event": "x", "matches": [{"id": 0}]},
        {"anchor_event": "x", "matches": "bad"},
        {"anchor_event": "x"},
    ]:
        class _BadMatchClient:
            def call_json(self, **_kw):
                return bad_resp, 50, 10, 0.0, "OK"

        p_bad, _, _, st_bad = run_pure_star_meso(
            "a3", 0, ds_a3, emb_a3, _BadMatchClient(), graph=g_a3, labels_init=lab_a3,
            max_seam_calls=0, max_star_calls=1,
        )
        assert st_bad["parse_errors"] == 1, (bad_resp, st_bad)
        assert st_bad["star_merges"] == 0 and st_bad["intra_cannot_link_pairs"] == [], (bad_resp, st_bad)

    class _StrDigitMatchClient:
        def call_json(self, **_kw):
            return {"anchor_event": "x", "matches": ["1"]}, 50, 10, 0.0, "OK"

    p_str, _, _, st_str = run_pure_star_meso(
        "a3", 0, ds_a3, emb_a3, _StrDigitMatchClient(), graph=g_a3, labels_init=lab_a3,
        max_seam_calls=0, max_star_calls=1,
    )
    assert st_str["parse_errors"] == 0 and st_str["star_merges"] == 1 and len(set(p_str.tolist())) == 1

    # A4 & B3 & B4: Deterministic hash regardless of budget spent, error responses not cached, usage_missing ledger tracking
    with tempfile.TemporaryDirectory() as tmp:
        led = TokenLedger(log_dir=Path(tmp))
        http_calls = []

        class _MockBudgetClient(BudgetedLLMClient):
            def _call_http(self, messages, max_retries=10, use_json_mode=None, max_tokens_override=None):
                http_calls.append(max_tokens_override)
                if len(http_calls) == 1:
                    return '{"anchor_event": "ok", "matches": []}', 0, 0, 0.01, "stop", True
                return "not valid json", 120, 15, 0.01, "stop", False

        cl_b = _MockBudgetClient("Qwen/Qwen3-8B", led, max_tokens=384)
        bk_a = CausalUniformBucket(tokens_per_block=2000, allow_carryover=False)
        bk_a.start_block()
        r_first = cl_b.call_json("q1", 0, "s", "star_merge", "sys", "usr_same", bucket=bk_a, est_completion_tokens=90)
        assert r_first is not None and "USAGE_MISSING" in r_first[4]
        assert http_calls[0] == 384

        # Second call with DIFFERENT spent budget must hit the exact same cache hash!
        bk_b = CausalUniformBucket(tokens_per_block=2000, allow_carryover=False)
        bk_b.start_block()
        bk_b.record_spend(1000, 0)
        r_replay = cl_b.call_json("q2", 0, "s", "star_merge", "sys", "usr_same", bucket=bk_b, est_completion_tokens=90)
        assert r_replay is not None and r_replay[4] == "CACHED_REPLAY|USAGE_MISSING"
        assert len(http_calls) == 1, "Budget spent should not alter cache hash"
        assert led.summary()["estimated_calls"] == 2 and led.summary()["estimated_tokens"] > 0

        # B3: Parse error with non-zero tokens must NOT be cached in responses.jsonl
        r_err = cl_b.call_json("q_err", 0, "s", "star_merge", "sys", "usr_err", bucket=bk_a, est_completion_tokens=90)
        assert r_err is not None and "error" in r_err[0]
        r_err_again = cl_b.call_json("q_err2", 0, "s", "star_merge", "sys", "usr_err", bucket=bk_a, est_completion_tokens=90)
        assert len(http_calls) == 3, "PARSE_ERROR with non-zero tokens must not be cached"

    # B1: Unnormalized embeddings (scaled x3) must not cause false-positive seams or distorted graph
    g_norm = build_message_graph(ds_a1, emb_a1, knn_k=5, knn_sim_threshold=0.50, compute_non_entity_tfidf=False)
    g_x3 = build_message_graph(ds_a1, emb_a1 * 3.0, knn_k=5, knn_sim_threshold=0.50, compute_non_entity_tfidf=False)
    assert np.allclose(g_norm.embeddings, g_x3.embeddings, atol=1e-6)
    s_norm = detect_clean_seams(g_norm, build_cluster_summaries(ds_a1, emb_a1, np.zeros(16, dtype=np.int64)))
    s_x3 = detect_clean_seams(g_x3, build_cluster_summaries(ds_a1, emb_a1 * 3.0, np.zeros(16, dtype=np.int64)))
    assert len(s_norm) == len(s_x3)
    if s_norm:
        assert abs(s_norm[0].sem_sep - s_x3[0].sem_sep) < 1e-5

    # B2: O(1) GuardedUnionFind.can_merge on two 500-leaf components + protos length check
    n_big = 1000
    cids_big = list(range(n_big))
    p_big = np.tile(u0, (n_big, 1)).astype(np.float32)
    uf_big = GuardedUnionFind(cids_big, np.zeros((1, 1)), guard_sim=0.40, protos=p_big, sizes=[1] * n_big)
    for k in range(1, 500):
        assert uf_big.merge(0, k)
    for k in range(501, 1000):
        assert uf_big.merge(500, k)
    uf_big.add_cannot_link(10, 510)
    t0_uf = _time.perf_counter()
    for _ in range(100):
        assert not uf_big.can_merge(0, 500)
    assert (_time.perf_counter() - t0_uf) < 0.1
    try:
        GuardedUnionFind([0, 1], np.eye(2), protos=np.ones((3, 2)))
        assert False, "Expected ValueError for mismatched protos length"
    except ValueError:
        pass

    # B5: embed_cache isolates same-length different texts + warns on legacy float16 mismatch
    with tempfile.TemporaryDirectory() as tmp:
        orig_enc = ec_mod.encode_texts_hf
        try:
            ec_mod.encode_texts_hf = lambda txts, **_kw: np.ones((len(txts), 8), dtype=np.float32) * (1.0 if txts[0] == "a" else 2.0)
            ea = ec_mod.load_or_extract_block_embeddings("b5", 0, ["a", "b"], cache_dir=Path(tmp))
            eb = ec_mod.load_or_extract_block_embeddings("b5", 0, ["c", "d"], cache_dir=Path(tmp))
            assert not np.allclose(ea, eb), "Same-length different texts wrongly reused embed cache"
            np.save(Path(tmp) / "leg_M00_sentence-transformers__all-MiniLM-L6-v2.npy", np.ones((2, 8), dtype=np.float16))
            with warnings.catch_warnings(record=True) as w_leg:
                warnings.simplefilter("always")
                ec_mod.load_or_extract_block_embeddings("leg", 0, ["x", "y"], cache_dir=Path(tmp), storage_dtype="float32")
                assert any("legacy float16" in str(w.message) for w in w_leg)
        finally:
            ec_mod.encode_texts_hf = orig_enc

    # Low-severity checks: key_loader deepseek-chat, cross_day out-of-range matched_id & budget_skipped,
    # anonymizer star_merge prompt builder, data_loader drop_unknown_dates
    assert OPEN_SOURCE_PROVIDERS["deepseek"][1] == "deepseek-chat"
    pfn = make_anonymized_prompt_fn("pseudonym")
    q_sm = QueryItem(
        query_id="sm0", query_type="star_merge", primary_nodes=[0, 4], candidate_clusters=[0, 1]
    )
    summs_a3 = build_cluster_summaries(ds_a3, emb_a3, lab_a3)
    _, um_sm = pfn(q_sm, ds_a3, summs_a3)
    assert "Anchor Cluster" in um_sm and "Candidate [1]" in um_sm

    class _BadCrossClient:
        def call_json(self, *, bucket=None, **_kw):
            if bucket is not None and not bucket.can_spend(10):
                return None
            return {"daily_event": "x", "matched_id": 99, "reason": "oob"}, 40, 10, 0.0, "OK"

    ds_cd_a = ds_a3.subset(list(range(4)))
    ds_cd_b = ds_a3.subset(list(range(4, 8)))
    emb_cd_a = emb_a3[:4]
    emb_cd_b = 0.75 * emb_a3[:4] + np.sqrt(1.0 - 0.75**2) * np.tile(u1, (4, 1))
    emb_cd_b /= np.linalg.norm(emb_cd_b, axis=1, keepdims=True)
    cd_oob = evaluate_cross_day_stream(
        [(0, ds_cd_a, emb_cd_a, np.zeros(4, dtype=np.int64)), (2, ds_cd_b, emb_cd_b, np.zeros(4, dtype=np.int64))],
        "sug_cross_star", eval_start_block=2, min_cross_size=2, client=_BadCrossClient(), budget_per_block=5000,
    )
    assert cd_oob["stats"]["parse_errors"] == 1 and cd_oob["stats"]["llm_linked"] == 0

    cd_bud0 = evaluate_cross_day_stream(
        [(0, ds_cd_a, emb_cd_a, np.zeros(4, dtype=np.int64)), (2, ds_cd_b, emb_cd_b, np.zeros(4, dtype=np.int64))],
        "sug_cross_star", eval_start_block=2, min_cross_size=2, client=_BadCrossClient(), budget_per_block=0,
    )
    assert cd_bud0["stats"]["budget_skipped"] == 1 and cd_bud0["stats"]["llm_queried"] == 0

    ds_unk = ds_a3.subset(list(range(4)))
    ds_unk.dates = ["2026-01-01", "2026-01-01", "unknown", "unknown"]
    assert len(split_daily_blocks(ds_unk, min_block_size=1, drop_unknown_dates=True)) == 1


if __name__ == "__main__":
    test_core_se_and_guarded_uf()
    test_star_meso_and_zero_llm_baselines()
    test_causal_bucket_and_anonymizer()
    test_all_llm_baselines_and_crossday_mock()
    test_e2e_deid_zero_truncation_and_clusterllm_band()
    test_priority1_audit_and_anonymizer_fixes()
    test_priority2_clusterllm_imbalance_and_dynamic_cannot_link()
    test_trackA_audit_bugfixes_and_refinements()
    test_trackA_round2_audit_fixes()
    print("ALL SUG-SED UNIT TESTS PASSED cleanly.")
