"""Unit tests verifying O(N) metrics_audit, exact 2D SE solver, token buckets, and surgery masks."""

from __future__ import annotations

import shutil
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_shared.data_loader import DEFAULT_DATASET_DIR
from core_shared.graph_builder import MessageGraph
from core_shared.metrics_audit import (
    event_macro_bcubed,
    item_bcubed_scores,
    pairwise_error_transition,
)
from core_shared.partition_cache import load_or_solve_2d_se
from core_shared.se_solver import _self_test as se_self_test
from core_shared.se_solver import solve_2d_se
from core_shared.token_ledger import (
    CausalSlidingBucket,
    CausalUniformBucket,
    CumulativeTokenBucket,
    TokenLedger,
)
from trackA_sug_sed.high_leverage_router import ClusterSummary, QueryItem
from trackA_sug_sed.llm_client import build_action_first_prompt, build_cross_block_prompt
from trackA_sug_sed.run_ablations_qwen3 import _build_action_first_prompt
from trackA_sug_sed.topology_surgery import (
    LLMDecision,
    apply_cequel_transitive_assignment,
    apply_direct_assignment,
    apply_topology_repair_and_recluster,
)


def _bruteforce_bcubed(y_true: np.ndarray, y_pred: np.ndarray):
    n = len(y_true)
    p = np.zeros(n)
    r = np.zeros(n)
    for i in range(n):
        same_c = y_pred == y_pred[i]
        same_y = y_true == y_true[i]
        both = same_c & same_y
        p[i] = both.sum() / same_c.sum()
        r[i] = both.sum() / same_y.sum()
    return p, r


def _bruteforce_pair_transition(y_true: np.ndarray, y_b: np.ndarray, y_a: np.ndarray):
    n = len(y_true)
    res_fm = int_fm = res_fs = int_fs = 0
    for i in range(n):
        for j in range(i + 1, n):
            same_y = y_true[i] == y_true[j]
            same_b = y_b[i] == y_b[j]
            same_a = y_a[i] == y_a[j]
            if not same_y:
                if same_b and not same_a:
                    res_fm += 1
                elif not same_b and same_a:
                    int_fm += 1
            else:
                if not same_b and same_a:
                    res_fs += 1
                elif same_b and not same_a:
                    int_fs += 1
    return res_fm, int_fm, res_fs, int_fs


def test_metrics_against_bruteforce() -> None:
    rng = np.random.default_rng(123)
    for _ in range(20):
        n = 60
        yt = rng.integers(0, 8, size=n)
        yb = rng.integers(0, 9, size=n)
        ya = rng.integers(0, 9, size=n)

        p_fast, r_fast, _ = item_bcubed_scores(yt, ya)
        p_bf, r_bf = _bruteforce_bcubed(yt, ya)
        assert np.allclose(p_fast, p_bf, atol=1e-12)
        assert np.allclose(r_fast, r_bf, atol=1e-12)

        tr = pairwise_error_transition(yt, yb, ya)
        rfm, ifm, rfs, ifs = _bruteforce_pair_transition(yt, yb, ya)
        assert tr["resolved_false_merges"] == rfm
        assert tr["introduced_false_merges"] == ifm
        assert tr["resolved_false_splits"] == rfs
        assert tr["introduced_false_splits"] == ifs

        mb = event_macro_bcubed(yt, ya)
        assert 0.0 <= mb["macro_b3_f1"] <= 1.0 + 1e-12
    print("[metrics_audit brute-force check PASSED]")


def test_cumulative_token_bucket() -> None:
    bucket = CumulativeTokenBucket(total_token_budget=10000, num_blocks=5)
    weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    uncertainties = [1.0, 2.5, 0.5, 3.0, 1.0]
    for w, u in zip(weights, uncertainties):
        cap = bucket.start_block(w, u)
        assert cap >= 0
        # Spend greedily in 400-token chunks
        while bucket.can_spend(400):
            bucket.record_spend(340, 60)
        assert bucket.cumulative_spent <= bucket.cumulative_allowance + 1e-6
    assert bucket.cumulative_spent <= 10000
    print(f"[CumulativeTokenBucket test PASSED] spent={bucket.cumulative_spent}/10000")


def test_data_loader_path() -> None:
    assert DEFAULT_DATASET_DIR.exists(), f"Missing dataset dir: {DEFAULT_DATASET_DIR}"
    assert (DEFAULT_DATASET_DIR / "Event2012.npy").exists()
    assert (DEFAULT_DATASET_DIR / "Event2018.npy").exists()
    print(f"[data_loader_path test PASSED] dir={DEFAULT_DATASET_DIR}")


def test_causal_token_buckets() -> None:
    u_fixed = CausalUniformBucket(tokens_per_block=1000, allow_carryover=False)
    for w in [0.05, 0.90, 0.25]:
        cap = u_fixed.start_block(block_weight=w, mean_uncertainty=5.0)
        assert cap == 1000
        u_fixed.record_spend(400, 100)
    assert u_fixed.cumulative_spent == 1500

    u_carry = CausalUniformBucket(
        tokens_per_block=1000, total_token_budget=3000, allow_carryover=True, max_carryover_mult=1.5
    )
    assert u_carry.start_block() == 1000
    u_carry.record_spend(500, 100)  # unused 400 carries over
    assert u_carry.start_block() == 1400
    assert not u_carry.can_spend(1401)
    u_carry.record_spend(1200, 200)
    assert u_carry.start_block() == 1000

    sliding = CausalSlidingBucket(tokens_per_block=1000, min_scale=0.5, max_scale=1.5)
    sliding.observe_warmup_block(1000)
    sliding.observe_warmup_block(1000)
    cap1 = sliding.start_block(n_posts=500)
    assert cap1 == 500
    sliding.record_spend(250, 50)
    cap2 = sliding.start_block(n_posts=2000)
    assert cap2 > 1000 and cap2 <= 1500
    print("[CausalTokenBuckets test PASSED]")


def test_prompt_shown_queried_mask() -> None:
    n = 20
    labels_init = np.array([0] * 10 + [1] * 10, dtype=np.int64)
    rng = np.random.default_rng(7)
    emb = rng.normal(size=(n, 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    adj = sparse.eye(n, format="csr", dtype=np.float64)
    adj = sparse.csr_matrix(adj - sparse.eye(n))
    # Build simple ring edges so topology repair can construct non-trivial CSR
    rows, cols, vals = [], [], []
    for i in range(n - 1):
        rows.extend([i, i + 1])
        cols.extend([i + 1, i])
        vals.extend([0.8, 0.8])
    adj = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)
    non_ent = sparse.csr_matrix((n, 2), dtype=np.float64)
    graph = MessageGraph(adj=adj, sem_adj=adj, attr_adj=sparse.csr_matrix((n, n)), embeddings=emb, non_entity_tfidf=non_ent)

    summaries = {
        0: ClusterSummary(cluster_id=0, members=list(range(0, 10)), size=10, prototype=emb[0], medoids=[0, 1, 2], top_keywords=[], mean_timestamp=0.0),
        1: ClusterSummary(cluster_id=1, members=list(range(10, 20)), size=10, prototype=emb[10], medoids=[10, 11, 12], top_keywords=[], mean_timestamp=0.0),
    }
    queries = [
        QueryItem(
            query_id="m0", strategy="sug_router", query_type="merge_pair", score=1.0,
            primary_nodes=[0, 10], affected_group=list(range(20)), candidate_clusters=[0, 1],
            merge_c1=0, merge_c2=1, merge_c1_medoids=[0, 1, 2], merge_c2_medoids=[10, 11, 12],
        ),
        QueryItem(
            query_id="s0", strategy="sug_router", query_type="seam", score=0.9,
            primary_nodes=[3, 6], affected_group=list(range(10)), candidate_clusters=[0],
            seam_cluster_id=0, seam_sub1=[0, 1, 2, 3, 4, 5], seam_sub2=[6, 7, 8, 9],
            seam_sub1_medoids=[3, 4, 5], seam_sub2_medoids=[6, 7, 8], seam_cut_edges=[(5, 6)],
        ),
        QueryItem(
            query_id="p0", strategy="cequel_edge", query_type="pair", score=0.8,
            primary_nodes=[13, 14], affected_group=[13, 14], candidate_clusters=[1],
        ),
        QueryItem(
            query_id="n0", strategy="random", query_type="node", score=0.7,
            primary_nodes=[15], affected_group=[15], candidate_clusters=[1],
        ),
    ]
    decisions = [
        LLMDecision(query_id="m0", query_type="merge_pair", action="SEPARATE", confidence=1.0),
        LLMDecision(query_id="s0", query_type="seam", action="SPLIT", target_cluster=0, confidence=1.0),
        LLMDecision(query_id="p0", query_type="pair", action="CANNOT_LINK", confidence=1.0),
        LLMDecision(query_id="n0", query_type="node", action="ASSIGN", target_cluster=1, confidence=1.0),
    ]

    expected_all = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15}
    _, qmask_all = apply_direct_assignment(labels_init, queries, decisions, summaries=summaries, mask_mode="all_shown")
    assert set(np.where(qmask_all)[0].tolist()) == expected_all

    _, qmask_prim = apply_direct_assignment(labels_init, queries, decisions, summaries=summaries, mask_mode="primary_only")
    assert set(np.where(qmask_prim)[0].tolist()) == {0, 3, 6, 10, 13, 14, 15}

    _, qmask_rep, _ = apply_topology_repair_and_recluster(graph, labels_init, summaries, queries, decisions, mask_mode="all_shown")
    assert set(np.where(qmask_rep)[0].tolist()) == expected_all
    print("[prompt_shown_queried_mask test PASSED]")


def test_cequel_transitive_assignment() -> None:
    labels_init = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    queries = [
        QueryItem(query_id="q1", strategy="cequel_edge", query_type="pair", score=1.0, primary_nodes=[0, 3], affected_group=[0, 3]),
        QueryItem(query_id="q2", strategy="cequel_edge", query_type="pair", score=0.9, primary_nodes=[1, 7], affected_group=[1, 7]),
        QueryItem(query_id="q3", strategy="cequel_edge", query_type="pair", score=0.8, primary_nodes=[4, 8], affected_group=[4, 8]),
        QueryItem(query_id="q4", strategy="cequel_edge", query_type="pair", score=0.7, primary_nodes=[0, 2], affected_group=[0, 2]),
    ]
    decisions = [
        LLMDecision(query_id="q1", query_type="pair", action="MUST_LINK", confidence=1.0),
        LLMDecision(query_id="q2", query_type="pair", action="CANNOT_LINK", confidence=1.0),
        LLMDecision(query_id="q3", query_type="pair", action="MUST_LINK", confidence=1.0),
        LLMDecision(query_id="q4", query_type="pair", action="CANNOT_LINK", confidence=1.0),
    ]
    out, qmask = apply_cequel_transitive_assignment(labels_init, queries, decisions)
    assert out[0] == out[3] and out[1] == out[4]  # clusters 0 & 1 merged
    assert out[7] != out[1]  # cluster 2 did NOT merge with 0/1 because CANNOT_LINK(1, 7) guarded it
    assert out[8] == out[4]  # fallback moved node 8 to cluster(4)
    assert out[2] != out[0]  # intra-cluster CANNOT_LINK detached node 2
    assert set(np.where(qmask)[0].tolist()) == {0, 1, 2, 3, 4, 7, 8}
    print("[cequel_transitive_assignment test PASSED]")


def test_partition_cache_roundtrip() -> None:
    rng = np.random.default_rng(11)
    a = rng.uniform(0.0, 1.0, size=(16, 16))
    a = (a + a.T) / 2.0
    a[a < 0.65] = 0.0
    np.fill_diagonal(a, 0.0)
    sp_a = sparse.csr_matrix(a)
    expected, _ = solve_2d_se(sp_a, refine_moves=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="part_cache_test_"))
    try:
        lab1, _ = load_or_solve_2d_se("Synth", 0, sp_a, cache_dir=tmp_dir)
        lab2, _ = load_or_solve_2d_se("Synth", 0, sp_a, cache_dir=tmp_dir)
        assert np.array_equal(lab1, expected)
        assert np.array_equal(lab2, expected)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print("[partition_cache_roundtrip test PASSED]")


def test_key_loader_and_action_first_prompt() -> None:
    from core_shared.data_loader import SocialDataset
    from core_shared.key_loader import DEFAULT_KEY_FILE
    from trackA_sug_sed.streaming_sed import HistoricalEvent

    uname = Path.home().name
    for rel in ["core_shared/key_loader.py", "core_shared/data_loader.py"]:
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert uname not in src, f"Hardcoded username leaked in {rel}"
    assert DEFAULT_KEY_FILE.exists(), f"Missing resolved key file: {DEFAULT_KEY_FILE}"

    ds = SocialDataset(
        name="SynthPrompt",
        tweet_ids=np.array(["1", "2", "3", "4"]),
        texts=["Team A beats Team B 2-0", "Goals by striker", "Mayor quits office", "Subway line opens"],
        labels=np.array([0, 0, 1, 2]),
        tokens_lemma=[["team", "beat"], ["goal"], ["mayor"], ["subway"]],
        entities=[["team a", "team b"], ["striker"], ["mayor"], ["subway"]],
        hashtags=[[], [], [], []],
        user_ids=["u1", "u2", "u3", "u4"],
        mentioned_users=[[], [], [], []],
        timestamps=np.array([1.0, 2.0, 3.0, 4.0]),
        dates=["2012-10-10"] * 4,
    )
    emb = np.eye(4, dtype=np.float32)
    summaries = {
        0: ClusterSummary(cluster_id=0, members=[0, 1], size=2, prototype=emb[0], medoids=[0, 1], top_keywords=["team", "beat"], mean_timestamp=1.5),
        1: ClusterSummary(cluster_id=1, members=[2], size=1, prototype=emb[2], medoids=[2], top_keywords=["mayor"], mean_timestamp=3.0),
    }
    qm = QueryItem(
        query_id="m_test", strategy="sug_router", query_type="merge_pair", score=1.0,
        primary_nodes=[0, 2], affected_group=[0, 1, 2], candidate_clusters=[0, 1],
        merge_c1=0, merge_c2=1, merge_c1_medoids=[0, 1], merge_c2_medoids=[2],
    )
    qs = QueryItem(
        query_id="s_test", strategy="sug_router", query_type="seam", score=0.9,
        primary_nodes=[0, 2], affected_group=[0, 1, 2], candidate_clusters=[0],
        seam_cluster_id=0, seam_sub1=[0, 1], seam_sub2=[2],
        seam_sub1_medoids=[0, 1], seam_sub2_medoids=[2],
    )
    assert build_action_first_prompt(qm, ds, summaries) == _build_action_first_prompt(qm, ds, summaries)
    assert build_action_first_prompt(qs, ds, summaries) == _build_action_first_prompt(qs, ds, summaries)

    hev = HistoricalEvent(global_id=7, prototype=emb[0], total_count=5, first_block=0, last_block=1, rep_texts=["Team A beats Team B 2-0"], top_keywords=["team"])
    s_ex, u_ex = build_cross_block_prompt(hev, summaries[0], ds, prompt_mode="extract_compare")
    s_af, u_af = build_cross_block_prompt(hev, summaries[0], ds, prompt_mode="action_first")
    assert "same_event" in u_ex and '"MERGE" | "SEPARATE"' in u_af and s_ex != s_af

    tmp_dir = Path(tempfile.mkdtemp(prefix="ledger_thread_test_"))
    try:
        led = TokenLedger(tmp_dir)
        errs: list[Exception] = []

        def _worker(tid: int) -> None:
            try:
                for k in range(10):
                    led.log_call(
                        request_id=f"t{tid}_{k}",
                        block_id=0,
                        strategy="test",
                        query_type="merge_pair",
                        model="m",
                        request_hash=f"h_{tid}_{k}",
                        prompt_tokens=10,
                        completion_tokens=5,
                        latency_sec=0.01,
                        status="OK",
                        response_payload={"action": "MERGE"},
                        from_cache=False,
                    )
            except Exception as exc:
                errs.append(exc)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert not errs, errs
        assert led.summary()["total_calls"] == 40
        assert led.summary()["total_tokens"] == 600
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print("[key_loader_and_action_first_prompt test PASSED]")


def test_vectorized_seam_detection_equivalence() -> None:
    from core_shared.data_loader import SocialDataset
    from core_shared.graph_builder import build_message_graph
    from trackA_sug_sed.high_leverage_router import build_cluster_summaries, detect_seam_queries

    n = 24
    texts = (
        ["Syria opposition delegates meet in Doha summit talks"] * 14
        + ["Ukraine election protests rally outside Kiev parliament"] * 10
    )
    lemmas = (
        [["syria", "opposition", "delegate", "meet", "doha", "summit", "talk"] for _ in range(14)]
        + [["ukraine", "election", "protest", "rally", "kiev", "parliament"] for _ in range(10)]
    )
    entities = [["opposition"] for _ in range(n)]
    ds = SocialDataset(
        name="SynthSeam",
        tweet_ids=np.array([str(i) for i in range(n)]),
        texts=texts,
        labels=np.array([0] * 14 + [1] * 10),
        tokens_lemma=lemmas,
        entities=entities,
        hashtags=[[] for _ in range(n)],
        user_ids=[f"u{i}" for i in range(n)],
        mentioned_users=[[] for _ in range(n)],
        timestamps=np.arange(n, dtype=np.float64) * 60.0,
        dates=["2012-10-10"] * n,
    )
    rng = np.random.default_rng(19)
    emb = np.vstack(
        [
            np.tile([1.0, 0.0, 0.0, 0.0], (14, 1)) + 0.05 * rng.normal(size=(14, 4)),
            np.tile([0.0, 1.0, 0.0, 0.0], (10, 1)) + 0.05 * rng.normal(size=(10, 4)),
        ]
    ).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    # Connect groups weakly via shared bridge edges so a cut exists inside a single cluster
    graph = build_message_graph(ds, emb, knn_k=5, knn_sim_threshold=0.0, attr_alpha=0.5)
    labels = np.zeros(n, dtype=np.int64)
    summaries = build_cluster_summaries(ds, emb, labels)
    seams = detect_seam_queries(ds, graph, labels, summaries, max_conductance=0.95)
    assert len(seams) == 1
    q = seams[0]
    assert len(q.seam_cut_edges) > 0
    scalar_sims = [
        float(graph.non_entity_tfidf[u].multiply(graph.non_entity_tfidf[v]).sum())
        for u, v in q.seam_cut_edges
    ]
    ref_phi = float(np.mean(scalar_sims))
    assert abs(q.meta["phi_non_ent"] - ref_phi) < 1e-12
    sem_sep = q.meta["intra_pock"] - q.meta["cross_sim"]
    cond = q.meta["conductance"]
    ref_score = sem_sep * ((1.0 - min(0.95, cond)) ** 2) * max(0.15, 1.0 - ref_phi) * float(np.log1p(len(q.seam_sub2)))
    assert abs(q.score - ref_score) < 1e-12
    # Verify ablation flags work without error
    s_nc = detect_seam_queries(ds, graph, labels, summaries, use_conductance=False)
    s_ne = detect_seam_queries(ds, graph, labels, summaries, use_non_entity=False)
    s_nl = detect_seam_queries(ds, graph, labels, summaries, use_leverage=False)
    assert len(s_nc) == 1 and len(s_ne) == 1 and len(s_nl) == 1
    assert abs(s_ne[0].meta["phi_non_ent"] - 0.0) < 1e-12
    print("[vectorized_seam_detection_equivalence test PASSED]")


def test_query_anonymizer_and_leakage_audit() -> None:
    from copy import deepcopy
    from core_shared.data_loader import SocialDataset
    from trackA_sug_sed.anonymizer import (
        AnonymizationAuditStore,
        check_summary_reconstruction_leakage,
        make_anonymized_prompt_fn,
    )

    # Mix EN + FR realistic tweets with multi-word entities, missed proper nouns,
    # acronyms, hashtags, mentions, URLs, and 4-digit years.
    texts = [
        "BBC journalist's Jimmy Savile warning amid #Libya protests in 2012 http://t.co/BP3LbK1D via @reuters",
        "Newsnight producer questioned over Savile scandal after Yash Chopra tribute",
        "#AffaireBenalla EN DIRECT : Alexandre Benalla répond sur LFI et l'Élysée en 2018 https://t.co/xLXQHRu1nI",
        "Le patron de CBS Leslie Moonves quitte après #AffaireBenalla? Non, Mondiaux de Wuxi en Chine!",
    ]
    ds = SocialDataset(
        name="SynthAnon_M06",
        tweet_ids=np.array(["10", "11", "12", "13"]),
        texts=list(texts),
        labels=np.array([0, 0, 1, 2]),
        tokens_lemma=[["journalist"], ["producer"], ["direct"], ["patron"]],
        entities=[
            ["bbc", "jimmy savile", "libya"],
            ["newsnight", "savile"],
            ["alexandre benalla", "lfi"],
            ["cbs", "leslie moonves", "mondiaux de wuxi", "chine"],
        ],
        hashtags=[["libya"], [], ["affairebenalla"], ["affairebenalla"]],
        user_ids=["u0", "u1", "u2", "u3"],
        mentioned_users=[["reuters"], [], [], []],
        timestamps=np.array([1.0, 2.0, 3.0, 4.0]),
        dates=["2012-10-22", "2012-10-22", "2018-07-28", "2018-07-28"],
    )
    ds_orig_texts = deepcopy(ds.texts)
    emb = np.eye(4, dtype=np.float32)
    summaries = {
        0: ClusterSummary(cluster_id=0, members=[0, 1], size=2, prototype=emb[0], medoids=[0, 1], top_keywords=["jimmy savile", "#libya", "warning"], mean_timestamp=1.5),
        1: ClusterSummary(cluster_id=1, members=[2], size=1, prototype=emb[2], medoids=[2], top_keywords=["#affairebenalla", "alexandre benalla", "lfi"], mean_timestamp=3.0),
        2: ClusterSummary(cluster_id=2, members=[3], size=1, prototype=emb[3], medoids=[3], top_keywords=["mondiaux de wuxi", "cbs"], mean_timestamp=4.0),
    }
    qm_en = QueryItem(
        query_id="merge_c0_c1", strategy="sug_router", query_type="merge_pair", score=0.9,
        primary_nodes=[0, 2], affected_group=[0, 1, 2], candidate_clusters=[0, 1],
        merge_c1=0, merge_c2=1, merge_c1_medoids=[0, 1], merge_c2_medoids=[2],
    )
    qm_fr = QueryItem(
        query_id="merge_c1_c2", strategy="sug_router", query_type="merge_pair", score=0.9,
        primary_nodes=[2, 3], affected_group=[2, 3], candidate_clusters=[1, 2],
        merge_c1=1, merge_c2=2, merge_c1_medoids=[2], merge_c2_medoids=[3],
    )

    # 1. Pseudonym mode: 100% masking + consistent intra-query co-reference + zero mutation
    st_p = AnonymizationAuditStore()
    fn_p = make_anonymized_prompt_fn("action_first", "pseudonym", audit_store=st_p)
    sys_en, usr_en = fn_p(qm_en, ds, summaries)
    sys_fr, usr_fr = fn_p(qm_fr, ds, summaries)
    assert ds.texts == ds_orig_texts, "Original dataset texts mutated!"
    assert "jimmy savile" in summaries[0].top_keywords, "Original summaries mutated!"

    for forbid in ["Jimmy Savile", "Savile", "BBC", "Libya", "reuters", "2012", "Yash Chopra", "BP3LbK1D", "Newsnight"]:
        assert forbid.lower() not in usr_en.lower(), f"EN pseudonym leaked {forbid!r}: {usr_en}"
    for forbid in ["AffaireBenalla", "Alexandre", "Benalla", "LFI", "Élysée", "2018", "CBS", "Leslie Moonves", "Mondiaux", "Wuxi", "Chine", "xLXQHRu1nI"]:
        assert forbid.lower() not in usr_fr.lower(), f"FR pseudonym leaked {forbid!r}: {usr_fr}"
    assert "[ENT_" in usr_en and "[URL]" in usr_en and "[YEAR]" in usr_en and "[USER_" in usr_en
    assert "[ENT_" in usr_fr and "[YEAR]" in usr_fr and "[URL]" in usr_fr
    # Co-reference check across Group A & Group B in EN (Savile appears in post 0 and post 1; #AffaireBenalla in post 2 and post 3)
    rec_en = st_p.records[("SynthAnon_M06", "merge_c0_c1")]
    rec_fr = st_p.records[("SynthAnon_M06", "merge_c1_c2")]
    assert rec_en["entity_masked"] == rec_en["entity_total"] >= 3
    assert rec_en["hashtag_masked"] == rec_en["hashtag_total"] >= 1
    assert rec_en["year_masked"] == rec_en["year_total"] >= 1
    assert rec_fr["entity_masked"] == rec_fr["entity_total"] >= 3
    assert rec_fr["hashtag_masked"] == rec_fr["hashtag_total"] >= 1
    assert rec_fr["year_masked"] == rec_fr["year_total"] >= 1
    summary_p = st_p.summarize_for_dataset("SynthAnon")
    assert summary_p["entity_mask_rate"] == 1.0
    assert summary_p["hashtag_mask_rate"] == 1.0
    assert summary_p["year_mask_rate"] == 1.0

    # 2. Blind mode: zero numbered IDs, generic placeholders only, 100% masking
    st_b = AnonymizationAuditStore()
    fn_b = make_anonymized_prompt_fn("action_first", "blind", audit_store=st_b)
    _, usr_blind = fn_b(qm_en, ds, summaries)
    assert "[ENT_1]" not in usr_blind and "[TAG_1]" not in usr_blind and "[USER_1]" not in usr_blind
    assert "[ENTITY]" in usr_blind and "[HASHTAG]" in usr_blind and "[URL]" in usr_blind and "[YEAR]" in usr_blind
    for forbid in ["Jimmy Savile", "Savile", "BBC", "Libya", "2012", "Yash Chopra"]:
        assert forbid.lower() not in usr_blind.lower()

    # 3. Pretraining reconstruction checker
    leaked = check_summary_reconstruction_leakage(
        "Jimmy Savile scandal reported by [ENT_1]",
        "Protests erupt in [ENT_2]",
        rec_en["masked_originals"],
    )
    assert "jimmy savile" in leaked or "savile" in leaked
    clean = check_summary_reconstruction_leakage(
        "[ENT_1] scandal investigation ongoing",
        "Protests erupt in #[ENT_3]",
        rec_en["masked_originals"],
    )
    assert clean == []
    print("[query_anonymizer_and_leakage_audit test PASSED]")


if __name__ == "__main__":
    se_self_test()
    test_metrics_against_bruteforce()
    test_cumulative_token_bucket()
    test_data_loader_path()
    test_causal_token_buckets()
    test_prompt_shown_queried_mask()
    test_cequel_transitive_assignment()
    test_partition_cache_roundtrip()
    test_key_loader_and_action_first_prompt()
    test_vectorized_seam_detection_equivalence()
    test_query_anonymizer_and_leakage_audit()
