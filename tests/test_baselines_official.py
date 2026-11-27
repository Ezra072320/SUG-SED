"""Unit and integration tests for the strictly aligned official baselines."""

from __future__ import annotations

import os

for _env_k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_k, "1")

import inspect
import sys
import tempfile
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_shared.data_loader import SocialDataset, _extract_raw_entities, _normalize_entities
from core_shared.token_ledger import CausalUniformBucket, TokenLedger
from trackA_sug_sed.baselines_official import (
    _RAGSEDE_DETECTOR_SYS_TMPL,
    _RAGSEDE_EVALUATOR_SYS,
    _cequel_compute_weights,
    _ole_time_vec,
    build_cequel_triangle_prompt,
    build_cobras_binary_prompt,
    build_lsed_official_prompt,
    cequel_wcsc,
    cequel_weighted_pckmeans,
    preprocess_for_dataset,
    preprocess_french_sentence,
    preprocess_social_sentence,
    run_cequel_official,
    run_cobras_official,
    run_hisevent_official,
    run_lsed_official,
    run_ragsede_official,
    run_sbert_kmeans_official,
    train_lsed_hyperbolic_encoder,
)
from trackA_sug_sed.embed_cache import load_or_extract_block_embeddings
from trackA_sug_sed.llm_client import BudgetedLLMClient


def _make_synthetic_ds(
    n_clusters: int = 3,
    per_cluster: int = 12,
    dim: int = 16,
    seed: int = 42,
) -> Tuple[SocialDataset, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    embs = []
    labels = []
    texts = []
    entities = []
    raw_entities = []
    hashtags = []
    user_ids = []
    mentions = []
    for c in range(n_clusters):
        for j in range(per_cluster):
            v = centers[c] + 0.05 * rng.normal(size=dim)
            v /= np.linalg.norm(v)
            embs.append(v)
            labels.append(c)
            texts.append(f"Breaking news about event cluster {c} incident #{j} with details " + ("x" * 200))
            entities.append([f"ent_{c}"])
            raw_entities.append([f"Next Year", f"Ent_{c}"] if c == 0 else [f"Ent_{c}"])
            hashtags.append([f"tag{c}"])
            user_ids.append(str(1000 + c))
            mentions.append([str(2000 + c)])

    n = len(texts)
    ds = SocialDataset(
        name="Synth",
        tweet_ids=np.array([str(i) for i in range(n)], dtype=object),
        texts=texts,
        labels=np.array(labels, dtype=np.int64),
        tokens_lemma=[t.lower().split()[:5] for t in texts],
        entities=entities,
        hashtags=hashtags,
        user_ids=user_ids,
        mentioned_users=mentions,
        timestamps=1350432000.0 + np.arange(n, dtype=np.float64) * 300.0,
        dates=["2012-10-17"] * n,
        raw_entities=raw_entities,
    )
    return ds, np.vstack(embs).astype(np.float32)


def test_data_loader_raw_entities() -> None:
    sample = [("UP", "ORG"), ("Next year", "DATE"), ("A", "GPE"), ("Dean", "PERSON")]
    norm = _normalize_entities(sample)
    raw = _extract_raw_entities(sample)
    assert norm == ["up", "dean"], f"Unexpected normalized entities: {norm}"
    assert raw == ["UP", "Next year", "A", "Dean"], f"Unexpected raw entities: {raw}"

    ds, _ = _make_synthetic_ds()
    sub = ds.subset([0, 1, 15])
    assert sub.raw_entities is not None and len(sub.raw_entities) == 3
    assert sub.raw_entities[0] == ["Next Year", "Ent_0"]


def test_sbert_kmeans_and_hisevent_raw_attributes() -> None:
    ds, emb = _make_synthetic_ds(n_clusters=3, per_cluster=10, dim=32, seed=7)
    lbl_km, st_km = run_sbert_kmeans_official(emb, k_oracle=3, seed=7)
    assert adjusted_rand_score(ds.labels, lbl_km) == 1.0
    assert st_km["n_pred_clusters"] == 3

    # Also test that use_raw_attributes connects nodes sharing DATE/case-preserved entities
    ds_attr = SocialDataset(
        name="AttrTest",
        tweet_ids=np.array(["0", "1", "2", "3"], dtype=object),
        texts=["a", "b", "c", "d"],
        labels=np.array([0, 0, 1, 1], dtype=np.int64),
        tokens_lemma=[[]] * 4,
        entities=[[], [], [], []],  # empty filtered entities
        hashtags=[[], [], [], []],
        user_ids=["999", "888", "777", "666"],
        mentioned_users=[[], ["999"], [], []],  # mention matches user_id 999
        timestamps=np.zeros(4, dtype=np.float64),
        dates=["2012-10-17"] * 4,
        raw_entities=[["2012"], ["2012"], ["Tuesday"], ["Tuesday"]],
    )
    emb4 = np.array(
        [
            [1.0, 0.2, 0.0, 0.0],
            [0.95, 0.3, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.2],
            [0.0, 0.0, 0.95, 0.3],
        ],
        dtype=np.float32,
    )
    lbl_raw, st_raw = run_hisevent_official(ds_attr, embeddings=emb4, use_raw_attributes=True)
    _, st_filt = run_hisevent_official(ds_attr, embeddings=emb4, use_raw_attributes=False)
    assert st_raw["n_attr_edges"] > st_filt["n_attr_edges"], "Raw attributes must preserve DATE & unified user links"
    assert len(lbl_raw) == 4

    fr_sample = "L'Algérie va réexaminer @user https://t.co/xyz #Event"
    assert "é" in preprocess_french_sentence(fr_sample)
    assert "é" in preprocess_for_dataset(fr_sample, "Event2018")
    assert "é" not in preprocess_for_dataset(fr_sample, "Event2012")


def test_ragsede_official_contract(monkeypatch_encode: bool = True) -> None:
    import trackA_sug_sed.baselines_official as bo

    assert inspect.signature(run_ragsede_official).parameters["max_retries"].default == 10

    # Mock encode_texts_hf so test runs fast & offline without downloading any model
    orig_encode = bo.encode_texts_hf

    def _fake_encode(texts, model_name="fake", batch_size=64, max_length=128, device=None):
        out = []
        for t in texts:
            tl = t.lower()
            v = np.zeros(8, dtype=np.float32)
            v[0] = 1.0 if "alpha" in tl else 0.1
            v[1] = 1.0 if "beta" in tl else 0.1
            v[2] = 1.0 if "event" in tl or "keyword" in tl else 0.0
            v /= np.linalg.norm(v)
            out.append(v)
        return np.vstack(out)

    bo.encode_texts_hf = _fake_encode
    try:
        ds = SocialDataset(
            name="Event2012",
            tweet_ids=np.array(["0", "1", "2"], dtype=object),
            texts=[
                "Alpha wildfire evacuation updates",
                "Beta stadium soccer championship final",
                "Random unrelated noisy chatter",
            ],
            labels=np.array([0, 1, 2], dtype=np.int64),
            tokens_lemma=[["alpha"], ["beta"], ["random"]],
            entities=[[], [], []],
            hashtags=[[], [], []],
            user_ids=["1", "2", "3"],
            mentioned_users=[[], [], []],
            timestamps=np.zeros(3, dtype=np.float64),
            dates=["2012-10-17"] * 3,
        )
        # Orthogonal cluster embeddings so 3 anchors form cleanly
        emb_prep = np.eye(3, 8, dtype=np.float32)

        class _MockRagClient:
            def __init__(self) -> None:
                self.calls: List[Dict[str, Any]] = []

            def call_json(
                self,
                request_id: str,
                block_id: int,
                strategy: str,
                query_type: str,
                sys_msg: str,
                user_msg: str,
                bucket=None,
                est_completion_tokens: int = 90,
                max_tokens_override=None,
                temperature_override=None,
                top_p_override=None,
                cache_salt=None,
                use_json_mode_override=None,
            ):
                self.calls.append(
                    {
                        "request_id": request_id,
                        "query_type": query_type,
                        "sys_msg": sys_msg,
                        "user_msg": user_msg,
                        "max_tokens_override": max_tokens_override,
                        "temperature_override": temperature_override,
                        "top_p_override": top_p_override,
                        "use_json_mode_override": use_json_mode_override,
                    }
                )
                if query_type == "ragsede_kb_update":
                    if "Alpha" in user_msg:
                        return {"EVENT_NAME": "Alpha Fire", "KEYWORDS": ["Alpha", "wildfire"]}, 10, 5, 0.01, "OK"
                    if "Beta" in user_msg:
                        return {"EVENT_NAME": "Beta Match", "KEYWORDS": ["Beta", "soccer"]}, 10, 5, 0.01, "OK"
                    return {"EVENT_NAME": "Noise Event", "KEYWORDS": ["random", "chatter"]}, 10, 5, 0.01, "OK"
                # ragsede_detect: return event if present in retrieved KB (sys_msg), else "Others" to trigger KB add
                if "Alpha" in user_msg:
                    ev = "Alpha Fire" if "EVENT: Alpha Fire." in sys_msg else "Others"
                    return {"INPUT": user_msg, "EVENT": ev}, 10, 5, 0.01, "OK"
                if "Beta" in user_msg:
                    ev = "Beta Match" if "EVENT: Beta Match." in sys_msg else "Others"
                    return {"INPUT": user_msg, "EVENT": ev}, 10, 5, 0.01, "OK"
                # Deliberately return invalid hallucinated event name -> must retry and fall back to Confusion
                return {"INPUT": user_msg, "EVENT": "Hallucinated_NonKB_Event"}, 10, 5, 0.01, "OK"

        client = _MockRagClient()
        lbl, shown, st = run_ragsede_official(
            "Event2012",
            1,
            ds,
            client,  # type: ignore[arg-type]
            embeddings_prep=emb_prep,
            similarity_threshold=0.90,
            max_retries=2,
        )
        assert len(lbl) == 3 and st["n_pred_clusters"] == 3, f"Expected 3 clusters (Alpha, Beta, Confusion), got {lbl}"
        det_calls = [c for c in client.calls if c["query_type"] == "ragsede_detect"]
        kb_calls = [c for c in client.calls if c["query_type"] == "ragsede_kb_update"]
        assert kb_calls[0]["sys_msg"] == _RAGSEDE_EVALUATOR_SYS
        assert "Here is the knowledge base:" in det_calls[0]["sys_msg"]
        assert all(c["max_tokens_override"] == 512 and c["top_p_override"] == 0.3 for c in client.calls)
        assert all(c["use_json_mode_override"] is False for c in client.calls)
        # Verify retry happened for the invalid event anchor with temperature_override=0.2
        noise_retries = [c for c in det_calls if "Random unrelated" in c["user_msg"]]
        assert len(noise_retries) == 2 and noise_retries[1]["temperature_override"] == 0.2
        assert st["parse_errors"] >= 2
    finally:
        bo.encode_texts_hf = orig_encode


def test_cobras_binary_oracle() -> None:
    ds, emb = _make_synthetic_ds(n_clusters=3, per_cluster=6, dim=16, seed=11)
    sys_b, usr_b = build_cobras_binary_prompt("hello\nworld", "foo\nbar")
    assert "\n" not in usr_b.split("\n")[0].replace("Tweet #1: ", "")
    assert "Yes" in sys_b and "No" in sys_b

    class _MockCobrasClient:
        def __init__(self) -> None:
            self.call_types: List[Tuple[str, Any]] = []

        def call_text(
            self,
            request_id: str,
            block_id: int,
            strategy: str,
            query_type: str,
            sys_msg: str,
            user_msg: str,
            bucket=None,
            est_completion_tokens: int = 64,
            max_tokens_override=None,
            **kwargs,
        ):
            self.call_types.append((query_type, max_tokens_override))
            parts = request_id.split("_")
            u = int(parts[-2][1:])
            v = int(parts[-1][1:])
            ans = "Yes" if ds.labels[u] == ds.labels[v] else "No"
            return ans, 20, 1, 0.001, "OK"

    cli = _MockCobrasClient()
    lbl, shown, st = run_cobras_official(1, ds, emb, cli, max_questions=50, seed=11)  # type: ignore[arg-type]
    assert adjusted_rand_score(ds.labels, lbl) == 1.0
    assert all(qt == "cobras_pair_bin" and mt == 8 for qt, mt in cli.call_types)
    assert shown.sum() > 0 and st["n_constraints"] > 0


def test_cequel_weights_wcsc_wckmeans_and_prompt() -> None:
    import scipy.linalg as spl
    from scipy import sparse as sp_mod
    from sklearn.cluster import KMeans

    ds, emb = _make_synthetic_ds(n_clusters=3, per_cluster=30, dim=16, seed=5)
    # Prompt must NOT truncate tweets at 180 chars
    _, usr = build_cequel_triangle_prompt(ds.texts[0], ds.texts[1], ds.texts[2])
    assert len(ds.texts[0]) > 200 and ds.texts[0] in usr

    # Add 1 bridge point per cluster pair so 15-NN graph is connected (nonzero cut cost > 1e-10)
    rng = np.random.default_rng(5)
    centers = [emb[0], emb[30], emb[60]]
    emb[29] = 0.55 * centers[0] + 0.45 * centers[1]
    emb[29] /= np.linalg.norm(emb[29])
    emb[59] = 0.55 * centers[1] + 0.45 * centers[2]
    emb[59] /= np.linalg.norm(emb[59])
    emb[89] = 0.55 * centers[2] + 0.45 * centers[0]
    emb[89] /= np.linalg.norm(emb[89])

    ml = [(0, 1), (0, 2), (30, 31), (60, 61)]
    cl = [(0, 30), (0, 60), (30, 60), (1, 31)]
    ml_wk, cl_wk = _cequel_compute_weights(emb, ml, cl, "log_degree_ratio", (0.01, 0.1), (0.0, 0.01))
    ml_sc, cl_sc = _cequel_compute_weights(emb, ml, cl, "log_degree_ratio", (0.5, 1.5), (0.5, 1.5))
    assert min(ml_wk.values()) >= 0.01 - 1e-9 and max(ml_wk.values()) <= 0.1 + 1e-9
    assert min(cl_wk.values()) >= 0.0 - 1e-9 and max(cl_wk.values()) <= 0.01 + 1e-9
    assert min(ml_sc.values()) >= 0.5 - 1e-9 and max(ml_sc.values()) <= 1.5 + 1e-9
    assert min(cl_sc.values()) >= 0.5 - 1e-9 and max(cl_sc.values()) <= 1.5 + 1e-9

    assert inspect.signature(cequel_weighted_pckmeans).parameters["init"].default == "random"
    lbl_wk = cequel_weighted_pckmeans(emb, 3, ml, cl, weight_method="log_degree_ratio", seed=5)
    lbl_sc = cequel_wcsc(emb, 3, ml, cl, weight_method="log_degree_ratio", seed=5)
    assert adjusted_rand_score(ds.labels, lbl_wk) >= 0.90
    assert adjusted_rand_score(ds.labels, lbl_sc) >= 0.85

    # Verify parity with direct dense scipy.linalg.eig(L_norm, Q1) from HKBU-LAGAS/Cequel main.py:240
    n = len(emb)
    x_n = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    sim_m = x_n @ x_n.T
    np.fill_diagonal(sim_m, -2.0)
    nbrs = np.argpartition(sim_m, -15, axis=1)[:, -15:]
    A = np.zeros((n, n))
    for i in range(n):
        A[i, nbrs[i]] = 1.0
    A = np.maximum(A, A.T)
    D_inv = np.diag(1.0 / np.sqrt(A.sum(axis=1)))
    L_d = np.eye(n) - D_inv @ A @ D_inv
    Q_d = np.eye(n)
    for (i, j), w in ml_sc.items():
        Q_d[i, j] = w; Q_d[j, i] = w
    for (i, j), w in cl_sc.items():
        Q_d[i, j] = -w; Q_d[j, i] = -w
    Qn_d = D_inv @ Q_d @ D_inv
    al = 0.1 * float(spl.svdvals(Qn_d)[3])
    val, vec = spl.eig(L_d, Qn_d - al * np.eye(n))
    vp = np.real(vec[:, np.real(val) >= 0])
    vn = vp / np.linalg.norm(vp, axis=0) * np.sqrt(A.sum())
    costs = np.sum((vn.T @ L_d) * vn.T, axis=1)
    ids = np.where(costs > 1e-10)[0]
    min_idx = np.argsort(costs[ids])[:3]
    u = D_inv @ vn[:, ids[min_idx]]
    lbl_dense_eig = KMeans(n_clusters=3, n_init=10, random_state=5).fit_predict(u)
    assert adjusted_rand_score(lbl_sc, lbl_dense_eig) >= 0.95

    # Test precomputed offline responses return both WCSC and WCKMeans
    class _FakeCli:
        def call_text(self, request_id, block_id, strategy, query_type, sys_msg, user_msg, **kw):
            return "e", 30, 1, 0.0, "OK"

    lbl_o, lbl_e, _shown, st = run_cequel_official(
        1, ds, emb, k_oracle=3, k_est=3, client=_FakeCli(), clustering_method="wcsc", include_wckmeans=True
    )  # type: ignore[arg-type]
    assert "wckmeans_oracleK" in st and len(lbl_o) == len(ds) and len(lbl_e) == len(ds)


def test_cequel_5vote_majority_and_call_text_n() -> None:
    ds, emb = _make_synthetic_ds(n_clusters=2, per_cluster=3, dim=8, seed=17)

    class _MockVoteClient:
        def __init__(self) -> None:
            self.calls: List[Dict[str, Any]] = []
            self.attempt_by_req: Dict[str, int] = {}

        def call_text_n(
            self,
            request_id: str,
            block_id: int,
            strategy: str,
            query_type: str,
            sys_msg: str,
            user_msg: str,
            n_completions: int = 5,
            bucket=None,
            est_completion_tokens: int = 5,
            max_tokens_override=None,
            temperature_override=None,
            cache_salt=None,
            **kw,
        ):
            base_id = request_id.split("_r")[0]
            att = self.attempt_by_req.get(base_id, 0)
            self.attempt_by_req[base_id] = att + 1
            self.calls.append(
                {
                    "n_completions": n_completions,
                    "max_tokens_override": max_tokens_override,
                    "temperature_override": temperature_override,
                    "attempt": att,
                }
            )
            if att == 0:
                # Only 1 valid vote (<= 2.5) -> must trigger retry
                return ["invalid", "b", "foo", "bar", "baz"], 20, 5, 0.01, "OK"
            # 4 valid votes (> 2.5) with majority "c" (ONLY #1 and #3 are Same Event)
            return ["c", "c", "b", "?", "c"], 20, 5, 0.01, "OK"

    vcli = _MockVoteClient()
    _lo, _le, _sh, st = run_cequel_official(
        1, ds, emb, k_oracle=2, k_est=2, client=vcli, clustering_method="wckmeans"  # type: ignore[arg-type]
    )
    assert len(vcli.calls) >= 2
    assert vcli.calls[0]["attempt"] == 0 and vcli.calls[1]["attempt"] == 1
    assert all(
        c["n_completions"] == 5 and c["max_tokens_override"] == 1 and c["temperature_override"] == 1.0
        for c in vcli.calls
    )
    assert st["n_ml"] >= 1 and st["n_cl"] >= 2

    # Test BudgetedLLMClient.call_text_n direct n=5 + HTTP 400 fallback without double-charging bucket
    with tempfile.TemporaryDirectory() as tmpd:
        ledger = TokenLedger(Path(tmpd) / "l1", shared_cache_file=Path(tmpd) / "shared.jsonl")
        cli = BudgetedLLMClient(model="Qwen/Qwen3-8B", ledger=ledger, api_key="k", min_pace_sec=0.0)
        bkt = CausalUniformBucket(tokens_per_block=500, allow_carryover=False)
        bkt.start_block()

        cli._call_http_multi = lambda msgs, **kw: (["a", "b", "a", "a", "e"], 15, 5, 0.01, "stop", False)  # type: ignore[method-assign]
        res1 = cli.call_text_n("r1", 1, "s", "q", "", "hi", n_completions=5, bucket=bkt)
        assert res1 is not None and res1[0] == ["a", "b", "a", "a", "e"] and bkt.current_block_spent == 20
        res1_cached = cli.call_text_n("r1_c", 1, "s", "q", "", "hi", n_completions=5, bucket=bkt)
        assert res1_cached is not None and "CACHED_REPLAY" in res1_cached[4] and bkt.current_block_spent == 40

        # HTTP 400 on n > 1 must fall back to 5 sequential call_text calls charged once (5 * 11 = 55)
        def _raise_400(msgs, **kw):
            raise urllib.error.HTTPError("http://x", 400, "Bad Request", None, None)  # type: ignore[arg-type]

        bkt2 = CausalUniformBucket(tokens_per_block=500, allow_carryover=False)
        bkt2.start_block()
        cli._call_http_multi = _raise_400  # type: ignore[method-assign]
        cli._call_http = lambda msgs, **kw: ("d", 10, 1, 0.01, "stop", False)  # type: ignore[method-assign]
        res2 = cli.call_text_n("r2", 1, "s", "q", "", "hi_fb", n_completions=5, bucket=bkt2)
        assert res2 is not None and res2[0] == ["d"] * 5 and res2[4] == "OK_FALLBACK_N"
        assert res2[1] == 50 and res2[2] == 5 and bkt2.current_block_spent == 55


def test_lsed_hgcn_poincare_and_ole() -> None:
    _, usr = build_lsed_official_prompt("hello UP")
    assert usr == "Summarize the following sentences:hello UP. If you come across some abbreviations, expand them. Please respond in English"
    tvec = _ole_time_vec(np.array([1350432000.0, np.nan]))
    assert tvec.shape == (2, 2) and 0.0 < tvec[0, 0] < 1.0 and tvec[1, 0] == 0.0

    ds, emb = _make_synthetic_ds(n_clusters=4, per_cluster=50, dim=16, seed=13)
    with tempfile.TemporaryDirectory() as tmpd:
        ckpt = Path(tmpd) / "lsed_test.pt"
        model, dev = train_lsed_hyperbolic_encoder(
            ds, emb, ckpt, hidden_dim=16, num_layers=3, epochs=120, batch_size=None, lr=0.02, tan_reg=0.02, seed=13
        )
        assert ckpt.exists()
        saved = torch.load(ckpt, map_location="cpu", weights_only=True)
        assert saved["arch"] == "hgcn_poincare_v2"
        assert saved["val_acc"] >= 0.85, f"Low val_acc={saved['val_acc']}"
        assert 0.05 < saved["mean_hyp_norm"] < 0.995, f"Saturated or collapsed hyp norm={saved['mean_hyp_norm']}"
        assert saved["sat_pct"] < 10.0, f"Excessive boundary saturation: {saved['sat_pct']}%"

        # Verify model is NOT Euclidean identity cancellation
        feat = np.hstack([emb, _ole_time_vec(ds.timestamps)]).astype(np.float32)
        with torch.no_grad():
            hyp = model.encode_hyperbolic(torch.from_numpy(feat).to(dev)).cpu().numpy()
        norms = np.linalg.norm(hyp, axis=1)
        assert float(np.mean(norms > 0.999)) < 0.1, f"Poincare ball boundary saturation detected: {norms[:10]}"

        class _FakeLsedCli:
            def call_text(self, request_id, block_id, strategy, query_type, sys_msg, user_msg, max_tokens_override=None, **kw):
                assert max_tokens_override == 256
                return "", 10, 1, 0.0, "OK"

        lbl_o, lbl_e, _shown, st = run_lsed_official(
            1, ds, emb, model, dev, k_oracle=4, k_est=4, client=_FakeLsedCli()  # type: ignore[arg-type]
        )
        assert adjusted_rand_score(ds.labels, lbl_o) >= 0.85
        assert st["open_set_window"] == "infty"
        assert 0.05 < st["mean_hyp_norm"] < 0.99
        raised = False
        try:
            run_lsed_official(1, ds, emb, model, dev, k_oracle=4, k_est=4, client=_FakeLsedCli(), window_size=1)  # type: ignore[arg-type]
        except NotImplementedError:
            raised = True
        assert raised, "Positive window_size must raise NotImplementedError"


def test_embed_cache_max_length_isolation() -> None:
    import trackA_sug_sed.embed_cache as ec

    orig_encode = ec.encode_texts_hf
    seen_lens: List[int] = []

    def _fake_encode(texts, model_name="fake", batch_size=64, max_length=128, device=None):
        seen_lens.append(int(max_length))
        return np.ones((len(texts), 4), dtype=np.float32) * 0.5

    ec.encode_texts_hf = _fake_encode
    try:
        with tempfile.TemporaryDirectory() as tmpd:
            cdir = Path(tmpd)
            e128 = load_or_extract_block_embeddings("D", 1, ["a", "b"], cache_dir=cdir, max_length=128)
            e256 = load_or_extract_block_embeddings("D", 1, ["a", "b"], cache_dir=cdir, max_length=256, allow_legacy_cache=False)
            assert seen_lens == [128, 256]
            files = sorted(p.name for p in cdir.glob("*.npy"))
            assert len(files) == 2 and any("_ml256_" in fn for fn in files)
            assert e128.shape == e256.shape == (2, 4)
    finally:
        ec.encode_texts_hf = orig_encode


def test_ragsede_french_accent_tokenizer() -> None:
    from trackA_sug_sed.baselines_official import _tokenize_kw_tf, preprocess_for_dataset

    cleaned = preprocess_for_dataset("La sortie de prison d'Algérie & vidéo @user #Rennes https://t.co/abc", "Event2018")
    assert "algérie" in cleaned.lower() and "vidéo" in cleaned.lower()
    tf = _tokenize_kw_tf("Président algérie vidéo événement")
    assert tf["président"] == 1
    assert tf["algérie"] == 1
    assert tf["vidéo"] == 1
    assert tf["événement"] == 1
    assert "alg" not in tf and "rie" not in tf


def main() -> None:
    tests = [
        test_data_loader_raw_entities,
        test_embed_cache_max_length_isolation,
        test_sbert_kmeans_and_hisevent_raw_attributes,
        test_ragsede_official_contract,
        test_ragsede_french_accent_tokenizer,
        test_cobras_binary_oracle,
        test_cequel_weights_wcsc_wckmeans_and_prompt,
        test_cequel_5vote_majority_and_call_text_n,
        test_lsed_hgcn_poincare_and_ole,
    ]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"ALL {len(tests)} OFFICIAL BASELINE TESTS PASSED")


if __name__ == "__main__":
    main()
