"""Strictly aligned implementations of the 6 baseline papers across 3 benchmark groups.

Active 3-Category Benchmark Structure:
  Group 1 — 0-LLM Original (0 Tokens):
    1. SBERT*    (Reimers & Gurevych, EMNLP 2019): preprocessed text + KMeans(K*).
                 (`run_sbert_agglom_dev` kept only for backward compatibility.)
    2. HISEvent  (Cao et al., AAAI 2024 — SELGroup/HISEvent):
                 - language-specific official preprocessing (English `preprocess_social_sentence` / French keeps accents) + SBERT(256) + Pearson corr
                 - unfiltered raw unified attributes (`user_id`, `mentioned_users`, `hashtags.lower()`, raw entities)
                 - 1D-SE first stable point (max_num_neighbors=200) + hierarchical mini-batch 2D-SE (`eps=0.0`, n_sub=400/300)
  Group 2 — Unbounded LLM-SED Original (Original Protocol, Token Cost Reported):
    3. RagSEDE   (Liu et al., WWW 2026 — SELGroup/RagSEDE):
                 - exact text preprocessing + Agglom(0.40/0.30) anchors + MMR top-3 sampling (lambda=0.7)
                 - in-memory hybrid Event KB indexed without template words, retrieved by Enc("INPUT: " + a_k[:200])
                 - verbatim Evaluator/Detector prompts, `max_tokens=512`, `top_p=0.3`, `use_json_mode=False`, `max_retries=10`, "Confusion" fallback
    4. LSED      (Qiu et al., ACL 2025 — 2025.acl-long.233):
                 - exact verbatim summarization prompt + SBERT(max_length=256) + OLE 2D TIME (386-d)
                 - genuine 3-layer HGCN Poincare hyperbolic MLP (HypLinear + HypAct, hidden_dim=64, c=1.0, numerical `tan_reg=0.02`)
                 - trained on summarized M_00, inferred via KMeans (Fig. 6 open-set `w=infty`) using Oracle-K / HISEvent Est-K
  Group 3 — Budgeted LLM Baselines (20k Tokens / Block):
    5. COBRAS    (Van Craenendonck et al., IDA 2018 — ML-KULeuven/cobras):
                 - exact COBRAS_kmeans top-down 2-means splitting (n_init=10) + bottom-up pairwise merging
                 - concise binary Yes/No ML/CL LLM oracle prompt (`build_cobras_binary_prompt`), stops at 20k budget
    6. Cequel    (Wang et al., CIKM 2025 — HKBU-LAGAS/Cequel):
                 - exact TriangleExploreConsolidate + uncapped 5-option triangle prompt + official 5-sample majority vote (`n=5, temp=1.0, max_tokens=1`), stops at 20k budget
                 - `cequel_wcsc` (WCSC / CSCK, primary method on Tweet, Q in float64 `[0.5, 1.5]`)
                 - `cequel_weighted_pckmeans` (WCKMeans, `init="random"`, ML in [0.01, 0.1], CL in [0.0, 0.01], empty-cluster fix)
"""

from __future__ import annotations

import copy
import heapq
import math
import re
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy import linalg as sp_linalg
from scipy import spatial, sparse
from sklearn.cluster import AgglomerativeClustering, KMeans, kmeans_plusplus
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

from core_shared.data_loader import SocialDataset
from core_shared.graph_builder import _ensure_l2_normalized
from core_shared.se_solver import minimize_2d_se_agglomerative
from trackA_sug_sed.baselines_sed import run_sbert_agglom
from trackA_sug_sed.embed_cache import encode_texts_hf, load_or_extract_block_embeddings
from trackA_sug_sed.llm_client import BudgetedLLMClient, build_pairwise_same_event_prompt

_TORCH_INFER_LOCK = threading.Lock()


# ===========================================================================
# Shared Official Preprocessing & 1. SBERT (Reimers & Gurevych, EMNLP 2019)
# ===========================================================================

def preprocess_social_sentence(s: str) -> str:
    """Exact tweet regex preprocessing from SELGroup/HISEvent & SELGroup/RagSEDE utils.py."""
    s = re.sub(r"((www\.[^\s]+)|(https?://[^\s]+))", "url", s)
    s = re.sub(r"#([^\s]+)", r"\1", s)
    s = re.sub(r"(\\u[0-9A-Fa-f]+)", "", s)
    s = re.sub(r"[^\x00-\x7f]", "", s)
    s = re.sub(r"(!)\1+", "!", s)
    s = re.sub(r"(\?)\1+", "?", s)
    s = re.sub(
        r":\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-*|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\*\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:",
        "",
        s,
    )
    s = re.sub(r"@[^\s]+|RT @[^\s]+", "", s)
    s = re.sub(r"\n", "", s)
    return s.strip()


def preprocess_french_sentence(s: str) -> str:
    """Exact French tweet regex preprocessing from SELGroup/HISEvent/utils.py:45-46 (keeps French accents)."""
    s = re.sub(r"((www\.[^\s]+)|(https?://[^\s]+))", "url", s)
    s = re.sub(r"#([^\s]+)", r"\1", s)
    s = re.sub(r"(!)\1+", "!", s)
    s = re.sub(r"(\?)\1+", "?", s)
    s = re.sub(
        r":\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-*|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\*\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:",
        "",
        s,
    )
    s = re.sub(r"@[^\s]+|RT @[^\s]+", "", s)
    s = re.sub(r"\n", "", s)
    return s.strip()


def preprocess_for_dataset(s: str, dataset_name: str = "Event2012") -> str:
    """Select dataset-specific official preprocessing (`Event2018` preserves French accented characters)."""
    return preprocess_french_sentence(s) if "2018" in dataset_name else preprocess_social_sentence(s)


_ragsede_preprocess_sentence = preprocess_social_sentence


def run_sbert_kmeans_official(
    embeddings_prep: np.ndarray,
    k_oracle: int,
    seed: int = 42,
    n_init: int = 10,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Literature-standard SBERT* baseline (HISEvent AAAI'24 / RagSEDE WWW'26): KMeans(K*) on preprocessed SBERT."""
    t0 = time.perf_counter()
    n = len(embeddings_prep)
    if n <= 1:
        return np.zeros(n, dtype=np.int64), {"runtime_sec": 0.0, "n_pred_clusters": int(n)}
    k_eff = max(1, min(int(k_oracle), n))
    emb = _ensure_l2_normalized(embeddings_prep).astype(np.float64)
    lbl = KMeans(n_clusters=k_eff, n_init=n_init, random_state=seed).fit_predict(emb)
    _, clean = np.unique(lbl, return_inverse=True)
    return clean.astype(np.int64), {
        "runtime_sec": round(time.perf_counter() - t0, 3),
        "n_pred_clusters": int(clean.max() + 1),
    }


def run_sbert_agglom_dev(embeddings: np.ndarray, distance_threshold: float = 0.60) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Dev-tuned SBERT + average-linkage cosine Agglomerative clustering (used as Est-K baseline/estimator)."""
    return run_sbert_agglom(embeddings, distance_threshold=distance_threshold)


def run_sbert_official(embeddings: np.ndarray, distance_threshold: float = 0.60) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Backward-compatible alias for `run_sbert_agglom_dev`."""
    return run_sbert_agglom_dev(embeddings, distance_threshold=distance_threshold)


# ===========================================================================
# 2. HISEvent (Cao et al., AAAI 2024 — SELGroup/HISEvent)
# ===========================================================================

def _hisevent_search_stable_point(
    corr_matrix: np.ndarray,
    max_num_neighbors: int = 200,
) -> Tuple[int, int]:
    """Vectorized exact reproduction of SELGroup/HISEvent search_stable_points."""
    n = corr_matrix.shape[0]
    if n <= 2:
        return 1, 1
    k_lim = min(max_num_neighbors, n - 1)
    sorted_idx = np.argsort(corr_matrix, axis=1)

    adj_mask = np.zeros((n, n), dtype=bool)
    row_deg = np.zeros(n, dtype=np.float64)
    all_1d_ses: List[float] = []
    rows_arange = np.arange(n)

    for step in range(k_lim):
        dst = sorted_idx[:, -(step + 1)]
        w = corr_matrix[rows_arange, dst]
        valid = w > 0.0
        if np.any(valid):
            u_v = rows_arange[valid]
            d_v = dst[valid]
            w_v = w[valid]
            new_e = ~adj_mask[u_v, d_v]
            if np.any(new_e):
                u_n = u_v[new_e]
                d_n = d_v[new_e]
                w_n = w_v[new_e]
                for ui, di, wi in zip(u_n.tolist(), d_n.tolist(), w_n.tolist()):
                    if not adj_mask[ui, di]:
                        adj_mask[ui, di] = True
                        adj_mask[di, ui] = True
                        row_deg[ui] += wi
                        row_deg[di] += wi
        vol = float(row_deg.sum())
        if vol <= 0.0:
            all_1d_ses.append(0.0)
        else:
            nz = row_deg > 0.0
            p = row_deg[nz] / vol
            all_1d_ses.append(float(-np.sum(p * np.log2(p))))

    stable_indices = [
        i for i in range(1, len(all_1d_ses) - 1)
        if all_1d_ses[i] < all_1d_ses[i - 1] and all_1d_ses[i] < all_1d_ses[i + 1]
    ]
    if not stable_indices:
        return 0, 0
    first_k = stable_indices[0] + 1
    best_i = min(stable_indices, key=lambda idx: all_1d_ses[idx])
    return first_k, best_i + 1


def run_hisevent_official(
    ds: SocialDataset,
    embeddings: Optional[np.ndarray] = None,
    n_sub: Optional[int] = None,
    max_num_neighbors: int = 200,
    dataset_name: str = "Event2012",
    block_id: Optional[int] = None,
    embed_model: Optional[str] = None,
    use_raw_attributes: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Faithful reproduction of SELGroup/HISEvent (AAAI 2024) open-set block clustering."""
    t0 = time.perf_counter()
    n = len(ds)
    if n <= 1:
        return np.zeros(n, dtype=np.int64), {"runtime_sec": 0.0, "n_edges": 0, "n_pred_clusters": int(n)}

    eff_nsub = int(n_sub) if n_sub is not None else (300 if "2018" in dataset_name else 400)
    if embeddings is None:
        eff_model = embed_model or (
            "sentence-transformers/distiluse-base-multilingual-cased-v1"
            if "2018" in dataset_name
            else "sentence-transformers/all-MiniLM-L6-v2"
        )
        prep_texts = [preprocess_for_dataset(t, dataset_name) for t in ds.texts]
        if block_id is not None:
            embeddings = load_or_extract_block_embeddings(
                f"{dataset_name}_hisevent_prep",
                block_id,
                prep_texts,
                model_name=eff_model,
                storage_dtype="float32",
                allow_legacy_cache=False,
                max_length=256,
            )
        else:
            embeddings = encode_texts_hf(prep_texts, model_name=eff_model, batch_size=64, max_length=256)

    emb = _ensure_l2_normalized(embeddings).astype(np.float64)
    corr_matrix = np.corrcoef(emb)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr_matrix, 0.0)

    first_k, _global_k = _hisevent_search_stable_point(corr_matrix, max_num_neighbors=max_num_neighbors)
    if first_k == 0:
        first_k = max(1, math.ceil((n / 1000.0) * 10))

    sorted_idx = np.argsort(corr_matrix, axis=1)
    edge_rows: List[np.ndarray] = []
    edge_cols: List[np.ndarray] = []

    top_k_idx = sorted_idx[:, -min(first_k, n - 1) :]
    r_grid = np.broadcast_to(np.arange(n)[:, None], top_k_idx.shape)
    top_w = corr_matrix[r_grid, top_k_idx]
    mask_k = top_w > 0.0
    rk = r_grid[mask_k]
    ck = top_k_idx[mask_k]
    edge_rows.append(np.minimum(rk, ck))
    edge_cols.append(np.maximum(rk, ck))

    # Attribute buckets matching SELGroup/HISEvent/run.py:51-56:
    # all_node_features = [[str(u)] + [str(um) for um in ums] + [h.lower() for h in hs] + e]
    attr_nodes: Dict[str, List[int]] = defaultdict(list)
    for i in range(n):
        seen_a: Set[str] = set()
        if use_raw_attributes:
            if ds.user_ids[i]:
                seen_a.add(str(ds.user_ids[i]))
            for mu in ds.mentioned_users[i]:
                if mu:
                    seen_a.add(str(mu))
            for h in ds.hashtags[i]:
                if h:
                    seen_a.add(str(h).lower())
            ent_list = ds.raw_entities[i] if ds.raw_entities is not None else ds.entities[i]
            for e in ent_list:
                if e:
                    seen_a.add(str(e))
        else:
            if ds.user_ids[i]:
                seen_a.add(f"u:{ds.user_ids[i]}")
            for mu in ds.mentioned_users[i]:
                if mu:
                    seen_a.add(f"u:{mu}")
            for h in ds.hashtags[i]:
                if h:
                    seen_a.add(f"h:{h.lower()}")
            for e in ds.entities[i]:
                if e:
                    seen_a.add(f"e:{e}")
        for term in seen_a:
            attr_nodes[term].append(i)

    # Sparse incidence matrix M (int32) avoids O(n^2) Python loops on frequent terms
    inc_rows: List[int] = []
    inc_cols: List[int] = []
    attr_col = 0
    for nodes in attr_nodes.values():
        if len(nodes) >= 2:
            inc_rows.extend(nodes)
            inc_cols.extend([attr_col] * len(nodes))
            attr_col += 1

    n_attr_edges = 0
    if attr_col > 0:
        inc_mat = sparse.csr_matrix(
            (np.ones(len(inc_rows), dtype=np.int32), (inc_rows, inc_cols)),
            shape=(n, attr_col),
        )
        co_mat = sparse.triu(inc_mat @ inc_mat.T, k=1).tocoo()
        pos = corr_matrix[co_mat.row, co_mat.col] > 0.0
        if np.any(pos):
            u_a = co_mat.row[pos].astype(np.int64)
            v_a = co_mat.col[pos].astype(np.int64)
            n_attr_edges = int(len(u_a))
            edge_rows.append(u_a)
            edge_cols.append(v_a)

    all_u = np.concatenate(edge_rows) if edge_rows else np.array([], dtype=np.int64)
    all_v = np.concatenate(edge_cols) if edge_cols else np.array([], dtype=np.int64)
    if len(all_u) > 0:
        keys = all_u.astype(np.int64) * n + all_v.astype(np.int64)
        uniq_keys = np.unique(keys)
        uu = uniq_keys // n
        vv = uniq_keys % n
        ww = corr_matrix[uu, vv]
        sym_r = np.concatenate([uu, vv])
        sym_c = np.concatenate([vv, uu])
        sym_w = np.concatenate([ww, ww])
        adj = sparse.csr_matrix((sym_w, (sym_r, sym_c)), shape=(n, n), dtype=np.float64)
        adj.setdiag(0.0)
        adj.eliminate_zeros()
    else:
        adj = sparse.csr_matrix((n, n), dtype=np.float64)

    # Hierarchical 2D-SE minimization (hier_2D_SE_mini with chunk size n_sub)
    clusters: List[List[int]] = [[i] for i in range(n)]
    cur_n = eff_nsub
    while True:
        n_cl = len(clusters)
        splits = [(s, min(s + cur_n, n_cl)) for s in range(0, n_cl, cur_n)]
        next_clusters: List[List[int]] = []
        for s, e in splits:
            sub_cl = clusters[s:e]
            sub_nodes: List[int] = []
            init_lbl: List[int] = []
            for local_cid, mems in enumerate(sub_cl):
                for u in mems:
                    sub_nodes.append(u)
                    init_lbl.append(local_cid)
            sub_arr = np.asarray(sub_nodes, dtype=np.int64)
            sub_adj = adj[sub_arr][:, sub_arr]
            merged_lbl, _ = minimize_2d_se_agglomerative(
                sub_adj, initial_labels=np.asarray(init_lbl, dtype=np.int64), eps=0.0
            )
            grouped: Dict[int, List[int]] = defaultdict(list)
            for pos_i, c_out in enumerate(merged_lbl.tolist()):
                grouped[int(c_out)].append(sub_nodes[pos_i])
            for c_out in sorted(grouped.keys()):
                next_clusters.append(grouped[c_out])

        if len(splits) == 1:
            clusters = next_clusters
            break
        if next_clusters == clusters:
            cur_n *= 2
        clusters = next_clusters

    labels = np.empty(n, dtype=np.int64)
    for cid, mems in enumerate(clusters):
        for u in mems:
            labels[u] = cid
    _, clean = np.unique(labels, return_inverse=True)
    return clean.astype(np.int64), {
        "runtime_sec": round(time.perf_counter() - t0, 3),
        "first_stable_k": int(first_k),
        "n_edges": int(adj.nnz // 2),
        "n_attr_edges": n_attr_edges,
        "n_pred_clusters": int(clean.max() + 1),
    }


# ===========================================================================
# 3. RagSEDE (Liu et al., WWW 2026 — SELGroup/RagSEDE)
# ===========================================================================

def _ragsede_select_representative_messages(
    cluster_vecs: np.ndarray,
    top_k: int = 3,
    lambda_val: float = 0.7,
) -> np.ndarray:
    """Exact MMR representative selection from SELGroup/RagSEDE/get_key_messages.py."""
    n_samples = cluster_vecs.shape[0]
    if n_samples <= top_k:
        return np.arange(n_samples, dtype=np.int64)
    vecs = _ensure_l2_normalized(cluster_vecs).astype(np.float64)
    center = vecs.mean(axis=0)
    c_norm = float(np.linalg.norm(center))
    if c_norm > 1e-12:
        center = center / c_norm
    rep = vecs @ center
    sim_mat = vecs @ vecs.T
    np.fill_diagonal(sim_mat, 0.0)
    redundancy = sim_mat.sum(axis=1) / max(1, n_samples - 1)
    scores = lambda_val * rep - (1.0 - lambda_val) * redundancy
    return np.argsort(scores)[-top_k:][::-1].astype(np.int64)


# Verbatim prompt templates from SELGroup/RagSEDE/main.py:205-206
_RAGSEDE_EVALUATOR_SYS = (
    "You are an event analysis assistant. Your task is to infer the event names discussed in the provided social media comments and extract keywords related to the event. \n"
    "First, carefully read all the COMMENTS and understand the core content they discuss. \n"
    "Second, summarize a concise and accurate EVENT NAME based on the COMMENTS. \n"
    "Third, extract no more than 10 KEYWORDS related to the event from the comments, which should cover the core theme, characters, location, time, or other vital information about the event. Each KEYWORD must be a single word that appears in the COMMENTS. \n"
    "Answer in JSON format, including EVENT_NAME (str) and KEYWORDS (list) attributes. Other than that, the answer must not include any other information."
)

_RAGSEDE_DETECTOR_SYS_TMPL = (
    "The knowledge base contains EVENTs and corresponding KEYWORDs. \n"
    "You are a social media comment classifier determining which one EVENT the INPUT belongs to in the knowledge base. \n"
    "Answer in JSON format, including INPUT and EVENT attributes. Other than that, the answer must not include any other information. \n"
    "When all knowledge base content is irrelevant to the INPUT, your EVENT answer must be 'Others'. \n"
    "When the knowledge base is empty, your EVENT answer must be 'Others'. \n"
    "Answers don't need to consider chat history. \n"
    "      Here is the knowledge base: \n"
    "      {knowledge}\n"
    "      The above is the knowledge base."
)


def _tokenize_kw_tf(text: str) -> Counter[str]:
    return Counter(w for w in re.findall(r"[^\W_]{2,}", text.lower()))


def run_ragsede_official(
    dataset_name: str,
    block_id: int,
    ds: SocialDataset,
    client: BudgetedLLMClient,
    bucket: Optional[Any] = None,
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embeddings_prep: Optional[np.ndarray] = None,
    kb_embed_model: Optional[str] = None,
    similarity_threshold: Optional[float] = None,
    max_cluster_size: int = 100,
    top_k_rep: int = 3,
    lambda_val: float = 0.7,
    max_messages_per_event: int = 10,
    detector_similarity_threshold: float = 0.0,
    top_n_kb: int = 8,
    kw_sim_weight: float = 0.5,
    max_retries: int = 10,
    strategy_tag: str = "ragsede_official",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Faithful in-memory reproduction of SELGroup/RagSEDE (WWW 2026)."""
    t0 = time.perf_counter()
    t_thread0 = time.thread_time()
    n = len(ds)
    if n == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=bool), {"total_calls": 0, "total_tokens": 0}

    eff_sim_thresh = (
        float(similarity_threshold)
        if similarity_threshold is not None
        else (0.30 if "2018" in dataset_name else 0.40)
    )
    kb_enc_model = kb_embed_model or embed_model

    if embeddings_prep is not None:
        emb = _ensure_l2_normalized(embeddings_prep)
    else:
        prep_texts = [preprocess_for_dataset(t, dataset_name) for t in ds.texts]
        emb = load_or_extract_block_embeddings(
            f"{dataset_name}_ragsede_prep",
            block_id,
            prep_texts,
            model_name=embed_model,
            storage_dtype="float32",
            allow_legacy_cache=False,
            max_length=256,
        )
        emb = _ensure_l2_normalized(emb)

    # Step 1: cluster_embeddings(similarity_threshold=0.40/0.30, max_cluster_size=100)
    sim_mat = np.clip(emb @ emb.T, -1.0, 1.0)
    dist_mat = np.clip(1.0 - sim_mat, 0.0, 2.0)
    cl_raw = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=1.0 - eff_sim_thresh,
    ).fit_predict(dist_mat)

    raw_clusters: Dict[int, List[int]] = defaultdict(list)
    for idx, cid in enumerate(cl_raw.tolist()):
        raw_clusters[int(cid)].append(idx)

    anchor_clusters: List[List[int]] = []
    for cid in sorted(raw_clusters.keys()):
        mems = raw_clusters[cid]
        if len(mems) <= max_cluster_size:
            anchor_clusters.append(mems)
        else:
            for s in range(0, len(mems), max_cluster_size):
                anchor_clusters.append(mems[s : s + max_cluster_size])

    # Step 2: MMR representative selection per anchor cluster & Paper Eq. 12 query encoding Enc(a_k)
    anchors_text: List[str] = []
    anchors_rep_indices: List[List[int]] = []
    anchors_query_inputs: List[str] = []
    anchors_kw_tfs: List[Counter[str]] = []

    for mems in anchor_clusters:
        m_arr = np.asarray(mems, dtype=np.int64)
        top_loc = _ragsede_select_representative_messages(emb[m_arr], top_k=top_k_rep, lambda_val=lambda_val)
        rep_idx = [int(mems[int(li)]) for li in top_loc]
        anchors_rep_indices.append(rep_idx)
        anc_str = " ".join(ds.texts[ri] for ri in rep_idx)
        anchors_text.append(anc_str)
        clean_q = anc_str.replace('"', "")[:200]
        anchors_query_inputs.append("INPUT: " + clean_q)
        anchors_kw_tfs.append(_tokenize_kw_tf(clean_q))

    if anchors_query_inputs:
        anchors_query_emb = _ensure_l2_normalized(
            encode_texts_hf(anchors_query_inputs, model_name=kb_enc_model, batch_size=64, max_length=256)
        ).astype(np.float32)
    else:
        anchors_query_emb = np.zeros((0, 384), dtype=np.float32)

    # In-memory Hybrid Knowledge Base
    kb_chunks: List[Dict[str, Any]] = []
    events_list: List[str] = ["Others"]
    event_to_id: Dict[str, int] = {}
    event_msgs: Dict[str, List[str]] = defaultdict(list)
    all_shown = np.zeros(n, dtype=bool)
    calls = pt_sum = ct_sum = errs = 0
    llm_sec = api_wall_sec = 0.0
    budget_stopped = False

    def _embed_kb_text(txt: str) -> np.ndarray:
        vec = encode_texts_hf([txt], model_name=kb_enc_model, batch_size=1, max_length=256)[0]
        return vec.astype(np.float32)

    def _build_chunk_record(event_name: str, kws: Sequence[str]) -> Dict[str, Any]:
        kw_body = ", ".join(kws)
        content = f"EVENT: {event_name}. KEYWORDs: {kw_body}."
        c_vec = _embed_kb_text(content)
        # Index semantic event name + keywords only (exclude "EVENT:" / "KEYWORDs:" boilerplate)
        c_tf = _tokenize_kw_tf(f"{event_name} {' '.join(kws)}")
        imp_toks = set(_tokenize_kw_tf(event_name).keys())
        return {
            "event": event_name,
            "content": content,
            "vec": c_vec,
            "tf": c_tf,
            "imp_toks": imp_toks,
        }

    def _kb_update(text_in: str, fixed_event_name: Optional[str] = None, tag_id: str = "init") -> Optional[str]:
        nonlocal calls, pt_sum, ct_sum, errs, llm_sec, api_wall_sec, budget_stopped
        if fixed_event_name == "Confusion":
            return "Confusion"
        user_msg = "COMMENTS: " + text_in[-400:]
        ev_name: Optional[str] = None
        kws: List[str] = []

        for att in range(1, max(1, max_retries) + 1):
            temp_ov = None if att == 1 else 0.2
            salt = None if att == 1 else f"r{att}"
            t_api0 = time.perf_counter()
            res = client.call_json(
                request_id=f"ragsede_kb_M{block_id:02d}_{tag_id}_a{att}",
                block_id=block_id,
                strategy=strategy_tag,
                query_type="ragsede_kb_update",
                sys_msg=_RAGSEDE_EVALUATOR_SYS,
                user_msg=user_msg,
                bucket=bucket,
                est_completion_tokens=96,
                max_tokens_override=512,
                temperature_override=temp_ov,
                top_p_override=0.3,
                cache_salt=salt,
                use_json_mode_override=False,
            )
            api_wall_sec += time.perf_counter() - t_api0
            if res is None:
                budget_stopped = True
                return None
            parsed, pt, ct, lat, _ = res
            pt_sum += pt
            ct_sum += ct
            llm_sec += lat
            calls += 1

            raw_ev = parsed.get("EVENT_NAME") if isinstance(parsed, dict) else None
            raw_kws = parsed.get("KEYWORDS") if isinstance(parsed, dict) else None
            if isinstance(parsed, dict) and "error" not in parsed and isinstance(raw_ev, str) and isinstance(raw_kws, list):
                cand = raw_ev.strip()
                if cand.endswith("."):
                    cand = cand[:-1].strip()
                if cand:
                    ev_name = cand
                    kws = [str(kw).strip() for kw in raw_kws if isinstance(kw, str) and kw.strip()][:10]
                    break
            errs += 1

        if ev_name is None:
            # Exact SELGroup/RagSEDE main.py:49-52 fallback ("Confusion" without adding chunk)
            return "Confusion"

        target_name = fixed_event_name or ev_name
        rec = _build_chunk_record(target_name, kws)
        if fixed_event_name is None:
            if target_name not in events_list:
                events_list.append(target_name)
            kb_chunks.append(rec)
        else:
            for ch in kb_chunks:
                if ch["event"] == fixed_event_name:
                    ch.update(rec)
                    break
        return target_name

    def _retrieve_knowledge(anc_vec: np.ndarray, anc_tf: Counter[str]) -> str:
        if not kb_chunks:
            return "(empty)"
        n_docs = len(kb_chunks)
        df: Counter[str] = Counter()
        for ch in kb_chunks:
            for tok in ch["tf"].keys():
                df[tok] += 1

        def _idf(tok: str) -> float:
            return math.log((1.0 + n_docs) / (1.0 + df.get(tok, 0))) + 1.0

        q_weights: Dict[str, float] = {t: (1.0 + math.log(max(1, c))) * _idf(t) for t, c in anc_tf.items()}
        q_norm = math.sqrt(sum(v * v for v in q_weights.values()))

        scored: List[Tuple[float, str]] = []
        for ch in kb_chunks:
            d_sim = max(0.0, float(np.dot(anc_vec, ch["vec"])))
            if q_norm > 0.0 and ch["tf"]:
                c_weights: Dict[str, float] = {}
                for t, c in ch["tf"].items():
                    boost = 1.5 if t in ch["imp_toks"] else 1.0
                    c_weights[t] = (1.0 + math.log(max(1, c))) * _idf(t) * boost
                c_norm = math.sqrt(sum(v * v for v in c_weights.values()))
                if c_norm > 0.0:
                    dot_k = sum(q_weights[t] * c_weights[t] for t in q_weights if t in c_weights)
                    k_sim = dot_k / (q_norm * c_norm)
                else:
                    k_sim = 0.0
            else:
                k_sim = 0.0
            score = (1.0 - kw_sim_weight) * d_sim + kw_sim_weight * k_sim
            if score >= detector_similarity_threshold:
                scored.append((score, ch["content"]))
        if not scored:
            return "(empty)"
        scored.sort(key=lambda x: -x[0])
        return "\n".join(c for _s, c in scored[:top_n_kb])

    # Initialize KB with anchor 0 (RagSEDE main.py:147-148)
    anchor_pred: List[Optional[int]] = [None] * len(anchor_clusters)
    if anchor_clusters:
        for ri in anchors_rep_indices[0]:
            all_shown[ri] = True
        _kb_update(anchors_text[0], fixed_event_name=None, tag_id="a0_init")

    cur_eid = 0
    for idx in range(len(anchor_clusters)):
        if budget_stopped:
            break
        anc_str = anchors_text[idx].replace('"', "")
        for ri in anchors_rep_indices[idx]:
            all_shown[ri] = True
        kb_ctx = _retrieve_knowledge(anchors_query_emb[idx], anchors_kw_tfs[idx])
        sys_msg = _RAGSEDE_DETECTOR_SYS_TMPL.replace("{knowledge}", kb_ctx)
        user_msg = "INPUT: " + anc_str[:200]

        ev_pred: Optional[str] = None
        for att in range(1, max(1, max_retries) + 1):
            temp_ov = None if att == 1 else 0.2
            salt = None if att == 1 else f"r{att}"
            t_api0 = time.perf_counter()
            res = client.call_json(
                request_id=f"ragsede_det_M{block_id:02d}_a{idx}_a{att}",
                block_id=block_id,
                strategy=strategy_tag,
                query_type="ragsede_detect",
                sys_msg=sys_msg,
                user_msg=user_msg,
                bucket=bucket,
                est_completion_tokens=64,
                max_tokens_override=512,
                temperature_override=temp_ov,
                top_p_override=0.3,
                cache_salt=salt,
                use_json_mode_override=False,
            )
            api_wall_sec += time.perf_counter() - t_api0
            if res is None:
                budget_stopped = True
                break
            parsed, pt, ct, lat, _ = res
            pt_sum += pt
            ct_sum += ct
            llm_sec += lat
            calls += 1

            if (
                isinstance(parsed, dict)
                and "error" not in parsed
                and "INPUT" in parsed
                and isinstance(parsed.get("EVENT"), str)
            ):
                cand_ev = parsed["EVENT"]
                if cand_ev.endswith("."):
                    cand_ev = cand_ev[:-1]
                if cand_ev.startswith(" "):
                    cand_ev = cand_ev[1:]
                cand_ev = cand_ev.strip()
                if cand_ev in events_list:
                    ev_pred = cand_ev
                    break
            errs += 1

        if budget_stopped:
            break
        if ev_pred is None:
            # Exact SELGroup/RagSEDE main.py:109 fallback when EVENT is persistently invalid
            ev_pred = "Confusion"

        if ev_pred == "Others":
            new_ev = _kb_update(anc_str, fixed_event_name=None, tag_id=f"a{idx}_new")
            if new_ev is None:
                budget_stopped = True
                break
            ev_pred = new_ev

        if ev_pred not in event_to_id:
            event_to_id[ev_pred] = cur_eid
            cur_eid += 1
        anchor_pred[idx] = event_to_id[ev_pred]

        event_msgs[ev_pred].append(anc_str)
        if len(event_msgs[ev_pred]) >= max_messages_per_event:
            joined = " ".join(event_msgs[ev_pred])
            upd = _kb_update(joined, fixed_event_name=ev_pred, tag_id=f"a{idx}_upd")
            if upd is None:
                budget_stopped = True
                break
            event_msgs[ev_pred] = []

    labels_out = np.empty(n, dtype=np.int64)
    fallback_offset = cur_eid + 10
    for idx, mems in enumerate(anchor_clusters):
        cid = anchor_pred[idx] if anchor_pred[idx] is not None else (fallback_offset + idx)
        for u in mems:
            labels_out[u] = cid
    _, clean = np.unique(labels_out, return_inverse=True)

    total_wall = time.perf_counter() - t0
    router_cpu = min(max(0.0, total_wall - api_wall_sec), max(0.0, time.thread_time() - t_thread0))
    return clean.astype(np.int64), all_shown, {
        "total_calls": calls,
        "n_anchors": len(anchor_clusters),
        "total_tokens": pt_sum + ct_sum,
        "prompt_tokens": pt_sum,
        "completion_tokens": ct_sum,
        "shown_posts": int(all_shown.sum()),
        "shown_pct": float(all_shown.sum() * 100.0 / max(1, n)),
        "parse_errors": errs,
        "n_pred_clusters": int(clean.max() + 1),
        "router_cpu_sec": round(router_cpu, 4),
        "api_wall_sec": round(api_wall_sec, 4),
        "llm_api_sec": round(llm_sec, 4),
        "total_wall_sec": round(total_wall, 4),
    }


# ===========================================================================
# 4. COBRAS (Van Craenendonck et al., IDA 2018 — ML-KULeuven/cobras)
# ===========================================================================

def build_cobras_binary_prompt(doc_u: str, doc_v: str) -> Tuple[str, str]:
    """Concise binary ML/CL oracle prompt matching active pairwise clustering queries."""
    sys_msg = (
        "Determine whether two tweets describe the EXACT SAME specific real-world event. "
        "Respond with ONLY 'Yes' or 'No' without explanation."
    )
    t1 = doc_u.replace("\n", " ").strip()
    t2 = doc_v.replace("\n", " ").strip()
    user_msg = f"Tweet #1: {t1}\nTweet #2: {t2}\nSame specific event? (Yes/No):"
    return sys_msg, user_msg


def _parse_yes_no(raw_text: str) -> Optional[bool]:
    s = raw_text.strip().lower()
    if not s:
        return None
    m = re.match(r"^(yes|no|true|false)\b", s)
    if m:
        return m.group(1) in {"yes", "true"}
    if "yes" in s and "no" not in s:
        return True
    if "no" in s and "yes" not in s:
        return False
    return None


class _CobrasSI:
    def __init__(self, data: np.ndarray, indices: List[int]) -> None:
        self.indices = list(indices)
        self.centroid = np.mean(data[self.indices], axis=0)
        dists = np.linalg.norm(data[self.indices] - self.centroid, axis=1)
        self.representative_idx = int(self.indices[int(np.argmin(dists))])
        self.tried_splitting = False

    def distance_to(self, other: "_CobrasSI") -> float:
        return float(np.linalg.norm(self.centroid - other.centroid))


class _CobrasCluster:
    def __init__(self, sis: List[_CobrasSI]) -> None:
        self.super_instances = list(sis)
        self.is_finished = False

    def distance_to(self, other: "_CobrasCluster") -> float:
        return min(s1.distance_to(s2) for s1 in self.super_instances for s2 in other.super_instances)

    def get_comparison_points(self, other: "_CobrasCluster") -> Tuple[_CobrasSI, _CobrasSI]:
        return self.super_instances[0], other.super_instances[0]

    def cannot_link_to(self, other: "_CobrasCluster", cl_set: Set[Tuple[int, int]]) -> bool:
        m1 = [s.representative_idx for s in self.super_instances]
        m2 = [s.representative_idx for s in other.super_instances]
        for x in m1:
            for y in m2:
                ek = (x, y) if x < y else (y, x)
                if ek in cl_set:
                    return True
        return False


def _cobras_labeling(clusters: Sequence[_CobrasCluster], n: int) -> np.ndarray:
    pred = np.empty(n, dtype=np.int64)
    for cid, cl in enumerate(clusters):
        for si in cl.super_instances:
            for pt in si.indices:
                pred[pt] = cid
    _, clean = np.unique(pred, return_inverse=True)
    return clean.astype(np.int64)


def run_cobras_official(
    block_id: int,
    ds: SocialDataset,
    embeddings: np.ndarray,
    client: BudgetedLLMClient,
    bucket: Optional[Any] = None,
    max_questions: int = 500,
    seed: int = 42,
    prompt_mode: str = "binary",
    strategy_tag: str = "cobras_official",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Faithful reproduction of ML-KULeuven/cobras (COBRAS_kmeans, IDA 2018) with binary LLM querier."""
    t0 = time.perf_counter()
    t_thread0 = time.thread_time()
    n = len(ds)
    if n <= 1:
        return np.zeros(n, dtype=np.int64), np.zeros(n, dtype=bool), {"total_calls": 0, "total_tokens": 0}

    data = _ensure_l2_normalized(embeddings).astype(np.float64)
    all_shown = np.zeros(n, dtype=bool)
    ml_set: Set[Tuple[int, int]] = set()
    cl_set: Set[Tuple[int, int]] = set()
    pair_cache: Dict[Tuple[int, int], bool] = {}

    calls = pt_sum = ct_sum = errs = 0
    llm_sec = api_wall_sec = 0.0
    budget_exhausted = False

    def _query_points(u: int, v: int) -> Optional[bool]:
        nonlocal calls, pt_sum, ct_sum, errs, llm_sec, api_wall_sec, budget_exhausted
        if u == v:
            return True
        ek = (u, v) if u < v else (v, u)
        if ek in pair_cache:
            return pair_cache[ek]
        if len(ml_set) + len(cl_set) >= max_questions:
            return None

        t_api0 = time.perf_counter()
        if prompt_mode == "cot_json":
            sys_msg, user_msg = build_pairwise_same_event_prompt(ds.texts[ek[0]], ds.texts[ek[1]], ds.dates[ek[0]], ds.dates[ek[1]])
            res = client.call_json(
                request_id=f"cobras_M{block_id:02d}_u{ek[0]}_v{ek[1]}",
                block_id=block_id,
                strategy=strategy_tag,
                query_type="cobras_pair",
                sys_msg=sys_msg,
                user_msg=user_msg,
                bucket=bucket,
                est_completion_tokens=75,
            )
            api_wall_sec += time.perf_counter() - t_api0
            if res is None:
                budget_exhausted = True
                return None
            parsed, pt, ct, lat, _ = res
            pt_sum += pt
            ct_sum += ct
            llm_sec += lat
            calls += 1
            all_shown[ek[0]] = True
            all_shown[ek[1]] = True
            if "error" in parsed or "same_event" not in parsed:
                errs += 1
                ans = False
            else:
                val = parsed["same_event"]
                ans = val.strip().lower() in {"true", "1", "yes"} if isinstance(val, str) else bool(val)
        else:
            sys_msg, user_msg = build_cobras_binary_prompt(ds.texts[ek[0]], ds.texts[ek[1]])
            res_t = client.call_text(
                request_id=f"cobras_M{block_id:02d}_u{ek[0]}_v{ek[1]}",
                block_id=block_id,
                strategy=strategy_tag,
                query_type="cobras_pair_bin",
                sys_msg=sys_msg,
                user_msg=user_msg,
                bucket=bucket,
                est_completion_tokens=8,
                max_tokens_override=8,
            )
            api_wall_sec += time.perf_counter() - t_api0
            if res_t is None:
                budget_exhausted = True
                return None
            raw_text, pt, ct, lat, status = res_t
            pt_sum += pt
            ct_sum += ct
            llm_sec += lat
            calls += 1
            all_shown[ek[0]] = True
            all_shown[ek[1]] = True
            parsed_bool = None if status.startswith("HTTP_ERROR") else _parse_yes_no(raw_text)
            if parsed_bool is None:
                errs += 1
                ans = False
            else:
                ans = parsed_bool

        pair_cache[ek] = ans
        if ans:
            ml_set.add(ek)
        else:
            cl_set.add(ek)
        return ans

    def _split_si(si: _CobrasSI, k_split: int) -> List[_CobrasSI]:
        k_eff = min(k_split, len(si.indices))
        if k_eff <= 1:
            return [_CobrasSI(data, si.indices)]
        sub = data[np.asarray(si.indices, dtype=np.int64)]
        # Exact match to ML-KULeuven/cobras_kmeans.py:15 (n_init=10)
        km = KMeans(n_clusters=k_eff, random_state=seed, n_init=10)
        lbl = km.fit_predict(sub)
        out: List[_CobrasSI] = []
        for c_id in sorted(set(lbl.tolist())):
            pts = [si.indices[pos] for pos, lval in enumerate(lbl.tolist()) if lval == c_id]
            if pts:
                out.append(_CobrasSI(data, pts))
        return out

    def _determine_split_level(si_in: _CobrasSI) -> int:
        cur_si = _CobrasSI(data, si_in.indices)
        max_split = len(cur_si.indices)
        split_level = 0
        while (len(ml_set) + len(cl_set) < max_questions) and not budget_exhausted:
            if len(cur_si.indices) <= 1:
                break
            if len(cur_si.indices) == 2:
                new_sis = [_CobrasSI(data, [cur_si.indices[0]]), _CobrasSI(data, [cur_si.indices[1]])]
            else:
                new_sis = _split_si(cur_si, 2)
            if len(new_sis) <= 1:
                break
            s1, s2 = new_sis[0], new_sis[1]
            ans = _query_points(s1.representative_idx, s2.representative_idx)
            if ans is None or ans:
                break
            split_level += 1
            cands = [s for s in (s1, s2) if len(s.indices) >= 2]
            if not cands:
                break
            cur_si = min(cands, key=lambda x: len(x.indices))
        split_n = 2 ** max(1, split_level)
        return min(max_split, split_n)

    def _merge_clusters(clusters_list: List[_CobrasCluster]) -> bool:
        query_limit_reached = False
        merged = True
        while merged and (len(ml_set) + len(cl_set) < max_questions) and not budget_exhausted:
            pairs: List[Tuple[float, int, int]] = []
            m_cl = len(clusters_list)
            for i in range(m_cl):
                if clusters_list[i].is_finished:
                    continue
                for j in range(i + 1, m_cl):
                    if clusters_list[j].is_finished:
                        continue
                    if clusters_list[i].cannot_link_to(clusters_list[j], cl_set):
                        continue
                    pairs.append((clusters_list[i].distance_to(clusters_list[j]), i, j))
            pairs.sort(key=lambda x: x[0])

            merged = False
            for _dist, i, j in pairs:
                ci, cj = clusters_list[i], clusters_list[j]
                if ci.cannot_link_to(cj, cl_set):
                    continue
                bc1, bc2 = ci.get_comparison_points(cj)
                pt1, pt2 = bc1.representative_idx, bc2.representative_idx
                ek = (pt1, pt2) if pt1 < pt2 else (pt2, pt1)
                if ek in ml_set:
                    ci.super_instances.extend(cj.super_instances)
                    clusters_list.pop(j)
                    merged = True
                    break
                if len(ml_set) + len(cl_set) >= max_questions or budget_exhausted:
                    query_limit_reached = True
                    break
                ans = _query_points(pt1, pt2)
                if ans is None:
                    query_limit_reached = True
                    break
                if ans:
                    ci.super_instances.extend(cj.super_instances)
                    clusters_list.pop(j)
                    merged = True
                    break
            if query_limit_reached:
                break
        return (not query_limit_reached) and (not merged) and (not budget_exhausted)

    init_si = _CobrasSI(data, list(range(n)))
    init_k = _determine_split_level(init_si)
    init_sis = _split_si(init_si, init_k)
    clusters: List[_CobrasCluster] = [_CobrasCluster([si]) for si in init_sis]
    _merge_clusters(clusters)
    last_valid_clusters = copy.deepcopy(clusters)

    while (len(ml_set) + len(cl_set) < max_questions) and not budget_exhausted:
        to_split: Optional[_CobrasSI] = None
        orig_cl: Optional[_CobrasCluster] = None
        max_len = -1
        for cl in clusters:
            if cl.is_finished:
                continue
            for si in cl.super_instances:
                if si.tried_splitting or len(si.indices) < 2:
                    continue
                if len(si.indices) > max_len:
                    max_len = len(si.indices)
                    to_split = si
                    orig_cl = cl
        if to_split is None or orig_cl is None:
            break

        orig_cl.super_instances.remove(to_split)
        if not orig_cl.super_instances:
            clusters.remove(orig_cl)

        sp_k = _determine_split_level(to_split)
        if budget_exhausted or (len(ml_set) + len(cl_set) >= max_questions):
            orig_cl.super_instances.append(to_split)
            if orig_cl not in clusters:
                clusters.append(orig_cl)
            break
        new_sis = _split_si(to_split, sp_k)
        if len(new_sis) <= 1:
            to_split.tried_splitting = True
            orig_cl.super_instances.append(to_split)
            if orig_cl not in clusters:
                clusters.append(orig_cl)
            continue
        for si_new in new_sis:
            clusters.append(_CobrasCluster([si_new]))

        fully_merged = _merge_clusters(clusters)
        if fully_merged:
            last_valid_clusters = copy.deepcopy(clusters)

    final_labels = _cobras_labeling(last_valid_clusters, n)
    total_wall = time.perf_counter() - t0
    router_cpu = min(max(0.0, total_wall - api_wall_sec), max(0.0, time.thread_time() - t_thread0))
    return final_labels, all_shown, {
        "total_calls": calls,
        "n_constraints": len(ml_set) + len(cl_set),
        "total_tokens": pt_sum + ct_sum,
        "prompt_tokens": pt_sum,
        "completion_tokens": ct_sum,
        "shown_posts": int(all_shown.sum()),
        "shown_pct": float(all_shown.sum() * 100.0 / max(1, n)),
        "parse_errors": errs,
        "n_pred_clusters": int(final_labels.max() + 1),
        "router_cpu_sec": round(router_cpu, 4),
        "api_wall_sec": round(api_wall_sec, 4),
        "llm_api_sec": round(llm_sec, 4),
        "total_wall_sec": round(total_wall, 4),
    }


# ===========================================================================
# 5. Cequel (Wang et al., CIKM 2025 — HKBU-LAGAS/Cequel)
# ===========================================================================

def cequel_select_triangles(X: np.ndarray, k_triangles: int) -> List[Tuple[int, int, int]]:
    """Faithful reproduction of HKBU-LAGAS/Cequel TriangleExploreConsolidate.select_triangles."""
    x_norm = _ensure_l2_normalized(X).astype(np.float64)
    n = len(x_norm)
    if n < 3 or k_triangles <= 0:
        return []

    dot_mat = x_norm @ x_norm.T
    dot_mat[dot_mat < 0.0] = 1e-10
    z = np.sum(dot_mat, axis=1)

    sorted_idx = sorted(range(n), key=lambda i: z[i])
    z_mapped = [float(z[idx]) for idx in sorted_idx]

    def _tri_score(tri: Tuple[int, int, int]) -> float:
        i, j, k = tri
        return (1.0 / z_mapped[i]) + (1.0 / z_mapped[j]) + (1.0 / z_mapped[k])

    def _heap_add(heap: List[int], elems: Sequence[int]) -> List[int]:
        if not heap:
            for e in elems:
                heapq.heappush(heap, e)
            return heap
        h = heap[0]
        for e in elems:
            heapq.heappush(heap, e)
        if any(e == h + 1 for e in elems):
            while len(heap) > 1:
                beta = heapq.heappop(heap)
                if heap[0] != beta + 1:
                    heapq.heappush(heap, beta)
                    break
        return heap

    def _greedy_scan(heap_x: List[int], heap_y: List[int]) -> Optional[int]:
        if not heap_y:
            return None
        min_y = heap_y[0]
        if min_y + 1 < n and min_y + 1 not in heap_x:
            return min_y + 1
        max_y = max(heap_y)
        for gamma in range(min_y + 2, min(n, max_y + 2)):
            if gamma not in heap_x and gamma not in heap_y:
                return gamma
        return None

    heaps: Dict[int, List[int]] = {t: [t] for t in range(n)}
    stack: List[Tuple[int, int, int]] = [(0, 1, 2)]
    selected: List[Tuple[int, int, int]] = [(0, 1, 2)]
    seen: Set[Tuple[int, int, int]] = {(0, 1, 2)}
    heaps[0] = _heap_add(heaps[0], [1, 2])
    heaps[1] = _heap_add(heaps[1], [2])

    while len(selected) < k_triangles:
        t_a, _t_b, _t_c = stack[-1]
        cands: List[Tuple[int, int, int]] = []
        for base_t in (t_a - 1, t_a, t_a + 1):
            if base_t < 0 or base_t >= n - 2:
                continue
            b_idx = heaps[base_t][0] + 1 if heaps.get(base_t) else None
            if b_idx is None or b_idx >= n or b_idx <= base_t:
                continue
            c_idx = _greedy_scan(heaps[base_t], heaps[b_idx])
            if c_idx is None or c_idx >= n or c_idx <= b_idx:
                continue
            tri = (base_t, b_idx, c_idx)
            if tri not in seen:
                cands.append(tri)
        if not cands:
            break
        best_tri = max(cands, key=_tri_score)
        stack.append(best_tri)
        selected.append(best_tri)
        seen.add(best_tri)
        tx, ty, tz = best_tri
        heaps[tx] = _heap_add(heaps[tx], [ty, tz])
        heaps[ty] = _heap_add(heaps[ty], [tz])

    return [
        (sorted_idx[t1], sorted_idx[t2], sorted_idx[t3])
        for t1, t2, t3 in selected
    ]


def build_cequel_triangle_prompt(doc_i: str, doc_j: str, doc_k: str) -> Tuple[str, str]:
    """Faithful reproduction of HKBU-LAGAS/Cequel TriangleLLM 5-option prompt (full text, no char cap)."""
    t1 = doc_i.replace("\n", " ").strip()
    t2 = doc_j.replace("\n", " ").strip()
    t3 = doc_k.replace("\n", " ").strip()
    user_msg = (
        "Cluster Tweets by whether they discuss the same specific real-world event. "
        "For each triangle, respond with a, b, c, d, or e without explanation.\n\n"
        f"Tweet #1: {t1}\n"
        f"Tweet #2: {t2}\n"
        f"Tweet #3: {t3}\n\n"
        "Given this context, do tweet #1, tweet #2 and tweet #3 likely discuss the same specific real-world event?\n\n"
        "a) All are Same Event\n"
        "b) ONLY #1 and #2 are Same Event\n"
        "c) ONLY #1 and #3 are Same Event\n"
        "d) ONLY #2 and #3 are Same Event\n"
        "e) NONE"
    )
    return "", user_msg


def _parse_cequel_choice(raw_text: str) -> Optional[str]:
    s = raw_text.strip().lower()
    if not s:
        return None
    m = re.match(r"^(?:option\s*)?([a-e])\b", s)
    if m:
        return m.group(1)
    m2 = re.search(r"\b([a-e])\)", s)
    if m2:
        return m2.group(1)
    return None


def _cequel_compute_weights(
    x_norm: np.ndarray,
    ml: Sequence[Tuple[int, int]],
    cl: Sequence[Tuple[int, int]],
    weight_method: str,
    ml_range: Tuple[float, float],
    cl_range: Tuple[float, float],
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
    """Compute exact Cequel constraint weights with specified MinMax feature ranges."""
    dot_mat = x_norm @ x_norm.T
    dot_mat[dot_mat < 0.0] = 1e-10
    d_vec = np.sum(dot_mat, axis=1)
    inv_d_sum = float(np.sum(1.0 / np.maximum(d_vec, 1e-12)))
    d_sum = float(np.sum(d_vec))

    def _edge_w(i: int, j: int) -> float:
        cos_ij = float(np.dot(x_norm[i], x_norm[j]))
        if cos_ij <= 1e-10:
            cos_ij = 1e-10
        if weight_method == "log_degree_ratio":
            ratio = (float(d_vec[i]) * float(d_vec[j])) / cos_ij
            return math.log(ratio * inv_d_sum + 1.0)
        if weight_method == "pmi_log":
            return math.log((cos_ij * d_sum) / max(float(d_vec[i]) * float(d_vec[j]), 1e-12) + 1.0)
        if weight_method == "sqrt_cosine":
            return math.sqrt(cos_ij)
        if weight_method == "log_cosine":
            return math.log(cos_ij + 1.0)
        if weight_method == "degree":
            return 1.0 / max(float(d_vec[i]), 1e-12) + 1.0 / max(float(d_vec[j]), 1e-12)
        if weight_method == "sqrt_degree":
            return math.sqrt(1.0 / max(float(d_vec[i]), 1e-12) + 1.0 / max(float(d_vec[j]), 1e-12))
        return cos_ij

    raw_ml = { (min(i, j), max(i, j)): _edge_w(i, j) for i, j in ml if i != j }
    raw_cl = { (min(i, j), max(i, j)): _edge_w(i, j) for i, j in cl if i != j }

    def _scale_dict(w_dict: Dict[Tuple[int, int], float], rng: Tuple[float, float]) -> Dict[Tuple[int, int], float]:
        if not w_dict:
            return {}
        lo, hi = float(rng[0]), float(rng[1])
        vals = np.array(list(w_dict.values()), dtype=np.float64)
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax - vmin < 1e-12:
            # Matches sklearn MinMaxScaler behavior when all values are equal (maps to feature_range[0])
            return {k: lo for k in w_dict}
        scale = (hi - lo) / (vmax - vmin)
        return {k: lo + (v - vmin) * scale for k, v in w_dict.items()}

    return _scale_dict(raw_ml, ml_range), _scale_dict(raw_cl, cl_range)


def cequel_weighted_pckmeans(
    X: np.ndarray,
    n_clusters: int,
    ml: Sequence[Tuple[int, int]],
    cl: Sequence[Tuple[int, int]],
    weight_method: str = "log_degree_ratio",
    init: str = "random",
    max_iter: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """Faithful reproduction of HKBU-LAGAS/Cequel WeightedPCKMeans (ML in [0.01, 0.1], CL in [0.0, 0.01])."""
    x_norm = _ensure_l2_normalized(X).astype(np.float64)
    n = len(x_norm)
    k_cl = max(1, min(int(n_clusters), n))
    if n <= 1 or k_cl == 1:
        return np.zeros(n, dtype=np.int64)

    n_ml_w, n_cl_w = _cequel_compute_weights(
        x_norm, ml, cl, weight_method=weight_method, ml_range=(0.01, 0.1), cl_range=(0.0, 0.01)
    )
    ml_adj: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    cl_adj: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for (i, j), w in n_ml_w.items():
        ml_adj[i].append((j, w))
        ml_adj[j].append((i, w))
    for (i, j), w in n_cl_w.items():
        cl_adj[i].append((j, w))
        cl_adj[j].append((i, w))

    rng = np.random.default_rng(seed)
    if init == "k-means++":
        centers, _ = kmeans_plusplus(x_norm, n_clusters=k_cl, random_state=seed)
        centers = centers.astype(np.float64, copy=True)
    else:
        chosen = rng.choice(n, size=k_cl, replace=False)
        centers = x_norm[chosen].astype(np.float64, copy=True)

    labels = np.zeros(n, dtype=np.int64)
    order = np.arange(n)

    for _it in range(max_iter):
        sq_dists = euclidean_distances(x_norm, centers, squared=True)
        new_labels = np.full(n, -1, dtype=np.int64)
        min_costs = np.zeros(n, dtype=np.float64)
        rng.shuffle(order)

        for xi in order.tolist():
            base_cost = 0.5 * sq_dists[xi]
            if xi in ml_adj or xi in cl_adj:
                pen = np.zeros(k_cl, dtype=np.float64)
                for yi, w in ml_adj.get(xi, ()):
                    ly = new_labels[yi]
                    if ly != -1:
                        pen += w
                        pen[ly] -= w
                for yi, w in cl_adj.get(xi, ()):
                    ly = new_labels[yi]
                    if ly != -1:
                        pen[ly] += w
                total_cost = base_cost + pen
            else:
                total_cost = base_cost
            best_c = int(np.argmin(total_cost))
            new_labels[xi] = best_c
            min_costs[xi] = float(total_cost[best_c])

        # Empty-cluster reassignment (weight_pckmeans.py:147-167)
        counts = np.bincount(new_labels, minlength=k_cl)
        empty = np.where(counts == 0)[0]
        if len(empty) > 0:
            farthest_order = np.argsort(-min_costs)
            pos = 0
            for _pass in range(10):
                if len(empty) == 0 or pos >= n:
                    break
                for cl_idx in empty.tolist():
                    while pos < n and counts[new_labels[farthest_order[pos]]] <= 1:
                        pos += 1
                    if pos >= n:
                        break
                    pt = int(farthest_order[pos])
                    old_c = int(new_labels[pt])
                    counts[old_c] -= 1
                    counts[cl_idx] += 1
                    new_labels[pt] = cl_idx
                    pos += 1
                empty = np.where(counts == 0)[0]

        new_centers = np.zeros_like(centers)
        for c_id in range(k_cl):
            mask = new_labels == c_id
            if np.any(mask):
                new_centers[c_id] = x_norm[mask].mean(axis=0)
            else:
                new_centers[c_id] = centers[c_id]

        labels = new_labels
        if np.allclose(new_centers, centers, atol=1e-6, rtol=0.0):
            break
        centers = new_centers

    _, clean = np.unique(labels, return_inverse=True)
    return clean.astype(np.int64)


def cequel_wcsc(
    X: np.ndarray,
    n_clusters: int,
    ml: Sequence[Tuple[int, int]],
    cl: Sequence[Tuple[int, int]],
    weight_method: str = "log_degree_ratio",
    eigen: float = 0.1,
    normalize_vectors: bool = False,
    seed: int = 42,
) -> np.ndarray:
    """Faithful reproduction of HKBU-LAGAS/Cequel WCSC / CSCK (`main.py:179-255`, Q weights in [0.5, 1.5])."""
    x_norm = _ensure_l2_normalized(X).astype(np.float64)
    n = len(x_norm)
    k_cl = max(1, min(int(n_clusters), n))
    if n <= 2 or k_cl == 1:
        return np.zeros(n, dtype=np.int64)
    if k_cl >= n:
        return np.arange(n, dtype=np.int64)

    X_work = StandardScaler().fit_transform(x_norm) if normalize_vectors else x_norm

    # 15-NN symmetric binary affinity matrix (for L2-normalized vectors, Euclidean 15-NN == dot 15-NN)
    k_nbr = min(15, n - 1)
    if not normalize_vectors:
        sim_mat = X_work @ X_work.T
        np.fill_diagonal(sim_mat, -2.0)
        nbrs = np.argpartition(sim_mat, -k_nbr, axis=1)[:, -k_nbr:]
    else:
        tree = spatial.KDTree(X_work)
        _dist, idx = tree.query(X_work, k=k_nbr + 1)
        nbrs = idx[:, 1:]
    r_idx = np.repeat(np.arange(n, dtype=np.int64), nbrs.shape[1])
    c_idx = nbrs.ravel().astype(np.int64)
    A_sp = sparse.csr_matrix((np.ones(len(r_idx), dtype=np.float64), (r_idx, c_idx)), shape=(n, n))
    A_sp = A_sp.maximum(A_sp.T)
    A_sp.setdiag(0.0)
    A_sp.eliminate_zeros()

    deg = np.asarray(A_sp.sum(axis=1)).ravel()
    vol = float(deg.sum())
    if vol <= 1e-12:
        return cequel_weighted_pckmeans(x_norm, k_cl, ml, cl, weight_method=weight_method, seed=seed)

    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    D_inv_sp = sparse.diags(d_inv_sqrt, dtype=np.float64)
    L_norm_sp = (sparse.eye(n, dtype=np.float64) - D_inv_sp @ A_sp @ D_inv_sp).tocsr()

    # Constraint matrix Q in float64 (matching Paper Eq. 4 Theta_{a,b} without int truncation bug)
    n_ml_w, n_cl_w = _cequel_compute_weights(
        x_norm, ml, cl, weight_method=weight_method, ml_range=(0.5, 1.5), cl_range=(0.5, 1.5)
    )
    q_r: List[int] = list(range(n))
    q_c: List[int] = list(range(n))
    q_v: List[float] = [1.0] * n
    for (i, j), w in n_ml_w.items():
        q_r.extend([i, j])
        q_c.extend([j, i])
        q_v.extend([w, w])
    for (i, j), w in n_cl_w.items():
        q_r.extend([i, j])
        q_c.extend([j, i])
        q_v.extend([-w, -w])
    Q_sp = sparse.csr_matrix((q_v, (q_r, q_c)), shape=(n, n), dtype=np.float64)
    Q_norm_sp = (D_inv_sp @ Q_sp @ D_inv_sp).tocsr()

    k_sv_idx = min(k_cl, n - 1)
    try:
        if n <= 2500:
            L_d = L_norm_sp.toarray()
            Q_d = Q_norm_sp.toarray()
            svals = sp_linalg.svdvals(Q_d)
            alpha = float(eigen) * float(svals[k_sv_idx])
            Q1_d = Q_d - alpha * np.eye(n, dtype=np.float64)
            # Generalized symmetric problem Q1 v = mu (L_norm + eps I) v <=> L_norm v = (1/mu) Q1 v
            mu, vec = sp_linalg.eigh(Q1_d, L_d + 1e-8 * np.eye(n, dtype=np.float64))
            vec_pos = vec[:, mu > 0.0]
        else:
            k_lm = min(max(k_sv_idx + 12, 20), n - 2)
            ev_lm = sparse.linalg.eigsh(Q_norm_sp, k=k_lm, which="LM", return_eigenvectors=False)
            svals = np.sort(np.abs(ev_lm))[::-1]
            alpha = float(eigen) * float(svals[min(k_sv_idx, len(svals) - 1)])
            Q1_sp = (Q_norm_sp - alpha * sparse.eye(n, dtype=np.float64)).tocsc()
            n_cc = sparse.csgraph.connected_components(A_sp, directed=False)[0]
            L_reg = (L_norm_sp + 1e-6 * sparse.eye(n, dtype=np.float64)).tocsc()
            lu = sparse.linalg.splu(L_reg, permc_spec="MMD_AT_PLUS_A")
            Minv = sparse.linalg.LinearOperator((n, n), matvec=lu.solve, dtype=np.float64)
            k_req = min(n - 2, max(n_cc + 2 * k_cl + 40, 120))
            mu, vec = sparse.linalg.eigsh(Q1_sp, k=k_req, M=L_reg, Minv=Minv, which="LA", tol=1e-5)
            vec_pos = vec[:, mu > 0.0]
    except Exception:
        return cequel_weighted_pckmeans(x_norm, k_cl, ml, cl, weight_method=weight_method, seed=seed)

    if vec_pos.shape[1] == 0:
        return cequel_weighted_pckmeans(x_norm, k_cl, ml, cl, weight_method=weight_method, seed=seed)

    v_norms = np.maximum(np.linalg.norm(vec_pos, axis=0), 1e-12)
    vec_norm = (vec_pos / v_norms[None, :]) * math.sqrt(vol)
    costs = np.sum((L_norm_sp @ vec_norm) * vec_norm, axis=0)
    ids = np.where(costs > 1e-10)[0]
    if len(ids) == 0:
        return cequel_weighted_pckmeans(x_norm, k_cl, ml, cl, weight_method=weight_method, seed=seed)

    take_k = min(k_cl, len(ids))
    min_idx = np.argsort(costs[ids])[:take_k]
    min_v = vec_norm[:, ids[min_idx]]
    u = d_inv_sqrt[:, None] * min_v

    lbl = KMeans(n_clusters=k_cl, n_init=10, random_state=seed).fit_predict(u)
    _, clean = np.unique(lbl, return_inverse=True)
    return clean.astype(np.int64)


def run_cequel_official(
    block_id: int,
    ds: SocialDataset,
    embeddings: np.ndarray,
    k_oracle: int,
    k_est: int,
    client: Optional[BudgetedLLMClient] = None,
    bucket: Optional[Any] = None,
    precomputed_responses: Optional[Dict[str, Dict[str, Any]]] = None,
    clustering_method: str = "wcsc",
    include_wckmeans: bool = True,
    num_votes: int = 5,
    vote_temperature: float = 1.0,
    max_vote_retries: int = 5,
    vote_max_tokens: int = 1,
    seed: int = 42,
    strategy_tag: str = "cequel_official",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Faithful TriangleLLM + WCSC (primary) / WCKMeans (returns labels_oracleK, labels_estK, all_shown, stats)."""
    t0 = time.perf_counter()
    n = len(ds)
    emb = _ensure_l2_normalized(embeddings)
    triangles = cequel_select_triangles(emb, k_triangles=n // 3)

    all_shown = np.zeros(n, dtype=bool)
    ml: List[Tuple[int, int]] = []
    cl: List[Tuple[int, int]] = []
    calls = pt_sum = ct_sum = errs = 0
    llm_sec = 0.0

    for idx_t, (i, j, k_node) in enumerate(triangles):
        req_id = f"cequel_M{block_id:02d}_t{idx_t}"
        sys_msg, user_msg = build_cequel_triangle_prompt(ds.texts[i], ds.texts[j], ds.texts[k_node])

        if precomputed_responses is not None and req_id in precomputed_responses:
            rec = precomputed_responses[req_id]
            pt = int(rec.get("prompt_tokens", int((len(sys_msg) + len(user_msg)) / 3.4)))
            ct = int(rec.get("completion_tokens", 2))
            if bucket is not None:
                est = int((len(sys_msg) + len(user_msg)) / 3.4) + max(16, int(vote_max_tokens))
                if not bucket.can_spend(est):
                    break
                bucket.record_spend(pt, ct)
            raw_text = str(rec.get("text", ""))
            pt_sum += pt
            ct_sum += ct
            calls += 1
            choice = _parse_cequel_choice(raw_text)
        else:
            assert client is not None
            n_v = max(1, int(num_votes))
            v_tok = max(1, int(vote_max_tokens))
            if n_v > 1 and hasattr(client, "call_text_n"):
                valid_votes: List[str] = []
                stopped_here = False
                for att in range(max(1, int(max_vote_retries))):
                    res_n = client.call_text_n(
                        request_id=req_id if att == 0 else f"{req_id}_r{att}",
                        block_id=block_id,
                        strategy=strategy_tag,
                        query_type="cequel_triangle",
                        sys_msg=sys_msg,
                        user_msg=user_msg,
                        n_completions=n_v,
                        bucket=bucket,
                        est_completion_tokens=n_v * v_tok,
                        max_tokens_override=v_tok,
                        temperature_override=float(vote_temperature),
                        cache_salt=None if att == 0 else f"r{att}",
                    )
                    if res_n is None:
                        stopped_here = True
                        break
                    raw_texts, pt, ct, lat, status = res_n
                    pt_sum += pt
                    ct_sum += ct
                    llm_sec += lat
                    calls += 1
                    if status.startswith("HTTP_ERROR"):
                        continue
                    cur_votes = [ch_v for rt in raw_texts if (ch_v := _parse_cequel_choice(rt)) is not None]
                    valid_votes = cur_votes
                    if len(cur_votes) > n_v / 2.0:
                        break
                if stopped_here:
                    break
                choice = (
                    max(["a", "b", "c", "d", "e"], key=lambda ch: valid_votes.count(ch))
                    if valid_votes
                    else None
                )
            else:
                votes: List[str] = []
                stopped_here = False
                for v_idx in range(n_v):
                    temp_ov = None if n_v == 1 else float(vote_temperature)
                    salt = None if n_v == 1 else f"v{v_idx}"
                    res = client.call_text(
                        request_id=req_id if n_v == 1 else f"{req_id}_v{v_idx}",
                        block_id=block_id,
                        strategy=strategy_tag,
                        query_type="cequel_triangle",
                        sys_msg=sys_msg,
                        user_msg=user_msg,
                        bucket=bucket,
                        est_completion_tokens=v_tok,
                        max_tokens_override=v_tok,
                        temperature_override=temp_ov,
                        cache_salt=salt,
                    )
                    if res is None:
                        stopped_here = True
                        break
                    raw_text, pt, ct, lat, status = res
                    pt_sum += pt
                    ct_sum += ct
                    llm_sec += lat
                    calls += 1
                    if not status.startswith("HTTP_ERROR"):
                        ch_v = _parse_cequel_choice(raw_text)
                        if ch_v is not None:
                            votes.append(ch_v)
                if stopped_here:
                    break
                choice = (
                    max(["a", "b", "c", "d", "e"], key=lambda ch: votes.count(ch))
                    if votes
                    else None
                )

        all_shown[i] = True
        all_shown[j] = True
        all_shown[k_node] = True
        if choice == "a":
            ml.extend([(i, j), (i, k_node), (j, k_node)])
        elif choice == "b":
            ml.append((i, j))
            cl.extend([(i, k_node), (j, k_node)])
        elif choice == "c":
            ml.append((i, k_node))
            cl.extend([(i, j), (j, k_node)])
        elif choice == "d":
            ml.append((j, k_node))
            cl.extend([(i, j), (i, k_node)])
        elif choice == "e":
            cl.extend([(i, j), (i, k_node), (j, k_node)])
        else:
            errs += 1

    if clustering_method == "wckmeans":
        lbl_oracle = cequel_weighted_pckmeans(emb, k_oracle, ml, cl, weight_method="log_degree_ratio", seed=seed)
        lbl_est = cequel_weighted_pckmeans(emb, k_est, ml, cl, weight_method="log_degree_ratio", seed=seed)
    else:
        lbl_oracle = cequel_wcsc(emb, k_oracle, ml, cl, weight_method="log_degree_ratio", eigen=0.1, seed=seed)
        lbl_est = cequel_wcsc(emb, k_est, ml, cl, weight_method="log_degree_ratio", eigen=0.1, seed=seed)

    stats: Dict[str, Any] = {
        "total_calls": calls,
        "n_triangles_total": len(triangles),
        "n_ml": len(ml),
        "n_cl": len(cl),
        "clustering_method": clustering_method,
        "total_tokens": pt_sum + ct_sum,
        "prompt_tokens": pt_sum,
        "completion_tokens": ct_sum,
        "shown_posts": int(all_shown.sum()),
        "parse_errors": errs,
        "llm_api_sec": round(llm_sec, 4),
        "total_wall_sec": round(time.perf_counter() - t0, 4),
    }
    if include_wckmeans and clustering_method != "wckmeans":
        stats["wckmeans_oracleK"] = cequel_weighted_pckmeans(emb, k_oracle, ml, cl, weight_method="log_degree_ratio", seed=seed)
        stats["wckmeans_estK"] = cequel_weighted_pckmeans(emb, k_est, ml, cl, weight_method="log_degree_ratio", seed=seed)

    return lbl_oracle, lbl_est, all_shown, stats


# ===========================================================================
# 6. LSED (Qiu et al., ACL 2025 — 2025.acl-long.233)
# ===========================================================================

def build_lsed_official_prompt(text: str) -> Tuple[str, str]:
    """Exact verbatim prompt from LSED Section 4.1 (ACL 2025)."""
    user_msg = f"Summarize the following sentences:{text}. If you come across some abbreviations, expand them. Please respond in English"
    return "", user_msg


def _ole_time_vec(timestamps: np.ndarray) -> np.ndarray:
    """Exact TIME(t) encoding from LSED Eq. (7): (t_days / 100000, t_seconds / 86400)."""
    ole_offset_sec = 2209161600.0
    valid = ~np.isnan(timestamps)
    out = np.zeros((len(timestamps), 2), dtype=np.float32)
    if np.any(valid):
        sec_since_ole = timestamps[valid] + ole_offset_sec
        t_days = np.floor(sec_since_ole / 86400.0)
        t_secs = sec_since_ole - t_days * 86400.0
        out[valid, 0] = (t_days / 100000.0).astype(np.float32)
        out[valid, 1] = (t_secs / 86400.0).astype(np.float32)
    return out


def _poincare_proj(x, c: float = 1.0, eps: float = 1e-5):
    import torch

    max_norm = (1.0 / math.sqrt(c)) - eps
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    return torch.where(norm > max_norm, x * (max_norm / norm.clamp(min=1e-9)), x)


def _poincare_exp0(v, c: float = 1.0):
    import torch

    sqrt_c = math.sqrt(c)
    norm = torch.norm(v, p=2, dim=-1, keepdim=True).clamp(min=1e-7)
    y = (torch.tanh(sqrt_c * norm) / (sqrt_c * norm)) * v
    return _poincare_proj(y, c=c)


def _poincare_log0(y, c: float = 1.0):
    import torch

    sqrt_c = math.sqrt(c)
    max_norm = (1.0 / sqrt_c) - 1e-5
    norm = torch.norm(y, p=2, dim=-1, keepdim=True).clamp(min=1e-7, max=max_norm)
    return (torch.atanh(sqrt_c * norm) / (sqrt_c * norm)) * y


def _poincare_mobius_add(x, y, c: float = 1.0):
    import torch

    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1.0 + 2.0 * c * xy + c * y2) * x + (1.0 - c * x2) * y
    denom = (1.0 + 2.0 * c * xy + (c * c) * x2 * y2).clamp(min=1e-7)
    return _poincare_proj(num / denom, c=c)


def _build_lsed_hgcn_model(in_dim: int, hidden_dim: int, n_cls: int, num_layers: int = 3, c: float = 1.0, dropout: float = 0.0):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _HypLinear(nn.Module):
        def __init__(self, d_in: int, d_out: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.empty(d_out, d_in))
            self.bias = nn.Parameter(torch.zeros(d_out))
            nn.init.xavier_uniform_(self.weight)
            self.drop = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            tan_x = self.drop(_poincare_log0(x, c=c))
            res = _poincare_exp0(F.linear(tan_x, self.weight), c=c)
            b_hyp = _poincare_exp0(self.bias, c=c)
            return _poincare_mobius_add(res, b_hyp, c=c)

    class _HypAct(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.act = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return _poincare_exp0(self.act(_poincare_log0(x, c=c)), c=c)

    class _LSEDHGCNPoincareEncoder(nn.Module):
        """Genuine 3-layer HGCN Poincare Hyperbolic MLP (Chami et al., NeurIPS 2019; LSED §4.3)."""

        def __init__(self) -> None:
            super().__init__()
            self.c = float(c)
            blocks: List[nn.Module] = []
            d_cur = in_dim
            for li in range(max(1, int(num_layers))):
                blocks.append(_HypLinear(d_cur, hidden_dim))
                if li < num_layers - 1:
                    blocks.append(_HypAct())
                d_cur = hidden_dim
            self.hyp_layers = nn.Sequential(*blocks)
            self.cls = nn.Linear(hidden_dim, n_cls)

        def encode_hyperbolic(self, x_euc: torch.Tensor) -> torch.Tensor:
            h0 = _poincare_exp0(x_euc, c=self.c)
            return _poincare_proj(self.hyp_layers(h0), c=self.c)

        def forward(self, x_euc: torch.Tensor) -> torch.Tensor:
            e_hyp = self.encode_hyperbolic(x_euc)
            tan_e = _poincare_log0(e_hyp, c=self.c)
            return self.cls(tan_e)

    return _LSEDHGCNPoincareEncoder()


def train_lsed_hyperbolic_encoder(
    ds_m0: SocialDataset,
    emb_m0: np.ndarray,
    ckpt_path: Path,
    hidden_dim: int = 64,
    num_layers: int = 3,
    curvature: float = 1.0,
    lr: float = 0.01,
    epochs: int = 200,
    batch_size: Optional[int] = None,
    weight_decay: float = 1e-4,
    tan_reg: float = 0.02,
    dropout: float = 0.0,
    patience: int = 30,
    seed: int = 42,
    precomputed_m0_responses: Optional[Dict[str, Dict[str, Any]]] = None,
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    """Train LSED's 3-layer HGCN Poincare encoder (hidden_dim=64) on M_00 (70/10/20 split)."""
    import torch
    import torch.nn as nn

    _, y_inv = np.unique(ds_m0.labels, return_inverse=True)
    n_cls = int(y_inv.max()) + 1
    in_dim = int(emb_m0.shape[1]) + 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = _build_lsed_hgcn_model(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        n_cls=n_cls,
        num_layers=num_layers,
        c=curvature,
        dropout=dropout,
    ).to(device)

    m0_summarized = False
    if ckpt_path.exists():
        loaded = torch.load(ckpt_path, map_location=device, weights_only=True)
        if (
            isinstance(loaded, dict)
            and loaded.get("arch") == "hgcn_poincare_v2"
            and int(loaded.get("in_dim", -1)) == in_dim
            and int(loaded.get("hidden_dim", -1)) == hidden_dim
            and int(loaded.get("num_layers", -1)) == num_layers
            and int(loaded.get("n_cls", -1)) == n_cls
        ):
            # Retrain if new M_00 summaries are provided while existing checkpoint used raw M_00
            has_pre = bool(precomputed_m0_responses)
            if loaded.get("m0_summarized", False) or not has_pre:
                model.load_state_dict(loaded["state_dict"])
                model.eval()
                return model, device

    emb_used = _ensure_l2_normalized(emb_m0).astype(np.float32).copy()
    if precomputed_m0_responses:
        rep_idx: List[int] = []
        rep_texts: List[str] = []
        for i in range(len(ds_m0)):
            rec = precomputed_m0_responses.get(f"lsed_M00_n{i}")
            if rec is not None:
                txt_out = str(rec.get("text", "")).strip()
                if txt_out:
                    rep_idx.append(i)
                    rep_texts.append(txt_out)
        if rep_idx:
            new_vecs = encode_texts_hf(rep_texts, model_name=embed_model, batch_size=64, max_length=256)
            new_vecs = _ensure_l2_normalized(new_vecs)
            for pos, idx in enumerate(rep_idx):
                emb_used[idx] = new_vecs[pos]
            m0_summarized = len(rep_idx) >= int(0.8 * len(ds_m0))

    feat = np.hstack([emb_used, _ole_time_vec(ds_m0.timestamps)]).astype(np.float32)

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(feat)
    perm = rng.permutation(n)
    n_tr = max(1, int(0.7 * n))
    n_va = max(1, int(0.1 * n))
    idx_tr = perm[:n_tr]
    idx_va = perm[n_tr : min(n, n_tr + n_va)]
    idx_te = perm[min(n, n_tr + n_va) :]
    if len(idx_va) == 0:
        idx_va = idx_tr
    if len(idx_te) == 0:
        idx_te = idx_va

    x_tr = torch.from_numpy(feat[idx_tr]).to(device)
    y_tr = torch.from_numpy(y_inv[idx_tr].astype(np.int64)).to(device)
    x_va = torch.from_numpy(feat[idx_va]).to(device)
    y_va = torch.from_numpy(y_inv[idx_va].astype(np.int64)).to(device)
    x_te = torch.from_numpy(feat[idx_te]).to(device)
    y_te = torch.from_numpy(y_inv[idx_te].astype(np.int64)).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0
    bs = len(x_tr) if (batch_size is None or int(batch_size) >= len(x_tr)) else max(64, int(batch_size))

    for _ep in range(epochs):
        model.train()
        if bs >= len(x_tr):
            opt.zero_grad()
            e_hyp = model.encode_hyperbolic(x_tr)
            tan_e = _poincare_log0(e_hyp, c=curvature)
            loss = loss_fn(model.cls(tan_e), y_tr)
            if tan_reg > 0.0:
                loss = loss + float(tan_reg) * tan_e.pow(2).sum(dim=-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        else:
            order_tr = torch.randperm(len(x_tr), device=device)
            for s in range(0, len(x_tr), bs):
                b_idx = order_tr[s : s + bs]
                opt.zero_grad()
                e_hyp = model.encode_hyperbolic(x_tr[b_idx])
                tan_e = _poincare_log0(e_hyp, c=curvature)
                loss = loss_fn(model.cls(tan_e), y_tr[b_idx])
                if tan_reg > 0.0:
                    loss = loss + float(tan_reg) * tan_e.pow(2).sum(dim=-1).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

        model.eval()
        with torch.no_grad():
            va_logits = model(x_va)
            va_loss = float(loss_fn(va_logits, y_va).item())
        if va_loss < best_loss - 1e-4:
            best_loss = va_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        va_acc = float((model(x_va).argmax(dim=-1) == y_va).float().mean().item())
        te_acc = float((model(x_te).argmax(dim=-1) == y_te).float().mean().item())
        hyp_va = model.encode_hyperbolic(x_va)
        hyp_norms = torch.norm(hyp_va, p=2, dim=-1)
        mean_hyp_norm = float(hyp_norms.mean().item())
        sat_pct = float((hyp_norms > (0.999 / math.sqrt(curvature))).float().mean().item() * 100.0)

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arch": "hgcn_poincare_v2",
            "state_dict": best_state,
            "in_dim": in_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "n_cls": n_cls,
            "curvature": float(curvature),
            "m0_summarized": m0_summarized,
            "val_acc": round(va_acc, 4),
            "test_acc": round(te_acc, 4),
            "mean_hyp_norm": round(mean_hyp_norm, 4),
            "sat_pct": round(sat_pct, 2),
        },
        ckpt_path,
    )
    print(
        f"  [LSED HGCN M_00] m0_summarized={m0_summarized} val_acc={va_acc:.4f} "
        f"test_acc={te_acc:.4f} mean||hyp||={mean_hyp_norm:.4f} sat={sat_pct:.2f}%"
    )
    return model, device


def run_lsed_official(
    block_id: int,
    ds: SocialDataset,
    embeddings_raw: np.ndarray,
    hyp_model: Any,
    device: str,
    k_oracle: int,
    k_est: int,
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    client: Optional[BudgetedLLMClient] = None,
    bucket: Optional[Any] = None,
    precomputed_responses: Optional[Dict[str, Dict[str, Any]]] = None,
    window_size: Optional[int] = None,
    seed: int = 42,
    strategy_tag: str = "lsed_official",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Faithful reproduction of LSED (ACL 2025): verbatim prompt -> SBERT(256)+TIME -> HGCN Poincare -> KMeans."""
    import torch

    t0 = time.perf_counter()
    n = len(ds)
    emb_base = _ensure_l2_normalized(embeddings_raw).astype(np.float32)
    all_shown = np.zeros(n, dtype=bool)
    rewritten_idx: List[int] = []
    rewritten_texts: List[str] = []
    calls = pt_sum = ct_sum = errs = 0
    llm_sec = 0.0

    for i in range(n):
        req_id = f"lsed_M{block_id:02d}_n{i}"
        sys_msg, user_msg = build_lsed_official_prompt(ds.texts[i])
        if precomputed_responses is not None and req_id in precomputed_responses:
            rec = precomputed_responses[req_id]
            pt = int(rec.get("prompt_tokens", int(len(user_msg) / 3.4)))
            ct = int(rec.get("completion_tokens", 20))
            if bucket is not None:
                est = int(len(user_msg) / 3.4) + 64
                if not bucket.can_spend(est):
                    break
                bucket.record_spend(pt, ct)
            txt_out = str(rec.get("text", "")).strip()
            pt_sum += pt
            ct_sum += ct
            calls += 1
        else:
            assert client is not None
            res = client.call_text(
                request_id=req_id,
                block_id=block_id,
                strategy=strategy_tag,
                query_type="lsed_summarize",
                sys_msg=sys_msg,
                user_msg=user_msg,
                bucket=bucket,
                est_completion_tokens=64,
                max_tokens_override=256,
            )
            if res is None:
                break
            txt_out, pt, ct, lat, status = res
            pt_sum += pt
            ct_sum += ct
            llm_sec += lat
            calls += 1
            if status.startswith("HTTP_ERROR"):
                errs += 1
                continue

        all_shown[i] = True
        if txt_out:
            rewritten_idx.append(i)
            rewritten_texts.append(txt_out)
        else:
            errs += 1

    emb_final = emb_base.copy()
    if rewritten_idx:
        new_vecs = encode_texts_hf(rewritten_texts, model_name=embed_model, batch_size=64, max_length=256)
        new_vecs = _ensure_l2_normalized(new_vecs)
        for pos, idx in enumerate(rewritten_idx):
            emb_final[idx] = new_vecs[pos]

    feat = np.hstack([emb_final, _ole_time_vec(ds.timestamps)]).astype(np.float32)
    with _TORCH_INFER_LOCK, torch.no_grad():
        x_t = torch.from_numpy(feat).to(device)
        hyp_e = hyp_model.encode_hyperbolic(x_t).cpu().numpy().astype(np.float64)
        mean_hyp_norm = float(np.linalg.norm(hyp_e, axis=1).mean()) if len(hyp_e) else 0.0

    # Paper Fig. 6 w=infty open-set default: unsupervised KMeans on hyperbolic embeddings
    if window_size is not None and int(window_size) > 0:
        raise NotImplementedError(
            "Supervised periodic retraining (window_size > 0) is disabled to prevent GT leakage; "
            "LSED runs under the Paper Fig. 6 open-set setting (w=infty)."
        )
    lbl_oracle = KMeans(n_clusters=max(1, min(k_oracle, n)), random_state=seed, n_init=10).fit_predict(hyp_e)
    lbl_est = KMeans(n_clusters=max(1, min(k_est, n)), random_state=seed, n_init=10).fit_predict(hyp_e)

    return lbl_oracle.astype(np.int64), lbl_est.astype(np.int64), all_shown, {
        "total_calls": calls,
        "rewritten_ok": len(rewritten_idx),
        "total_tokens": pt_sum + ct_sum,
        "prompt_tokens": pt_sum,
        "completion_tokens": ct_sum,
        "shown_posts": int(all_shown.sum()),
        "parse_errors": errs,
        "mean_hyp_norm": round(mean_hyp_norm, 4),
        "open_set_window": "infty",
        "llm_api_sec": round(llm_sec, 4),
        "total_wall_sec": round(time.perf_counter() - t0, 4),
    }
