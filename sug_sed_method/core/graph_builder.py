"""Sparse heterogeneous + semantic message graph construction for SED.

Constructs undirected weighted message graphs G = (V, E, W) combining:
  - Semantic top-k nearest neighbors (cosine similarity on normalized embeddings)
  - Attribute co-occurrence edges (shared entities, hashtags, users/mentions)
    with IDF / high-frequency hub capping so giant hubs do not explode O(n^2)
  - Exponential temporal decay exp(-lambda * |tau_i - tau_j|)

Also tracks edge provenance (attribute-only vs semantic) and non-entity text
similarity to power Track A's Intra-Cluster False-Merge Seam detector (U_seam).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from sug_sed_method.core.data_loader import SocialDataset


@dataclass
class MessageGraph:
    adj: sparse.csr_matrix              # Combined symmetric weighted adjacency (n x n)
    sem_adj: sparse.csr_matrix          # Pure semantic k-NN adjacency
    attr_adj: sparse.csr_matrix         # Pure attribute-sharing adjacency
    embeddings: np.ndarray              # L2-normalized dense embeddings (n x d)
    non_entity_tfidf: sparse.csr_matrix # L2-normalized TF-IDF with entity tokens stripped
    knn_sim_threshold: Optional[float] = None


def build_non_entity_tfidf(ds: SocialDataset) -> sparse.csr_matrix:
    """Build L2-normalized TF-IDF vectors after stripping entity tokens from each tweet."""
    docs: List[str] = []
    for lemmas, ents in zip(ds.tokens_lemma, ds.entities):
        ent_tokens = set()
        for e in ents:
            for tok in e.split():
                ent_tokens.add(tok.lower())
        kept = [t for t in lemmas if t not in ent_tokens and len(t) >= 2]
        docs.append(" ".join(kept))

    vec = TfidfVectorizer(min_df=2, max_df=0.8, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
    try:
        mat = vec.fit_transform(docs)
    except ValueError:
        mat = sparse.csr_matrix((len(docs), 1), dtype=np.float64)
    return normalize(mat, norm="l2", axis=1).tocsr()


def compute_tfidf_svd_embeddings(
    ds: SocialDataset,
    n_components: int = 256,
    seed: int = 42,
) -> np.ndarray:
    """Fast zero-download lexical/subword baseline embeddings (TF-IDF + TruncatedSVD)."""
    from sklearn.decomposition import TruncatedSVD

    vec = TfidfVectorizer(
        min_df=2,
        max_df=0.85,
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=30000,
    )
    tfidf = vec.fit_transform(ds.texts)
    dim = min(n_components, max(2, min(tfidf.shape) - 1))
    svd = TruncatedSVD(n_components=dim, random_state=seed)
    emb = svd.fit_transform(tfidf).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = np.divide(emb, np.maximum(norms, 1e-12))
    return emb


def compute_hf_embeddings(
    texts: List[str],
    model_name: str,
    batch_size: int = 64,
    max_length: int = 128,
    device: Optional[str] = None,
) -> np.ndarray:
    """Extract L2-normalized mean-pooled embeddings using HuggingFace transformers."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval().to(device)

    all_embs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            out = model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            all_embs.append(pooled.cpu().numpy().astype(np.float32))

    return np.vstack(all_embs)


def _ensure_l2_normalized(embeddings: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    """Return float32 embeddings with every non-zero row L2-normalized to unit length."""
    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim != 2 or emb.shape[0] == 0:
        return emb
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    nz = norms.ravel() > 1e-12
    if np.any(nz) and float(np.max(np.abs(norms.ravel()[nz] - 1.0))) > tol:
        emb = np.divide(emb, np.maximum(norms, 1e-12)).astype(np.float32)
    return emb


def _build_semantic_knn(
    embeddings: np.ndarray,
    timestamps: np.ndarray,
    k: int = 15,
    sim_threshold: float = 0.25,
    time_decay_per_day: float = 0.0,
    batch_size: int = 1024,
) -> sparse.csr_matrix:
    """Exact cosine top-k neighbor graph computed in memory-safe batches."""
    n = embeddings.shape[0]
    if n <= 1:
        return sparse.csr_matrix((n, n), dtype=np.float64)
    k_eff = min(k, n - 1)
    emb = _ensure_l2_normalized(embeddings)

    rows_list: List[np.ndarray] = []
    cols_list: List[np.ndarray] = []
    vals_list: List[np.ndarray] = []

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        sims = emb[start:end] @ emb.T  # (B, N)
        # Mask self-similarity
        for bi in range(end - start):
            sims[bi, start + bi] = -2.0

        # Argpartition top-k per row
        top_idx = np.argpartition(-sims, kth=k_eff - 1, axis=1)[:, :k_eff]
        row_arange = np.arange(end - start)[:, None]
        top_vals = sims[row_arange, top_idx]

        mask = top_vals >= sim_threshold
        r_idx = np.broadcast_to(np.arange(start, end)[:, None], top_idx.shape)[mask]
        c_idx = top_idx[mask]
        v_val = top_vals[mask].astype(np.float64)

        if time_decay_per_day > 0.0 and len(r_idx) > 0:
            dt_days = np.abs(timestamps[r_idx] - timestamps[c_idx]) / 86400.0
            decay = np.where(np.isnan(dt_days), 1.0, np.exp(-time_decay_per_day * dt_days))
            v_val = v_val * decay

        rows_list.append(r_idx)
        cols_list.append(c_idx)
        vals_list.append(v_val)

    rows = np.concatenate(rows_list) if rows_list else np.array([], dtype=np.int64)
    cols = np.concatenate(cols_list) if cols_list else np.array([], dtype=np.int64)
    vals = np.concatenate(vals_list) if vals_list else np.array([], dtype=np.float64)

    mat = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)
    # Symmetrize via max(A, A^T) (mutual or directed top-k union)
    mat = mat.maximum(mat.T)
    mat.setdiag(0.0)
    mat.eliminate_zeros()
    return mat.tocsr()


def _build_attribute_graph(
    ds: SocialDataset,
    embeddings: np.ndarray,
    max_bucket_size: int = 300,
    max_neighbors_per_attr: int = 12,
    min_sem_sim: float = 0.05,
    time_decay_per_day: float = 0.0,
    entity_weight: float = 1.0,
    hashtag_weight: float = 1.2,
    user_weight: float = 0.6,
) -> sparse.csr_matrix:
    """Build sparse attribute-cooccurrence graph with hub capping & semantic gating.

    Connects each post to its temporally closest `max_neighbors_per_attr` peers
    sharing that attribute, weighted by cosine similarity and IDF-like bucket
    dampening. For large hub buckets (`m > max_bucket_size`), tightens the local
    temporal window and semantic similarity floor instead of dropping the bucket.
    """
    n = len(ds)
    if n <= 1:
        return sparse.csr_matrix((n, n), dtype=np.float64)

    attr_buckets: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for i in range(n):
        for e in ds.entities[i]:
            attr_buckets[("ent", e)].append(i)
        for h in ds.hashtags[i]:
            attr_buckets[("tag", h)].append(i)
        if ds.user_ids[i]:
            attr_buckets[("usr", ds.user_ids[i])].append(i)
        for mu in ds.mentioned_users[i]:
            attr_buckets[("usr", mu)].append(i)

    type_weights = {"ent": entity_weight, "tag": hashtag_weight, "usr": user_weight}
    edge_weights: Dict[Tuple[int, int], float] = {}
    ts = ds.timestamps

    for (atype, _aval), nodes in attr_buckets.items():
        m = len(nodes)
        if m < 2:
            continue
        # Sort bucket chronologically (NaN timestamps placed at end) so temporal linking is O(m * window)
        nodes_sorted = sorted(
            nodes,
            key=lambda idx: (bool(np.isnan(ts[idx])), 0.0 if np.isnan(ts[idx]) else float(ts[idx])),
        )
        # IDF-like dampening for frequent attributes
        base_w = type_weights[atype] / np.log2(2.0 + m)
        half_win = max(1, max_neighbors_per_attr // 2)
        if m > max_bucket_size:
            eff_half_win = max(1, min(half_win, 4))
            eff_min_sim = max(min_sem_sim, 0.35)
        else:
            eff_half_win = half_win
            eff_min_sim = min_sem_sim

        for pos, u in enumerate(nodes_sorted):
            hi = min(m, pos + eff_half_win + 1)
            for pos2 in range(pos + 1, hi):
                v = nodes_sorted[pos2]
                sim = float(np.dot(embeddings[u], embeddings[v]))
                if sim < eff_min_sim:
                    continue
                w = base_w * max(0.0, sim)
                if time_decay_per_day > 0.0 and not (np.isnan(ts[u]) or np.isnan(ts[v])):
                    dt_days = abs(ts[u] - ts[v]) / 86400.0
                    w *= float(np.exp(-time_decay_per_day * dt_days))
                if w <= 0.0:
                    continue
                key = (u, v) if u < v else (v, u)
                edge_weights[key] = edge_weights.get(key, 0.0) + w

    if not edge_weights:
        return sparse.csr_matrix((n, n), dtype=np.float64)

    rows, cols, vals = [], [], []
    for (u, v), w in edge_weights.items():
        rows.extend([u, v])
        cols.extend([v, u])
        vals.extend([w, w])

    mat = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)
    mat.setdiag(0.0)
    mat.eliminate_zeros()
    return mat.tocsr()


def build_message_graph(
    ds: SocialDataset,
    embeddings: np.ndarray,
    knn_k: int = 15,
    knn_sim_threshold: float = 0.25,
    attr_alpha: float = 0.5,
    time_decay_per_day: float = 0.0,
    max_attr_bucket_size: int = 300,
    compute_non_entity_tfidf: bool = True,
) -> MessageGraph:
    """Build combined sparse heterogeneous + semantic message graph."""
    embeddings = _ensure_l2_normalized(embeddings)
    sem_adj = _build_semantic_knn(
        embeddings=embeddings,
        timestamps=ds.timestamps,
        k=knn_k,
        sim_threshold=knn_sim_threshold,
        time_decay_per_day=time_decay_per_day,
    )
    attr_adj = _build_attribute_graph(
        ds=ds,
        embeddings=embeddings,
        max_bucket_size=max_attr_bucket_size,
        time_decay_per_day=time_decay_per_day,
    )
    combined = (sem_adj + attr_alpha * attr_adj).tocsr()
    combined.setdiag(0.0)
    combined.eliminate_zeros()
    non_ent_tfidf = (
        build_non_entity_tfidf(ds)
        if compute_non_entity_tfidf
        else sparse.csr_matrix((len(ds), 1), dtype=np.float64)
    )
    return MessageGraph(
        adj=combined,
        sem_adj=sem_adj,
        attr_adj=attr_adj,
        embeddings=embeddings,
        non_entity_tfidf=non_ent_tfidf,
        knn_sim_threshold=knn_sim_threshold,
    )
