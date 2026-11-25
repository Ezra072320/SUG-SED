"""Disk cache for deterministic Stage-1 2D Structural Entropy initial partitions."""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy import sparse

from sug_sed_method.core.se_solver import solve_2d_se

_REPO_PARTITION_CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "partitions"
DEFAULT_PARTITION_CACHE_DIR = (
    _REPO_PARTITION_CACHE_DIR
    if _REPO_PARTITION_CACHE_DIR.exists()
    else Path(__file__).resolve().parents[1] / "cache" / "partitions"
)


def _adj_fingerprint(adj: sparse.spmatrix) -> str:
    """Compute a fast deterministic SHA-256 fingerprint of a CSR graph structure and weights."""
    csr = adj.tocsr()
    h = hashlib.sha256()
    h.update(f"{csr.shape[0]}x{csr.shape[1]}:{csr.nnz}".encode("ascii"))
    h.update(csr.indptr.astype(np.int64, copy=False).tobytes())
    h.update(csr.indices.astype(np.int64, copy=False).tobytes())
    h.update(np.round(csr.data.astype(np.float64, copy=False), 5).tobytes())
    return h.hexdigest()[:12]


def load_or_solve_2d_se(
    dataset_name: str,
    block_id: int,
    adj: sparse.csr_matrix,
    knn_k: int = 15,
    knn_sim_threshold: float = 0.35,
    attr_alpha: float = 0.5,
    refine_moves: bool = True,
    cache_dir: Optional[Path] = None,
    embed_model: Optional[str] = None,
) -> Tuple[np.ndarray, float]:
    """Load cached Stage-1 2D SE partition or solve and persist it atomically."""
    cdir = Path(cache_dir) if cache_dir is not None else DEFAULT_PARTITION_CACHE_DIR
    cdir.mkdir(parents=True, exist_ok=True)
    enc_tag = f"_enc_{embed_model.replace('/', '__')}" if embed_model else ""
    adj_hash = _adj_fingerprint(adj)
    fname = (
        f"{dataset_name}_M{int(block_id):02d}_k{int(knn_k)}"
        f"_tau{float(knn_sim_threshold):.3f}_a{float(attr_alpha):.3f}_ref{int(bool(refine_moves))}"
        f"{enc_tag}_g{adj_hash}.npz"
    )
    path = cdir / fname
    n = int(adj.shape[0])
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as data:
                labels = np.asarray(data["labels"], dtype=np.int64)
                solve_sec = float(data["solve_sec"])
            if len(labels) == n:
                return labels, solve_sec
        except Exception:
            pass

    t0 = time.time()
    labels, _ = solve_2d_se(adj, refine_moves=refine_moves)
    solve_sec = round(time.time() - t0, 3)
    tmp_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(dir=cdir, prefix=".tmp_", suffix=".npz", delete=False) as tf:
            tmp_name = tf.name
        np.savez_compressed(tmp_name, labels=np.asarray(labels, dtype=np.int64), solve_sec=np.float64(solve_sec))
        os.replace(tmp_name, path)
        tmp_name = None
    finally:
        if tmp_name is not None and os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except OSError:
                pass
    return np.asarray(labels, dtype=np.int64), solve_sec
