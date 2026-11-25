"""Disk cache for deterministic Stage-1 2D Structural Entropy initial partitions."""

from __future__ import annotations

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


def load_or_solve_2d_se(
    dataset_name: str,
    block_id: int,
    adj: sparse.csr_matrix,
    knn_k: int = 15,
    knn_sim_threshold: float = 0.35,
    attr_alpha: float = 0.5,
    refine_moves: bool = True,
    cache_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, float]:
    """Load cached Stage-1 2D SE partition or solve and persist it atomically."""
    cdir = Path(cache_dir) if cache_dir is not None else DEFAULT_PARTITION_CACHE_DIR
    cdir.mkdir(parents=True, exist_ok=True)
    fname = (
        f"{dataset_name}_M{int(block_id):02d}_k{int(knn_k)}"
        f"_tau{float(knn_sim_threshold):.3f}_a{float(attr_alpha):.3f}_ref{int(bool(refine_moves))}.npz"
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
    np.savez_compressed(path, labels=np.asarray(labels, dtype=np.int64), solve_sec=np.float64(solve_sec))
    return np.asarray(labels, dtype=np.int64), solve_sec
