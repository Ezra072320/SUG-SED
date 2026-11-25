"""Disk-cached and in-memory cached HuggingFace sentence embedding utilities."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EMBED_CACHE_DIR = ROOT / "cache" / "embeddings"

_HF_MODEL_CACHE: Dict[Tuple[str, str], Tuple[Any, Any]] = {}


def encode_texts_hf(
    texts: Sequence[str],
    model_name: str,
    batch_size: int = 64,
    max_length: int = 128,
    device: Optional[str] = None,
) -> np.ndarray:
    """Extract L2-normalized mean-pooled embeddings using cached HuggingFace encoders."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)

    import torch
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_key = (model_name, device)
    if cache_key not in _HF_MODEL_CACHE:
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModel.from_pretrained(model_name)
        mdl.eval().to(device)
        _HF_MODEL_CACHE[cache_key] = (tok, mdl)
    tok, mdl = _HF_MODEL_CACHE[cache_key]

    out_chunks = []
    texts_list = list(texts)
    with torch.no_grad():
        for start in range(0, len(texts_list), batch_size):
            batch = texts_list[start : start + batch_size]
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            out = mdl(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            out_chunks.append(pooled.cpu().numpy().astype(np.float32))
    return np.vstack(out_chunks)


def load_or_extract_block_embeddings(
    dataset_name: str,
    block_id: int,
    texts: Sequence[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    cache_dir: Optional[Path] = None,
    storage_dtype: str = "float32",
) -> np.ndarray:
    """Load precomputed block embeddings from disk cache or extract on GPU/CPU."""
    cdir = Path(cache_dir) if cache_dir is not None else EMBED_CACHE_DIR
    cdir.mkdir(parents=True, exist_ok=True)
    safe_model = model_name.replace("/", "__")
    cache_path = cdir / f"{dataset_name}_M{int(block_id):02d}_{safe_model}.npy"
    if cache_path.exists():
        emb = np.load(cache_path).astype(np.float32)
        if len(emb) == len(texts):
            return emb

    t0 = time.time()
    emb = encode_texts_hf(texts, model_name=model_name, batch_size=64, max_length=128)
    save_dtype = np.float16 if storage_dtype == "float16" else np.float32
    emb_saved = emb.astype(save_dtype)
    np.save(cache_path, emb_saved)
    print(f"  [Emb] Extracted & cached {cache_path.name} shape={emb_saved.shape} ({save_dtype.__name__}) in {time.time()-t0:.1f}s")
    return emb_saved.astype(np.float32)
