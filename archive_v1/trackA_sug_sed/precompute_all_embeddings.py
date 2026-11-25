"""Pre-extract and cache HuggingFace embeddings for all streaming blocks locally on GPU.

Strictly offline-safe when HF_HUB_OFFLINE=1 (models already cached locally).
Does NOT touch any labels or run any evaluation; purely caches dense embeddings
for M_00..M_21 (Event2012) and M_00..M_16 (Event2018) so full streaming runs
take seconds instead of minutes.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_shared.data_loader import load_socialed_npy, split_daily_blocks

CACHE_DIR = ROOT / "cache" / "embeddings"


def extract_dataset_blocks(dataset_name: str, model_name: str, batch_size: int = 128, max_length: int = 128) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = model_name.replace("/", "__")

    ds = load_socialed_npy(dataset_name, sort_chronologically=True)
    blocks = split_daily_blocks(ds, merge_initial_days=7, min_block_size=200)

    missing = []
    for bid, (bname, bidx) in enumerate(blocks):
        cp = CACHE_DIR / f"{dataset_name}_M{bid:02d}_{safe_model}.npy"
        if cp.exists():
            arr = np.load(cp)
            if arr.shape[0] == len(bidx):
                continue
        missing.append((bid, bname, bidx, cp))

    if not missing:
        print(f"[Cache OK] All {len(blocks)} blocks of {dataset_name} ({model_name}) already cached.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Extract] {dataset_name} ({model_name}) on {device}: {len(missing)}/{len(blocks)} blocks to extract...")
    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    print(f"  Loaded model in {time.time() - t_load:.1f}s")

    for bid, bname, bidx, cp in missing:
        texts = [ds.texts[i] for i in bidx.tolist()]
        t0 = time.time()
        all_embs = []
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
        emb = np.vstack(all_embs)
        np.save(cp, emb.astype(np.float16))
        print(f"  M_{bid:02d} [{bname}] n={len(texts):5d} -> {cp.name} ({time.time() - t0:.1f}s)")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    extract_dataset_blocks("Event2012", "sentence-transformers/all-MiniLM-L6-v2", batch_size=128)
    extract_dataset_blocks("Event2018", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", batch_size=128)
    print("[Done] All embeddings cached cleanly.")


if __name__ == "__main__":
    main()
