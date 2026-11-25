"""Pre-extract and cache offline text & CLIP embeddings + 64-bit image dHash for NED splits."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPTokenizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trackB_evid_msed.ned_loader import compute_dhash64, dhash_hamming, load_image_rgb, load_ned_split

CACHE_DIR = ROOT / "cache" / "trackB"

# OpenAI CLIP exact normalization constants
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(1, 3, 1, 1)


def _preprocess_rgb_for_clip(rgb: np.ndarray) -> torch.Tensor:
    """Center-crop shortest side and bicubic resize to 224x224 matching CLIPImageProcessor."""
    h, w, _ = rgb.shape
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = rgb[y0 : y0 + side, x0 : x0 + side]
    t = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    t = F.interpolate(t, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
    return t.clamp(0.0, 1.0)


def extract_and_cache(split: str = "development", batch_size: int = 64) -> Path:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    out_dir = CACHE_DIR / f"ned_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    p_txt_minilm = out_dir / "text_minilm.npy"
    p_clip_img = out_dir / "clip_img.npy"
    p_clip_txt = out_dir / "clip_txt.npy"
    p_meta = out_dir / "image_meta.npz"

    if p_txt_minilm.exists() and p_clip_img.exists() and p_clip_txt.exists() and p_meta.exists():
        print(f"[Cache OK] All NED '{split}' features already cached in {out_dir}")
        return out_dir

    ds = load_ned_split(split=split)
    n = len(ds)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[NED Extract] split={split} n={n} raw_img_ok={int(ds.image_ok.sum())} on {device}...")

    # 1. Extract dense MiniLM text embeddings
    t0 = time.time()
    minilm_name = "sentence-transformers/all-MiniLM-L6-v2"
    tok_m = AutoTokenizer.from_pretrained(minilm_name)
    mod_m = AutoModel.from_pretrained(minilm_name).eval().to(device)
    embs_m = []
    with torch.no_grad():
        for st in range(0, n, batch_size):
            batch = [x if x else " " for x in ds.texts[st : st + batch_size]]
            enc = tok_m(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            out = mod_m(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = F.normalize(pooled, p=2, dim=-1)
            embs_m.append(pooled.cpu().numpy().astype(np.float32))
    txt_minilm = np.vstack(embs_m)
    del mod_m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  MiniLM text shape={txt_minilm.shape} in {time.time()-t0:.1f}s")

    # 2. Extract CLIP ViT-B/32 text & image embeddings + 64-bit dHash
    t1 = time.time()
    clip_name = "openai/clip-vit-base-patch32"
    tok_c = CLIPTokenizer.from_pretrained(clip_name)
    mod_c = CLIPModel.from_pretrained(clip_name).eval().to(device)

    embs_ct = []
    with torch.no_grad():
        for st in range(0, n, batch_size):
            batch = [x if x else " " for x in ds.texts[st : st + batch_size]]
            enc = tok_c(batch, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
            feat = mod_c.get_text_features(**enc)
            if not isinstance(feat, torch.Tensor):
                feat = feat.pooler_output
            feat = F.normalize(feat, p=2, dim=-1)
            embs_ct.append(feat.cpu().numpy().astype(np.float32))
    clip_txt = np.vstack(embs_ct)
    print(f"  CLIP text shape={clip_txt.shape} in {time.time()-t1:.1f}s")

    t2 = time.time()
    clip_img = np.zeros((n, 512), dtype=np.float32)
    decoded_ok = np.zeros(n, dtype=bool)
    dhash64 = np.zeros(n, dtype=np.uint64)
    decode_failures = 0

    batch_tensors = []
    batch_indices = []

    def _flush_img_batch() -> None:
        if not batch_tensors:
            return
        px = (torch.cat(batch_tensors, dim=0) - CLIP_MEAN) / CLIP_STD
        px = px.to(device)
        with torch.no_grad():
            feat = mod_c.get_image_features(pixel_values=px)
            if not isinstance(feat, torch.Tensor):
                feat = feat.pooler_output
            feat = F.normalize(feat, p=2, dim=-1)
            arr = feat.cpu().numpy().astype(np.float32)
        for pos, gidx in enumerate(batch_indices):
            clip_img[gidx] = arr[pos]
        batch_tensors.clear()
        batch_indices.clear()

    for i in range(n):
        if not ds.image_ok[i] or ds.image_paths[i] is None:
            continue
        rgb = load_image_rgb(ds.image_paths[i])
        if rgb is None:
            decode_failures += 1
            continue
        decoded_ok[i] = True
        dhash64[i] = np.uint64(compute_dhash64(rgb))
        batch_tensors.append(_preprocess_rgb_for_clip(rgb))
        batch_indices.append(i)
        if len(batch_tensors) >= batch_size:
            _flush_img_batch()
    _flush_img_batch()

    del mod_c
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    n_valid = int(decoded_ok.sum())
    print(f"  CLIP image valid={n_valid}/{int(ds.image_ok.sum())} (decode_fail={decode_failures}) in {time.time()-t2:.1f}s")

    # Quick audit of image exact/near-duplicates inside the split
    valid_hashes = dhash64[decoded_ok].tolist()
    uniq_exact = len(set(valid_hashes))
    print(f"  Image dHash64 unique={uniq_exact}/{n_valid} ({(1 - uniq_exact/max(1, n_valid))*100:.1f}% exact dHash repeats)")

    np.save(p_txt_minilm, txt_minilm.astype(np.float32))
    np.save(p_clip_img, clip_img.astype(np.float32))
    np.save(p_clip_txt, clip_txt.astype(np.float32))
    np.savez(p_meta, decoded_ok=decoded_ok, dhash64=dhash64)
    print(f"[Saved] Cached all features cleanly to {out_dir}")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-extract and cache offline features for NED splits.")
    parser.add_argument("--splits", nargs="+", default=["development"], choices=["development", "validation", "holdout"])
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    for sp in args.splits:
        extract_and_cache(split=sp, batch_size=args.batch_size)
