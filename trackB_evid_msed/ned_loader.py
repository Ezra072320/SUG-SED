"""Leakage-safe concurrent NED loader, competitor pairs, and image dHash utilities.

Loads ONLY the requested split (`development` for Gate B) from NED_with_images.jsonl
and constructs a time-shortcut-free concurrent stream (`rel_timestamp = ts - min_ts(event)`)
so all events in the split overlap temporally. Never exposes `event_name` or calendar
year/date metadata to any model or prompt.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_NED_JSONL = Path(r"C:\Users\32730\Desktop\SMU\dataset\NED_with_images.jsonl")
DEFAULT_NED_ROOT = Path(r"C:\Users\32730\Desktop\SMU\dataset")
SPLITS_PATH = Path(__file__).resolve().parent / "ned_splits.json"

# Direct same-theme competing event pairs inside NED development split
DEV_COMPETITOR_PAIRS: Dict[str, str] = {
    "E15": "E23",  # World Series (baseball)
    "E23": "E15",
    "E22": "E33",  # Cyclones / storms (Hurricane vs Typhoon)
    "E33": "E22",
    "E38": "E40",  # Acute civil/natural disasters
    "E40": "E38",
    "E11": "E24",  # Urban attacks / protests
    "E24": "E11",
}


def load_image_rgb(path: Path) -> Optional[np.ndarray]:
    """Decode local image to uint8 RGB ndarray (H, W, 3) without relying on broken PIL DLL."""
    if not path.exists():
        return None
    try:
        import imageio.v3 as iio

        arr = iio.imread(path)
        if arr is None:
            return None
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            # Composite RGBA onto white background
            rgb = arr[..., :3].astype(np.float32)
            alpha = arr[..., 3:4].astype(np.float32) / 255.0
            arr = np.clip(rgb * alpha + 255.0 * (1.0 - alpha), 0, 255).astype(np.uint8)
        elif arr.ndim == 3 and arr.shape[2] > 3:
            arr = arr[..., :3]
        elif arr.ndim == 4:
            # Animated GIF -> first frame
            arr = arr[0][..., :3]
        return np.ascontiguousarray(arr, dtype=np.uint8)
    except Exception:
        pass

    # Fallback via PySide6 QImage
    try:
        from PySide6.QtGui import QImage

        qimg = QImage(str(path))
        if qimg.isNull():
            return None
        qimg = qimg.convertToFormat(QImage.Format.Format_RGB888)
        w, h = qimg.width(), qimg.height()
        bpl = qimg.bytesPerLine()
        buf = qimg.constBits()
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, bpl))[:, : w * 3].reshape((h, w, 3))
        return np.ascontiguousarray(arr, dtype=np.uint8)
    except Exception:
        return None


def encode_image_resized_jpeg_bytes(path: Path, max_side: int = 512, quality: int = 85) -> bytes:
    """Resize image cleanly via QImage to max_side and return compact JPEG bytes."""
    from PySide6.QtCore import QBuffer, QIODevice, Qt
    from PySide6.QtGui import QImage

    qimg = QImage(str(path))
    if qimg.isNull():
        # Fallback through imageio RGB -> QImage
        rgb = load_image_rgb(path)
        if rgb is None:
            raise ValueError(f"Unreadable image: {path}")
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()

    w, h = qimg.width(), qimg.height()
    if max(w, h) > max_side:
        if w >= h:
            qimg = qimg.scaledToWidth(max_side, Qt.TransformationMode.SmoothTransformation)
        else:
            qimg = qimg.scaledToHeight(max_side, Qt.TransformationMode.SmoothTransformation)

    qbuf = QBuffer()
    qbuf.open(QIODevice.OpenModeFlag.WriteOnly)
    ok = qimg.save(qbuf, "JPEG", quality)
    if not ok:
        raise RuntimeError(f"Failed to encode JPEG for {path}")
    return bytes(qbuf.data())


def compute_dhash64(rgb: np.ndarray) -> int:
    """Compute 64-bit difference hash (9x8 grayscale luminance) purely in NumPy."""
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
    h, w = gray.shape
    ys = np.linspace(0, max(0, h - 1), num=8).astype(np.int64)
    xs = np.linspace(0, max(0, w - 1), num=9).astype(np.int64)
    small = gray[np.ix_(ys, xs)]
    bits = small[:, 1:] > small[:, :-1]
    val = 0
    for idx, b in enumerate(bits.ravel().tolist()):
        if b:
            val |= 1 << idx
    return int(val)


def dhash_hamming(h1: int, h2: int) -> int:
    return int((int(h1) ^ int(h2)).bit_count())


def normalize_text_for_hash(text: str) -> str:
    s = re.sub(r"https?://\S+", "", text.lower())
    s = re.sub(r"\brt\s+@\w+:?", "", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def extract_surface_keywords(text: str) -> Tuple[List[str], List[str], List[str]]:
    """Extract lowercase hashtags, @mentions, and content tokens without external NLP deps."""
    clean = re.sub(r"https?://\S+", " ", text)
    hashtags = [h.lower() for h in re.findall(r"#([A-Za-z0-9_]+)", clean)]
    mentions = [m.lower() for m in re.findall(r"@([A-Za-z0-9_]+)", clean)]
    no_social = re.sub(r"[#@]\w+", " ", clean.lower())
    stop = {
        "the", "and", "for", "that", "with", "this", "from", "are", "was", "were",
        "have", "has", "had", "you", "your", "just", "now", "all", "out", "about",
        "into", "after", "over", "via", "more", "not", "but", "what", "when", "who",
        "how", "our", "their", "they", "will", "can", "one", "new", "day", "rt",
    }
    toks = [t for t in re.findall(r"[a-z]{3,}", no_social) if t not in stop]
    return list(dict.fromkeys(hashtags)), list(dict.fromkeys(mentions)), toks


@dataclass
class NEDDataset:
    split: str
    post_ids: List[str]
    event_ids: List[str]            # string e.g. 'E15'
    labels: np.ndarray              # integer 0..C-1 ground-truth event index
    label_to_event: List[str]       # mapping int -> 'E15'
    texts: List[str]
    text_hashes: List[str]
    raw_timestamps: np.ndarray      # unix seconds (not exposed to models)
    rel_timestamps: np.ndarray      # event-relative concurrent timeline (eliminates 72h shortcut)
    image_paths: List[Optional[Path]]
    image_ok: np.ndarray            # bool array indicating decoded local image exists
    hashtags: List[List[str]]
    mentions: List[List[str]]
    tokens: List[List[str]]

    def __len__(self) -> int:
        return len(self.post_ids)


def _parse_ts(time_str: Any) -> float:
    if not time_str:
        return 0.0
    s = str(time_str).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s[:19], fmt).timestamp()
        except ValueError:
            continue
    return 0.0


def load_ned_split(
    split: str = "development",
    jsonl_path: Path = DEFAULT_NED_JSONL,
    dataset_root: Path = DEFAULT_NED_ROOT,
    splits_path: Path = SPLITS_PATH,
) -> NEDDataset:
    """Load strictly one NED event split (`development`, `validation`, or `holdout`)."""
    splits_obj = json.loads(splits_path.read_text(encoding="utf-8"))
    allowed = set(splits_obj["splits"][split])

    raw_rows: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("event_id") in allowed:
                raw_rows.append(obj)

    # Sort deterministically by event_id then raw timestamp then post_id
    for r in raw_rows:
        r["_ts"] = _parse_ts(r.get("time"))
    raw_rows.sort(key=lambda r: (r["event_id"], r["_ts"], r["id"]))

    # Compute per-event relative timestamps so all events run concurrently from t=0
    min_ts_by_ev: Dict[str, float] = {}
    for r in raw_rows:
        ev = r["event_id"]
        if ev not in min_ts_by_ev or r["_ts"] < min_ts_by_ev[ev]:
            min_ts_by_ev[ev] = r["_ts"]
    for r in raw_rows:
        r["_rel_ts"] = float(r["_ts"] - min_ts_by_ev[r["event_id"]])

    # Re-sort chronologically along concurrent timeline!
    raw_rows.sort(key=lambda r: (r["_rel_ts"], r["id"]))

    sorted_evs = sorted(allowed)
    ev_to_int = {e: i for i, e in enumerate(sorted_evs)}

    post_ids: List[str] = []
    event_ids: List[str] = []
    labels: List[int] = []
    texts: List[str] = []
    text_hashes: List[str] = []
    raw_ts: List[float] = []
    rel_ts: List[float] = []
    img_paths: List[Optional[Path]] = []
    img_ok: List[bool] = []
    hashtags: List[List[str]] = []
    mentions: List[List[str]] = []
    tokens: List[List[str]] = []

    for r in raw_rows:
        ev = str(r["event_id"])
        txt = str(r.get("text") or "").strip()
        rel_p = r.get("image_path")
        ok = r.get("image_status") == "ok" and bool(rel_p)
        abs_p: Optional[Path] = (dataset_root / str(rel_p)) if ok else None
        if abs_p is not None and not abs_p.exists():
            abs_p = None
            ok = False

        ht, mn, tk = extract_surface_keywords(txt)
        th = hashlib.sha256(normalize_text_for_hash(txt).encode("utf-8")).hexdigest()[:16]

        post_ids.append(str(r["id"]))
        event_ids.append(ev)
        labels.append(ev_to_int[ev])
        texts.append(txt)
        text_hashes.append(th)
        raw_ts.append(float(r["_ts"]))
        rel_ts.append(float(r["_rel_ts"]))
        img_paths.append(abs_p)
        img_ok.append(bool(ok))
        hashtags.append(ht)
        mentions.append(mn)
        tokens.append(tk)

    return NEDDataset(
        split=split,
        post_ids=post_ids,
        event_ids=event_ids,
        labels=np.asarray(labels, dtype=np.int64),
        label_to_event=sorted_evs,
        texts=texts,
        text_hashes=text_hashes,
        raw_timestamps=np.asarray(raw_ts, dtype=np.float64),
        rel_timestamps=np.asarray(rel_ts, dtype=np.float64),
        image_paths=img_paths,
        image_ok=np.asarray(img_ok, dtype=bool),
        hashtags=hashtags,
        mentions=mentions,
        tokens=tokens,
    )
