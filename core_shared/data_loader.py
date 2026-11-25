"""Unified data loader and leakage-safe streaming block splitter for SED datasets.

Supports SocialED-format .npy datasets already stored locally:
  - Event2012.npy (English Twitter, 68,841 tweets, 503 events)
  - Event2012_100.npy (English Twitter 100-event subset, 15,019 tweets)
  - Event2018.npy (French Twitter, 64,516 tweets, 257 events)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_DATASET_DIR = Path(
    os.environ.get(
        "SED_DATASET_DIR",
        str(Path(__file__).resolve().parent.parent.parent / "dataset" / "Others"),
    )
)


@dataclass
class SocialDataset:
    name: str
    tweet_ids: np.ndarray            # int64 / str
    texts: List[str]
    labels: np.ndarray               # int64 ground-truth event labels
    tokens_lemma: List[List[str]]
    entities: List[List[str]]        # normalized lowercase entity names
    hashtags: List[List[str]]        # normalized lowercase hashtags (without '#')
    user_ids: List[str]
    mentioned_users: List[List[str]]
    timestamps: np.ndarray           # float64 unix epoch seconds
    dates: List[str]                 # YYYY-MM-DD strings

    def __len__(self) -> int:
        return len(self.texts)

    def subset(self, indices: Sequence[int], name_suffix: str = "") -> "SocialDataset":
        idx = np.asarray(indices, dtype=np.int64)
        return SocialDataset(
            name=f"{self.name}{name_suffix}",
            tweet_ids=self.tweet_ids[idx],
            texts=[self.texts[i] for i in idx],
            labels=self.labels[idx],
            tokens_lemma=[self.tokens_lemma[i] for i in idx],
            entities=[self.entities[i] for i in idx],
            hashtags=[self.hashtags[i] for i in idx],
            user_ids=[self.user_ids[i] for i in idx],
            mentioned_users=[self.mentioned_users[i] for i in idx],
            timestamps=self.timestamps[idx],
            dates=[self.dates[i] for i in idx],
        )


def _normalize_entities(raw: Any) -> List[str]:
    """Handle both Event2012 [('UP', 'ORG'), ...] and Event2018 ['Djamel Beghal', ...]."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    out: List[str] = []
    if isinstance(raw, (list, tuple, np.ndarray)):
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                ent = str(item[0]).strip().lower()
                ent_type = str(item[1]).upper() if len(item) >= 2 else ""
                # Skip noisy generic DATE/TIME/CARDINAL/ORDINAL entities if typed
                if ent_type in {"DATE", "TIME", "CARDINAL", "ORDINAL", "QUANTITY", "PERCENT", "MONEY"}:
                    continue
                if len(ent) >= 2:
                    out.append(ent)
            else:
                ent = str(item).strip().lower()
                if len(ent) >= 2:
                    out.append(ent)
    return list(dict.fromkeys(out))


def _normalize_str_list(raw: Any, strip_prefix: str = "") -> List[str]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    out: List[str] = []
    if isinstance(raw, (list, tuple, np.ndarray)):
        for item in raw:
            s = str(item).strip().lower()
            if strip_prefix and s.startswith(strip_prefix):
                s = s[len(strip_prefix):]
            if s and s != "-pron-":
                out.append(s)
    return list(dict.fromkeys(out))


def _normalize_user(raw: Any) -> str:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ""
    if isinstance(raw, (list, tuple, np.ndarray)):
        return str(raw[0]).strip() if len(raw) > 0 else ""
    s = str(raw).strip()
    return "" if s in {"", "[]", "nan", "None"} else s


def load_socialed_npy(
    dataset_name: str = "Event2012",
    dataset_dir: Path | str = DEFAULT_DATASET_DIR,
    sort_chronologically: bool = True,
) -> SocialDataset:
    """Load Event2012 / Event2012_100 / Event2018 .npy into a normalized SocialDataset."""
    path = Path(dataset_dir) / f"{dataset_name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    arr = np.load(path, allow_pickle=True)
    if arr.ndim != 2 or arr.shape[1] < 11:
        raise ValueError(f"Unexpected array shape {arr.shape} in {path}")

    n = arr.shape[0]
    tweet_ids = np.array([str(x) for x in arr[:, 0]], dtype=object)
    texts = [str(x) for x in arr[:, 1]]
    labels = np.array([int(x) for x in arr[:, 2]], dtype=np.int64)
    tokens_lemma = [_normalize_str_list(x) for x in arr[:, 4]]
    entities = [_normalize_entities(x) for x in arr[:, 5]]
    user_ids = [_normalize_user(x) for x in arr[:, 6]]

    ts_series = pd.to_datetime(arr[:, 7], utc=True, errors="coerce")
    is_na_ts = np.asarray(pd.isna(ts_series), dtype=bool)
    ts_ns = np.asarray(ts_series.astype("int64"), dtype=np.int64)
    timestamps = np.where(is_na_ts, np.nan, (ts_ns // 10**9).astype(np.float64))
    dates = [t.strftime("%Y-%m-%d") if pd.notna(t) else "unknown" for t in ts_series]

    hashtags = [_normalize_str_list(x, strip_prefix="#") for x in arr[:, 9]]
    mentioned_users = [_normalize_str_list(x, strip_prefix="@") for x in arr[:, 10]]

    ds = SocialDataset(
        name=dataset_name,
        tweet_ids=tweet_ids,
        texts=texts,
        labels=labels,
        tokens_lemma=tokens_lemma,
        entities=entities,
        hashtags=hashtags,
        user_ids=user_ids,
        mentioned_users=mentioned_users,
        timestamps=timestamps,
        dates=dates,
    )

    if sort_chronologically:
        sort_ts = np.where(np.isnan(timestamps), np.inf, timestamps)
        order = np.lexsort((tweet_ids.astype(str), sort_ts))
        ds = ds.subset(order)
    return ds


def split_daily_blocks(
    ds: SocialDataset,
    merge_initial_days: int = 1,
    min_block_size: int = 200,
    drop_unknown_dates: bool = False,
) -> List[Tuple[str, np.ndarray]]:
    """Split a chronologically sorted SocialDataset into daily streaming blocks.

    Small trailing/edge days (< min_block_size) are merged into the adjacent block
    so every streaming block has non-trivial event competition.
    """
    unique_dates: List[str] = []
    date_to_indices: Dict[str, List[int]] = {}
    for i, d in enumerate(ds.dates):
        if d not in date_to_indices:
            unique_dates.append(d)
            date_to_indices[d] = []
        date_to_indices[d].append(i)

    unknown_idx = date_to_indices.pop("unknown", [])
    unique_dates = [d for d in unique_dates if d != "unknown"]

    raw_blocks: List[Tuple[str, List[int]]] = []
    if merge_initial_days > 1 and len(unique_dates) >= merge_initial_days:
        init_dates = unique_dates[:merge_initial_days]
        init_idx = [i for d in init_dates for i in date_to_indices[d]]
        raw_blocks.append((f"{init_dates[0]}..{init_dates[-1]}", init_idx))
        rest_dates = unique_dates[merge_initial_days:]
    else:
        rest_dates = unique_dates

    for d in rest_dates:
        raw_blocks.append((d, date_to_indices[d]))

    merged: List[Tuple[str, np.ndarray]] = []
    buf_name = ""
    buf_idx: List[int] = []
    for name, idxs in raw_blocks:
        if not buf_idx:
            buf_name = name
            buf_idx = list(idxs)
        else:
            buf_name = f"{buf_name}+{name}"
            buf_idx.extend(idxs)
        if len(buf_idx) >= min_block_size:
            merged.append((buf_name, np.asarray(buf_idx, dtype=np.int64)))
            buf_name = ""
            buf_idx = []
    if buf_idx:
        if merged:
            prev_name, prev_idx = merged[-1]
            merged[-1] = (
                f"{prev_name}+{buf_name}",
                np.concatenate([prev_idx, np.asarray(buf_idx, dtype=np.int64)]),
            )
        else:
            merged.append((buf_name, np.asarray(buf_idx, dtype=np.int64)))
    if unknown_idx and not drop_unknown_dates:
        merged.append(("unknown", np.asarray(unknown_idx, dtype=np.int64)))
    return merged


def split_fixed_size_blocks(
    ds: SocialDataset,
    block_size: int = 2000,
) -> List[Tuple[str, np.ndarray]]:
    """Split chronologically sorted dataset into fixed-size streaming message blocks."""
    n = len(ds)
    blocks: List[Tuple[str, np.ndarray]] = []
    for b_idx, start in enumerate(range(0, n, block_size)):
        end = min(n, start + block_size)
        if end - start < block_size // 4 and blocks:
            prev_name, prev_idx = blocks[-1]
            blocks[-1] = (
                f"{prev_name}_ext",
                np.concatenate([prev_idx, np.arange(start, end, dtype=np.int64)]),
            )
        else:
            blocks.append((f"B{b_idx:02d}({ds.dates[start]}..{ds.dates[end-1]})", np.arange(start, end, dtype=np.int64)))
    return blocks
