"""Probe available lightweight Qwen open-source models on Aliyun DashScope key."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_shared.key_loader import load_provider_key


def main() -> None:
    url, _, key = load_provider_key("aliyun")
    try:
        req = urllib.request.Request(f"{url}/models", headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode("utf-8"))
        ids = [m["id"] for m in data.get("data", [])]
        print(f"/models returned {len(ids)} models:")
        matches = [m for m in ids if any(k in m.lower() for k in ["qwen3", "qwen2.5", "7b", "8b", "9b", "4b", "14b", "llama"])]
        for x in sorted(matches):
            print(" ", x)
    except Exception as e:
        print("/models failed:", type(e).__name__, str(e)[:150])

    print("\nProbing candidate lightweight Qwen open-source IDs:")
    cands = [
        "qwen3-8b",
        "qwen3-4b",
        "qwen3-14b",
        "qwen3.5-9b",
        "qwen3.5-8b",
        "qwen3.5-4b",
        "qwen3-8b-instruct",
        "qwen2.5-7b-instruct",
        "qwen2.5-14b-instruct",
        "qwen-flash",
        "qwen-turbo",
    ]
    for m in cands:
        payload = {
            "model": m,
            "messages": [{"role": "user", "content": "Reply ok"}],
            "enable_thinking": False,
            "max_tokens": 10,
        }
        req = urllib.request.Request(
            f"{url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                body = json.loads(r.read().decode("utf-8"))
            actual = body.get("model")
            content = body["choices"][0]["message"]["content"]
            print(f"  [OK] {m:22s} -> actual={actual} content={content!r}")
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="ignore")[:140]
            print(f"  [HTTP {e.code}] {m:22s} -> {err}")
        except Exception as e:
            print(f"  [ERR] {m:22s} -> {type(e).__name__}: {str(e)[:100]}")


if __name__ == "__main__":
    main()
