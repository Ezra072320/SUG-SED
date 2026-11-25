"""1-call smoke test for SiliconFlow Qwen/Qwen3-8B (never prints secrets)."""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_shared.key_loader import DEFAULT_KEY_FILE, load_provider_key


def main() -> None:
    lines = [l.strip() for l in DEFAULT_KEY_FILE.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
    print("Last 6 lines of APIkey.txt (masked):")
    for i, l in enumerate(lines[-6:], start=max(1, len(lines) - 5)):
        print(f"  L{i}:", re.sub(r"([A-Za-z0-9_-]{4})[A-Za-z0-9_-]{8,}([A-Za-z0-9_-]{3})", r"\1***\2", l))

    try:
        url, model, key = load_provider_key("siliconflow")
    except Exception as e:
        print("[PARSE ERR]", type(e).__name__, e)
        return

    print(f"Loaded siliconflow: url={url}, model={model}")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Summarize event_a and event_b in 5-8 words and return strict JSON "
                    '{"event_a": "...", "event_b": "...", "same_event": true or false}.\n'
                    "A) Newsweek ends print publication after 80 years.\n"
                    "B) Google stock drops 9% after premature Q3 earnings release."
                ),
            }
        ],
        "temperature": 0.0,
        "max_tokens": 120,
        "enable_thinking": False,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        f"{url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=25) as r:
            body = json.loads(r.read().decode("utf-8"))
        msg = body["choices"][0]["message"]
        rc = msg.get("reasoning_content") or ""
        print(
            f"[OK] lat={time.time()-t0:.2f}s usage={body.get('usage')} "
            f"reasoning_len={len(rc)} content={msg.get('content')!r}"
        )
    except urllib.error.HTTPError as e:
        print("[HTTP ERR]", e.code, e.read().decode("utf-8", errors="ignore")[:260])
    except Exception as e:
        print("[ERR]", type(e).__name__, str(e)[:200])


if __name__ == "__main__":
    main()
