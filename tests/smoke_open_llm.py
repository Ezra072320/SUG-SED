"""1-call smoke test for lightweight open-source LLM API endpoints (never prints secrets)."""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_shared.key_loader import DEFAULT_KEY_FILE


def _parse_cf(key_file: Path):
    lines = [l.strip() for l in key_file.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
    acct = tok = ""
    for i, l in enumerate(lines):
        if "帐户 ID" in l or "账户 ID" in l:
            acct = lines[i + 1].strip()
        if "API 令牌" in l:
            tok = lines[i + 1].strip()
    return acct, tok


def main() -> None:
    cf_acct, cf_tok = _parse_cf(DEFAULT_KEY_FILE)
    if cf_acct and cf_tok:
        url = f"https://api.cloudflare.com/client/v4/accounts/{cf_acct}/ai/v1/chat/completions"
        models = [
            "@cf/meta/llama-3.1-8b-instruct",
            "@cf/meta/llama-3.1-8b-instruct-fast",
            "@cf/utter-project/eurollm-9b-it",
            "@cf/swiss-ai/apertus-v1.5-8b",
            "@cf/google/gemma-4-26b-a4b-it",
        ]
        for model in models:
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": 'Return valid JSON only: {"action": "MERGE", "confidence": 0.9}',
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 60,
            }
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {cf_tok}"},
                    method="POST",
                )
                t0 = time.time()
                with urllib.request.urlopen(req, timeout=25) as r:
                    body = json.loads(r.read().decode("utf-8"))
                msg = body["choices"][0]["message"]
                usage = body.get("usage")
                content = repr(msg.get("content"))[:100]
                print(f"[OK] {model} lat={time.time()-t0:.2f}s usage={usage} content={content}")
            except Exception as e:
                print(f"[FAIL] {model}: {type(e).__name__}: {str(e)[:140]}")


if __name__ == "__main__":
    main()
