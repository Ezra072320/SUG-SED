"""Secure local API key reader for lightweight open-source LLM endpoints.

Reads provider credentials from ~/Desktop/APIkey.txt (or SED_API_KEY_FILE / env vars)
in memory without ever printing or persisting secrets to logs/reports.

Supported lightweight open-source providers:
  - cloudflare : Cloudflare Workers AI (e.g. @cf/meta/llama-3.1-8b-instruct-fast)
  - dashscope  : Aliyun Bailian / DashScope official Qwen API (qwen2.5-7b-instruct, qwen3-8b)
  - siliconflow: SiliconFlow OpenAI-compatible API (Qwen/Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct)
  - modelscope : ModelScope official API (Qwen/Qwen2.5-7B-Instruct)
  - vllm       : Self-hosted local/cloud vLLM OpenAI-compatible server
  - deepseek   : DeepSeek API (legacy reference)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

DEFAULT_KEY_FILE = Path(
    os.environ.get("SED_API_KEY_FILE", str(Path.home() / "Desktop" / "APIkey.txt"))
)

OPEN_SOURCE_PROVIDERS: Dict[str, Tuple[str, str]] = {
    "dashscope": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen3-8b"),
    "aliyun": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen3-8b"),
    "siliconflow": ("https://api.siliconflow.cn/v1", "Qwen/Qwen3-8B"),
    "modelscope": ("https://api-inference.modelscope.cn/v1", "Qwen/Qwen3-8B"),
    "cloudflare": ("https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1", "@cf/meta/llama-3.1-8b-instruct-fast"),
    "vllm": ("http://127.0.0.1:8000/v1", "Qwen/Qwen3-8B"),
    "deepseek": ("https://api.deepseek.com/v1", "deepseek-chat"),
}


def load_provider_key(
    provider: str = "dashscope",
    key_file: Path | str = DEFAULT_KEY_FILE,
) -> Tuple[str, str, str]:
    """Return (base_url, default_open_source_model, api_key) without logging the key."""
    provider_l = provider.strip().lower()
    if provider_l not in OPEN_SOURCE_PROVIDERS:
        raise ValueError(
            f"Unsupported open-source provider '{provider}'. Choose from {list(OPEN_SOURCE_PROVIDERS)}."
        )

    base_url, default_model = OPEN_SOURCE_PROVIDERS[provider_l]

    if provider_l == "vllm":
        return os.environ.get("VLLM_BASE_URL", base_url), default_model, os.environ.get("VLLM_API_KEY", "EMPTY")

    # Check env override first
    env_key = os.environ.get(f"{provider_l.upper()}_API_KEY") or os.environ.get("LLM_API_KEY")
    if env_key and provider_l != "cloudflare":
        return base_url, default_model, env_key.strip()

    path = Path(key_file)
    if not path.exists():
        raise FileNotFoundError(f"Key file not found: {path}")

    lines = [l.strip() for l in path.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]

    if provider_l == "cloudflare":
        acct = tok = ""
        for i, line in enumerate(lines):
            if ("帐户 id" in line.lower() or "账户 id" in line.lower()) and i + 1 < len(lines):
                acct = lines[i + 1].strip()
            if "api 令牌" in line.lower() and i + 1 < len(lines):
                tok = lines[i + 1].strip()
        if not acct or not tok:
            raise RuntimeError(f"Could not parse Cloudflare Account ID & API Token from {path}")
        return base_url.format(account_id=acct), default_model, tok

    # Standard label -> next-line key parsing (supports aliases like dashscope/bailian/aliyun)
    aliases = {provider_l}
    if provider_l in {"dashscope", "aliyun"}:
        aliases.update({"dashscope", "aliyun", "bailian", "百炼", "阿里云百炼"})
    elif provider_l == "siliconflow":
        aliases.update({"siliconflow", "硅基流动"})
    elif provider_l == "modelscope":
        aliases.update({"modelscope", "魔搭"})

    api_key = ""
    for i, line in enumerate(lines):
        low = line.lower().rstrip(":： ")
        if low in aliases and i + 1 < len(lines):
            j = i + 1
            while j < len(lines):
                cand = lines[j].strip()
                low_c = cand.lower()
                if low_c.startswith("for gemini:") or low_c.startswith("for antigravity:"):
                    api_key = cand.split(":", 1)[1].strip()
                    break
                if low_c.startswith("for ") and ":" in cand:
                    j += 1
                    continue
                if cand.endswith(":") or cand.endswith("："):
                    break
                # Handle generic "<tag>:sk-..." single-line formatting
                if not cand.startswith("sk-") and ":sk-" in cand:
                    api_key = cand[cand.index(":sk-") + 1 :].strip()
                else:
                    api_key = cand
                break
            break

    if not api_key:
        raise RuntimeError(f"Could not find key for provider '{provider}' in {path}")

    return base_url, default_model, api_key
