from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


class SecretError(RuntimeError):
    pass


def load_openai_api_key_from_file(path: Path) -> str:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise SecretError(f"OPENAI_API_KEY_FILE not found: {p}")
    key = p.read_text(encoding="utf-8").strip()
    # minimal sanity check
    if not key or not key.startswith("sk-"):
        raise SecretError(f"Key file does not look like an OpenAI API key: {p}")
    return key


def get_openai_api_key(
    *,
    env_var: str = "OPENAI_API_KEY",
    file_env_var: str = "OPENAI_API_KEY_FILE",
) -> str:
    """Return OpenAI API key from env var or from a local file pointed to by env var."""
    key = os.getenv(env_var)
    if key:
        return key.strip()

    file_path = os.getenv(file_env_var)
    if file_path:
        return load_openai_api_key_from_file(Path(file_path))

    raise SecretError(f"{env_var} is not set and {file_env_var} is not set")

