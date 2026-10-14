"""Configuração via variáveis de ambiente (.env na raiz do repo)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    chat_model: str
    embed_model: str
    cache_dir: Path

    @classmethod
    def load(cls) -> Settings:
        load_dotenv(ROOT / ".env", override=False)
        cache = Path(os.getenv("ENTER_CACHE_DIR", ".cache/llm"))
        if not cache.is_absolute():
            cache = ROOT / cache
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            chat_model=os.getenv("ENTER_CHAT_MODEL", "gpt-4.1-mini"),
            embed_model=os.getenv("ENTER_EMBED_MODEL", "text-embedding-3-small"),
            cache_dir=cache,
        )
