"""CLI de utilitários: `uv run enter --help`."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import typer
from pydantic import BaseModel

from common.llm import LLM
from common.text import mask_entities, normalize

app = typer.Typer(no_args_is_help=True, help="Case Enter: utilitários compartilhados.")


class _Smoke(BaseModel):
    tese: str
    area: Literal["consumidor", "trabalhista", "outro"]


@app.command()
def smoke() -> None:
    """Uma chamada estruturada + um embedding reais (requer OPENAI_API_KEY)."""
    llm = LLM()
    parsed, rec = llm.structured(
        _Smoke,
        system="Classifique a petição. Responda em pt-BR.",
        user="A autora teve o voo cancelado sem aviso prévio e pede danos morais.",
        prompt_version="smoke-v1",
    )
    typer.echo(f"chat: {parsed.model_dump()}  {rec}")
    _, rec = llm.embed(["voo cancelado", "cobrança indevida"])
    typer.echo(f"embed: {rec}")
    typer.echo(f"resumo: {llm.summary()}")


@app.command()
def mask(path: Path, numbers: bool = True) -> None:
    """Normaliza e mascara entidades de um arquivo de texto."""
    masked = mask_entities(normalize(path.read_text("utf-8")), numbers=numbers)
    typer.echo(masked.text)
    typer.echo("\n--- entidades ---")
    for e in masked.entities:
        typer.echo(f"{e.kind:9} [{e.start}:{e.end}] {e.value}")
