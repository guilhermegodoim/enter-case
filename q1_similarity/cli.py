"""CLI da Q1: `uv run q1 --help`."""

from __future__ import annotations

import typer

from common.config import ROOT
from common.llm import LLM

app = typer.Typer(no_args_is_help=True, help="Q1: similaridade de documentos.")


@app.callback()
def _main() -> None:
    """Força modo de subcomandos mesmo com um único comando registrado."""


@app.command("gen-templates")
def gen_templates(model: str = "gpt-4.1", workers: int = 6) -> None:
    """Gera (via LLM) os templates que faltam em data/synthetic/templates. Executar uma vez."""
    from q1_similarity.generate import generate_all

    llm = LLM(ledger_path=ROOT / ".cache" / "ledger-q1-templates.jsonl")
    stats = generate_all(llm, model=model, workers=workers)
    typer.echo(f"{stats}  {llm.summary()}")


@app.command()
def build(seed: int = 13, cases: int = 60) -> None:
    """Monta casos, documentos e pares rotulados em data/synthetic/q1 (determinístico, sem LLM)."""
    from q1_similarity.build import build as run_build

    meta = run_build(seed=seed, n_cases=cases)
    typer.echo(f"{meta}\nestatísticas: data/synthetic/q1/STATS.md")
