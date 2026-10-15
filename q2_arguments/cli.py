"""CLI da Q2: `uv run q2 --help`."""

from __future__ import annotations

import subprocess
import sys
from collections import Counter
from typing import Annotated

import typer

from common.config import ROOT
from common.llm import LLM

app = typer.Typer(no_args_is_help=True, help="Q2: argumentos de sentenças.")


@app.command()
def scrape() -> None:
    """Baixa as páginas do TJSP CJPG listadas em data/rulings/queries.json (cache em raw/)."""
    from q2_arguments.scrape import load_queries
    from q2_arguments.scrape import scrape as run

    queries, pages = load_queries()
    typer.echo(run(queries, pages))


@app.command()
def build() -> None:
    """Seleciona, limpa e anonimiza o corpus em data/rulings (usa LLM para listar nomes)."""
    from q2_arguments.corpus import build as run

    llm = LLM(ledger_path=ROOT / ".cache" / "ledger-q2-build.jsonl")
    typer.echo(f"{run(llm)}\n{llm.summary()}\nestatísticas: data/rulings/STATS.md")


@app.command("draft-gold")
def draft_gold(
    model: str = "gpt-5",
    force: bool = typer.Option(False, help="Refaz rascunhos existentes (nunca os revisados)."),
    only: Annotated[list[str] | None, typer.Option(help="Só estes ruling_id (repetível).")] = None,
    workers: int = 4,
) -> None:
    """Gera rascunhos do gold via LLM em data/gold/ para revisão humana."""
    from q2_arguments.gold import draft_all

    llm = LLM(ledger_path=ROOT / ".cache" / "ledger-q2-gold.jsonl")
    for rid, msg in draft_all(llm, model=model, only=only, force=force, workers=workers).items():
        typer.echo(f"{rid}  {msg}")
    typer.echo(llm.summary())


@app.command("gold-status")
def gold_status() -> None:
    """Status do gold por sentença (rascunho/revisado, citações ancoradas)."""
    from q2_arguments.gold import status_rows

    rows = status_rows()
    for r in rows:
        flag = "  TEXTO MUDOU: reancorar" if r["texto_mudou"] else ""
        typer.echo(
            f"{r['ruling_id']:34} {r['split']:4} {r['faixa']:11} {r['status']:9} "
            f"{r['resultado']:24} args={r['argumentos']:<3} citações={r['ancoradas']}{flag}"
        )
    c = Counter(r["status"] for r in rows)
    typer.echo(f"\n{dict(c)}  (meta da fase: >= 20 revisados)")


@app.command()
def review(port: int = 8501) -> None:
    """Abre o editor do gold (Streamlit)."""
    app_path = ROOT / "q2_arguments" / "gold_app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)]
    raise typer.Exit(subprocess.call(cmd))
