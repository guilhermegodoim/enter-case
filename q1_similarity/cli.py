"""CLI da Q1: `uv run q1 --help`."""

from __future__ import annotations

from typing import Annotated

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


@app.command()
def bench(
    only: Annotated[list[str] | None, typer.Option(help="Só estes métodos (repetível).")] = None,
    judge_model: str = "gpt-4.1-mini",
) -> None:
    """Benchmark: calibra no dev, mede no test, grava results/<run_id>/."""
    from q1_similarity.bench import run

    w = run(only=only or None, judge_model=judge_model)
    typer.echo(f"resultados: {w.dir}")


@app.command()
def demo(case_id: str = "caso-001", llm: bool = False, max_diff: int = 3) -> None:
    """Ingere os documentos de um caso em ordem e mostra ação, família e diff de cada um."""
    from q1_similarity.bench import Corpus, methods
    from q1_similarity.classify import fit_prefilter
    from q1_similarity.data import Dataset
    from q1_similarity.pipeline import CaseIndex, Config

    ds = Dataset()
    client = LLM()
    m = next(x for x in methods() if x.key == ("cascata_llm" if llm else "cascata"))
    dev = Corpus(ds, client).dev_rows(m, frozenset())
    cfg = Config(m.classifier.fit(dev), prefilter=fit_prefilter(dev), use_llm=llm)
    idx = CaseIndex(cfg, client)
    for d in ds.case_docs(case_id):
        dec = idx.ingest(d["doc_id"], ds.text(d["doc_id"]))
        flag = "  [revisão humana]" if dec.needs_review else ""
        typer.echo(
            f"{d['doc_id']}  {dec.action:8} casado={dec.match or '-':14} "
            f"família={dec.canonical}{flag}   (gabarito: {d['op']})"
        )
        if dec.action == "versao":
            new, old = idx.sigs[d["doc_id"]].prep, idx.sigs[dec.match].prep
            for r in dec.diff()[:max_diff]:
                a = old.norm[slice(*r["anterior"])] if r["anterior"] else "∅"
                b = new.norm[slice(*r["novo"])] if r["novo"] else "∅"
                typer.echo(f"    - antes: {a[:110]}\n      agora: {b[:110]}")
