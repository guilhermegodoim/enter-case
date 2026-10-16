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

# limiar do merge calibrado no dev pelo benchmark (results/q2-bench-*/report.md): 1.01 = só
# sobreposição de citações. Mesmo valor do bench => mesmo run_id da extração.
MERGE_THRESHOLD = 1.01


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


@app.command()
def extract(
    ruling_id: str,
    config: str = typer.Option("chunk_mini_verif", help="Chave de q2_arguments.bench.configs()."),
) -> None:
    """Extrai e mostra os argumentos de uma sentença do corpus."""
    from q2_arguments.bench import configs
    from q2_arguments.corpus import ruling_text
    from q2_arguments.extract import extract as run

    cfg = next(c for c in configs() if c.key == config)
    text = ruling_text(ruling_id)
    e = run(LLM(), cfg, ruling_id, text)
    typer.echo(
        f"run {e.run_id}  chunks={e.n_chunks}  brutos={e.brutos}  "
        f"sem âncora={e.descartados_sem_ancora}  fundidos={e.fundidos}  "
        f"US$ {e.cost_usd:.4f}  {e.latency_ms:.0f} ms\n"
    )
    for a in e.argumentos:
        tags = f"{a.tipo}/{a.decisao}/{a.suscitado_por}"
        typer.echo(f"{a.id} [{tags}] conf={a.confianca} ({a.confianca_nivel})")
        typer.echo(f"   {a.titulo}\n   {a.resumo}")
        for c in a.citacoes:
            src = text[c.char_start : c.char_end][:160] if c.ancorada else "(não ancorada)"
            typer.echo(f"   [{c.char_start}:{c.char_end} score {c.score}] {src}")


@app.command()
def bench(
    only: Annotated[list[str] | None, typer.Option(help="Só estas configs (repetível).")] = None,
    gold: str = typer.Option("todos", help="todos (rascunho+revisado) | revisado"),
) -> None:
    """Benchmark de extração: calibra merge no dev, mede no test, grava results/<run_id>/."""
    from q2_arguments.bench import run

    w = run(only=only or None, gold_mode=gold)
    typer.echo(f"resultados: {w.dir}")


# --------------------------------------------------------------------------- fase 5: store


def _store(db: str | None):
    from q2_arguments.store import Store

    return Store(db)


@app.command()
def ingest(
    config: str = typer.Option("chunk_mini", help="Chave de q2_arguments.bench.configs()."),
    only: Annotated[list[str] | None, typer.Option(help="Só estes ruling_id (repetível).")] = None,
    split: str = typer.Option("todos", help="dev | test | todos"),
    db: str | None = typer.Option(None, help="Caminho do SQLite (padrão data/store/q2.sqlite)."),
) -> None:
    """Extrai e grava no store. Run já gravada para a sentença: mantida como está."""
    from concurrent.futures import ThreadPoolExecutor

    from q2_arguments.bench import configs
    from q2_arguments.corpus import load_rulings, ruling_text
    from q2_arguments.extract import extract as run

    cfg = next(c for c in configs() if c.key == config)
    cfg = cfg.model_copy(update={"merge_threshold": MERGE_THRESHOLD})
    rows = [
        r for r in load_rulings()
        if (not only or r["ruling_id"] in only) and split in ("todos", r["split"])
    ]  # fmt: skip
    llm = LLM(ledger_path=ROOT / ".cache" / "ledger-q2-ingest.jsonl")
    store = _store(db)

    def one(meta):
        text = ruling_text(meta["ruling_id"])
        return meta, text, run(llm, cfg, meta["ruling_id"], text)

    with ThreadPoolExecutor(max_workers=4) as ex:
        for meta, text, e in ex.map(one, rows):
            eid, new = store.save_extraction(meta, text, cfg, e)
            typer.echo(f"{meta['ruling_id']}  run {e.run_id}  {len(e.argumentos)} args  "
                       f"{'gravada' if new else 'já existia'}")  # fmt: skip
    typer.echo(llm.summary())


@app.command("import-bench")
def import_bench(
    bench_run: str = typer.Argument(..., help="Diretório em results/ (ex.: q2-bench-4d5403af77)."),
    config: str = typer.Option("chunk_mini", help="Config cujas predições importar."),
    db: str | None = None,
) -> None:
    """Importa predições de um benchmark antigo no schema original (lido via upcast)."""
    typer.echo(_store(db).import_bench(ROOT / "results" / bench_run, config))


@app.command()
def runs(ruling_id: str, db: str | None = None) -> None:
    """Runs de extração gravadas para uma sentença."""
    for r in _store(db).runs_for(ruling_id):
        typer.echo(
            f"{r['run_id']}  {r['params'].get('key', '?'):16} {r['model']:13} "
            f"{r['prompt_version']:28} {r['schema_version']:16} snapshot={r['model_snapshot']}  "
            f"{r['revisados']}/{r['n_args']} revisados"
        )


@app.command()
def diff(ruling_id: str, run_a: str, run_b: str, db: str | None = None) -> None:
    """Diff entre duas runs da mesma sentença (pareamento por citação ou título)."""
    d = _store(db).diff_runs(ruling_id, run_a, run_b)
    for p in d.pairs:
        mark = "=" if not p.changes else ("~" if p.equivalent else "!")
        typer.echo(f"{mark} {p.a.local_id}[{p.a.status}] -> {p.b.local_id}[{p.b.status}] "
                   f"({p.how})  {p.b.arg.titulo}")  # fmt: skip
        for f, (x, y) in p.changes.items():
            typer.echo(f"     {f}: {x!s:.70} -> {y!s:.70}")
    for x in d.only_a:
        typer.echo(f"- só em A: {x.local_id}[{x.status}] {x.arg.titulo}")
    for y in d.only_b:
        typer.echo(f"+ só em B: {y.local_id}[{y.status}] {y.arg.titulo}")
    typer.echo(d.counts())


@app.command("review-args")
def review_args(port: int = 8502, db: str | None = None) -> None:
    """Abre a revisão de argumentos extraídos (Streamlit, grava no store)."""
    import os

    env = {**os.environ, **({"ENTER_Q2_DB": db} if db else {})}
    app_path = ROOT / "q2_arguments" / "review_app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)]
    raise typer.Exit(subprocess.call(cmd, env=env))


@app.command()
def compare(
    bench_runs: Annotated[list[str], typer.Argument(help="Diretórios em results/.")],
    only: Annotated[list[str] | None, typer.Option(help="Só estas configs (repetível).")] = None,
) -> None:
    """Junta configs de benchmarks diferentes numa tabela (gold e juiz conferidos)."""
    from q2_arguments.compare import run as run_compare

    w = run_compare(bench_runs, only or None)
    typer.echo(f"resultados: {w.dir}")
