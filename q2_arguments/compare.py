"""Comparação entre execuções de benchmark da Q2: `uv run q2 compare <bench_run>...`.

Junta, numa tabela, configs de benchmarks diferentes (ex.: antes e depois de mudar prompt,
modelo ou schema). Cada linha mostra o que define a extração (run de extração, prompt + hash,
schema, modelo) e o que define a medição (gold, juiz). Linhas só são comparáveis quando gold e
juiz são os mesmos: o relatório marca a referência e avisa quando não forem.
"""

from __future__ import annotations

import json
from pathlib import Path

from common.config import ROOT
from common.runs import RunSpec, RunWriter

RESULTS = ROOT / "results"


def load_bench(bench_dir: Path) -> dict:
    spec = json.loads((bench_dir / "spec.json").read_text("utf-8"))
    metrics = json.loads((bench_dir / "metrics.json").read_text("utf-8"))
    return {"dir": bench_dir.name, "spec": spec, "metrics": metrics}


def rows(benches: list[dict], only: list[str] | None = None) -> list[dict]:
    out = []
    for b in benches:
        sp = b["spec"]
        for key, r in b["metrics"]["configs"].items():
            if only and key not in only:
                continue
            out.append({
                "bench": b["dir"], "config": key, "extraction_run_id": r["extraction_run_id"],
                "model": r["config"]["model"], "mode": r["config"]["mode"],
                "prompt_version": r["prompt_version"],
                "prompt_sha256": r.get("prompt_sha256") or "—",
                # benchmarks anteriores à fase 5 não gravavam o schema por config: eram v1
                "schema_version": r.get("schema_version", "q2-extracted-v1"),
                "gold": sp["dataset"], "juiz": f"{sp['model']}/{sp['prompt_version']}",
                "provisorio": b["metrics"]["meta"]["provisorio"],
                "precision": r["precision"], "recall": r["recall"], "f1": r["f1"],
                "f1_ic95": r["f1_ic95"], "custo": r["custo_usd_por_sentenca"],
                "p50": r["latencia_ms_p50"],
            })  # fmt: skip
    return out


def comparable(rs: list[dict]) -> bool:
    return len({(r["gold"], r["juiz"]) for r in rs}) <= 1


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def report(rs: list[dict], benches: list[dict]) -> str:
    ref = (rs[0]["gold"], rs[0]["juiz"]) if rs else None
    header = ["bench", "config", "run extração", "modelo", "modo", "prompt", "sha prompt",
              "schema", "gold", "P", "R", "F1 (IC95)", "US$/sentença", "p50 ms"]  # fmt: skip
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rs:
        lo, hi = r["f1_ic95"]
        flag = "" if (r["gold"], r["juiz"]) == ref else " ⚠️"
        lines.append("| " + " | ".join([
            f"`{r['bench']}`", f"`{r['config']}`", f"`{r['extraction_run_id']}`", r["model"],
            r["mode"], f"`{r['prompt_version']}`", f"`{r['prompt_sha256']}`",
            f"`{r['schema_version']}`", f"`{r['gold']}`{flag}", _pct(r["precision"]),
            _pct(r["recall"]), f"{_pct(r['f1'])} ({_pct(lo)}–{_pct(hi)})", f"{r['custo']:.4f}",
            f"{r['p50']:.0f}",
        ]) + " |")  # fmt: skip
    # mesma config lógica (modelo, modo, versão do prompt) em benches diferentes: o run_id
    # muda com schema/código, a saída só muda se o modelo receber outro prompt
    groups: dict[tuple, list[dict]] = {}
    for r in rs:
        groups.setdefault((r["config"], r["model"], r["mode"], r["prompt_version"]), []).append(r)
    reruns = [
        f"`{k[0]}` ({len(v)} runs, schemas {', '.join(sorted({x['schema_version'] for x in v}))}"
        f"): F1 {' / '.join(_pct(x['f1']) for x in v)}"
        for k, v in groups.items()
        if len({x["bench"] for x in v}) > 1
    ]
    note = (
        "Gold e juiz iguais em todas as linhas: números diretamente comparáveis."
        if comparable(rs)
        else "⚠️ Gold ou juiz diferem entre linhas (marcadas): compare só linhas com o mesmo gold."
    )
    prov = any(r["provisorio"] for r in rs)
    return f"""# Q2: comparação entre execuções

Gerado por `uv run q2 compare {" ".join(b["dir"] for b in benches)}`.

{note}
{"> **PROVISÓRIO**: gold ainda contém rascunhos LLM não revisados." if prov else ""}

O `run_id` da extração é o hash de modelo, versão e hash do prompt, schema, parâmetros e código:
mudar o schema ou o código gera run nova. Mesma config (modelo, modo, versão do prompt)
reexecutada em benchmarks diferentes, com o cache LLM, deve dar o mesmo F1:
{"; ".join(reruns) or "nenhuma config repetida"}.

{chr(10).join(lines)}
"""


def run(bench_runs: list[str], only: list[str] | None = None) -> RunWriter:
    benches = [load_bench(RESULTS / b) for b in bench_runs]
    rs = rows(benches, only)
    spec = RunSpec(
        task="q2-compare",
        dataset=",".join(sorted({r["gold"] for r in rs})),
        params={"benches": bench_runs, "only": only},
    )
    w = RunWriter(spec)
    w.write_metrics({"comparavel": comparable(rs), "linhas": rs})
    w.write_report(report(rs, benches))
    return w
