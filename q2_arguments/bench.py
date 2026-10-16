"""Benchmark da Q2: qualidade, ancoragem, custo e latência da extração de argumentos.

Protocolo:
- Gold = data/gold. Por padrão usa rascunhos + revisados e marca o relatório como PROVISÓRIO
  enquanto houver rascunho (gold ainda não corrigido por humano). `--gold revisado` usa só os
  revisados. O fingerprint do gold entra no run_id: revisar o gold gera outro run.
- Limiar do merge calibrado no split dev; números reportados no split test.
- Pareamento extraído × gold por sentença: juiz LLM decide se dois itens tratam da MESMA
  questão decidida (one-to-one forçado em código). Um pareador determinístico (embeddings +
  sobreposição de citações, atribuição húngara) roda junto; a concordância dos dois é reportada.
- Latência por sentença = latência original das chamadas (chunks em paralelo). IC95 por
  bootstrap sobre sentenças.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import linear_sum_assignment

from common.config import ROOT
from common.llm import LLM
from common.runs import RunSpec, RunWriter
from common.text import sha256_text
from q2_arguments.corpus import load_rulings, ruling_text
from q2_arguments.extract import EXTRACT_SCHEMA_VERSION, Config, Extraction, extract
from q2_arguments.gold import GOLD_DIR, load_gold
from q2_arguments.schema import Argumento

TASK = "q2-bench"
SCHEMA_VERSION = "q2-bench-v1"
JUDGE_MODEL = "gpt-4.1"
JUDGE_PROMPT_VERSION = "q2-match-v2"
BOOTSTRAP = 1000
MERGE_GRID = (0.80, 0.85, 0.88, 0.92, 1.01)  # 1.01 = só sobreposição de citações
BASE = "chunk_mini"
EMB_MATCH = 0.80  # pareador determinístico: cosseno mínimo
EMB_MATCH_OVERLAP = 0.60  # ... ou cosseno mínimo quando as citações se sobrepõem

# --------------------------------------------------------------------------- configurações


def configs() -> list[Config]:
    mini, v = "gpt-4.1-mini", {"verifier": "gpt-4.1-mini"}
    return [
        Config(key="full_mini", mode="full", model=mini),
        Config(key="truncado_mini", mode="truncate", model=mini),
        Config(key="chunk_mini", mode="chunk", model=mini),
        Config(key="chunk_mini_basico", mode="chunk", model=mini, prompt="basico"),
        Config(key="chunk_mini_v2", mode="chunk", model=mini, prompt="guia2"),
        Config(key="full_41_v2", mode="full", model="gpt-4.1", prompt="guia2"),
        Config(key="chunk_mini_sem_filtro", mode="chunk", model=mini, ground_filter=False),
        Config(key="chunk_mini_sem_merge", mode="chunk", model=mini, merge=False),
        Config(key="chunk_mini_verif", mode="chunk", model=mini, **v),
        Config(key="chunk_mini_verif_drop", mode="chunk", model=mini, verifier_drop=True, **v),
        Config(key="chunk_nano", mode="chunk", model="gpt-4.1-nano"),
        Config(key="chunk_5mini", mode="chunk", model="gpt-5-mini"),
        Config(key="chunk_41", mode="chunk", model="gpt-4.1"),
        Config(key="full_41", mode="full", model="gpt-4.1"),
    ]


GROUPS = {
    "Contexto longo": ["full_mini", "truncado_mini", "chunk_mini"],
    "Modelos": ["chunk_nano", "chunk_mini", "chunk_5mini", "chunk_41", "full_41"],
    "Prompt e ablações": [
        "chunk_mini", "chunk_mini_basico", "chunk_mini_sem_filtro", "chunk_mini_sem_merge",
    ],
    "Verificador": ["chunk_mini", "chunk_mini_verif", "chunk_mini_verif_drop"],
    "Versão do prompt (v1 × v2)": ["chunk_mini", "chunk_mini_v2", "full_41", "full_41_v2"],
}  # fmt: skip

# --------------------------------------------------------------------------- dados


def load_items(split: str, gold_mode: str) -> list[dict]:
    items = []
    for r in load_rulings():
        if r["split"] != split:
            continue
        g = load_gold(r["ruling_id"])
        if g is None or (gold_mode == "revisado" and g.status != "revisado"):
            continue
        items.append({"meta": r, "text": ruling_text(r["ruling_id"]), "gold": g})
    return items


def gold_fingerprint(items: list[dict]) -> str:
    h = hashlib.sha256()
    for it in sorted(items, key=lambda x: x["meta"]["ruling_id"]):
        h.update((GOLD_DIR / f"{it['meta']['ruling_id']}.json").read_bytes())
        h.update(sha256_text(it["text"]).encode())
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------- pareamento


class Par(BaseModel):
    gold: int = Field(description="Número do item da lista G (ex.: 2 para G2)")
    extraido: int = Field(description="Número do item da lista E (ex.: 3 para E3)")


class Pareamento(BaseModel):
    pares: list[Par]


JUDGE_SYSTEM = """Você compara duas listas de argumentos extraídos da MESMA sentença judicial: \
G (referência) e E (extração automática). Pareie G com E quando tratam da MESMA questão \
decidida pelo juiz (mesma tese), ainda que com redação, tipo ou decisão diferentes.

Regras:
- cada item de G e de E aparece em no máximo um par;
- não pareie questões apenas relacionadas: "há dano moral?" e "qual o valor do dano moral?" \
são questões diferentes; "ilegitimidade passiva" e "responsabilidade solidária" também;
- se um item de E junta duas questões de G, pareie com a principal;
- itens sem correspondente ficam de fora;
- responda só com os números dos itens (2 para G2, 3 para E3)."""


def _desc(prefix: str, args: list[Argumento]) -> str:
    return "\n".join(
        f"{prefix}{i}: [{a.tipo}] {a.titulo} :: {a.resumo}" for i, a in enumerate(args, 1)
    )


def match_llm(llm: LLM, gold: list[Argumento], pred: list[Argumento]) -> list[tuple[int, int]]:
    if not gold or not pred:
        return []
    out, _ = llm.structured(
        Pareamento,
        system=JUDGE_SYSTEM,
        user=f"Lista G:\n{_desc('G', gold)}\n\nLista E:\n{_desc('E', pred)}",
        model=JUDGE_MODEL,
        prompt_version=JUDGE_PROMPT_VERSION,
    )
    pairs, used_g, used_p = [], set(), set()
    for p in out.pares:  # one-to-one forçado; números fora da lista ignorados
        gi, pi = p.gold - 1, p.extraido - 1
        if 0 <= gi < len(gold) and 0 <= pi < len(pred) and gi not in used_g and pi not in used_p:
            pairs.append((gi, pi))
            used_g.add(gi)
            used_p.add(pi)
    return pairs


def _spans(a: Argumento) -> list[tuple[int, int]]:
    return [(c.char_start, c.char_end) for c in a.citacoes if c.ancorada]


def quotes_overlap(a: Argumento, b: Argumento) -> bool:
    return any(s1 < e2 and s2 < e1 for s1, e1 in _spans(a) for s2, e2 in _spans(b))


def match_emb(llm: LLM, gold: list[Argumento], pred: list[Argumento]) -> list[tuple[int, int]]:
    if not gold or not pred:
        return []
    vecs, _ = llm.embed([f"{a.titulo}. {a.resumo}" for a in gold + pred])
    m = np.asarray(vecs, dtype=float)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    sim = m[: len(gold)] @ m[len(gold) :].T
    ov = np.array([[quotes_overlap(g, p) for p in pred] for g in gold])
    ok = (sim >= EMB_MATCH) | (ov & (sim >= EMB_MATCH_OVERLAP))
    rows, cols = linear_sum_assignment(-(sim + 0.2 * ov))
    return [(int(g), int(p)) for g, p in zip(rows, cols, strict=True) if ok[g, p]]


# --------------------------------------------------------------------------- métricas


def score_ruling(llm: LLM, item: dict, ext: Extraction, *, with_emb: bool = True) -> dict[str, Any]:
    gold, pred = item["gold"].argumentos, ext.argumentos
    pairs = match_llm(llm, gold, pred)
    emb_pairs = set(match_emb(llm, gold, pred)) if with_emb else set()
    matched_p = {p for _, p in pairs}
    fields = Counter()
    for gi, pi in pairs:
        g, p = gold[gi], pred[pi]
        fields["tipo"] += g.tipo == p.tipo
        fields["decisao"] += g.decisao == p.decisao
        fields["suscitado_por"] += g.suscitado_por == p.suscitado_por
        fields["citacao_confere"] += quotes_overlap(g, p)
    levels = defaultdict(lambda: [0, 0])
    for i, a in enumerate(pred):
        levels[a.confianca_nivel][0] += i in matched_p
        levels[a.confianca_nivel][1] += 1
    raw_unanchored = ext.descartados_sem_ancora + sum(
        1 for a in pred if not any(c.ancorada for c in a.citacoes)
    )
    return {
        "ruling_id": item["meta"]["ruling_id"],
        "faixa": item["meta"]["faixa"],
        "dominio": item["meta"]["dominio"],
        "n_gold": len(gold),
        "n_pred": len(pred),
        "tp": len(pairs),
        "pairs": pairs,
        "emb_pairs": sorted(emb_pairs),
        "concordancia": [len(set(pairs) & emb_pairs), len(set(pairs) | emb_pairs)],
        "fields": dict(fields),
        "levels": {k: v for k, v in levels.items()},
        "brutos": ext.brutos,
        "brutos_sem_ancora": raw_unanchored,
        "citacoes": [ext.citacoes_ancoradas, ext.citacoes_brutas],
        "fundidos": ext.fundidos,
        "descartados_verificador": ext.descartados_verificador,
        "chunks_falhos": ext.chunks_falhos,
        "chunks_com_retry": ext.chunks_com_retry,
        "n_chunks": ext.n_chunks,
        "latency_ms": ext.latency_ms,
        "cost_usd": ext.cost_usd,
        "input_tokens": sum(r.input_tokens for r in ext.calls),
        "output_tokens": sum(r.output_tokens for r in ext.calls),
    }


def prf(rows: list[dict]) -> tuple[float, float, float]:
    tp = sum(r["tp"] for r in rows)
    npred, ngold = sum(r["n_pred"] for r in rows), sum(r["n_gold"] for r in rows)
    p = tp / npred if npred else 0.0
    r = tp / ngold if ngold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def bootstrap_f1(rows: list[dict], seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = [
        prf([rows[i] for i in rng.integers(0, len(rows), len(rows))])[2] for _ in range(BOOTSTRAP)
    ]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def aggregate(rows: list[dict]) -> dict[str, Any]:
    p, r, f1 = prf(rows)
    tp = sum(x["tp"] for x in rows) or 1
    fields = Counter()
    for x in rows:
        fields.update(x["fields"])
    levels = defaultdict(lambda: [0, 0])
    for x in rows:
        for k, (ok, n) in x["levels"].items():
            levels[k][0] += ok
            levels[k][1] += n
    bands = {}
    for b in ("curto", "medio", "longo", "muito_longo"):
        sub = [x for x in rows if x["faixa"] == b]
        if sub:
            bp, br, bf = prf(sub)
            bands[b] = {"n": len(sub), "precision": bp, "recall": br, "f1": bf}
    agree = [sum(x["concordancia"][0] for x in rows), sum(x["concordancia"][1] for x in rows)]
    brutos = sum(x["brutos"] for x in rows)
    cq = [sum(x["citacoes"][0] for x in rows), sum(x["citacoes"][1] for x in rows)]
    lat = [x["latency_ms"] for x in rows]
    zero = [x for x in rows if x["n_gold"] == 0]
    return {
        "n_sentencas": len(rows),
        "n_gold": sum(x["n_gold"] for x in rows),
        "n_pred": sum(x["n_pred"] for x in rows),
        "precision": p,
        "recall": r,
        "f1": f1,
        "f1_ic95": bootstrap_f1(rows),
        "acuracia_campos": {k: fields[k] / tp for k in ("tipo", "decisao", "suscitado_por")},
        "citacao_confere": fields["citacao_confere"] / tp,
        "citacoes_ancoradas": cq[0] / cq[1] if cq[1] else None,
        "brutos_sem_ancora": sum(x["brutos_sem_ancora"] for x in rows) / brutos if brutos else None,
        "args_em_sentenca_sem_argumento": sum(x["n_pred"] for x in zero),
        "sentencas_sem_argumento": len(zero),
        "fundidos": sum(x["fundidos"] for x in rows),
        "descartados_verificador": sum(x["descartados_verificador"] for x in rows),
        "chunks_falhos": sum(x["chunks_falhos"] for x in rows),
        "chunks_com_retry": sum(x["chunks_com_retry"] for x in rows),
        "precisao_por_confianca": {
            k: {"precision": ok / n if n else None, "n": n} for k, (ok, n) in sorted(levels.items())
        },
        "por_faixa": bands,
        "concordancia_pareadores": agree[0] / agree[1] if agree[1] else None,
        "f1_pareador_embeddings": prf([{**x, "tp": len(x["emb_pairs"])} for x in rows])[2],
        "chunks_por_sentenca": float(np.mean([x["n_chunks"] for x in rows])),
        "latencia_ms_p50": float(np.percentile(lat, 50)),
        "latencia_ms_p95": float(np.percentile(lat, 95)),
        "custo_usd_por_sentenca": float(np.mean([x["cost_usd"] for x in rows])),
        "tokens_entrada_por_sentenca": float(np.mean([x["input_tokens"] for x in rows])),
    }


# --------------------------------------------------------------------------- execução


WORKERS = 4  # sentenças em paralelo (os chunks de cada uma também rodam em paralelo)


def evaluate(
    llm: LLM, cfg: Config, items: list[dict], *, with_emb: bool = True
) -> tuple[list[dict], list[Extraction]]:
    def one(it: dict) -> tuple[dict, Extraction]:
        e = extract(llm, cfg, it["meta"]["ruling_id"], it["text"])
        return score_ruling(llm, it, e, with_emb=with_emb), e

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        out = list(ex.map(one, items))
    return [r for r, _ in out], [e for _, e in out]


def calibrate_merge(llm: LLM, base: Config, dev: list[dict]) -> tuple[float, list[dict]]:
    grid = []
    for t in MERGE_GRID:
        cfg = base.model_copy(update={"merge_threshold": t})
        rows, _ = evaluate(llm, cfg, dev, with_emb=False)
        p, r, f1 = prf(rows)
        grid.append({"limiar": t, "precision": p, "recall": r, "f1": f1})
    best = max(grid, key=lambda g: (round(g["f1"], 4), g["limiar"]))
    return best["limiar"], grid


def run(only: list[str] | None = None, gold_mode: str = "todos") -> RunWriter:
    llm = LLM(ledger_path=ROOT / ".cache" / "ledger-q2-bench.jsonl")
    dev, test = load_items("dev", gold_mode), load_items("test", gold_mode)
    if not test:
        raise SystemExit(f"nenhuma sentença de test com gold ({gold_mode})")
    status = Counter(it["gold"].status for it in dev + test)
    cfgs = {c.key: c for c in configs()}
    threshold, grid = calibrate_merge(llm, cfgs[BASE], dev)
    print(f"merge: limiar calibrado no dev = {threshold}")
    cfgs = {k: c.model_copy(update={"merge_threshold": threshold}) for k, c in cfgs.items()}
    keys = [k for k in cfgs if not only or k in only]
    spec = RunSpec(
        task=TASK,
        model=JUDGE_MODEL,
        prompt_version=JUDGE_PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        dataset=gold_fingerprint(dev + test),
        params={
            "gold": gold_mode,
            "configs": {k: cfgs[k].spec().run_id for k in keys},
            "merge_grid": list(MERGE_GRID),
            "emb_match": [EMB_MATCH, EMB_MATCH_OVERLAP],
        },
    )
    results, preds = {}, []
    for k in keys:
        rows, exts = evaluate(llm, cfgs[k], test)
        agg = aggregate(rows)
        results[k] = {
            "config": cfgs[k].model_dump(),
            "extraction_run_id": cfgs[k].spec().run_id,
            "prompt_version": cfgs[k].prompt_version,
            "prompt_sha256": cfgs[k].prompt_sha256,
            "schema_version": EXTRACT_SCHEMA_VERSION,
            **agg,
        }
        for row, e in zip(rows, exts, strict=True):
            preds.append({
                "config": k, "ruling_id": e.ruling_id, "run_id": e.run_id,
                "pares_gold_extraido": row["pairs"],
                "argumentos": [a.model_dump() for a in e.argumentos],
                "descartados_sem_ancora": e.sem_ancora,
                "chunks_falhos": e.falhas,
            })  # fmt: skip
        cost, p50 = agg["custo_usd_por_sentenca"], agg["latencia_ms_p50"]
        print(f"{k:24} P {agg['precision']:.3f} R {agg['recall']:.3f} F1 {agg['f1']:.3f}  "
              f"${cost:.4f}/sent  p50 {p50:.0f}ms")  # fmt: skip
    meta = {
        "gold_status": dict(status),
        "provisorio": status.get("rascunho", 0) > 0,
        "n_dev": len(dev),
        "n_test": len(test),
        "merge_limiar": threshold,
        "merge_grid": grid,
        "gold_modelo": sorted({it["gold"].rascunho.get("modelo", "?") for it in dev + test}),
    }
    w = RunWriter(spec)
    w.write_metrics({"meta": meta, "configs": results})
    with (w.dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    w.write_report(report(results, meta, spec))
    return w


# --------------------------------------------------------------------------- relatório


def _params(cfg: dict) -> str:
    return json.dumps({k: v for k, v in cfg.items() if k != "key"}, ensure_ascii=False)


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}"


def _table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def report(results: dict[str, dict], meta: dict, spec: RunSpec) -> str:
    header = [
        "config", "modelo", "modo", "P", "R", "F1 (IC95)", "citações ancoradas",
        "brutos sem âncora", "tipo", "decisão", "citação confere", "US$/sentença",
        "latência ms p50/p95",
    ]  # fmt: skip

    def row(k: str) -> list:
        r = results[k]
        lo, hi = r["f1_ic95"]
        c = r["config"]
        return [
            f"`{k}`", c["model"], c["mode"], _pct(r["precision"]), _pct(r["recall"]),
            f"{_pct(r['f1'])} ({_pct(lo)}–{_pct(hi)})", _pct(r["citacoes_ancoradas"]),
            _pct(r["brutos_sem_ancora"]), _pct(r["acuracia_campos"]["tipo"]),
            _pct(r["acuracia_campos"]["decisao"]), _pct(r["citacao_confere"]),
            f"{r['custo_usd_por_sentenca']:.4f}",
            f"{r['latencia_ms_p50']:.0f} / {r['latencia_ms_p95']:.0f}",
        ]  # fmt: skip

    sections = []
    for title, ks in GROUPS.items():
        ks = [k for k in ks if k in results]
        if ks:
            sections.append(f"### {title}\n\n{_table(header, [row(k) for k in ks])}")
    band_keys = [
        k for k in ("full_mini", "truncado_mini", "chunk_mini", "chunk_41") if k in results
    ]
    bands = ("curto", "medio", "longo", "muito_longo")

    def band_cell(k: str, b: str) -> str:
        fb = results[k]["por_faixa"].get(b)
        return f"{_pct(fb['recall'])} / {_pct(fb['precision'])}" if fb else "—"

    band_rows = [[f"`{k}`"] + [band_cell(k, b) for b in bands] for k in band_keys]
    nb = {}
    for b in bands:
        nb[b] = next((r["por_faixa"][b]["n"] for r in results.values() if b in r["por_faixa"]), 0)
    conf_rows = []
    for k in ("chunk_mini", "chunk_mini_verif"):
        if k in results:
            pc = results[k]["precisao_por_confianca"]
            conf_rows.append([f"`{k}`"] + [
                f"{_pct(pc[lv]['precision'])} (n={pc[lv]['n']})" if lv in pc else "—"
                for lv in ("alta", "media", "baixa")
            ])  # fmt: skip
    grid_rows = [[g["limiar"], _pct(g["precision"]), _pct(g["recall"]), _pct(g["f1"])]
                 for g in meta["merge_grid"]]  # fmt: skip
    misc_rows = [
        [f"`{k}`", r["n_pred"], r["fundidos"], r["descartados_verificador"],
         f"{r['chunks_falhos']} / {r['chunks_com_retry']}",
         r["args_em_sentenca_sem_argumento"], f"{r['chunks_por_sentenca']:.1f}",
         f"{r['tokens_entrada_por_sentenca']:.0f}", _pct(r["concordancia_pareadores"]),
         _pct(r["f1_pareador_embeddings"])]
        for k, r in results.items()
    ]  # fmt: skip
    gs = meta["gold_status"]
    warn = (
        f"> **PROVISÓRIO.** O gold ainda contém {gs.get('rascunho', 0)} rascunho(s) LLM "
        f"(`{', '.join(meta['gold_modelo'])}`) não revisados por humano "
        f"({gs.get('revisado', 0)} revisados). Números medem concordância com o rascunho; "
        "modelos da mesma família do rascunho tendem a ser favorecidos. Reexecutar após a revisão."
        if meta["provisorio"]
        else f"Gold: {gs.get('revisado', 0)} sentenças revisadas por humano."
    )
    any_r = next(iter(results.values()))
    cfg_lines = "\n".join(
        f"- `{k}`: run `{r['extraction_run_id']}`, prompt `{r['prompt_version']}` "
        f"(sha `{r.get('prompt_sha256', '?')}`), schema `{r.get('schema_version', '?')}`, "
        f"params `{_params(r['config'])}`"
        for k, r in results.items()
    )
    return f"""# Q2: benchmark de extração de argumentos

Gerado por `uv run q2 bench`. Run `{spec.run_id}`, gold `{spec.dataset}`.

{warn}

Split test: {meta["n_test"]} sentenças, {any_r["n_gold"]} argumentos no gold. Limiar do merge
calibrado no dev ({meta["n_dev"]} sentenças): **{meta["merge_limiar"]}**.

Pareamento extraído × gold: juiz `{spec.model}` (`{spec.prompt_version}`) decide se tratam da
mesma questão; one-to-one forçado em código. P = extraídos pareados / extraídos; R = gold
pareados / gold. Campos (tipo, decisão) e "citação confere" (alguma citação extraída sobrepõe
uma citação do gold no texto) são medidos sobre os pares. "Brutos sem âncora" = argumentos
devolvidos pelo LLM sem nenhuma citação encontrada no texto (proxy de invenção), antes do
filtro. Latência = chamadas originais, chunks em paralelo. IC95 = bootstrap sobre sentenças.

## Resultados

{chr(10).join(sections)}

## Por faixa de tamanho (recall / precisão %)

Sentenças no test por faixa: {", ".join(f"{b} {nb[b]}" for b in bands)}. Faixas com poucas
sentenças indicam tendência, não prova.

{_table(["config", *bands], band_rows)}

## Confiança calibrada? (precisão % por nível)

Confiança = média do score de ancoragem × fator do verificador (sem verificador, teto 0,8:
nada sai "alta").

{_table(["config", "alta", "média", "baixa"], conf_rows)}

## Calibração do merge (dev, `{BASE}`)

{_table(["limiar cosseno", "P", "R", "F1"], grid_rows)}

## Outros números

{
        _table(
            [
                "config",
                "extraídos",
                "fundidos no merge",
                "descartados pelo verificador",
                "chunks falhos / com retry",
                "extraídos em sentença sem argumento",
                "chunks/sentença",
                "tokens entrada/sentença",
                "concordância pareadores %",
                "F1 com pareador embeddings",
            ],
            misc_rows,
        )
    }

## Configurações

{cfg_lines}
"""
