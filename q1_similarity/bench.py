"""Benchmark da Q1: métodos, ablações, custo e latência por documento.

Protocolo:
- Limiares e modelos são calibrados no split dev; números reportados são do split test.
- Cada documento de teste é ingerido na ordem de chegada do seu caso (CaseIndex), como em
  produção. Métricas por par (novo × anterior) e por documento (ação tomada).
- Latência = CPU local medida + latência original das chamadas de API (cache preserva a
  latência da primeira chamada, então reexecutar devolve os mesmos números de custo).
- IC 95% por bootstrap sobre casos (os pares de um caso não são independentes).
"""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from datasketch import MinHashLSH
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from common.config import ROOT
from common.llm import LLM
from common.runs import RunSpec, RunWriter
from q1_similarity import judge as judge_mod
from q1_similarity.classify import EMB_FEATURES, LEX_FEATURES, LogReg, Rule, fit_prefilter
from q1_similarity.data import LABELS, Dataset
from q1_similarity.features import (
    NUM_PERM,
    boilerplate_shingles,
    chunk_texts,
    pair_features,
    shingles,
    signature,
)
from q1_similarity.pipeline import CaseIndex, Config
from q1_similarity.preprocess import prepare

TASK = "q1-bench"
SCHEMA_VERSION = "q1-bench-v1"
BOOTSTRAP = 1000
GRAY_FACTOR = 2.0
_RANK = {"copia": 2, "versao": 1, "diferente": 0}


# --------------------------------------------------------------------------- métodos


@dataclass
class Method:
    key: str
    descricao: str
    classifier: Any
    prep: dict[str, bool] = field(default_factory=lambda: {"layout": True, "mask": True})
    regions: bool = True
    fast_hash: bool = True
    prefilter: bool = False
    use_llm: bool = False
    embeddings: bool = False
    boilerplate: bool = False
    group: str = "metodo"  # metodo | ablacao


def methods() -> list[Method]:
    raw = {"layout": False, "mask": False}
    return [
        Method("jaccard_bruto", "Jaccard de shingles sobre texto só normalizado",
               Rule("jaccard_bruto", "jaccard", "jaccard", -1, fast_path=False),
               prep=raw, regions=False, fast_hash=False),
        Method("minhash", "MinHash (128 perm.) sobre texto limpo e mascarado",
               Rule("minhash", "minhash", "minhash", -1),
               regions=False),
        Method("embeddings", "Cosseno médio de embeddings por trecho (text-embedding-3-small)",
               Rule("embeddings", "emb_cos", "emb_cos", -1, fast_path=False),
               regions=False, fast_hash=False, embeddings=True),
        Method("alinhamento", "Hash + alinhamento de tokens com trechos classificados (regra)",
               Rule("alinhamento", "align_cov", "content_ratio")),
        Method("logreg", "Regressão logística sobre todas as features léxicas + embeddings",
               LogReg("logreg", tuple(LEX_FEATURES + EMB_FEATURES)), embeddings=True),
        Method("cascata", "Hash → pré-filtro MinHash → alinhamento (regra)",
               Rule("cascata", "align_cov", "content_ratio"), prefilter=True),
        Method("cascata_llm", "Cascata + juiz LLM na zona cinzenta cópia × versão",
               Rule("cascata_llm", "align_cov", "content_ratio", gray_factor=GRAY_FACTOR),
               prefilter=True, use_llm=True),
        # ablações
        Method("sem_mascara", "Alinhamento sem máscara de entidades",
               Rule("sem_mascara", "align_cov", "content_ratio"),
               prep={"layout": True, "mask": False}, group="ablacao"),
        Method("sem_layout", "Alinhamento sem limpeza de cabeçalho/paginação",
               Rule("sem_layout", "align_cov", "content_ratio"),
               prep={"layout": False, "mask": True}, group="ablacao"),
        Method("minhash_boilerplate", "MinHash com shingles de boilerplate removidos",
               Rule("minhash_boilerplate", "minhash_f", "minhash_f", -1),
               regions=False, boilerplate=True, group="ablacao"),
        Method("alinhamento_sem_hash", "Alinhamento sem atalho de hash",
               Rule("alinhamento_sem_hash", "align_cov", "content_ratio", fast_path=False),
               fast_hash=False, group="ablacao"),
    ]  # fmt: skip


# --------------------------------------------------------------------------- dados


class Corpus:
    """Textos, embeddings (com latência original) e features por variante de preparo."""

    def __init__(self, ds: Dataset, llm: LLM):
        self.ds, self.llm = ds, llm
        self._emb: dict[str, tuple[np.ndarray, float, float]] = {}
        self._prep: dict[tuple, dict] = {}

    def emb(self, doc_id: str) -> tuple[np.ndarray, float, float]:
        if doc_id not in self._emb:
            vecs, rec = self.llm.embed(chunk_texts(prepare(self.ds.text(doc_id))))
            self._emb[doc_id] = (np.array(vecs), rec.latency_ms, rec.cost_usd or 0.0)
        return self._emb[doc_id]

    def preps(self, prep: dict[str, bool]) -> dict:
        key = tuple(sorted(prep.items()))
        if key not in self._prep:
            self._prep[key] = {i: prepare(self.ds.text(i), **prep) for i in self.ds.docs}
        return self._prep[key]

    def boilerplate(self, prep: dict[str, bool], max_df: float = 0.1) -> frozenset[int]:
        p = self.preps(prep)
        by_case: dict[str, list[set[int]]] = defaultdict(list)
        for i, d in self.ds.docs.items():
            if self.ds.cases[d["case_id"]]["split"] == "dev":
                by_case[d["case_id"]].append(shingles(p[i].tokens))
        return boilerplate_shingles(by_case, max_df)

    def dev_rows(self, m: Method, bp: frozenset[int]) -> list[dict]:
        p = self.preps(m.prep)
        sigs = {}
        rows = []
        for pair in self.ds.split_pairs("dev"):
            for i in (pair["a"], pair["b"]):
                if i not in sigs:
                    sigs[i] = signature(p[i], bp)
                    if m.embeddings:
                        sigs[i].emb = self.emb(i)[0]
            f, _ = pair_features(sigs[pair["a"]], sigs[pair["b"]], regions=m.regions)
            rows.append({**pair, **f.as_dict()})
        return rows


# --------------------------------------------------------------------------- avaliação


def _prf(y: list[str], p: list[str]) -> dict[str, Any]:
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=list(LABELS), zero_division=0)
    per = {
        lb: {"precision": float(pr[k]), "recall": float(rc[k]), "f1": float(f1[k]),
             "support": int(sup[k])}
        for k, lb in enumerate(LABELS)
    }  # fmt: skip
    return {
        "macro_f1": float(np.mean(f1)),
        "accuracy": float(np.mean(np.array(y) == np.array(p))),
        "per_class": per,
    }


def _bootstrap(rows: list[dict], seed: int = 0) -> tuple[float, float]:
    by_case: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_case[r["case_id"]].append(r)
    cases = sorted(by_case)
    rng = random.Random(seed)
    vals = []
    for _ in range(BOOTSTRAP):
        sample = [r for c in rng.choices(cases, k=len(cases)) for r in by_case[c]]
        vals.append(_prf([r["label"] for r in sample], [r["pred"] for r in sample])["macro_f1"])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def doc_gold(ds: Dataset, split: str) -> dict[str, tuple[str, set[str]]]:
    """Por documento: ação correta (copia > versao > novo) e documentos que a justificam."""
    best: dict[str, tuple[int, set[str]]] = {}
    for p in ds.split_pairs(split):
        r = _RANK[p["label"]]
        cur = best.get(p["b"], (0, set()))
        if r > cur[0]:
            best[p["b"]] = (r, {p["a"]})
        elif r == cur[0] and r > 0:
            cur[1].add(p["a"])
    out = {}
    for c in ds.split_cases(split):
        for d in ds.case_docs(c):
            r, m = best.get(d["doc_id"], (0, set()))
            out[d["doc_id"]] = ({2: "copia", 1: "versao", 0: "novo"}[r], m)
    return out


def evaluate(m: Method, corpus: Corpus, llm: LLM, split: str = "test") -> dict[str, Any]:
    ds = corpus.ds
    bp = corpus.boilerplate(m.prep) if m.boilerplate else frozenset()
    dev = corpus.dev_rows(m, bp)
    clf = m.classifier.fit(dev)
    pre = fit_prefilter(dev) if m.prefilter else 0.0
    cfg = Config(clf, prefilter=pre, use_llm=m.use_llm, prep=m.prep, fast_hash=m.fast_hash,
                 regions=m.regions, boilerplate=bp)  # fmt: skip
    gold_pairs = {(p["a"], p["b"]): p for p in ds.split_pairs(split)}
    gold_docs = doc_gold(ds, split)
    rows, docs = [], []
    for c in ds.split_cases(split):
        idx = CaseIndex(cfg, llm)
        for d in ds.case_docs(c):
            emb, emb_ms, emb_cost = corpus.emb(d["doc_id"]) if m.embeddings else (None, 0, 0)
            dec = idx.ingest(d["doc_id"], ds.text(d["doc_id"]), emb=emb, extra_ms=emb_ms)
            for pr in dec.pairs:
                g = gold_pairs[(pr.a, pr.b)]
                rows.append({**g, "pred": pr.label, "stage": pr.stage, "gray": pr.gray})
            act, matches = gold_docs[d["doc_id"]]
            docs.append({
                "doc_id": d["doc_id"], "case_id": c, "gold": act, "pred": dec.action,
                "match": dec.match, "match_ok": dec.match in matches if dec.match else None,
                "review": dec.needs_review, "latency_ms": dec.latency_ms,
                "cost_usd": dec.llm_cost + emb_cost, "n_prev": len(dec.pairs),
            })  # fmt: skip
    y, p = [r["label"] for r in rows], [r["pred"] for r in rows]
    by_rel: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_rel[f"{r['label']}/{r['relation']}"].append(r["label"] == r["pred"])
    lat = [d["latency_ms"] for d in docs]
    copy_act = [d for d in docs if d["pred"] == "copia"]
    stages = defaultdict(int)
    for r in rows:
        stages[r["stage"]] += 1
    return {
        "metodo": m.key,
        "descricao": m.descricao,
        "grupo": m.group,
        "params": {**clf.params(), "prefiltro_minhash": round(pre, 4), "llm": m.use_llm},
        "pares": {
            **_prf(y, p),
            "macro_f1_ic95": _bootstrap(rows),
            "confusao": confusion_matrix(y, p, labels=list(LABELS)).tolist(),
            "por_relacao": {k: float(np.mean(v)) for k, v in sorted(by_rel.items())},
            "estagios": dict(stages),
            "zona_cinzenta": sum(r["gray"] for r in rows),
        },
        "documentos": {
            "n": len(docs),
            "acuracia_acao": float(np.mean([d["gold"] == d["pred"] for d in docs])),
            "precisao_copia": (
                float(np.mean([d["gold"] == "copia" and d["match_ok"] for d in copy_act]))
                if copy_act else None
            ),
            "revisao_humana": float(np.mean([d["review"] for d in docs])),
            "latencia_ms_p50": float(np.percentile(lat, 50)),
            "latencia_ms_p95": float(np.percentile(lat, 95)),
            "custo_usd_por_doc": float(np.mean([d["cost_usd"] for d in docs])),
        },
        "_rows": rows,
        "_docs": docs,
    }  # fmt: skip


# --------------------------------------------------------------------------- escala (LSH)


def lsh_scale(corpus: Corpus, threshold: float = 0.5) -> dict[str, Any]:
    """Se a busca fosse no corpus inteiro, quanto MinHash-LSH reduz candidatos sem perder recall.

    Escopo da Q1 é o caso (poucos documentos), então o LSH não está no caminho crítico; isto
    mostra que a mesma assinatura escala para deduplicação entre casos/clientes.
    """
    preps = corpus.preps({"layout": True, "mask": True})
    sigs = {i: signature(p) for i, p in preps.items()}
    t0 = time.perf_counter()
    lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    for i, s in sigs.items():
        lsh.insert(i, s.mh)
    build_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    cands = {i: set(lsh.query(s.mh)) - {i} for i, s in sigs.items()}
    query_ms = (time.perf_counter() - t0) * 1000 / len(sigs)
    same = [p for p in corpus.ds.pairs if p["label"] != "diferente"]
    recall = np.mean([p["a"] in cands[p["b"]] for p in same])
    n = len(sigs)
    return {
        "limiar": threshold,
        "documentos": n,
        "recall_mesma_familia": float(recall),
        "candidatos_por_doc": float(np.mean([len(v) for v in cands.values()])),
        "fracao_do_corpus": float(np.mean([len(v) for v in cands.values()]) / (n - 1)),
        "build_ms": build_ms,
        "query_ms_por_doc": query_ms,
    }


# --------------------------------------------------------------------------- relatório


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}"


def _table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def report(results: list[dict], lsh: dict, meta: dict, judge_summary: dict) -> str:
    def main_rows(group: str) -> list[list]:
        rows = []
        for r in results:
            if r["grupo"] != group:
                continue
            pc, d = r["pares"]["per_class"], r["documentos"]
            lo, hi = r["pares"]["macro_f1_ic95"]
            rows.append([
                f'`{r["metodo"]}`', f'{_pct(r["pares"]["macro_f1"])} ({_pct(lo)}–{_pct(hi)})',
                _pct(pc["copia"]["precision"]), _pct(pc["copia"]["recall"]),
                _pct(pc["versao"]["f1"]), _pct(pc["diferente"]["f1"]),
                _pct(d["acuracia_acao"]), _pct(d["precisao_copia"]),
                f'{d["latencia_ms_p50"]:.0f} / {d["latencia_ms_p95"]:.0f}',
                f'{d["custo_usd_por_doc"]:.5f}',
            ])  # fmt: skip
        return rows

    header = [
        "método", "macro-F1 % (IC95)", "P cópia", "R cópia", "F1 versão", "F1 dif.",
        "acurácia ação (doc)", "precisão ação cópia", "latência ms p50/p95", "US$/doc",
    ]  # fmt: skip
    rels = sorted({k for r in results for k in r["pares"]["por_relacao"]})
    keys = [r["metodo"] for r in results if r["grupo"] == "metodo"]
    rel_rows = [
        [rel] + [_pct(next(r for r in results if r["metodo"] == k)["pares"]["por_relacao"][rel])
                 for k in keys]
        for rel in rels
    ]  # fmt: skip
    best = next(r for r in results if r["metodo"] == "cascata_llm")
    conf = best["pares"]["confusao"]
    conf_rows = [[f"**{lb}**"] + conf[i] for i, lb in enumerate(LABELS)]
    params = "\n".join(
        f"- `{r['metodo']}`: {r['descricao']}. "
        f"Parâmetros: `{json.dumps(r['params'], ensure_ascii=False)}`"
        for r in results
    )
    n_pairs = sum(best["pares"]["per_class"][k]["support"] for k in LABELS)
    cand = f"{lsh['candidatos_por_doc']:.1f} ({_pct(lsh['fracao_do_corpus'])}% do corpus)"
    return f"""# Q1: benchmark de similaridade (cópia / versão / diferente)

Gerado por `uv run q1 bench`. Dataset `{meta["dataset_version"]}` (fingerprint
`{meta["fingerprint"]}`). Calibração no split dev, números no split test
({best["documentos"]["n"]} documentos, {n_pairs} pares).

Cada documento de teste é ingerido na ordem de chegada do caso e comparado com os anteriores.
Métricas por par e por documento (ação tomada: cópia, versão ou novo). Latência por documento =
CPU local + latência original das chamadas de API. IC95 = bootstrap sobre casos.

## Métodos

{_table(header, main_rows("metodo"))}

## Ablações

{_table(header, main_rows("ablacao"))}

## Acerto por relação (% dos pares)

Linhas `diferente/*` são os negativos difíceis: vocabulário, partes ou boilerplate em comum.

{_table(["relação"] + [f"`{k}`" for k in keys], rel_rows)}

## Matriz de confusão: `cascata_llm` (linhas = verdade, colunas = predição)

{_table(["", *LABELS], conf_rows)}

Estágios que decidiram os pares: `{json.dumps(best["pares"]["estagios"])}`.
Pares na zona cinzenta enviados ao LLM: {best["pares"]["zona_cinzenta"]}.
Juiz: `{judge_summary["modelo"]}`, prompt `{judge_summary["prompt_version"]}`,
{judge_summary["calls"]} chamadas, {judge_summary["input_tokens"]} tokens de entrada no total.

## Escala: MinHash-LSH sobre o corpus inteiro

Fora do escopo da Q1 (comparação é dentro do caso), mas mostra que a mesma assinatura serve
para deduplicação entre casos: limiar {lsh["limiar"]}, {lsh["documentos"]} documentos.

| métrica | valor |
|---|---|
| recall de pares da mesma família | {_pct(lsh["recall_mesma_familia"])}% |
| candidatos por documento | {cand} |
| consulta por documento | {lsh["query_ms_por_doc"]:.2f} ms |

## Configuração

{params}
"""


# --------------------------------------------------------------------------- execução


def run(only: list[str] | None = None, judge_model: str = judge_mod.DEFAULT_MODEL) -> RunWriter:
    ds = Dataset()
    llm = LLM(ledger_path=ROOT / ".cache" / "ledger-q1-bench.jsonl")
    corpus = Corpus(ds, llm)
    ms = [m for m in methods() if not only or m.key in only]
    spec = RunSpec(
        task=TASK,
        model=judge_model,
        prompt_version=judge_mod.PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        dataset=ds.meta["fingerprint"],
        params={"metodos": [m.key for m in ms], "gray_factor": GRAY_FACTOR,
                "embed_model": llm.embed_model},
    )  # fmt: skip
    results = []
    for m in ms:
        t0 = time.perf_counter()
        llm.reset()
        r = evaluate(m, corpus, llm)
        if m.use_llm:
            s = llm.summary()
            judge_summary = {"modelo": judge_model, "prompt_version": judge_mod.PROMPT_VERSION,
                             "calls": sum(x.kind == "chat" for x in llm.records),
                             "input_tokens": sum(x.input_tokens for x in llm.records
                                                 if x.kind == "chat"), **s}  # fmt: skip
        results.append(r)
        f1, acc = r["pares"]["macro_f1"], r["documentos"]["acuracia_acao"]
        print(f"{m.key:22} macroF1 {f1:.3f}  doc acc {acc:.3f}  {time.perf_counter() - t0:.1f}s")
    lsh = lsh_scale(corpus)
    w = RunWriter(spec)
    w.write_metrics({"lsh": lsh, "metodos": [
        {k: v for k, v in r.items() if not k.startswith("_")} for r in results]})  # fmt: skip
    with (w.dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            for row in r["_rows"]:
                f.write(json.dumps({"metodo": r["metodo"], **row}, ensure_ascii=False) + "\n")
    if any(m.use_llm for m in ms):
        w.write_report(report(results, lsh, ds.meta, judge_summary))
    return w
