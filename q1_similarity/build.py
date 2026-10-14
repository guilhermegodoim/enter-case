"""Montagem determinística do dataset da Q1: casos, documentos e pares rotulados.

Rótulos (ver data/synthetic/q1/README.md):
- cópia: mesmo conteúdo (mesma sequência de blocos); só mudam valores de entidades e/ou
  a superfície (formatação, paginação, OCR).
- versão: mesma linhagem (mesmo template de origem), mas blocos adicionados, removidos ou
  reescritos.
- diferente: linhagens distintas, mesmo que compartilhem tese, partes ou boilerplate.

O rótulo vem da proveniência dos blocos, nunca de uma métrica de texto.
"""

from __future__ import annotations

import json
import random
import shutil
import statistics
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path

from common.config import ROOT
from common.runs import fingerprint_files
from q1_similarity import spec
from q1_similarity.noise import reformat, scan
from q1_similarity.render import (
    DEFENSE_STYLE,
    FIRM_STYLE,
    Block,
    DocSpec,
    base_contestacao,
    base_peticao,
    case_entities,
    doc_date,
    ementa_blocks,
    ementas,
    family,
    firm_lawyers,
    render,
    stable_seed,
    strip_marker,
    with_filing_date,
)

OUT_DIR = ROOT / "data" / "synthetic" / "q1"
DATASET_VERSION = "q1-synth-v1"
DEFAULT_SEED = 13
DEFAULT_CASES = 60

SURFACES = ["digital", "pdf", "scan_leve", "scan_media", "scan_forte"]
LEVELS = {"leve": (0.03, 0.10), "media": (0.12, 0.25), "forte": (0.28, 0.45)}


@dataclass
class Doc:
    doc_id: str
    case_id: str
    spec: DocSpec
    surface: str
    op: str  # original | copia:<tipo> | versao:<nivel> | contestacao | neg:<tipo>
    parent: str | None = None
    text: str = ""


# --------------------------------------------------------------------------- superfície


def apply_surface(text: str, surface: str, rng: random.Random, header: str | None) -> str:
    if surface == "digital":
        return text
    if surface == "pdf":
        return reformat(text, rng, header)
    return scan(text, rng, header, surface.removeprefix("scan_"))


def header_for(doc: DocSpec) -> str:
    if doc.doc_type == "contestacao":
        return f"Processo nº {doc.ents['PROCESSO_NUMERO']}"
    return spec.FIRMS[doc.firm].nome if doc.firm else ""


# --------------------------------------------------------------------------- versões


def weighted_distance(a: DocSpec, b: DocSpec) -> float:
    """1 - Jaccard ponderado por palavras dos ids de bloco (ids reescritos contam como novos)."""
    wa = {x.id: len(x.tpl.split()) or 1 for x in a.blocks}
    wb = {x.id: len(x.tpl.split()) or 1 for x in b.blocks}
    inter = sum(min(wa[k], wb[k]) for k in wa.keys() & wb.keys())
    union = sum(max(wa.get(k, 0), wb.get(k, 0)) for k in wa.keys() | wb.keys())
    return 1 - inter / union


def _is_editable(b: Block) -> bool:
    return b.id.startswith("fam:") and b.kind in ("para", "item")


def _rewrite(blocks: list[Block], rng: random.Random, tese: str, firm: str) -> bool:
    para = family(tese, firm).parafrases
    cands = [
        i
        for i, b in enumerate(blocks)
        if _is_editable(b) and "~" not in b.id and b.id.rsplit(":", 1)[1] in para
    ]
    if not cands:
        return False
    i = rng.choice(cands)
    b = blocks[i]
    text = para[b.id.rsplit(":", 1)[1]]
    blocks[i] = replace(b, id=b.id + "~r", tpl=strip_marker(text) if b.kind == "item" else text)
    return True


def _add(blocks: list[Block], rng: random.Random, tese: str, firm: str) -> bool:
    fk = spec.family_key(tese, firm)
    extras = family(tese, firm).extras
    used = {b.id for b in blocks}
    cands = [k for k in range(len(extras)) if f"x:{fk}:{k}" not in used]
    if not cands:
        return False
    k = rng.choice(cands)
    ex = extras[k]
    anchors = [
        i for i, b in enumerate(blocks) if b.section == ex.secao_alvo and b.kind in ("para", "item")
    ]
    if not anchors:
        return False
    i = rng.choice(anchors)
    kind = "item" if ex.secao_alvo == "pedidos" else "para"
    tpl = strip_marker(ex.texto) if kind == "item" else ex.texto
    blocks.insert(i + 1, Block(f"x:{fk}:{k}", kind, tpl, ex.secao_alvo, blocks[i].group))
    return True


def _drop(blocks: list[Block], rng: random.Random, tese: str, firm: str) -> bool:
    def count(section: str, group: str) -> int:
        return sum(
            1 for b in blocks if b.section == section and b.group == group and b.kind != "title2"
        )

    cands = [i for i, b in enumerate(blocks) if _is_editable(b) and count(b.section, b.group) > 1]
    if not cands:
        return False
    del blocks[rng.choice(cands)]
    return True


def _swap_ementa(blocks: list[Block], rng: random.Random, tese: str, firm: str) -> bool:
    present = sorted({int(b.id.split(":")[2]) for b in blocks if b.id.startswith(f"em:{tese}:")})
    unused = [k for k in range(len(ementas(tese).itens)) if k not in present]
    if not present or not unused:
        return False
    old, new = rng.choice(present), rng.choice(unused)
    pos = next(i for i, b in enumerate(blocks) if b.id == f"em:{tese}:{old}:ref")
    blocks[pos : pos + 2] = ementa_blocks(tese, new)
    return True


_OPS = [(_rewrite, 0.45), (_add, 0.25), (_drop, 0.2), (_swap_ementa, 0.1)]


def make_version(parent: DocSpec, level: str, rng: random.Random) -> DocSpec:
    lo, hi = LEVELS[level]
    target = rng.uniform(lo, hi)
    blocks = list(parent.blocks)
    child = parent
    for _ in range(200):
        op = rng.choices([o for o, _ in _OPS], weights=[w for _, w in _OPS])[0]
        if not op(blocks, rng, parent.tese, parent.firm):
            continue
        child = replace(parent, blocks=tuple(blocks))
        if weighted_distance(parent, child) >= target:
            break
    return child


# --------------------------------------------------------------------------- caso


def _petition(tese: spec.Tese, firm: spec.Firm, client: spec.Client, ents, seed: int) -> DocSpec:
    return DocSpec(
        doc_type="peticao_inicial",
        family=spec.family_key(tese.key, firm.key),
        tese=tese.key,
        firm=firm.key,
        client=client.key,
        party=ents["AUTOR_CPF"],
        blocks=base_peticao(tese, firm, seed),
        ents=ents,
        style=FIRM_STYLE[firm.key],
    )


def _later(ents: dict[str, str], rng: random.Random, lo: int, hi: int) -> dict[str, str]:
    return with_filing_date(ents, doc_date(ents) + timedelta(days=rng.randint(lo, hi)))


def build_case(case_id: str, tese: spec.Tese, rng: random.Random, seed: int) -> list[Doc]:
    client = rng.choice(spec.clients_for(tese))
    firm = rng.choice(list(spec.FIRMS.values()))
    ents = case_entities(rng, tese, client, firm)
    original = _petition(tese, firm, client, ents, seed)

    docs: list[Doc] = []

    def add(s: DocSpec, surface: str, op: str, parent: Doc | None = None) -> Doc:
        d = Doc("", case_id, s, surface, op, parent.doc_id if parent else None)
        d.doc_id = f"{case_id}-tmp{len(docs)}"
        docs.append(d)
        return d

    root = add(original, rng.choice(["digital", "pdf"]), "original")

    # cópias do original
    copy_kinds = ["identica", "pdf", "scan_leve", "scan_media", "scan_forte", "troca_entidades"]
    n_copies = rng.choice([0, 1, 1, 2, 2, 3])
    for kind in rng.sample(copy_kinds, n_copies):
        if kind == "identica":
            add(original, root.surface, "copia:identica", root)
        elif kind == "troca_entidades":
            other = case_entities(rng, tese, client, firm)
            add(replace(original, ents=other, party=other["AUTOR_CPF"]),
                rng.choice(["digital", "pdf"]), "copia:troca_entidades", root)  # fmt: skip
        else:
            add(original, kind, f"copia:{kind}", root)

    # cadeia de versões (emendas, revisões)
    latest = root
    for _ in range(rng.choice([0, 1, 1, 2, 3])):
        parent = latest if rng.random() < 0.7 else root
        level = rng.choices(list(LEVELS), weights=[0.35, 0.4, 0.25])[0]
        v = make_version(parent.spec, level, rng)
        v = replace(v, ents=_later(parent.spec.ents, rng, 5, 60))
        surface = "scan_leve" if rng.random() < 0.2 else rng.choice(["digital", "pdf"])
        latest = add(v, surface, f"versao:{level}", parent)
    if latest is not root and rng.random() < 0.3:
        add(latest.spec, rng.choice(["pdf", "scan_media"]), "copia:de_versao", latest)

    # contestação da ré: mesmas partes, valores e tema; outro documento
    if rng.random() < 0.75:
        ct = DocSpec(
            doc_type="contestacao",
            family=f"ct:{tese.key}",
            tese=tese.key,
            firm=None,
            client=client.key,
            party=ents["AUTOR_CPF"],
            blocks=base_contestacao(tese),
            ents=_later(ents, rng, 30, 120),
            style=DEFENSE_STYLE,
        )
        c = add(ct, rng.choice(["digital", "pdf", "scan_leve"]), "contestacao")
        if rng.random() < 0.2:
            add(ct, "scan_media", "copia:contestacao", c)

    # negativos difíceis: vocabulário, partes e boilerplate compartilhados
    if rng.random() < 0.45:
        viz = spec.TESES[tese.vizinha]
        add(_petition(viz, firm, client, _later(ents, rng, 1, 90), seed),
            rng.choice(["digital", "pdf"]), "neg:tese_vizinha_mesmo_escritorio")  # fmt: skip
    if rng.random() < 0.4:
        other_firm = next(f for f in spec.FIRMS.values() if f.key != firm.key)
        lawyer, oab = rng.choice(firm_lawyers(other_firm.key))
        e2 = _later(ents, rng, 1, 90) | {"ADVOGADO_NOME": lawyer, "ADVOGADO_OAB": oab}
        add(_petition(tese, other_firm, client, e2, seed),
            rng.choice(["digital", "pdf"]), "neg:mesma_tese_outro_escritorio")  # fmt: skip
    if tese.area == "consumidor" and rng.random() < 0.25:
        others = [t for t in spec.TESES.values() if t.area == tese.area and t.setor != tese.setor]
        t3 = rng.choice(others)
        c3 = rng.choice(spec.clients_for(t3))
        e3 = case_entities(rng, t3, c3, firm)
        add(_petition(t3, firm, c3, e3, seed),
            rng.choice(["digital", "pdf"]), "neg:outra_tese_mesmo_escritorio")  # fmt: skip

    # ordem de chegada: original primeiro; depois aleatória respeitando pai antes do filho
    placed = [root]
    pending = docs[1:]
    while pending:
        ready = [d for d in pending if d.parent is None or d.parent in {p.doc_id for p in placed}]
        pick = rng.choice(ready)
        pending.remove(pick)
        placed.append(pick)
    rename = {d.doc_id: f"{case_id}-d{i:02d}" for i, d in enumerate(placed)}
    for d in placed:
        d.doc_id = rename[d.doc_id]
        d.parent = rename[d.parent] if d.parent else None
        clean = render(d.spec)
        srng = random.Random(stable_seed(seed, "surface", d.doc_id))
        d.text = apply_surface(clean, d.surface, srng, header_for(d.spec))
    # cópia idêntica é byte a byte igual à origem
    by_id = {d.doc_id: d for d in placed}
    for d in placed:
        if d.op == "copia:identica":
            d.text = by_id[d.parent].text
    return placed


# --------------------------------------------------------------------------- pares


_NOISE_RANK = {s: i for i, s in enumerate(SURFACES)}


def _bucket(dist: float) -> str:
    if dist < 0.11:
        return "leve"
    return "media" if dist < 0.265 else "forte"


def label_pair(a: Doc, b: Doc) -> dict:
    sa, sb = a.spec, b.spec
    noise = max([a.surface, b.surface], key=_NOISE_RANK.__getitem__)
    dist = round(weighted_distance(sa, sb), 4)
    if sa.family != sb.family:
        if sa.doc_type != sb.doc_type:
            rel = "tipo_diferente"
        elif sa.tese == sb.tese:
            rel = "mesma_tese_outro_escritorio"
        elif sa.firm == sb.firm and spec.TESES[sa.tese].vizinha == sb.tese:
            rel = "tese_vizinha_mesmo_escritorio"
        elif sa.firm == sb.firm:
            rel = "outra_tese_mesmo_escritorio"
        else:
            rel = "outra_tese_outro_escritorio"
        label = "diferente"
    elif sa.block_ids == sb.block_ids:
        label = "copia"
        if sa.ents["AUTOR_CPF"] != sb.ents["AUTOR_CPF"]:
            rel = "troca_entidades"
        elif a.text == b.text:
            rel = "identica"
        else:
            rel = f"ruido_{noise}"
    else:
        label = "versao"
        rel = f"versao_{_bucket(dist)}"
    return {
        "a": a.doc_id,
        "b": b.doc_id,
        "label": label,
        "relation": rel,
        "block_distance": dist,
        "noise": noise,
    }


# --------------------------------------------------------------------------- dataset


def build(out: Path = OUT_DIR, seed: int = DEFAULT_SEED, n_cases: int = DEFAULT_CASES) -> dict:
    rng = random.Random(seed)
    teses = list(spec.TESES.values())
    cases, docs_out, pairs = [], [], []
    texts: dict[str, str] = {}
    for i in range(n_cases):
        tese = teses[i % len(teses)]  # balanceado por tese
        case_id = f"caso-{i:03d}"
        split = "dev" if (i // len(teses)) % 2 == 0 else "test"  # estratificado por tese
        docs = build_case(case_id, tese, random.Random(rng.getrandbits(63)), seed)
        cases.append(
            {
                "case_id": case_id,
                "client": docs[0].spec.client,
                "tese": tese.key,
                "firm": docs[0].spec.firm,
                "split": split,
                "n_docs": len(docs),
            }
        )
        for d in docs:
            texts[d.doc_id] = d.text
            docs_out.append(
                {
                    "doc_id": d.doc_id,
                    "case_id": case_id,
                    "seq": int(d.doc_id.rsplit("-d", 1)[1]),
                    "doc_type": d.spec.doc_type,
                    "family": d.spec.family,
                    "tese": d.spec.tese,
                    "firm": d.spec.firm,
                    "client": d.spec.client,
                    "op": d.op,
                    "parent": d.parent,
                    "surface": d.surface,
                    "n_words": len(d.text.split()),
                    "n_chars": len(d.text),
                    "blocks": d.spec.block_ids,
                }
            )
        # cada documento novo é comparado com os anteriores do mesmo caso
        for j, b in enumerate(docs):
            for a in docs[:j]:
                pairs.append({"case_id": case_id, "split": split, **label_pair(a, b)})

    if (out / "docs").exists():
        shutil.rmtree(out / "docs")
    (out / "docs").mkdir(parents=True)
    for doc_id, text in texts.items():
        (out / "docs" / f"{doc_id}.txt").write_text(text + "\n", "utf-8")
    _jsonl(out / "cases.jsonl", cases)
    _jsonl(out / "docs.jsonl", docs_out)
    _jsonl(out / "pairs.jsonl", pairs)
    files = sorted((out / "docs").glob("*.txt")) + [out / f for f in ("docs.jsonl", "pairs.jsonl")]
    meta = {
        "dataset_version": DATASET_VERSION,
        "seed": seed,
        "n_cases": n_cases,
        "fingerprint": fingerprint_files(files),
        "templates_fingerprint": fingerprint_files(
            sorted((ROOT / "data" / "synthetic" / "templates").rglob("*.json"))
        ),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", "utf-8")
    (out / "STATS.md").write_text(stats_markdown(meta, cases, docs_out, pairs), "utf-8")
    return meta


def _jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text("utf-8").splitlines() if ln.strip()]


# --------------------------------------------------------------------------- estatísticas


def _table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def _count(rows: list[dict], key: str, split: str | None = None) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        if split is None or r["split"] == split:
            out[r[key]] = out.get(r[key], 0) + 1
    return out


def _quantiles(values: list[int]) -> list[int]:
    if len(values) < 2:
        return [values[0]] * 4 if values else [0] * 4
    q = statistics.quantiles(values, n=10, method="inclusive")
    return [min(values), round(statistics.median(values)), round(q[8]), max(values)]


def stats_markdown(meta: dict, cases: list[dict], docs: list[dict], pairs: list[dict]) -> str:
    total = _count(pairs, "label")
    dev, test = _count(pairs, "label", "dev"), _count(pairs, "label", "test")
    labels = ["copia", "versao", "diferente"]
    by_label = _table(
        ["rótulo", "dev", "test", "total", "%"],
        [
            [lb, dev.get(lb, 0), test.get(lb, 0), total.get(lb, 0),
             f"{100 * total.get(lb, 0) / len(pairs):.1f}"]
            for lb in labels
        ]
        + [["**total**", sum(dev.values()), sum(test.values()), len(pairs), "100"]],
    )  # fmt: skip
    rel_rows = []
    for lb in labels:
        rels = _count([p for p in pairs if p["label"] == lb], "relation")
        for rel, n in sorted(rels.items(), key=lambda kv: -kv[1]):
            dists = [p["block_distance"] for p in pairs if p["relation"] == rel]
            rel_rows.append([lb, rel, n, f"{statistics.mean(dists):.3f}"])
    by_rel = _table(["rótulo", "relação", "pares", "distância de blocos média"], rel_rows)

    groups: dict[tuple[str, str], list[int]] = {}
    for d in docs:
        key = (d["doc_type"], d["firm"] or "ré")
        groups.setdefault(key, []).append(d["n_words"])
    length = _table(
        ["tipo", "escritório", "docs", "palavras min", "mediana", "p90", "max"],
        [[t, f, len(v), *_quantiles(v)] for (t, f), v in sorted(groups.items())],
    )
    surf = _count(docs, "surface")
    surfaces = _table(["superfície", "docs"], [[s, surf.get(s, 0)] for s in SURFACES])
    ops = _count(docs, "op")
    op_table = _table(["operação", "docs"], sorted(ops.items(), key=lambda kv: -kv[1]))
    per_case = [c["n_docs"] for c in cases]
    chars = [d["n_chars"] for d in docs]

    return f"""# Dataset sintético Q1: estatísticas

Gerado por `uv run q1 build`. Não editar à mão.

| campo | valor |
|---|---|
| versão | `{meta["dataset_version"]}` |
| seed | {meta["seed"]} |
| fingerprint (docs + pares) | `{meta["fingerprint"]}` |
| fingerprint dos templates | `{meta["templates_fingerprint"]}` |
| casos | {len(cases)} ({_count(cases, "split").get("dev", 0)} dev, \
{_count(cases, "split").get("test", 0)} test) |
| documentos | {len(docs)} |
| documentos por caso | min {min(per_case)}, mediana {statistics.median(per_case):g}, \
max {max(per_case)} |
| caracteres por documento | mediana {round(statistics.median(chars))}, max {max(chars)} |

Pares = cada documento comparado com todos os anteriores do mesmo caso, na ordem de chegada.

## Pares por rótulo

{by_label}

## Pares por relação

Distância de blocos = 1 − Jaccard ponderado por palavras entre os blocos de origem dos dois
documentos (0 = mesmo conteúdo). Ruído de superfície e troca de entidades não alteram blocos.

{by_rel}

## Tamanho dos documentos (palavras)

{length}

## Documentos por operação

{op_table}

## Documentos por superfície

{surfaces}
"""
