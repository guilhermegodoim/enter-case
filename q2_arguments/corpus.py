"""Corpus da Q2: limpeza, anonimização e seleção determinística de sentenças do pool coletado.

Texto armazenado = limpo + anonimizado. Offsets de citações (gold e extração) referem-se a ele.

Anonimização (higiene LGPD, mesmo sendo decisões públicas):
- Pessoas físicas privadas: partes do cabeçalho (Requerente/Requerido...) + nomes listados por
  LLM com papel (testemunhas, terceiros, advogados, servidores). O LLM só lista; a troca é
  feita em código: nome completo e, depois, prenome/sobrenome isolado (ver `anonymize`).
- Identificadores por regex: CPF, RG, telefone, e-mail, CEP, OAB.
- Mantidos: pessoas jurídicas (réus são empresas; o banco de argumentos é por réu), magistrados
  (agentes públicos em ato de ofício), autores de doutrina citados, número do
  processo (proveniência; a coleta é reprodutível a partir das consultas de qualquer forma).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from common.config import ROOT
from common.llm import LLM
from common.text import find_entities, normalize
from common.text import fold_keep_offsets as fold_map
from q2_arguments.scrape import QUERIES_PATH, RAW_DIR, Hit, load_pool, load_queries

RULINGS_DIR = ROOT / "data" / "rulings"
CORPUS_VERSION = "q2-tjsp-v1"
NAMES_PROMPT_VERSION = "q2-pii-names-v2"
NAMES_MODEL = "gpt-4.1"
INLINE_CAP = 49_000  # o CJPG corta o inteiro teor inline perto de 50k caracteres

ALLOWED_CLASSES = (
    "Procedimento Comum Cível",
    "Procedimento do Juizado Especial Cível",
    "Procedimento de Repactuação de Dívidas (Superendividamento)",
    "Busca e Apreensão em Alienação Fiduciária",
)
# faixas por tokens (o200k): "muito_longo" passa de um orçamento típico de chunk (~8k tokens)
LENGTH_BUCKETS = (("curto", 0), ("medio", 3_000), ("longo", 6_000), ("muito_longo", 9_000))
QUOTAS = {"curto": 6, "medio": 10, "longo": 8, "muito_longo": 4}  # com mérito
N_SEM_MERITO = 2  # negativos: sentença sem argumentos de mérito (testa alucinação)
MAX_PER_JUDGE = 2
DEV_EVERY = 3  # 1 em cada 3 vai para dev (ajuste de prompt); o resto é test

# --------------------------------------------------------------------------- limpeza


def clean(raw: str) -> str:
    """Cada linha do inteiro teor do CJPG é um parágrafo: vira bloco separado por linha vazia."""
    lines = (ln.strip() for ln in raw.replace("\r", "\n").split("\n"))
    return normalize("\n\n".join(ln for ln in lines if ln))


def text_tokens(text: str) -> int:
    return len(_encoder().encode(text))


@lru_cache(maxsize=1)
def _encoder():
    import tiktoken

    return tiktoken.get_encoding("o200k_base")


# --------------------------------------------------------------------------- cabeçalho / partes

_ACTIVE = r"requerente|exequente|autora?|reclamante|embargante|parte ativa|devedor"
_PASSIVE = r"requerid[oa]|executad[oa]|r[ée]u|r[ée]|reclamad[oa]|embargad[oa]|parte passiva|credor"
_HEADER = re.compile(
    rf"^(?P<label>(?i:{_ACTIVE}|{_PASSIVE})[^:\n]{{0,40}}):\s*(?P<value>[^\n]+)$", re.MULTILINE
)
_ORG = re.compile(
    r"(?i)\b(?:s\.?/?a\.?|ltda|eireli|me|epp|banco|bank|cia|companhia|associa\w*|institui\w*|"
    r"caixa|cooperativa|servi[çc]os|linhas|a[ée]reas?|airlines?|pagamentos?|financ\w*|"
    r"cr[ée]dito|segur\w*|telef\w*|telecom\w*|claro|vivo|tim|oi|munic[íi]pio|estado|fazenda|inss|"
    r"uni[ãa]o|fundo|condom[íi]nio|empreendimentos|com[ée]rcio|ind[úu]stria|participa[çc][õo]es|"
    r"holding|group|grupo|brasil|administradora|cons[óo]rcio|investimentos?|sociedade|empresa|"
    r"tecnologia|digital|mercado|log[íi]stica|transportes?|editora|universidade|faculdade|"
    r"escola|hospital|cl[íi]nica|sabesp|eletropaulo|energia|saneamento|net|online|internet|"
    r"facebook|google|apple|amazon|meta|ifood|uber|nubank|picpay|santander|bradesco|ita[úu]|"
    r"sicredi|sicoob|outros?|outras?)\b"
)
_TRAIL = re.compile(r"\s+e\s+outr[oa]s?\.?$", re.IGNORECASE)
# "e" não é conector: "FULANO DA SILVA e BELTRANO SOUZA" são dois nomes, não um
_CONNECTORS = {"da", "de", "do", "das", "dos", "d"}
_WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+")
# palavra isolada precedida disto é lugar, não pessoa (São Paulo, Rua Silva...)
_PLACE_BEFORE = re.compile(
    r"(?i:\b(?:s[ãa]o|santo|santa|rio|vila|jardim|rua|avenida|av\.|pra[çc]a|bairro|"
    r"comarca\s+de|foro\s+de)\s+)$"
)
MIN_SINGLE = 4  # palavra isolada de nome só é trocada com >= 4 letras


def header_parties(text: str) -> list[tuple[str, str]]:
    """[(lado, nome)] das partes pessoa física no cabeçalho. lado: 'autor' | 'reu'."""
    out = []
    for m in _HEADER.finditer(text[:3000]):
        value = m.group("value").split(",")[0]  # "Fulano, registrado civilmente como ..."
        name = _TRAIL.sub("", value.strip()).strip(" .")
        side = "autor" if re.match(rf"(?i:{_ACTIVE})", m.group("label")) else "reu"
        if is_person_name(name):
            out.append((side, name))
    return out


def is_person_name(name: str, min_words: int = 2) -> bool:
    words = _WORD.findall(name)
    core = [w for w in words if w.lower() not in _CONNECTORS]
    if not min_words <= len(core) <= 7 or _ORG.search(name) or re.search(r"\d|/", name):
        return False
    return all(w[0].isupper() for w in core)


# --------------------------------------------------------------------------- nomes via LLM

PapelPrivado = Literal[
    "parte", "representante", "testemunha", "advogado", "perito", "servidor", "terceiro"
]
PapelPublico = Literal["magistrado", "autor_doutrina"]
PUBLIC_ROLES = set(PapelPublico.__args__)


class PessoaCitada(BaseModel):
    nome: str = Field(description="Nome exatamente como aparece no texto")
    papel: PapelPrivado | PapelPublico


class PessoasCitadas(BaseModel):
    pessoas: list[PessoaCitada]


NAMES_SYSTEM = """Você ajuda a anonimizar sentenças judiciais públicas (LGPD). Liste TODAS as \
pessoas físicas mencionadas pelo nome no texto, cada uma com seu papel:
- parte: autor(a), réu/ré pessoa física, litisconsortes, menores representados;
- representante: pais, tutores, curadores, sócios ou prepostos que agem pela parte;
- testemunha (inclui informantes e depoentes), advogado, perito;
- servidor: escrevente, assistente judiciário, oficial de justiça;
- terceiro: qualquer outra pessoa privada (parentes, titulares de contas, golpistas...);
- magistrado: juiz(a) que assina, juízes, desembargadores, ministros e relatores citados;
- autor_doutrina: autores de livros e artigos jurídicos citados.

Escreva cada nome exatamente como aparece no texto, uma entrada por grafia diferente (inclusive \
só o prenome ou só o sobrenome, quando a pessoa é chamada assim). Não liste pessoas jurídicas, \
órgãos públicos, marcas e produtos."""


def llm_person_names(llm: LLM, text: str, model: str = NAMES_MODEL) -> list[PessoaCitada]:
    out, _ = llm.structured(
        PessoasCitadas,
        system=NAMES_SYSTEM,
        user=text,
        model=model,
        prompt_version=NAMES_PROMPT_VERSION,
    )
    return out.pessoas


# --------------------------------------------------------------------------- anonimização

_ID_KINDS = {"CPF", "RG", "TELEFONE", "EMAIL", "CEP", "OAB"}


@dataclass
class Anonymized:
    text: str
    replacements: Counter = field(default_factory=Counter)  # rótulo -> nº de trocas
    persons: list[str] = field(default_factory=list)  # rótulos atribuídos (sem os nomes)
    residuos: int = 0  # palavras de nomes privados que sobraram (protegidas por uma regra)


def _core_words(name: str) -> list[str]:
    return [fold_map(w) for w in _WORD.findall(name) if w.lower() not in _CONNECTORS]


def _name_spans(text: str, folded: str, name: str) -> list[tuple[int, int]]:
    """Sequências no texto com >= 2 palavras capitalizadas do nome (conectores permitidos)."""
    vocab = {v for v in _core_words(name) if len(v) >= 2}
    if len(vocab) < 2:
        return []
    words = [(m.start(), m.end()) for m in _WORD.finditer(text)]
    spans, i = [], 0
    while i < len(words):
        j, hits, last = i, 0, None
        while j < len(words):
            s, e = words[j]
            w = folded[s:e]
            gap = text[words[j - 1][1] : s] if j > i else ""
            if gap.strip() != "" or "\n" in gap:  # nome não atravessa pontuação nem parágrafo
                break
            if w in vocab and text[s].isupper():
                hits, last = hits + 1, j
            elif not (w in _CONNECTORS and hits > 0):
                break
            j += 1
        if hits >= 2 and last is not None:
            spans.append((words[i][0], words[last][1]))
            i = last + 1
        else:
            i += 1
    return spans


def anonymize(
    text: str,
    header: list[tuple[str, str]],
    private_names: list[str],
    keep: list[str] = (),  # type: ignore[assignment]
) -> Anonymized:
    """Troca nomes privados e identificadores.

    keep: nomes que ficam de propósito (magistrados, autores de doutrina, comarca/foro). Trecho
    de nome mantido vence trecho privado mais curto que o cruze.
    Passo 1: nome completo (>= 2 palavras do nome em sequência). Passo 2: palavra isolada do
    nome (prenome/sobrenome, >= 4 letras, capitalizada), exceto quando precedida de indicador de
    lugar ("São Paulo") ou quando é um nome mantido de 1 palavra.
    """
    folded = fold_map(text)
    labels: dict[str, str] = {}  # chave do nome -> rótulo
    counts = {"autor": 0, "reu": 0, "pessoa": 0}
    spans: list[tuple[int, int, str]] = []

    def label_for(name: str, side: str) -> str:
        key = " ".join(_core_words(name))
        if key not in labels:
            counts[side] += 1
            base = {"autor": "AUTOR", "reu": "REU", "pessoa": "PESSOA"}[side]
            first = side != "pessoa" and counts[side] == 1
            labels[key] = base if first else f"{base}_{counts[side]}"
        return labels[key]

    named = [(side, n) for side, n in header] + [
        ("pessoa", n) for n in private_names if is_person_name(n, min_words=1)
    ]
    # nome de 1 palavra ("o autor Michel") herda o rótulo do nome completo que a contém
    full = [(s, n) for s, n in named if len(_core_words(n)) >= 2]
    singles = [(s, n) for s, n in named if len(_core_words(n)) == 1]
    word_labels: dict[str, set[str]] = {}
    for side, name in full:
        lab = label_for(name, side)
        spans += [(s, e, lab) for s, e in _name_spans(text, folded, name)]
        for w in _core_words(name):
            word_labels.setdefault(w, set()).add(lab)
    for side, name in singles:
        w = _core_words(name)[0]
        if w not in word_labels:
            word_labels[w] = {label_for(name, side)}

    # nome mantido de 1 palavra (comarca "Santos", "Liebman") protege a palavra em todo o texto;
    # nomes mantidos compostos protegem só os próprios trechos (kept_spans abaixo)
    protected = {ws[0] for k in keep if len(ws := _core_words(k)) == 1}
    # nomes mantidos ("Carlos Roberto Gonçalves") bloqueiam trocas mais curtas que os cruzem
    kept_spans = [sp for k in keep if k for sp in _name_spans(text, folded, k)]
    for m in _WORD.finditer(text):
        w = folded[m.start() : m.end()]
        labs = word_labels.get(w)
        if not labs or len(w) < MIN_SINGLE or not text[m.start()].isupper():
            continue
        if w in protected or _PLACE_BEFORE.search(text[max(0, m.start() - 20) : m.start()]):
            continue
        spans.append((m.start(), m.end(), next(iter(labs)) if len(labs) == 1 else "NOME"))

    for ent in find_entities(text, numbers=False):
        if ent.kind in _ID_KINDS:
            spans.append((ent.start, ent.end, ent.kind))

    # vence o trecho mais longo: "Roberto Carlos Paulo" (privado) > "Roberto Carlos" (público)
    spans = [
        sp
        for sp in spans
        if not any(sp[0] < ke and ks < sp[1] and ke - ks >= sp[1] - sp[0] for ks, ke in kept_spans)
    ]
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    out, pos, reps = [], 0, Counter()
    for s, e, lab in spans:
        if s < pos:
            continue
        out += [text[pos:s], f"[{lab}]"]
        reps[lab] += 1
        pos = e
    out.append(text[pos:])
    result = "".join(out)
    rfold = fold_map(result)
    residuos = sum(
        1
        for m in _WORD.finditer(result)
        if result[m.start()].isupper()
        and len(m.group()) >= MIN_SINGLE
        and rfold[m.start() : m.end()] in word_labels
    )
    return Anonymized(result, reps, sorted(set(labels.values())), residuos)


# --------------------------------------------------------------------------- seleção

_RESULT = [
    (
        "embargos",
        re.compile(
            r"(?i)embargos\s+de\s+declara[çc][ãa]o\s+opost|acolho\s+os\s+embargos|rejeito\s+os\s+embargos"
        ),
    ),
    ("parcial", re.compile(r"(?i)julgo\s+parcialmente\s+procedentes?")),
    ("improcedente", re.compile(r"(?i)julgo\s+improcedentes?")),
    ("procedente", re.compile(r"(?i)julgo\s+procedentes?")),
    ("homologacao", re.compile(r"(?i)homologo\b[^.]{0,80}acordo")),
    (
        "extinto",
        re.compile(r"(?i)julgo\s+extint|extingo\s+o\s+processo|julgo\s+o\s+processo\s+extinto"),
    ),
]


def guess_result(text: str) -> str:
    """Heurística só para diversificar a seleção; o resultado real é rotulado no gold."""
    for name, pat in _RESULT:
        if pat.search(text):
            return name
    return "outro"


def length_bucket(tokens: int) -> str:
    return [name for name, lo in LENGTH_BUCKETS if tokens >= lo][-1]


def ruling_id(processo: str) -> str:
    return f"tjsp-{processo}"


def _order(processo: str) -> str:
    return hashlib.sha256(f"{CORPUS_VERSION}:{processo}".encode()).hexdigest()


@dataclass
class Candidate:
    hit: Hit
    dominio: str
    text: str  # limpo, ainda não anonimizado
    tokens: int
    bucket: str
    resultado: str

    @property
    def merito(self) -> bool:
        return self.resultado in {"procedente", "improcedente", "parcial"}


def candidates(pool: list[Hit], dominios: dict[str, str]) -> list[Candidate]:
    seen, out = set(), []
    for h in pool:
        if h.processo in seen:
            continue
        seen.add(h.processo)
        if h.classe not in ALLOWED_CLASSES or len(h.texto) >= INLINE_CAP:
            continue
        text = clean(h.texto)
        tok = text_tokens(text)
        out.append(
            Candidate(h, dominios[h.query_id], text, tok, length_bucket(tok), guess_result(text))
        )
    return out


def select(cands: list[Candidate]) -> list[Candidate]:
    """Estratifica por faixa de tamanho; dentro da faixa, alterna domínios em ordem de hash."""
    picked: list[Candidate] = []
    per_judge: Counter = Counter()

    def take(pool: list[Candidate], n: int) -> None:
        by_dom: dict[str, list[Candidate]] = {}
        for c in sorted(pool, key=lambda c: _order(c.hit.processo)):
            by_dom.setdefault(c.dominio, []).append(c)
        doms = sorted(by_dom)
        got = 0
        while got < n and any(by_dom.values()):
            for d in doms:
                while by_dom[d]:
                    c = by_dom[d].pop(0)
                    if per_judge[c.hit.magistrado] < MAX_PER_JUDGE:
                        picked.append(c)
                        per_judge[c.hit.magistrado] += 1
                        got += 1
                        break
                if got >= n:
                    break

    for bucket, n in QUOTAS.items():
        take([c for c in cands if c.merito and c.bucket == bucket], n)
    take([c for c in cands if c.resultado == "extinto"], N_SEM_MERITO)
    return picked


# --------------------------------------------------------------------------- build


def build(llm: LLM, out_dir: Path = RULINGS_DIR, raw_dir: Path = RAW_DIR) -> dict:
    queries, _ = load_queries(QUERIES_PATH)
    dominios = {q.id: q.dominio for q in queries}
    pool = load_pool(raw_dir)
    cands = candidates(pool, dominios)
    chosen = select(cands)

    text_dir = out_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    for old in text_dir.glob("*.txt"):
        old.unlink()
    rows = []
    for i, c in enumerate(sorted(chosen, key=lambda c: c.hit.processo)):
        rid = ruling_id(c.hit.processo)
        people = llm_person_names(llm, c.text)
        private = [p.nome for p in people if p.papel not in PUBLIC_ROLES]
        h = c.hit
        keep = [p.nome for p in people if p.papel in PUBLIC_ROLES]
        keep += [h.magistrado, h.comarca, h.foro, h.vara]
        anon = anonymize(c.text, header_parties(c.text), private, keep=keep)
        (text_dir / f"{rid}.txt").write_text(anon.text + "\n", "utf-8")
        rows.append(
            {
                "ruling_id": rid,
                "split": "dev" if i % DEV_EVERY == 0 else "test",
                "tribunal": "TJSP",
                "processo": h.processo,
                "classe": h.classe,
                "assunto": h.assunto,
                "magistrado": h.magistrado,
                "comarca": h.comarca,
                "foro": h.foro,
                "vara": h.vara,
                "data_disponibilizacao": h.data_disponibilizacao,
                "dominio": c.dominio,
                "query_id": h.query_id,
                "resultado_heuristico": c.resultado,
                "faixa": c.bucket,
                "n_chars": len(anon.text),
                "n_tokens": text_tokens(anon.text),
                "source_sha256": hashlib.sha256(h.texto.encode()).hexdigest(),
                "text_sha256": hashlib.sha256(anon.text.encode()).hexdigest(),
                "anonimizacao": dict(sorted(anon.replacements.items())),
                "residuos_nome": anon.residuos,
            }
        )
    (out_dir / "rulings.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8"
    )
    meta = {
        "corpus_version": CORPUS_VERSION,
        "pool": len(pool),
        "pool_unique": len({h.processo for h in pool}),
        "candidatos": len(cands),
        "selecionadas": len(rows),
        "quotas": QUOTAS,
        "sem_merito": N_SEM_MERITO,
        "names_model": NAMES_MODEL,
        "names_prompt_version": NAMES_PROMPT_VERSION,
        "queries_sha256": hashlib.sha256(QUERIES_PATH.read_bytes()).hexdigest()[:12],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    (out_dir / "STATS.md").write_text(stats_md(rows, cands, meta), "utf-8")
    return meta


def load_rulings(root: Path = RULINGS_DIR) -> list[dict]:
    path = root / "rulings.jsonl"
    return [json.loads(ln) for ln in path.read_text("utf-8").splitlines() if ln.strip()]


def ruling_text(rid: str, root: Path = RULINGS_DIR) -> str:
    return (root / "text" / f"{rid}.txt").read_text("utf-8").rstrip("\n")


def stats_md(rows: list[dict], cands: list[Candidate], meta: dict) -> str:
    def table(key: str, title: str) -> list[str]:
        c = Counter(r[key] for r in rows)
        return [
            f"| {title} | n |",
            "|---|---|",
            *[f"| {k} | {v} |" for k, v in sorted(c.items())],
            "",
        ]

    n_dev = sum(r["split"] == "dev" for r in rows)
    toks = sorted(r["n_tokens"] for r in rows)
    pct = lambda q: toks[min(len(toks) - 1, int(q * len(toks)))]  # noqa: E731
    reps = Counter()
    for r in rows:
        reps.update(r["anonimizacao"])
    by_kind = Counter()
    for k, v in reps.items():
        by_kind[re.sub(r"_\d+$", "", k)] += v
    lines = [
        f"# Corpus Q2 ({meta['corpus_version']})",
        "",
        "Gerado por `uv run q2 build`. Não editar à mão.",
        "",
        f"- Pool coletado: {meta['pool']} resultados, {meta['pool_unique']} processos únicos.",
        f"- Candidatos (classe cível permitida, texto inline completo): {meta['candidatos']}.",
        f"- Selecionadas: {meta['selecionadas']} ({n_dev} dev, {len(rows) - n_dev} test).",
        "",
        "## Tamanho (tokens o200k)",
        "",
        "| mín | p25 | mediana | p75 | máx |",
        "|---|---|---|---|---|",
        f"| {toks[0]} | {pct(0.25)} | {pct(0.5)} | {pct(0.75)} | {toks[-1]} |",
        "",
        *table("faixa", "faixa"),
        *table("dominio", "domínio"),
        *table("resultado_heuristico", "resultado (heurística)"),
        *table("classe", "classe"),
        "## Anonimização (trocas no corpus selecionado)",
        "",
        "| rótulo | trocas |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in sorted(by_kind.items())],
        "",
        f"Palavras de nomes privados que sobraram (protegidas por regra de lugar/nome público): "
        f"{sum(r['residuos_nome'] for r in rows)} no corpus todo.",
        "",
    ]
    return "\n".join(lines)


def candidate_summary(cands: list[Candidate]) -> dict:
    return {
        "faixa": Counter(c.bucket for c in cands if c.merito),
        "resultado": Counter(c.resultado for c in cands),
        "dominio": Counter(c.dominio for c in cands),
    }
