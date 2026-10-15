"""Sinais de similaridade entre dois documentos já preparados (ver preprocess.py).

Do mais barato ao mais caro:
- hashes exatos (texto normalizado; esqueleto mascarado);
- shingles de 5 tokens: Jaccard, contenção e MinHash, com filtro opcional de boilerplate;
- alinhamento de tokens (Indel/LCS) e regiões de diferença classificadas como ruído
  (OCR, troca de dados) ou mudança de conteúdo;
- embeddings por trecho (opcional, chamada de API).
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np
from datasketch import MinHash
from rapidfuzz import fuzz
from rapidfuzz.distance import Indel

from q1_similarity.preprocess import Prepared

SHINGLE = 5
NUM_PERM = 128
GAP = 2  # trechos iguais menores que isso não separam regiões de diferença
NOISE_CHAR_SIM = 75  # similaridade de caracteres (0-100) acima disso = ruído de OCR
MOVED_SHARE = 0.8  # trecho "novo" cujos trigramas existem em outro ponto do outro documento


# --------------------------------------------------------------------------- shingles


def _h64(s: str) -> int:
    return struct.unpack("<q", hashlib.blake2b(s.encode(), digest_size=8).digest())[0]


def shingles(tokens: Iterable[str], k: int = SHINGLE) -> set[int]:
    toks = list(tokens)
    if len(toks) < k:
        return {_h64(" ".join(toks))} if toks else set()
    return {_h64(" ".join(toks[i : i + k])) for i in range(len(toks) - k + 1)}


def minhash(sh: set[int], num_perm: int = NUM_PERM) -> MinHash:
    m = MinHash(num_perm=num_perm, seed=1)
    m.update_batch([struct.pack("<q", x) for x in sh])
    return m


def jaccard(a: set[int], b: set[int]) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


def containment(a: set[int], b: set[int]) -> float:
    """Fração de b contida em a."""
    return len(a & b) / len(b) if b else 1.0


# --------------------------------------------------------------------------- alinhamento


@dataclass(frozen=True)
class Region:
    """Trecho onde os documentos diferem; offsets de tokens e de caracteres nos dois lados."""

    a_tok: tuple[int, int]
    b_tok: tuple[int, int]
    a_chr: tuple[int, int] | None
    b_chr: tuple[int, int] | None
    kind: str  # ruido | dado | movido | conteudo
    char_sim: float

    @property
    def size(self) -> int:
        return max(self.a_tok[1] - self.a_tok[0], self.b_tok[1] - self.b_tok[0])


def _ids(a: Prepared, b: Prepared) -> tuple[list[int], list[int]]:
    vocab: dict[str, int] = {}
    ia = [vocab.setdefault(t, len(vocab)) for t in a.tokens]
    ib = [vocab.setdefault(t, len(vocab)) for t in b.tokens]
    return ia, ib


def _chr(p: Prepared, i: int, j: int) -> tuple[int, int] | None:
    return p.span(i, j) if j > i else None


def _is_data(p: Prepared, i: int, j: int) -> bool:
    """Região dominada por entidades: máscaras, números ou nomes próprios.

    Nome próprio = inicial maiúscula seguida de minúsculas. Palavras TODAS EM CAIXA ALTA não
    contam: ementas e títulos são conteúdo jurídico, não dado variável.
    """
    if j <= i:
        return True
    hits = 0
    for k in range(i, j):
        tok = p.tokens[k]
        word = p.norm[p.starts[k] : p.ends[k]]
        titlecase = word[:1].isupper() and not word.isupper()
        if tok.startswith("<") or tok.isdigit() or titlecase:
            hits += 1
    return hits / (j - i) >= 0.5


def _grams(tokens: tuple[str, ...] | list[str], n: int = 3) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _moved(frag: tuple[str, ...], other: set[tuple[str, ...]]) -> bool:
    """Trecho cujo conteúdo existe em outro ponto do outro documento (item deslocado/repetido).

    Lado vazio conta como movido. Fragmentos curtos demais para trigramas não contam.
    """
    if not frag:
        return True
    grams = _grams(frag)
    if not grams:
        return False
    return len(grams & other) / len(grams) >= MOVED_SHARE


def align(a: Prepared, b: Prepared) -> tuple[int, list[Region]]:
    """Alinha tokens; devolve (tokens iguais, regiões de diferença classificadas)."""
    ia, ib = _ids(a, b)
    ops = [o for o in Indel.opcodes(ia, ib)]
    matched = sum(o.src_end - o.src_start for o in ops if o.tag == "equal")
    # agrupa operações de diferença separadas por menos de GAP tokens iguais
    groups: list[list] = []
    cur: list = []
    for o in ops:
        if o.tag == "equal" and o.src_end - o.src_start >= GAP:
            if cur:
                groups.append(cur)
                cur = []
        elif o.tag != "equal" or cur:
            cur.append(o)
    if cur:
        groups.append(cur)
    regions: list[Region] = []
    grams_a, grams_b = _grams(a.tokens), _grams(b.tokens)
    for g in groups:
        while g and g[-1].tag == "equal":
            g.pop()
        if not g:
            continue
        a0, a1 = g[0].src_start, g[-1].src_end
        b0, b1 = g[0].dest_start, g[-1].dest_end
        sa, sb = " ".join(a.tokens[a0:a1]), " ".join(b.tokens[b0:b1])
        sim = fuzz.ratio(sa, sb) if sa and sb else 0.0
        if sim >= NOISE_CHAR_SIM:
            kind = "ruido"
        elif _is_data(a, a0, a1) and _is_data(b, b0, b1):
            kind = "dado"
        elif _moved(a.tokens[a0:a1], grams_b) and _moved(b.tokens[b0:b1], grams_a):
            kind = "movido"  # alinhamento cruzou itens repetidos/deslocados; conteúdo existe
        else:
            kind = "conteudo"
        regions.append(Region((a0, a1), (b0, b1), _chr(a, a0, a1), _chr(b, b0, b1), kind, sim))
    return matched, regions


# --------------------------------------------------------------------------- embeddings


def chunk_texts(p: Prepared, words: int = 300) -> list[str]:
    """Trechos de ~`words` palavras do texto normalizado (sem máscara), para embedding."""
    out = []
    for i in range(0, len(p.tokens), words):
        s, e = p.span(i, min(i + words, len(p.tokens)))
        out.append(p.norm[s:e])
    return out or [p.norm or " "]


def _unit(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


def embedding_scores(ea: np.ndarray, eb: np.ndarray, hit: float = 0.95) -> dict[str, float]:
    """ea, eb: matrizes (trechos × dim). Cosseno do documento médio + cobertura de trechos."""
    ea, eb = _unit(ea), _unit(eb)
    doc = float(_unit(ea.mean(0)) @ _unit(eb.mean(0)))
    sims = eb @ ea.T
    return {"emb_cos": doc, "emb_chunk_cov": float((sims.max(1) >= hit).mean())}


# --------------------------------------------------------------------------- vetor de features


@dataclass(frozen=True)
class PairFeatures:
    exact: bool  # texto normalizado idêntico
    skeleton: bool  # idêntico após máscara de entidades/layout
    jaccard: float  # shingles completos
    contain_b: float  # fração do novo contida no antigo
    contain_a: float
    minhash: float  # estimativa MinHash do jaccard (pré-filtro barato)
    jaccard_f: float  # shingles sem boilerplate
    contain_b_f: float
    contain_a_f: float
    minhash_f: float  # estimativa MinHash do jaccard_f
    align_sim: float  # 2·iguais / (|a|+|b|)
    align_cov: float  # iguais / max(|a|, |b|)
    content_ratio: float  # tokens em regiões de conteúdo / max(|a|, |b|)
    data_ratio: float
    noise_ratio: float
    moved_ratio: float
    n_content_regions: int
    max_content_region: int  # maior trecho de conteúdo divergente (tokens)
    len_ratio: float
    emb_cos: float | None = None
    emb_chunk_cov: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class DocSig:
    """Assinatura de um documento, calculada uma vez na ingestão e reutilizada por par."""

    prep: Prepared
    sh: set[int]
    sh_f: set[int]  # sem shingles de boilerplate (ablação)
    mh: MinHash
    mh_f: MinHash
    emb: np.ndarray | None = None


def signature(p: Prepared, boilerplate: frozenset[int] = frozenset()) -> DocSig:
    sh = shingles(p.tokens)
    sh_f = sh - boilerplate
    mh = minhash(sh)
    return DocSig(p, sh, sh_f, mh, minhash(sh_f) if boilerplate else mh)


def pair_features(a: DocSig, b: DocSig, *, regions: bool = True) -> tuple[PairFeatures, list]:
    pa, pb = a.prep, b.prep
    la, lb = len(pa.tokens), len(pb.tokens)
    big = max(la, lb, 1)
    if regions:
        matched, regs = align(pa, pb)
    else:
        matched, regs = 0, []
    by = {
        k: sum(r.size for r in regs if r.kind == k) for k in ("conteudo", "dado", "ruido", "movido")
    }
    emb = embedding_scores(a.emb, b.emb) if a.emb is not None and b.emb is not None else {}
    f = PairFeatures(
        exact=pa.norm_hash == pb.norm_hash,
        skeleton=pa.skeleton_hash == pb.skeleton_hash,
        jaccard=jaccard(a.sh, b.sh),
        contain_b=containment(a.sh, b.sh),
        contain_a=containment(b.sh, a.sh),
        minhash=float(a.mh.jaccard(b.mh)) if a.sh and b.sh else 0.0,
        jaccard_f=jaccard(a.sh_f, b.sh_f),
        contain_b_f=containment(a.sh_f, b.sh_f),
        contain_a_f=containment(b.sh_f, a.sh_f),
        minhash_f=float(a.mh_f.jaccard(b.mh_f)) if a.sh_f and b.sh_f else 0.0,
        align_sim=2 * matched / max(la + lb, 1),
        align_cov=matched / big,
        content_ratio=by["conteudo"] / big,
        data_ratio=by["dado"] / big,
        noise_ratio=by["ruido"] / big,
        moved_ratio=by["movido"] / big,
        n_content_regions=sum(r.kind == "conteudo" for r in regs),
        max_content_region=max((r.size for r in regs if r.kind == "conteudo"), default=0),
        len_ratio=min(la, lb) / big,
        **emb,
    )
    return f, regs


# --------------------------------------------------------------------------- boilerplate


def boilerplate_shingles(docs_by_case: dict[str, list[set[int]]], max_df: float) -> frozenset[int]:
    """Shingles presentes em mais de `max_df` dos casos de referência (texto-padrão).

    Frequência por caso, não por documento: cópias dentro de um caso não inflam a contagem.
    """
    df: dict[int, int] = {}
    for sets in docs_by_case.values():
        for x in set().union(*sets):
            df[x] = df.get(x, 0) + 1
    limit = max_df * len(docs_by_case)
    return frozenset(x for x, n in df.items() if n > limit)
