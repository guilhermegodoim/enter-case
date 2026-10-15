"""Ancoragem de citações no texto-fonte.

O LLM devolve só o texto da citação; posição (char_start, char_end) é sempre calculada aqui.
Busca literal primeiro e, se falhar, similaridade (rapidfuzz) sobre texto sem acento, sem caixa
e com espaços colapsados, mapeando de volta para offsets do texto original.
"""

from __future__ import annotations

from rapidfuzz import fuzz

from common.text import fold_keep_offsets, normalize
from q2_arguments.schema import Citacao

MIN_SCORE = 90.0


class Source:
    """Texto-fonte pré-processado para várias buscas."""

    def __init__(self, text: str):
        self.text = text
        self.squashed, self.index = squash(text)


def squash(text: str) -> tuple[str, list[int]]:
    """Texto dobrado com espaços colapsados + mapa posição -> offset no original."""
    chars: list[str] = []
    index: list[int] = []
    for i, c in enumerate(fold_keep_offsets(text)):
        if c.isspace():
            if not chars or chars[-1] == " ":
                continue
            c = " "
        chars.append(c)
        index.append(i)
    return "".join(chars), index


def anchor(quote: str, source: Source | str) -> Citacao:
    src = source if isinstance(source, Source) else Source(source)
    q = normalize(quote).replace("\n\n", " ")
    if not q:
        return Citacao(texto=quote)
    i = src.text.find(q)
    if i >= 0:
        return Citacao(texto=quote, char_start=i, char_end=i + len(q), score=100.0)
    sq, _ = squash(q)
    sq = sq.strip()
    j = src.squashed.find(sq)
    if j >= 0:
        start, end, score = j, j + len(sq), 100.0
    else:
        al = fuzz.partial_ratio_alignment(sq, src.squashed, score_cutoff=MIN_SCORE)
        if al is None:
            return Citacao(texto=quote)
        start, end, score = al.dest_start, al.dest_end, round(al.score, 1)
    if end <= start:
        return Citacao(texto=quote)
    return Citacao(
        texto=quote,
        char_start=src.index[start],
        char_end=src.index[end - 1] + 1,
        score=score,
    )
