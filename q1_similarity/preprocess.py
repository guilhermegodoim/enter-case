"""Pré-processamento para comparação: limpeza de layout, máscara de entidades e tokens.

Parágrafos não servem como unidade de comparação: quebras de página partem parágrafos e OCR
sem linhas em branco funde o documento inteiro. A unidade é o token, com offsets no texto
normalizado (o texto canônico mostrado ao revisor), então todo diff aponta para a fonte.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import cached_property

from rapidfuzz import fuzz

from common.text import find_entities, fold, normalize

# marcadores de página, tolerando erros de OCR (Págína, fIs., dc)
_FUZZY_PAGE = re.compile(
    r"^\s*(?:p\S{3,6}\s+\d+\s+d\S\s+\d+|\d{1,4}\s*/\s*\d{1,4}|f\S{1,3}\.?\s*\d{1,5})\s*$",
    re.IGNORECASE,
)
_WORD = re.compile(r"\w+")


def strip_layout(raw: str, *, min_repeats: int = 2, max_len: int = 80) -> str:
    """Remove marcadores de página e cabeçalhos corridos, e costura parágrafos partidos.

    Cabeçalho = linha curta logo após um marcador de página, repetida (fuzzy, por causa do
    OCR) em ao menos `min_repeats` páginas. Ancorar no marcador evita apagar linhas curtas
    parecidas do próprio conteúdo (pedidos, títulos).
    """
    lines = raw.replace("\r\n", "\n").split("\n")
    keep = [not _FUZZY_PAGE.match(ln) for ln in lines]
    anchored: list[int] = []
    prev_marker = False
    for i, ln in enumerate(lines):
        if not ln.strip():
            continue
        if not keep[i]:
            prev_marker = True
            continue
        if prev_marker and 3 <= len(ln.strip()) <= max_len:
            anchored.append(i)
        prev_marker = False
    groups: list[list[int]] = []
    for i in anchored:
        key = lines[i].strip()
        for g in groups:
            if fuzz.ratio(key, lines[g[0]].strip()) >= 85:
                g.append(i)
                break
        else:
            groups.append([i])
    for g in groups:
        if len(g) >= min_repeats:
            for i in g:
                keep[i] = False
    return _heal_breaks(lines, keep)


def _heal_breaks(lines: list[str], keep: list[bool]) -> str:
    """Remove as linhas descartadas e costura o parágrafo que a quebra de página partiu.

    Uma região (linhas descartadas + brancas ao redor) vira quebra de parágrafo só se a linha
    anterior termina com pontuação final; senão o parágrafo continua na página seguinte.
    """
    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        if keep[i] and lines[i].strip():
            out.append(lines[i])
            i += 1
            continue
        j, dropped = i, False
        while j < n and (not keep[j] or not lines[j].strip()):
            dropped |= not keep[j]
            j += 1
        if not dropped:
            out.extend(lines[i:j])
        elif out and j < n and not re.search(r"[.:;!?]\s*$", out[-1]):
            pass  # costura: a próxima linha continua o parágrafo
        else:
            out.append("")
        i = j
    return "\n".join(out)


@dataclass(frozen=True)
class Prepared:
    norm: str  # texto normalizado (canônico; offsets referem-se a ele)
    tokens: tuple[str, ...]  # palavras sem acento/minúsculas; entidades viram <TIPO>
    starts: tuple[int, ...]  # offset de cada token em `norm`
    ends: tuple[int, ...]

    @cached_property
    def norm_hash(self) -> str:
        return hashlib.sha256(self.norm.encode()).hexdigest()

    @cached_property
    def skeleton_hash(self) -> str:
        """Igual para cópias que só diferem em entidades, espaços, pontuação ou layout."""
        return hashlib.sha256(" ".join(self.tokens).encode()).hexdigest()

    def span(self, i: int, j: int) -> tuple[int, int]:
        """Offsets em `norm` dos tokens [i, j)."""
        return self.starts[i], self.ends[j - 1]


def tokenize(norm: str, *, mask: bool) -> tuple[list[str], list[int], list[int]]:
    entities = find_entities(norm) if mask else []
    toks: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    k = 0
    for m in _WORD.finditer(norm):
        while k < len(entities) and entities[k].end <= m.start():
            k += 1
        if k < len(entities) and entities[k].start < m.end():
            ent = entities[k]
            if starts and starts[-1] == ent.start:  # entidade já emitida: estende o token
                ends[-1] = max(ends[-1], m.end())
                continue
            toks.append(f"<{ent.kind.lower()}>")
            starts.append(ent.start)
            ends.append(max(ent.end, m.end()))
            continue
        toks.append(fold(m.group()))
        starts.append(m.start())
        ends.append(m.end())
    return toks, starts, ends


def prepare(raw: str, *, layout: bool = True, mask: bool = True) -> Prepared:
    """layout=False/mask=False reproduzem o baseline 'texto bruto' (só normalize)."""
    norm = normalize(strip_layout(raw) if layout else raw)
    toks, starts, ends = tokenize(norm, mask=mask)
    return Prepared(norm, tuple(toks), tuple(starts), tuple(ends))
