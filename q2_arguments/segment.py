"""Segmentação da sentença (relatório / fundamentação / dispositivo) e chunking da fundamentação.

Tudo determinístico e por offsets do texto armazenado:
- seções por marcadores textuais típicos ("É o relatório", "Fundamento e decido", "Ante o
  exposto"); sem marcador, a seção fica vazia e o texto inteiro conta como fundamentação;
- chunks = parágrafos inteiros da fundamentação empacotados até um orçamento de tokens, com
  sobreposição de parágrafos entre chunks vizinhos (argumento na fronteira aparece inteiro em
  pelo menos um chunk). A deduplicação dessa sobreposição fica em merge.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from q2_arguments.corpus import text_tokens

_REL_END = re.compile(
    r"(?i)\b(?:é\s+o\s+(?:breve\s+)?relat[óo]rio|relat[óo]rio\s+dispensado|dispensado\s+o\s+"
    r"relat[óo]rio|é\s+a\s+s[íi]ntese|é\s+o\s+necess[áa]rio|fundamento\s+e\s+decido|"
    r"passo\s+a\s+(?:decidir|fundamentar)|decido\.|^\W*fundamenta[çc][ãa]o\W*$)",
    re.MULTILINE,
)
_DISP = re.compile(
    r"(?i)\b(?:ante\s+o\s+exposto|diante\s+do\s+exposto|pelo\s+exposto|em\s+face\s+do\s+exposto|"
    r"por\s+todo\s+o\s+exposto|do\s+exposto|posto\s+isso|isto\s+posto|ante\s+o\s+quadro|"
    r"diante\s+do\s+quadro|^\W*dispositivo\W*$)",
    re.MULTILINE,
)
_JULGO = re.compile(
    r"(?i)\bjulgo\s+(?:parcialmente\s+)?(?:im)?procedentes?|\bjulgo\s+extint|\bindefiro\s+a\s+"
    r"peti[çc][ãa]o|\bhomologo\b"
)
REL_MAX_FRAC = 0.6  # fim do relatório só é aceito na primeira parte do texto
DISP_MIN_FRAC = 0.4  # início do dispositivo só é aceito depois disto


@dataclass(frozen=True)
class Sections:
    n: int
    fund_start: int  # relatório = [0, fund_start)
    disp_start: int  # fundamentação = [fund_start, disp_start); dispositivo = [disp_start, n)
    found: tuple[bool, bool]  # (marcador de fim do relatório, marcador de dispositivo)

    def name_at(self, pos: int) -> str:
        if pos < self.fund_start:
            return "relatorio"
        return "fundamentacao" if pos < self.disp_start else "dispositivo"


def _para_end(text: str, pos: int) -> int:
    i = text.find("\n\n", pos)
    return len(text) if i < 0 else i + 2


def _para_start(text: str, pos: int) -> int:
    i = text.rfind("\n\n", 0, pos)
    return 0 if i < 0 else i + 2


def sections(text: str) -> Sections:
    n = len(text)
    rel = [m for m in _REL_END.finditer(text) if m.start() < REL_MAX_FRAC * n]
    fund_start = _para_end(text, rel[0].end()) if rel else 0
    floor = max(fund_start, int(DISP_MIN_FRAC * n))
    disp = [m for m in _DISP.finditer(text) if m.start() >= floor]
    if not disp:  # sem "ante o exposto": parágrafo do último "julgo procedente/extinto"
        disp = [m for m in _JULGO.finditer(text) if m.start() >= floor]
    disp_start = _para_start(text, disp[-1].start()) if disp else n
    disp_start = max(disp_start, fund_start)
    return Sections(n, fund_start, disp_start, (bool(rel), bool(disp)))


def section_of(sec: Sections, spans: list[tuple[int, int]]) -> str | None:
    """Seção com mais caracteres citados; None sem spans."""
    if not spans:
        return None
    acc: dict[str, int] = {}
    for s, e in spans:
        for name, lo, hi in (
            ("relatorio", 0, sec.fund_start),
            ("fundamentacao", sec.fund_start, sec.disp_start),
            ("dispositivo", sec.disp_start, sec.n),
        ):
            acc[name] = acc.get(name, 0) + max(0, min(e, hi) - max(s, lo))
    return max(acc, key=acc.get)


@dataclass(frozen=True)
class Chunk:
    idx: int
    start: int
    end: int


def paragraph_spans(text: str, lo: int = 0, hi: int | None = None) -> list[tuple[int, int]]:
    hi = len(text) if hi is None else hi
    out, pos = [], lo
    while pos < hi:
        nxt = text.find("\n\n", pos, hi)
        end = hi if nxt < 0 else nxt
        if text[pos:end].strip():
            out.append((pos, end))
        pos = end + 2 if nxt >= 0 else hi
    return out


def chunk_range(text: str, lo: int, hi: int, budget: int, overlap: int) -> list[Chunk]:
    """Parágrafos de text[lo:hi] em chunks de ≤ budget tokens (parágrafo maior fica sozinho).

    Cada chunk novo recomeça pelos últimos parágrafos do anterior até somar `overlap` tokens.
    """
    paras = paragraph_spans(text, lo, hi)
    if not paras:
        return []
    toks = [text_tokens(text[s:e]) for s, e in paras]
    chunks: list[Chunk] = []
    i = 0
    while i < len(paras):
        j, used = i, 0
        while j < len(paras) and (j == i or used + toks[j] <= budget):
            used += toks[j]
            j += 1
        chunks.append(Chunk(len(chunks), paras[i][0], paras[j - 1][1]))
        if j >= len(paras):
            break
        k, back = j, 0  # recua parágrafos para a sobreposição, sem voltar ao início do chunk
        while k - 1 > i and back + toks[k - 1] <= overlap:
            k -= 1
            back += toks[k]
        i = k
    return chunks


def chunk_fundamentacao(text: str, sec: Sections, budget: int, overlap: int) -> list[Chunk]:
    return chunk_range(text, sec.fund_start, sec.disp_start, budget, overlap)


def head_tokens(text: str, max_tokens: int) -> str:
    """Prefixo com no máximo max_tokens (corta em parágrafo quando possível)."""
    out, used = [], 0
    for s, e in paragraph_spans(text):
        t = text_tokens(text[s:e])
        if used + t > max_tokens:
            break
        out.append(text[s:e])
        used += t
    return "\n\n".join(out) if out else text[: max_tokens * 4]


def tail_tokens(text: str, max_tokens: int) -> str:
    out, used = [], 0
    for s, e in reversed(paragraph_spans(text)):
        t = text_tokens(text[s:e])
        if used + t > max_tokens:
            break
        out.insert(0, text[s:e])
        used += t
    return "\n\n".join(out) if out else text[-max_tokens * 4 :]
