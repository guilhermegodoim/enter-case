"""Normalização de texto jurídico pt-BR e mascaramento de entidades.

O texto normalizado é o texto canônico armazenado: offsets de entidades (e, na Q2,
de citações) referem-se a ele, nunca ao texto bruto/OCR.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass

# --------------------------------------------------------------------------- normalização

_PAGE_MARKER = re.compile(
    r"^[ \t]*(?:p[áa]gina\s+\d+(?:\s+de\s+\d+)?|fls?\.\s*\d+|\d+\s*/\s*\d+)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_QUOTES = str.maketrans({"“": '"', "”": '"', "„": '"', "‘": "'", "’": "'", "–": "-", "—": "-"})


def normalize(text: str) -> str:
    """Limpa artefatos de OCR/PDF e devolve parágrafos separados por linha em branco."""
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text).replace("\t", " ").translate(_QUOTES)
    text = _PAGE_MARKER.sub("", text)
    text = re.sub(r"(\w)-\n[ \t]*([a-zà-ÿ])", r"\1\2", text)  # hifenização de quebra de linha
    if "\n\n" not in text:  # OCR sem linhas em branco: pontuação + maiúscula = novo parágrafo
        text = re.sub(r"([.:;!?])[ \t]*\n(?=[ \t]*[A-ZÀ-Ý0-9])", r"\1\n\n", text)
    paras = (re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n[ \t]*\n", text))
    return "\n\n".join(p for p in paras if p)


def paragraphs(text: str) -> list[str]:
    return [p for p in text.split("\n\n") if p.strip()]


def fold(text: str) -> str:
    """Minúsculas sem acentos, para shingling/hash."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- entidades


@dataclass(frozen=True)
class Entity:
    kind: str
    value: str
    start: int
    end: int


@dataclass(frozen=True)
class Masked:
    text: str
    entities: list[Entity]


_MONTHS = (
    "janeiro|fevereiro|março|marco|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro"
)
_UP = "A-ZÀ-ÖØ-Ý"
_LO = "a-zà-öø-ÿ"
_WORD = rf"(?:[{_UP}][{_UP}']+|[{_UP}][{_LO}']+)"
_CONN = r"(?:d[aeo]s?|e|D[AEO]S?|E)"
_COMPANY = r"(?:\s+(?:S\.?/?A\.?|LTDA\.?|Ltda\.?|EIRELI|ME|EPP))?"
_NAME = rf"{_WORD}(?:\s+(?:{_CONN}\s+)?{_WORD})+{_COMPANY}"
_QUALIF = (
    r"(?=\s*,\s*(?i:brasileir|portador|inscrit|pessoa\s+jur|solteir|casad|divorciad|"
    r"vi[uú]v|nacionalidade|empresa|sociedade|com\s+sede|residente))"
)
_ROLE = (
    r"(?i:\b(?:autora?|r[ée]u|r[ée]|requerente|requerid[oa]|reclamante|reclamad[oa]|"
    r"exequente|executad[oa]|apelante|apelad[oa])\s*:\s*)"
)

# Ordem = prioridade de desempate. _KEEP protege citações legais (art. 186, Lei 8.078/90)
# do mascaramento numérico: são substância jurídica, não entidade.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "_KEEP",
        re.compile(
            r"(?i:\b(?:arts?\.|artigos?|leis?|s[úu]mulas?|decreto|tema|inciso|resolu[çc][ãa]o|"
            r"par[áa]grafo|lc)\s*(?:n[º°o.]*\s*)?|§+\s*)\d[\d.,/ºª°-]*(?:\s*(?:e|,)\s*\d[\d.,/ºª°-]*)*"
        ),
    ),
    ("PROCESSO", re.compile(r"\b\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}\b")),
    ("CNPJ", re.compile(r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b")),
    ("CPF", re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b")),
    ("RG", re.compile(r"(?i:\bRG\b\D{0,15})(?P<v>\d{1,2}\.?\d{3}\.?\d{3}-?[\dxX])")),
    ("OAB", re.compile(r"(?i:\bOAB)\s*/?\s*[A-Z]{2}\s*(?:n[º°o.]*\s*)?\d{1,3}(?:\.?\d{3})*")),
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("VALOR", re.compile(r"R\$\s*\d+(?:\.\d{3})*(?:,\d{1,2})?")),
    ("VALOR", re.compile(r"\((?=[^()]*\breais\b)[^()]{3,120}\)")),
    ("DATA", re.compile(rf"(?i:\b\d{{1,2}}º?\s+de\s+(?:{_MONTHS})\s+de\s+\d{{4}}\b)")),
    ("DATA", re.compile(r"\b\d{1,2}[/.]\d{1,2}[/.]\d{2,4}\b")),
    ("CEP", re.compile(r"\b\d{2}\.?\d{3}-\d{3}\b")),
    ("TELEFONE", re.compile(r"\(\d{2}\)\s*9?\d{4}-?\d{4}")),
    ("VOO", re.compile(r"(?i:\bvoo\s+(?:n[º°o.]*\s*)?)(?P<v>[A-Z0-9]{2}\s?\d{3,4})\b")),
    ("NOME", re.compile(rf"\b{_NAME}{_QUALIF}")),
    ("NOME", re.compile(rf"(?i:\bem\s+face\s+d[aoe]s?\s+)(?P<v>{_NAME})")),
    ("NOME", re.compile(rf"{_ROLE}(?P<v>{_NAME})")),
]
_NUM = re.compile(r"\b\d+(?:[.,]\d+)*\b")
_NAME_STOP = {
    "autor",
    "autora",
    "requerente",
    "reclamante",
    "o",
    "a",
    "sr",
    "sra",
    "senhor",
    "senhora",
    "eu",
    "dr",
    "dra",
    "parte",
    "ré",
    "réu",
}


def _trim_name(value: str) -> str:
    words = value.split()
    while words and words[0].lower().strip(".") in _NAME_STOP:
        words.pop(0)
    return " ".join(words)


def find_entities(text: str, *, numbers: bool = True) -> list[Entity]:
    spans: list[tuple[int, int, int, str]] = []  # (start, end, prioridade, kind)
    names: set[str] = set()
    for prio, (kind, pat) in enumerate(_PATTERNS):
        for m in pat.finditer(text):
            g = "v" if "v" in pat.groupindex else 0
            start, end = m.span(g)
            if kind == "NOME":
                trimmed = _trim_name(m.group(g))
                if len(trimmed) < 3:
                    continue
                start = end - len(trimmed)
                names.add(trimmed)
            spans.append((start, end, prio, kind))
    # propaga nomes detectados por âncora para todas as ocorrências no documento
    for name in names:
        for m in re.finditer(rf"\b{re.escape(name)}\b", text, re.IGNORECASE):
            spans.append((m.start(), m.end(), len(_PATTERNS), "NOME"))
    if numbers:
        for m in _NUM.finditer(text):
            spans.append((m.start(), m.end(), len(_PATTERNS) + 1, "NUM"))

    # leftmost-longest, desempate por prioridade; sem sobreposição
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0]), s[2]))
    kept: list[Entity] = []
    cursor = -1
    for start, end, _, kind in spans:
        if start >= cursor:
            kept.append(Entity(kind, text[start:end], start, end))
            cursor = end
    return [e for e in kept if e.kind != "_KEEP"]


def mask_entities(text: str, *, numbers: bool = True) -> Masked:
    """Troca entidades variáveis (nomes, valores, datas, documentos) por <TIPO>.

    Offsets das entidades referem-se ao `text` de entrada, para rastreabilidade.
    """
    entities = find_entities(text, numbers=numbers)
    out, pos = [], 0
    for e in entities:
        out.append(text[pos : e.start])
        out.append(f"<{e.kind}>")
        pos = e.end
    out.append(text[pos:])
    return Masked("".join(out), entities)


def skeleton(raw: str) -> str:
    """Forma canônica para detectar cópias: normaliza, mascara entidades, minúsculas sem acento."""
    return fold(mask_entities(normalize(raw)).text)


def fold_keep_offsets(text: str) -> str:
    """Como `fold`, mas com 1 caractere de saída por caractere de entrada (offsets preservados)."""
    out = []
    for c in text:
        f = "".join(
            x for x in unicodedata.normalize("NFD", c.lower()) if not unicodedata.combining(x)
        )
        out.append(f[0] if len(f) == 1 else c.lower() if len(c.lower()) == 1 else c)
    return "".join(out)


_NULL_ESCAPE = re.compile(r"\x00([0-9a-fA-F]{2})")


def fix_null_escapes(text: str) -> str:
    """Conserta saída de LLM com acentos como NUL + 2 dígitos hex ("rela\\x00e7\\x00e3o")."""
    return _NULL_ESCAPE.sub(lambda m: chr(int(m.group(1), 16)), text).replace("\x00", "")


def fix_strings(obj):
    """Aplica fix_null_escapes em todas as strings de dict/list aninhados."""
    if isinstance(obj, str):
        return fix_null_escapes(obj)
    if isinstance(obj, list):
        return [fix_strings(x) for x in obj]
    if isinstance(obj, dict):
        return {k: fix_strings(v) for k, v in obj.items()}
    return obj
