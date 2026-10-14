"""Ruído de superfície determinístico: quebra de linha, marcadores de página e erros de OCR.

Nada aqui muda o conteúdo jurídico. Um documento com ruído continua sendo cópia do original.
"""

from __future__ import annotations

import random
import textwrap

_CONFUSIONS = [
    ("rn", "m"),
    ("m", "rn"),
    ("l", "1"),
    ("I", "l"),
    ("O", "0"),
    ("e", "c"),
    ("ç", "c"),
    ("ã", "a"),
    ("é", "e"),
    ("ê", "e"),
    ("í", "i"),
    ("ó", "o"),
    ("õ", "o"),
    ("á", "a"),
    (",", "."),
]


def hard_wrap(text: str, rng: random.Random, *, hyphenate: bool) -> list[str]:
    """Quebra cada parágrafo em linhas de largura fixa, como num PDF exportado."""
    width = rng.randint(68, 92)
    lines: list[str] = []
    for para in text.split("\n\n"):
        wrapped = textwrap.wrap(para, width=width, break_long_words=False, break_on_hyphens=False)
        if hyphenate:
            _hyphenate(wrapped, rng)
        lines += wrapped
        lines.append("")
    return lines[:-1]


def _hyphenate(wrapped: list[str], rng: random.Random) -> None:
    """Às vezes parte a última palavra da linha com hífen e leva o resto para a próxima."""
    for k in range(len(wrapped) - 1):
        head, _, last = wrapped[k].rpartition(" ")
        if head and len(last) >= 8 and last.isalpha() and rng.random() < 0.25:
            cut = rng.randint(3, len(last) - 3)
            wrapped[k] = f"{head} {last[:cut]}-"
            wrapped[k + 1] = f"{last[cut:]} {wrapped[k + 1]}"


def paginate(lines: list[str], rng: random.Random, header: str | None) -> str:
    per_page = rng.randint(38, 52)
    pages = [lines[i : i + per_page] for i in range(0, len(lines), per_page)]
    total = len(pages)
    fmt = rng.choice(["Página {n} de {t}", "{n}/{t}", "fls. {f}"])
    first_fl = rng.randint(1, 40)
    out: list[str] = []
    for n, page in enumerate(pages, 1):
        if header and n > 1:
            out += [header, ""]
        out += page
        out += ["", fmt.format(n=n, t=total, f=first_fl + n - 1), ""]
    return "\n".join(out).strip()


def ocr_errors(text: str, rng: random.Random, rate: float) -> str:
    """Troca caracteres a uma taxa `rate` por posição elegível; junta/separa palavras."""
    chars = list(text)
    i, out = 0, []
    while i < len(chars):
        c = chars[i]
        r = rng.random()
        if r < rate:
            two = "".join(chars[i : i + 2])
            options = [(a, b) for a, b in _CONFUSIONS if two.startswith(a)]
            if options:
                a, b = rng.choice(options)
                out.append(b)
                i += len(a)
                continue
        if c == " " and r < rate * 0.6:  # palavras coladas
            i += 1
            continue
        out.append(c)
        if c.isalpha() and r > 1 - rate * 0.2:  # espaço espúrio
            out.append(" ")
        i += 1
    return "".join(out)


def drop_blank_lines(text: str) -> str:
    """OCR frequentemente perde as linhas em branco entre parágrafos."""
    return "\n".join(ln for ln in text.split("\n") if ln.strip())


def reformat(text: str, rng: random.Random, header: str | None) -> str:
    """Exportação para PDF e de volta: quebras de linha + paginação, sem erro de caractere."""
    return paginate(hard_wrap(text, rng, hyphenate=False), rng, header)


def scan(text: str, rng: random.Random, header: str | None, level: str) -> str:
    """Digitalização + OCR: quebras com hifenização, paginação e erros de caractere."""
    rate = {"leve": 0.002, "media": 0.006, "forte": 0.015}[level]
    paged = paginate(hard_wrap(text, rng, hyphenate=True), rng, header)
    noisy = ocr_errors(paged, rng, rate)
    return drop_blank_lines(noisy) if rng.random() < 0.5 else noisy
