"""Deduplicação de argumentos (sobreposição de chunks e repetições do próprio modelo).

Dois argumentos são o mesmo se têm o mesmo tipo E (citações que se sobrepõem no texto OU
cosseno de "título. resumo" >= limiar). Componentes conexos viram um argumento: fica o de maior
confiança, com a união das citações (sem repetir trecho) e dos fundamentos legais.
"""

from __future__ import annotations

import numpy as np

from q2_arguments.schema import Argumento

MAX_QUOTES = 3


def _spans(a: Argumento) -> list[tuple[int, int]]:
    return [(c.char_start, c.char_end) for c in a.citacoes if c.ancorada]


def spans_overlap(a: Argumento, b: Argumento) -> bool:
    return any(s1 < e2 and s2 < e1 for s1, e1 in _spans(a) for s2, e2 in _spans(b))


def merge[A: Argumento](
    args: list[A], vecs: list[list[float]] | None, threshold: float
) -> tuple[list[A], int]:
    n = len(args)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    sims = None
    if vecs is not None and n > 1:
        m = np.asarray(vecs, dtype=float)
        m /= np.linalg.norm(m, axis=1, keepdims=True)
        sims = m @ m.T
    for i in range(n):
        for j in range(i + 1, n):
            if args[i].tipo != args[j].tipo:
                continue
            if spans_overlap(args[i], args[j]) or (sims is not None and sims[i, j] >= threshold):
                parent[find(j)] = find(i)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out: list[A] = []
    for members in sorted(groups.values(), key=min):
        best = max(members, key=lambda i: (getattr(args[i], "confianca", 0.0), -i))
        rep = args[best].model_copy(deep=True)
        if len(members) > 1:
            cits = list(rep.citacoes)
            fund = list(rep.fundamento_legal)
            chunks = list(getattr(rep, "sinais", {}).get("chunks", []))
            for i in members:
                if i == best:
                    continue
                for c in args[i].citacoes:
                    dup = any(
                        c.ancorada and k.ancorada and c.char_start < k.char_end
                        and k.char_start < c.char_end for k in cits
                    )  # fmt: skip
                    if not dup and len(cits) < MAX_QUOTES:
                        cits.append(c)
                fund += [f for f in args[i].fundamento_legal if f not in fund]
                chunks += getattr(args[i], "sinais", {}).get("chunks", [])
            rep.citacoes = cits
            rep.fundamento_legal = fund
            if hasattr(rep, "sinais"):
                rep.sinais = {**rep.sinais, "chunks": sorted(set(chunks)), "fundidos": len(members)}
        out.append(rep)
    return out, n - len(out)
