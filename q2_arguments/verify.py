"""Verificador (modelo barato): cada argumento é sustentado pelos trechos citados?

O verificador vê só o argumento e os trechos-fonte ancorados (lidos do texto armazenado, não
do que o extrator escreveu), com contexto. Uma chamada por sentença.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from common.llm import LLM, CallRecord
from q2_arguments.schema import Argumento

PROMPT_VERSION = "q2-verify-v1"
CONTEXT_CHARS = 350


class Checagem(BaseModel):
    id: str
    suporte: Literal["sustentado", "parcial", "nao_sustentado"]
    decisao_confere: bool
    motivo: str


class Checagens(BaseModel):
    itens: list[Checagem]


SYSTEM = """Você audita argumentos extraídos de uma sentença judicial. Para cada argumento, \
você recebe título, resumo, decisão atribuída ao juiz e os trechos da sentença usados como \
prova (com contexto; o trecho citado está entre ⟦ ⟧).

Responda, com base APENAS nos trechos mostrados:
- suporte: "sustentado" se os trechos mostram o juiz decidindo a questão do título como \
resumido; "parcial" se tratam da questão mas o resumo extrapola ou mistura outra questão; \
"nao_sustentado" se os trechos não tratam dessa questão ou são só alegação de parte/ementa sem \
decisão do juiz.
- decisao_confere: a decisão atribuída (acolhido/rejeitado/parcial a tese do título) confere \
com os trechos?
- motivo: uma frase curta."""


def evidence(a: Argumento, text: str) -> str:
    out = []
    for c in a.citacoes:
        if not c.ancorada:
            continue
        lo, hi = max(0, c.char_start - CONTEXT_CHARS), min(len(text), c.char_end + CONTEXT_CHARS)
        out.append(
            " ".join(
                (text[lo : c.char_start] + "⟦" + text[c.char_start : c.char_end] + "⟧"
                 + text[c.char_end : hi]).split()
            )
        )  # fmt: skip
    return "\n".join(f"  - …{e}…" for e in out)


def verify(
    llm: LLM, model: str, args: list[Argumento], text: str
) -> tuple[dict[str, Checagem], CallRecord]:
    blocks = [
        f"[{a.id}] título: {a.titulo}\n  resumo: {a.resumo}\n  decisão atribuída: {a.decisao}"
        f" (suscitado por {a.suscitado_por})\n  trechos:\n{evidence(a, text)}"
        for a in args
    ]
    out, rec = llm.structured(
        Checagens,
        system=SYSTEM,
        user="\n\n".join(blocks),
        model=model,
        temperature=None if model.startswith(("gpt-5", "o")) else 0.0,
        prompt_version=PROMPT_VERSION,
    )
    return {c.id: c for c in out.itens}, rec
