"""Juiz LLM para a zona cinzenta cópia × versão.

O modelo nunca vê o documento inteiro: recebe só os trechos divergentes do alinhamento, com
um pouco de contexto. Custo e latência ficam fixos e pequenos, e documento longo deixa de ser
problema de janela de contexto.

Ancoragem: o modelo classifica cada trecho por id. A decisão final é derivada em código
(versão sse algum trecho for 'conteudo'), então todo veredito aponta para trechos da fonte.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from common.llm import LLM, CallRecord
from q1_similarity.features import Region
from q1_similarity.preprocess import Prepared

PROMPT_VERSION = "q1-judge-v1"
DEFAULT_MODEL = "gpt-4.1-mini"
CONTEXT_TOKENS = 8
MAX_REGIONS = 25

Tipo = Literal["ruido_ocr", "dado_variavel", "formatacao", "conteudo"]


class TrechoVeredito(BaseModel):
    id: int
    tipo: Tipo
    motivo: str = Field(description="Uma frase curta")


class Veredito(BaseModel):
    trechos: list[TrechoVeredito]
    decisao: Literal["copia", "versao"]


SYSTEM = """Você compara duas peças jurídicas do mesmo processo para decidir se a segunda é \
CÓPIA da primeira ou uma VERSÃO modificada. Você recebe só os trechos em que elas diferem, \
com contexto; o resto do texto é igual.

Classifique cada trecho divergente:
- ruido_ocr: erro de digitalização (letras trocadas, palavras coladas ou partidas, hífen).
- dado_variavel: nome, valor, data, endereço, qualificação, número de documento/voo/contrato, \
foro. Trocar esses dados mantém a peça uma cópia do mesmo modelo.
- formatacao: numeração, marcadores, ordem ou repetição de itens sem texto novo.
- conteudo: fato, argumento, fundamento, pedido ou jurisprudência acrescentado, removido ou \
reescrito com outras palavras.

decisao = "versao" se algum trecho for conteudo; senão "copia". Baseie-se apenas nos \
trechos mostrados."""


def _ctx(p: Prepared, i: int, j: int, k: int = CONTEXT_TOKENS) -> str:
    """Trecho [i, j) de tokens com k tokens de contexto de cada lado, marcado com ⟦ ⟧."""
    n = len(p.tokens)
    lo, hi = max(0, i - k), min(n, j + k)
    mid_s = p.starts[i] if i < n else len(p.norm)
    mid_e = p.ends[j - 1] if j > i else mid_s
    before = p.norm[p.starts[lo] : mid_s] if i > lo else ""
    after = p.norm[mid_e : p.ends[hi - 1]] if hi > j else ""
    mid = p.norm[mid_s:mid_e]
    return f"…{before}⟦{mid or '∅'}⟧{after}…"


def candidate_regions(regions: list[Region]) -> list[Region]:
    """Trechos enviados ao juiz: tudo que não é ruído óbvio, maiores primeiro."""
    regs = [r for r in regions if r.kind != "ruido"]
    return sorted(regs, key=lambda r: -r.size)[:MAX_REGIONS]


def render_prompt(a: Prepared, b: Prepared, regions: list[Region]) -> str:
    parts = []
    for n, r in enumerate(regions):
        parts.append(f"[{n}]\nANTERIOR: {_ctx(a, *r.a_tok)}\nNOVO:     {_ctx(b, *r.b_tok)}")
    return "Trechos divergentes (⟦ ⟧ marca a diferença, ∅ = ausente):\n\n" + "\n\n".join(parts)


@dataclass(frozen=True)
class Judgement:
    label: str  # copia | versao
    content_regions: list[int]  # índices em `regions` classificados como conteúdo
    verdict: Veredito | None
    record: CallRecord | None


def judge(
    llm: LLM, a: Prepared, b: Prepared, regions: list[Region], model: str = DEFAULT_MODEL
) -> Judgement:
    regs = candidate_regions(regions)
    if not regs:
        return Judgement("copia", [], None, None)
    out, rec = llm.structured(
        Veredito,
        system=SYSTEM,
        user=render_prompt(a, b, regs),
        model=model,
        temperature=0.0,
        prompt_version=PROMPT_VERSION,
    )
    valid = {t.id for t in out.trechos if 0 <= t.id < len(regs)}
    content = sorted(t.id for t in out.trechos if t.tipo == "conteudo" and t.id in valid)
    # decisão derivada dos trechos (ancorada), não do campo livre do modelo
    label = "versao" if content else "copia"
    idx = [regions.index(regs[i]) for i in content]
    return Judgement(label, idx, out, rec)
