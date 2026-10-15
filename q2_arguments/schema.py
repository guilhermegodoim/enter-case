"""Schema dos argumentos da Q2.

Compartilhado por gold (fase 3), extração (fase 4) e revisão (fase 5).

Mudança incompatível de campo/enum => novo SCHEMA_VERSION (+ função de upcast na fase 5).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "q2-args-v1"

TipoArgumento = Literal["preliminar", "prejudicial", "merito", "quantum", "acessorio"]
SuscitadoPor = Literal["autor", "reu", "juizo"]
Decisao = Literal["acolhido", "rejeitado", "parcial"]
Secao = Literal["relatorio", "fundamentacao", "dispositivo"]
Resultado = Literal[
    "procedente",
    "parcialmente_procedente",
    "improcedente",
    "extinto_sem_merito",
    "homologacao",
    "outro",
]


class Citacao(BaseModel):
    """Trecho literal da sentença. Offsets no texto armazenado, calculados em código."""

    texto: str
    char_start: int | None = None
    char_end: int | None = None
    score: float = 0.0  # 100 = literal; >= ground.MIN_SCORE = ancorada por similaridade

    @property
    def ancorada(self) -> bool:
        return self.char_start is not None


class Argumento(BaseModel):
    id: str
    titulo: str = Field(description="A tese, em uma linha (ex.: 'Ilegitimidade passiva da ré')")
    resumo: str = Field(description="1 a 3 frases: o que o juiz decidiu sobre a tese e por quê")
    tipo: TipoArgumento
    tema: str = Field(description="Rótulo curto do assunto (ex.: 'dano moral', 'fortuito externo')")
    suscitado_por: SuscitadoPor
    decisao: Decisao
    fundamento_legal: list[str] = Field(default_factory=list)
    citacoes: list[Citacao]
    secao: Secao = "fundamentacao"
