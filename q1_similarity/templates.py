"""Schemas dos templates sintéticos (gerados uma vez por LLM e versionados em data/).

Um template é texto com placeholders {NOME_EM_MAIUSCULAS}. Toda variação depois disso é
determinística (render + mutadores), então rótulos de cópia/versão são exatos.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from common.config import ROOT

TEMPLATES_DIR = ROOT / "data" / "synthetic" / "templates"
PLACEHOLDER = re.compile(r"\{([A-Z][A-Z0-9_]*)\}")


class Secao(BaseModel):
    titulo: str = Field(description="Título da seção, sem numeração")
    paragrafos: list[str]


class Boilerplate(BaseModel):
    """Trechos-padrão de um escritório numa área: repetidos em todas as teses."""

    enderecamento: str
    qualificacao: str
    gratuidade: Secao
    aplicabilidade: list[Secao]
    pedidos_gerais: list[str]
    fecho: list[str]


class Especifico(BaseModel):
    """Conteúdo próprio de uma tese escrita por um escritório."""

    fatos: list[str]
    direito: list[Secao]
    pedidos: list[str]


class ParafraseItem(BaseModel):
    id: str
    texto: str


class Extra(BaseModel):
    secao_alvo: Literal["fatos", "direito", "pedidos"]
    texto: str


class Variantes(BaseModel):
    parafrases: list[ParafraseItem]
    extras: list[Extra]


class Ementa(BaseModel):
    referencia: str = Field(description="Tribunal, classe, número, relator e data (fictícios)")
    texto: str


class Ementas(BaseModel):
    itens: list[Ementa]


class Contestacao(BaseModel):
    enderecamento: str
    qualificacao: str
    sintese: list[str]
    preliminares: list[Secao]
    merito: list[Secao]
    pedidos: list[str]
    fecho: list[str]


class Family(BaseModel):
    """Template completo de uma petição (tese × escritório), com variantes para versões."""

    tese: str
    firm: str
    especifico: Especifico
    parafrases: dict[str, str]  # id do parágrafo -> paráfrase
    extras: list[Extra]


def placeholders_in(obj: BaseModel | str | list | dict) -> set[str]:
    blob = obj if isinstance(obj, str) else json.dumps(_dump(obj), ensure_ascii=False)
    return set(PLACEHOLDER.findall(blob))


def _dump(obj):
    return obj.model_dump() if isinstance(obj, BaseModel) else obj


def save(model: BaseModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2) + "\n", "utf-8")


def paths() -> dict[str, Path]:
    return {
        "boilerplate": TEMPLATES_DIR / "boilerplate",
        "families": TEMPLATES_DIR / "families",
        "ementas": TEMPLATES_DIR / "ementas",
        "contestacoes": TEMPLATES_DIR / "contestacoes",
    }


def specific_paragraph_ids(esp: Especifico) -> dict[str, str]:
    """Ids estáveis: f3 = fatos[3], d1.2 = direito[1].paragrafos[2], p0 = pedidos[0]."""
    out = {f"f{i}": p for i, p in enumerate(esp.fatos)}
    for s, sec in enumerate(esp.direito):
        out.update({f"d{s}.{i}": p for i, p in enumerate(sec.paragrafos)})
    out.update({f"p{i}": p for i, p in enumerate(esp.pedidos)})
    return out
