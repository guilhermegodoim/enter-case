"""Gold da Q2: rascunho por LLM, leitura/escrita e status. A correção é humana (gold_app.py).

O rascunho usa um modelo forte com o texto inteiro (as sentenças do corpus cabem no contexto),
porque o gold não tem restrição de custo. As citações são ancoradas em código.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from common.config import ROOT
from common.llm import LLM
from common.text import sha256_text
from q2_arguments.corpus import load_rulings, ruling_text
from q2_arguments.ground import Source, anchor
from q2_arguments.schema import (
    SCHEMA_VERSION,
    Argumento,
    Decisao,
    Resultado,
    Secao,
    SuscitadoPor,
    TipoArgumento,
)

GOLD_DIR = ROOT / "data" / "gold"


def gold_dir(root: Path | None = None) -> Path:
    """Diretório do gold; ENTER_GOLD_DIR sobrescreve (testes do editor)."""
    return root or Path(os.getenv("ENTER_GOLD_DIR") or GOLD_DIR)


DRAFT_MODEL = "gpt-5"
DRAFT_PROMPT_VERSION = "q2-gold-draft-v1"

Status = Literal["rascunho", "revisado"]


class GoldRuling(BaseModel):
    ruling_id: str
    text_sha256: str
    schema_version: str = SCHEMA_VERSION
    status: Status = "rascunho"
    resultado: Resultado | None = None
    argumentos: list[Argumento] = Field(default_factory=list)
    rascunho: dict = Field(default_factory=dict)  # proveniência do rascunho LLM
    revisor: str | None = None
    revisado_em: str | None = None
    notas: str = ""


# --------------------------------------------------------------------------- io


def gold_path(rid: str, root: Path | None = None) -> Path:
    return gold_dir(root) / f"{rid}.json"


def load_gold(rid: str, root: Path | None = None) -> GoldRuling | None:
    path = gold_path(rid, root)
    if not path.exists():
        return None
    return GoldRuling.model_validate_json(path.read_text("utf-8"))


def save_gold(g: GoldRuling, root: Path | None = None) -> Path:
    path = gold_path(g.ruling_id, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(g.model_dump(), indent=2, ensure_ascii=False) + "\n", "utf-8")
    return path


def reanchor(g: GoldRuling, text: str) -> GoldRuling:
    src = Source(text)
    for a in g.argumentos:
        a.citacoes = [anchor(c.texto, src) for c in a.citacoes if c.texto.strip()]
    g.text_sha256 = sha256_text(text)
    return g


def mark_reviewed(g: GoldRuling, reviewer: str) -> GoldRuling:
    bad = [a.id for a in g.argumentos for c in a.citacoes if not c.ancorada]
    empty = [a.id for a in g.argumentos if not a.citacoes]
    if bad or empty:
        raise ValueError(f"citações não ancoradas em {sorted(set(bad))}, sem citação em {empty}")
    if not reviewer.strip():
        raise ValueError("informe o revisor")
    if g.resultado is None:
        raise ValueError("informe o resultado da sentença")
    g.status = "revisado"
    g.revisor = reviewer.strip()
    g.revisado_em = datetime.now(UTC).isoformat(timespec="seconds")
    return g


# --------------------------------------------------------------------------- rascunho LLM


class ArgumentoRascunho(BaseModel):
    titulo: str
    resumo: str
    tipo: TipoArgumento
    tema: str
    suscitado_por: SuscitadoPor
    decisao: Decisao
    fundamento_legal: list[str]
    citacoes: list[str] = Field(description="1 a 3 trechos copiados literalmente da sentença")
    secao: Secao


class RascunhoSentenca(BaseModel):
    resultado: Resultado
    argumentos: list[ArgumentoRascunho]


DRAFT_SYSTEM = """Você é advogado(a) brasileiro(a) montando um banco de argumentos a partir de \
sentenças cíveis. Leia a sentença inteira e liste cada questão que o juiz enfrentou e decidiu, \
com a razão da decisão.

Um argumento = uma tese sobre uma questão decidida, com fundamentação própria:
- cada preliminar (ilegitimidade, interesse de agir, incompetência, inépcia, impugnação à \
gratuidade...) e cada prejudicial (prescrição, decadência) é um argumento;
- no mérito, um argumento por questão decidida (contratação/relação jurídica, falha do serviço, \
excludentes como fortuito externo ou culpa exclusiva, dano moral, dano material, repetição em \
dobro, abusividade de cláusula ou taxa, ônus da prova quando discutido...);
- o valor da indenização é argumento separado (tipo quantum) quando o juiz fundamenta o valor;
- não divida uma questão em vários argumentos; não junte questões diferentes em um só.

Não são argumentos: resumo das alegações sem decisão do juiz; custas, honorários, correção e \
juros aplicados de forma padrão; fórmulas de encerramento. Sentença sem mérito pode ter zero \
argumentos.

Campos:
- titulo: a tese em uma linha, do ponto de vista de quem a levantou.
- resumo: 1 a 3 frases com o que o juiz decidiu e por quê, com suas palavras.
- tipo: preliminar | prejudicial | merito | quantum | acessorio (juros, correção, honorários, \
gratuidade só quando discutidos).
- tema: rótulo curto e reutilizável (ex.: "ilegitimidade passiva", "fortuito externo").
- suscitado_por: autor | reu | juizo (de ofício).
- decisao: o juiz acolheu, rejeitou ou acolheu em parte a tese do título (de ofício: acolhido).
- fundamento_legal: dispositivos, súmulas, temas e precedentes citados para esta questão, na \
forma curta ("art. 14 do CDC", "Súmula 479 do STJ").
- citacoes: 1 a 3 trechos COPIADOS LITERALMENTE da sentença (uma ou duas frases cada, sem \
reticências, sem alterar uma letra), de preferência da fundamentação, que sustentam a decisão.
- secao: relatorio | fundamentacao | dispositivo, onde a questão é decidida.

resultado: procedente | parcialmente_procedente | improcedente | extinto_sem_merito | \
homologacao | outro.

Use apenas o que está escrito na sentença. O texto está anonimizado ([AUTOR], [PESSOA_1]...): \
mantenha os marcadores como estão."""


def draft_one(llm: LLM, rid: str, model: str = DRAFT_MODEL) -> GoldRuling:
    text = ruling_text(rid)
    parsed, rec = llm.structured(
        RascunhoSentenca,
        system=DRAFT_SYSTEM,
        user=f"Sentença (texto integral):\n<<<\n{text}\n>>>",
        model=model,
        temperature=None,
        prompt_version=DRAFT_PROMPT_VERSION,
    )
    src = Source(text)
    args = [
        Argumento(
            id=f"a{i:02d}",
            **a.model_dump(exclude={"citacoes"}),
            citacoes=[anchor(q, src) for q in a.citacoes],
        )
        for i, a in enumerate(parsed.argumentos, start=1)
    ]
    quotes = [c for a in args for c in a.citacoes]
    return GoldRuling(
        ruling_id=rid,
        text_sha256=sha256_text(text),
        resultado=parsed.resultado,
        argumentos=args,
        rascunho={
            "modelo": model,
            "snapshot": rec.model_snapshot,
            "prompt_version": DRAFT_PROMPT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "input_tokens": rec.input_tokens,
            "output_tokens": rec.output_tokens,
            "cost_usd": rec.cost_usd,
            "latency_ms": round(rec.latency_ms),
            "citacoes": len(quotes),
            "citacoes_ancoradas": sum(c.ancorada for c in quotes),
        },
    )


def draft_all(
    llm: LLM,
    *,
    model: str = DRAFT_MODEL,
    only: list[str] | None = None,
    force: bool = False,
    workers: int = 4,
    root: Path | None = None,
) -> dict[str, str]:
    """Gera rascunhos que faltam. Nunca sobrescreve revisado; rascunho só com force."""
    rids = [r["ruling_id"] for r in load_rulings() if not only or r["ruling_id"] in only]
    todo, outcome = [], {}
    for rid in rids:
        g = load_gold(rid, root)
        if g is not None and (g.status == "revisado" or not force):
            outcome[rid] = f"mantido ({g.status})"
        else:
            todo.append(rid)

    def work(rid: str) -> tuple[str, str]:
        g = draft_one(llm, rid, model)
        save_gold(g, root)
        r = g.rascunho
        return (
            rid,
            f"rascunho: {len(g.argumentos)} args, "
            f"{r['citacoes_ancoradas']}/{r['citacoes']} ancoradas",
        )

    with ThreadPoolExecutor(max_workers=workers) as ex:
        outcome.update(dict(ex.map(work, todo)))
    return outcome


def status_rows(root: Path | None = None) -> list[dict]:
    rows = []
    for r in load_rulings():
        rid = r["ruling_id"]
        g = load_gold(rid, root)
        quotes = [c for a in g.argumentos for c in a.citacoes] if g else []
        stale = g is not None and g.text_sha256 != sha256_text(ruling_text(rid))
        rows.append(
            {
                "ruling_id": rid,
                "split": r["split"],
                "faixa": r["faixa"],
                "status": g.status if g else "-",
                "resultado": (g.resultado or "-") if g else "-",
                "argumentos": len(g.argumentos) if g else 0,
                "ancoradas": f"{sum(c.ancorada for c in quotes)}/{len(quotes)}",
                "texto_mudou": stale,
            }
        )
    return rows
