"""Geração (única) dos templates sintéticos via LLM. Saída versionada em data/synthetic/templates.

Uso: `uv run q1 gen-templates`. Reexecutar é grátis (cache em disco); só regenera o que falta.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel

from common.llm import LLM
from q1_similarity import spec
from q1_similarity.templates import (
    Boilerplate,
    Contestacao,
    Ementas,
    Especifico,
    Family,
    Variantes,
    paths,
    placeholders_in,
    save,
    specific_paragraph_ids,
)

PROMPT_VERSION = "q1-templates-v1"
DEFAULT_MODEL = "gpt-4.1"
TEMPERATURE = 0.7
MAX_ATTEMPTS = 3

SYSTEM = """Você é advogado(a) brasileiro(a) experiente redigindo peças FICTÍCIAS para um \
benchmark de software jurídico. Escreva em português do Brasil, com linguagem forense real.

Regras obrigatórias:
- Todo dado variável (nomes, documentos, valores, datas, números de voo/contrato, cidades, \
endereços) deve aparecer SOMENTE como placeholder entre chaves, exatamente como listado, ex.: \
{AUTOR_NOME}. Nunca escreva valores concretos (R$ 1.000,00, 12/03/2024, CPF etc.).
- Use apenas os placeholders permitidos. Não invente outros.
- Citações de lei, artigos, súmulas e resoluções são permitidas por número (art. 14 do CDC).
- Refira-se ao autor de forma neutra quanto ao gênero: "a parte autora", "a parte reclamante".
- Não use nomes reais de pessoas, empresas, escritórios ou processos.
- Cada parágrafo é um bloco de texto corrido, sem quebras de linha internas e sem numeração \
de títulos."""

_LITERAL_DATA = re.compile(
    r"R\$\s*\d|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{3}\.\d{3}\.\d{3}-\d{2}\b|\b\d{2}\.\d{3}\.\d{3}/"
)


def _allowed(names: frozenset[str]) -> str:
    return ", ".join("{" + n + "}" for n in sorted(names))


def _problems(
    obj: BaseModel, allowed: frozenset[str], required: set[str], literals: bool
) -> list[str]:
    found = placeholders_in(obj)
    out = []
    if bad := found - allowed:
        out.append(f"placeholders não permitidos: {sorted(bad)}")
    if missing := required - found:
        out.append(f"placeholders obrigatórios ausentes: {sorted(missing)}")
    if not literals:
        return out
    blob = json.dumps(obj.model_dump(), ensure_ascii=False)
    # citações legais (Lei 8.078/90) não casam: o padrão exige R$, data completa ou CPF/CNPJ
    if lit := _LITERAL_DATA.findall(blob):
        out.append(f"dados concretos escritos sem placeholder: {lit[:5]}")
    return out


def _ask[T: BaseModel](
    llm: LLM,
    schema: type[T],
    user: str,
    *,
    allowed: frozenset[str],
    required: set[str],
    model: str,
    check: Callable[[T], list[str]] | None = None,
    literals: bool = True,
) -> T:
    """Chamada estruturada com validação; rejeições voltam ao modelo como feedback."""
    feedback = ""
    for _ in range(MAX_ATTEMPTS):
        out, _rec = llm.structured(
            schema,
            system=SYSTEM,
            user=user + feedback,
            model=model,
            temperature=TEMPERATURE,
            prompt_version=PROMPT_VERSION,
        )
        problems = _problems(out, allowed, required, literals) + (check(out) if check else [])
        if not problems:
            return out
        feedback = "\n\nSua resposta anterior foi rejeitada: " + "; ".join(problems) + ". Corrija."
    raise RuntimeError(f"{schema.__name__}: falhou após {MAX_ATTEMPTS} tentativas: {problems}")


def _count(label: str, items: list, lo: int, hi: int) -> list[str]:
    return [] if lo <= len(items) <= hi else [f"{label}: {len(items)} itens, esperado {lo}-{hi}"]


# --------------------------------------------------------------------------- prompts


def gen_boilerplate(llm: LLM, firm: spec.Firm, area: str, model: str) -> Boilerplate:
    allowed = spec.AREA_PLACEHOLDERS[area]
    area_txt = {
        "consumidor": "ações de consumidor na Justiça Estadual (Vara Cível ou Juizado Especial)",
        "trabalhista": "reclamações trabalhistas na Justiça do Trabalho",
    }[area]
    aplic = {
        "consumidor": "aplicabilidade do CDC e relação de consumo; inversão do ônus da prova "
        "(art. 6º, VIII); responsabilidade objetiva do fornecedor; celeridade se cabível",
        "trabalhista": "aplicação dos princípios protetivos do Direito do Trabalho; ônus da "
        "prova do empregador quanto a documentos (Súmula 338 do TST); honorários "
        "sucumbenciais (art. 791-A da CLT); correção monetária e juros",
    }[area]
    user = f"""Escreva as partes-PADRÃO que o escritório "{firm.nome}" reutiliza em TODAS as \
suas {area_txt}, independentemente da tese. Esses trechos são colados de petição em petição.

Estilo do escritório: {firm.estilo}

Placeholders permitidos: {_allowed(allowed)}

Campos:
- enderecamento: uma linha de endereçamento ao juízo, usando {{VARA}} e {{COMARCA}}.
- qualificacao: um parágrafo qualificando a parte autora ({{AUTOR_NOME}}, \
{{AUTOR_QUALIFICACAO}}, {{AUTOR_CPF}}, {{AUTOR_RG}}, {{AUTOR_ENDERECO}}, {{AUTOR_EMAIL}}), \
mencionando o advogado ({{ADVOGADO_NOME}}) e propondo {{NOME_ACAO}} em face de {{REU_NOME}} \
({{REU_CNPJ}}, {{REU_ENDERECO}}).
- gratuidade: seção sobre justiça gratuita, 2 a 3 parágrafos.
- aplicabilidade: 3 seções genéricas ({aplic}), cada uma com 2 a 3 parágrafos.
- pedidos_gerais: 4 a 6 pedidos genéricos (citação, provas, gratuidade, honorários etc.).
- fecho: 3 a 5 linhas finais (valor da causa com {{VALOR_CAUSA}} e {{VALOR_CAUSA_EXTENSO}}, \
"Nestes termos, pede deferimento", local {{COMARCA}} e {{DATA_PETICAO}}, assinatura \
{{ADVOGADO_NOME}} e {{ADVOGADO_OAB}}).

Não mencione fatos de nenhuma tese específica."""
    required = {"AUTOR_NOME", "REU_NOME", "VALOR_CAUSA", "ADVOGADO_OAB", "VARA", "COMARCA"}

    def check(b: Boilerplate) -> list[str]:
        return _count("aplicabilidade", b.aplicabilidade, 3, 3) + _count(
            "pedidos_gerais", b.pedidos_gerais, 4, 6
        )

    return _ask(
        llm, Boilerplate, user, allowed=allowed, required=required, model=model, check=check
    )


def gen_especifico(llm: LLM, tese: spec.Tese, firm: spec.Firm, model: str) -> Especifico:
    allowed = tese.placeholders
    user = f"""Escreva o conteúdo ESPECÍFICO de uma petição inicial do escritório \
"{firm.nome}" para a tese abaixo. Endereçamento, qualificação, justiça gratuita, \
aplicabilidade genérica da lei, pedidos genéricos, fecho e jurisprudência já existem em \
outros blocos: NÃO os repita aqui.

Tese: {tese.descricao}
Ação: {tese.nome_acao}
Estilo do escritório: {firm.estilo}
Placeholders permitidos: {_allowed(allowed)}

Campos:
- fatos: 8 a 12 parágrafos narrando os fatos concretos, usando os placeholders do caso.
- direito: 4 a 6 seções de fundamentação jurídica específicas da tese, cada uma com 2 a 4 \
parágrafos.
- pedidos: 6 a 10 pedidos específicos da tese, com valores via placeholders."""
    required = set(tese.extra_placeholders) & {
        "VOO_NUMERO",
        "CONTRATO_NUMERO",
        "CARGO",
        "SALARIO",
    }

    def check(e: Especifico) -> list[str]:
        return (
            _count("fatos", e.fatos, 8, 12)
            + _count("direito", e.direito, 4, 6)
            + _count("pedidos", e.pedidos, 6, 10)
        )

    return _ask(llm, Especifico, user, allowed=allowed, required=required, model=model, check=check)


def gen_variantes(llm: LLM, tese: spec.Tese, esp: Especifico, model: str) -> Variantes:
    ids = specific_paragraph_ids(esp)
    listing = "\n".join(f"[{k}] {v}" for k, v in ids.items())
    user = f"""Abaixo estão os parágrafos específicos de uma petição, cada um com um id. \
Gere duas coisas para simular versões revisadas desta petição:

1. parafrases: para CADA id, reescreva o parágrafo com o mesmo sentido jurídico mas com \
redação substancialmente diferente (troque estrutura das frases e vocabulário, como um \
revisor reescrevendo). Mantenha exatamente os mesmos placeholders do original.
2. extras: 8 parágrafos NOVOS que um revisor acrescentaria (fato adicional, argumento \
complementar, pedido adicional), cada um com secao_alvo "fatos", "direito" ou "pedidos".

Tese: {tese.descricao}
Placeholders permitidos: {_allowed(tese.placeholders)}

Parágrafos:
{listing}"""

    def check(v: Variantes) -> list[str]:
        got = {p.id for p in v.parafrases}
        out = []
        if missing := set(ids) - got:
            out.append(f"faltam paráfrases para os ids {sorted(missing)}")
        return out + _count("extras", v.extras, 8, 8)

    return _ask(
        llm,
        Variantes,
        user,
        allowed=tese.placeholders,
        required=set(),
        model=model,
        check=check,
    )


def gen_ementas(llm: LLM, tese: spec.Tese, model: str) -> Ementas:
    user = f"""Escreva 8 ementas de jurisprudência FICTÍCIAS (mas verossímeis) favoráveis à \
tese abaixo, no formato usado em petições brasileiras.

Tese: {tese.descricao}

- referencia: tribunal, classe, número fictício, relator(a) fictício(a) e data, ex.: \
"TJSP, Apelação Cível nº 1000000-00.2023.8.26.0000, Rel. Des. Fulano de Tal, j. 10/05/2023". \
Na referência (e só nela) números e datas concretos são permitidos.
- texto: a ementa em caixa alta no início (palavras-chave separadas por travessão) seguida \
do texto, 60 a 140 palavras. Sem placeholders."""

    def check(e: Ementas) -> list[str]:
        return _count("ementas", e.itens, 8, 8)

    # referências contêm datas/números de propósito: sem filtro de dados concretos
    return _ask(
        llm,
        Ementas,
        user,
        allowed=frozenset(),
        required=set(),
        model=model,
        check=check,
        literals=False,
    )


def gen_contestacao(llm: LLM, tese: spec.Tese, model: str) -> Contestacao:
    allowed = tese.defense_placeholders
    user = f"""Escreva uma CONTESTAÇÃO completa da empresa ré contra a petição inicial com a \
tese abaixo. {spec.DEFENSE_FIRM}

Tese da parte autora: {tese.descricao}
Placeholders permitidos: {_allowed(allowed)}

Campos:
- enderecamento: endereçamento ao juízo ({{VARA}}, {{COMARCA}}) e referência ao processo \
{{PROCESSO_NUMERO}}.
- qualificacao: um parágrafo com {{REU_NOME}}, {{REU_CNPJ}}, {{REU_ENDERECO}}, por seu \
advogado {{REU_ADVOGADO_NOME}}, apresentando contestação à ação movida por {{AUTOR_NOME}}.
- sintese: 3 a 5 parágrafos resumindo as alegações da inicial (com os placeholders do caso).
- preliminares: 1 a 3 seções (ex.: ilegitimidade, falta de interesse, impugnação à \
gratuidade), 1 a 3 parágrafos cada.
- merito: 4 a 6 seções rebatendo a tese, 2 a 4 parágrafos cada.
- pedidos: 4 a 7 pedidos.
- fecho: 3 a 4 linhas finais com {{COMARCA}}, {{DATA_PETICAO}}, {{REU_ADVOGADO_NOME}} e \
{{REU_ADVOGADO_OAB}}."""
    required = {"AUTOR_NOME", "REU_NOME", "PROCESSO_NUMERO", "REU_ADVOGADO_OAB"}

    def check(c: Contestacao) -> list[str]:
        return _count("merito", c.merito, 4, 6) + _count("sintese", c.sintese, 3, 5)

    return _ask(
        llm, Contestacao, user, allowed=allowed, required=required, model=model, check=check
    )


# --------------------------------------------------------------------------- orquestração


def _family(llm: LLM, tese: spec.Tese, firm: spec.Firm, model: str) -> Family:
    esp = gen_especifico(llm, tese, firm, model)
    var = gen_variantes(llm, tese, esp, model)
    originals = specific_paragraph_ids(esp)
    # descarta paráfrases que perderam/criaram placeholders: não seriam "mesmo conteúdo"
    para = {
        p.id: p.texto
        for p in var.parafrases
        if p.id in originals and placeholders_in(p.texto) == placeholders_in(originals[p.id])
    }
    return Family(tese=tese.key, firm=firm.key, especifico=esp, parafrases=para, extras=var.extras)


def generate_all(llm: LLM, model: str = DEFAULT_MODEL, workers: int = 6) -> dict[str, int]:
    p = paths()
    jobs: list[tuple[str, Callable[[], BaseModel]]] = []
    for firm in spec.FIRMS.values():
        for area in spec.AREA_PLACEHOLDERS:
            jobs.append(
                (
                    str(p["boilerplate"] / f"{firm.key}__{area}.json"),
                    lambda f=firm, a=area: gen_boilerplate(llm, f, a, model),
                )
            )
    for tese in spec.TESES.values():
        jobs.append(
            (str(p["ementas"] / f"{tese.key}.json"), lambda t=tese: gen_ementas(llm, t, model))
        )
        jobs.append(
            (
                str(p["contestacoes"] / f"{tese.key}.json"),
                lambda t=tese: gen_contestacao(llm, t, model),
            )
        )
        for firm in spec.FIRMS.values():
            jobs.append(
                (
                    str(p["families"] / f"{spec.family_key(tese.key, firm.key)}.json"),
                    lambda t=tese, f=firm: _family(llm, t, f, model),
                )
            )

    todo = [(path, fn) for path, fn in jobs if not Path(path).exists()]

    def run(job: tuple[str, Callable[[], BaseModel]]) -> str:
        path, fn = job
        save(fn(), Path(path))
        return path

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done in pool.map(run, todo):
            print(f"ok {done}")
    return {"total": len(jobs), "generated": len(todo)}
