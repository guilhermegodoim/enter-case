"""Extração de argumentos de uma sentença: LLM -> ancoragem -> merge -> verificador -> confiança.

Princípios (ver docs/q2.md, fase 6):
- O LLM devolve texto; posição, seção e confiança são calculadas em código.
- Argumento sem nenhuma citação ancorada no texto é descartado (não existe na fonte).
- Sentença longa: modo "chunk" divide fundamentação + dispositivo em pedaços de orçamento fixo,
  com o início do relatório e o dispositivo como contexto; os pedaços rodam em paralelo.
- Toda saída carrega o run_id da configuração (modelo, prompt, schema, parâmetros, código).
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from common.llm import LLM, CallRecord, LLMError
from common.runs import RunSpec
from common.text import fix_strings
from q2_arguments.ground import Source, anchor
from q2_arguments.merge import merge
from q2_arguments.schema import Argumento, Citacao
from q2_arguments.segment import (
    chunk_range,
    head_tokens,
    section_of,
    sections,
    tail_tokens,
)
from q2_arguments.verify import PROMPT_VERSION as VERIFY_PROMPT_VERSION
from q2_arguments.verify import SYSTEM as VERIFY_SYSTEM
from q2_arguments.verify import Checagem, verify

# v1: Argumento (q2-args-v1) + confiança/sinais/run_id
# v2: + posicao (envelope das citações ancoradas); upcaster v1 -> v2 em versions.py
EXTRACT_SCHEMA_VERSION = "q2-extracted-v2"
MAX_QUOTES = 3

# --------------------------------------------------------------------------- saída do LLM


class ArgumentoLLM(BaseModel):
    titulo: str
    resumo: str
    tipo: Literal["preliminar", "prejudicial", "merito", "quantum", "acessorio"]
    tema: str
    suscitado_por: Literal["autor", "reu", "juizo"]
    decisao: Literal["acolhido", "rejeitado", "parcial"]
    fundamento_legal: list[str]
    citacoes: list[str] = Field(description="1 a 3 trechos copiados literalmente do texto")


class SaidaLLM(BaseModel):
    argumentos: list[ArgumentoLLM]


_RULES = """Um argumento = uma tese sobre uma questão que o juiz enfrentou e decidiu, com \
fundamentação própria:
- cada preliminar (ilegitimidade, interesse de agir, incompetência, inépcia, impugnação à \
gratuidade, cerceamento de defesa...) e cada prejudicial (prescrição, decadência) é um argumento;
- no mérito, um argumento por questão decidida (contratação/relação jurídica, falha do serviço, \
excludentes como fortuito externo ou culpa exclusiva, dano moral, dano material, repetição em \
dobro, abusividade de cláusula ou taxa, ônus da prova quando discutido...);
- o valor da indenização é argumento separado (tipo quantum) quando o juiz fundamenta o valor;
- não divida uma questão em vários argumentos; não junte questões diferentes em um só.

Não são argumentos: resumo das alegações sem decisão do juiz; custas, honorários, correção e \
juros aplicados de forma padrão; fórmulas de encerramento. Pode haver zero argumentos.

Campos:
- titulo: a tese em uma linha, do ponto de vista de quem a levantou.
- resumo: 1 a 3 frases com o que o juiz decidiu e por quê, com suas palavras.
- tipo: preliminar | prejudicial | merito | quantum | acessorio (juros, correção, honorários, \
gratuidade só quando discutidos).
- tema: rótulo curto e reutilizável (ex.: "ilegitimidade passiva", "fortuito externo").
- suscitado_por: autor | reu | juizo (de ofício).
- decisao: o juiz acolheu, rejeitou ou acolheu em parte a tese do título (de ofício: acolhido).
- fundamento_legal: dispositivos, súmulas, temas e precedentes citados para esta questão, \
forma curta ("art. 14 do CDC", "Súmula 479 do STJ").
- citacoes: 1 a 3 trechos COPIADOS LITERALMENTE (uma ou duas frases cada, sem reticências, \
sem corrigir nada), de preferência do raciocínio do próprio juiz, que sustentam a decisão."""

SYSTEM_GUIA = f"""Você extrai, de sentenças cíveis brasileiras, os argumentos decididos pelo \
juiz, para um banco de argumentos de defesa. Use apenas o que está escrito no texto.

{_RULES}

O texto está anonimizado ([AUTOR], [PESSOA_1]...): mantenha os marcadores como estão."""

SYSTEM_BASICO = """Liste os principais argumentos usados pelo juiz nesta sentença. Para cada \
um: título, resumo, tipo (preliminar | prejudicial | merito | quantum | acessorio), tema, quem \
suscitou (autor | reu | juizo), decisão (acolhido | rejeitado | parcial), fundamento legal e 1 a \
3 citações literais do texto."""

CHUNK_NOTE = """

Você recebe só UM TRECHO da sentença (a sentença é longa e foi dividida). Extraia apenas \
argumentos cuja decisão está NESTE TRECHO, com citações copiadas DESTE TRECHO. O início da \
sentença (relatório) e o dispositivo vêm só como contexto, para saber partes, pedidos e \
resultado: não extraia deles."""

# v2: ajustada só com erros do split dev de chunk_mini (fase 5): faltavam valor da indenização,
# aplicabilidade do CDC, ônus da prova e solidariedade; sobravam princípios gerais (boa-fé,
# função social) listados como argumento próprio.
_RULES_V2 = _RULES.replace(
    "- não divida uma questão em vários argumentos; não junte questões diferentes em um só.",
    """- quando o juiz decide sobre elas, também são argumentos (um cada): aplicabilidade do CDC, \
inversão ou distribuição do ônus da prova, responsabilidade solidária de outra ré ou da cadeia \
de fornecimento, efeito de acordo ou pagamento de corréu;
- quando o juiz fixa um valor de indenização (dano moral, dano material), a fixação do valor é \
um argumento quantum separado, mesmo com fundamentação curta;
- princípios gerais citados como apoio (boa-fé objetiva, função social do contrato, pacta sunt \
servanda) não são argumentos próprios: entram no argumento que ajudam a decidir;
- não divida uma questão em vários argumentos; não junte questões diferentes em um só.""",
)
assert _RULES_V2 != _RULES

SYSTEM_GUIA_V2 = SYSTEM_GUIA.replace(_RULES, _RULES_V2)

PROMPTS = {"guia": SYSTEM_GUIA, "guia2": SYSTEM_GUIA_V2, "basico": SYSTEM_BASICO}
PROMPT_VERSIONS = {
    "guia": "q2-extract-guia-v1",
    "guia2": "q2-extract-guia-v2",
    "basico": "q2-extract-basico-v1",
}

# --------------------------------------------------------------------------- configuração


class Config(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    # full: texto inteiro numa chamada; chunk: pedaços em paralelo; truncate: só os primeiros
    # `budget` tokens (baseline ingênuo de "não cabe no contexto")
    mode: Literal["full", "chunk", "truncate"] = "chunk"
    model: str = "gpt-4.1-mini"
    prompt: Literal["guia", "guia2", "basico"] = "guia"
    budget: int = 3000  # tokens por chunk
    overlap: int = 300  # tokens de sobreposição entre chunks
    head: int = 800  # tokens do relatório como contexto (modo chunk)
    tail: int = 600  # tokens do dispositivo como contexto (modo chunk)
    ground_filter: bool = True  # descarta argumento sem citação ancorada
    merge: bool = True
    merge_threshold: float = 0.88  # cosseno título+resumo (mesmo tipo); calibrado no dev
    verifier: str | None = None  # modelo do verificador; None = desligado
    verifier_drop: bool = False  # descarta "nao_sustentado" (senão só reduz a confiança)

    @property
    def prompt_version(self) -> str:
        v = PROMPT_VERSIONS[self.prompt]
        return v + ("+chunk" if self.mode == "chunk" else "")

    @property
    def prompt_sha256(self) -> str:
        """Hash do texto real dos prompts: pega edição de prompt sem troca de versão."""
        parts = [PROMPTS[self.prompt], CHUNK_NOTE if self.mode == "chunk" else ""]
        if self.verifier:
            parts.append(VERIFY_SYSTEM)
        return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:16]

    @property
    def temperature(self) -> float | None:
        return None if self.model.startswith(("gpt-5", "o")) else 0.0

    def spec(self) -> RunSpec:
        return RunSpec(
            task="q2-extract",
            model=self.model,
            prompt_version=self.prompt_version,
            schema_version=EXTRACT_SCHEMA_VERSION,
            params={
                **self.model_dump(exclude={"key"}),
                "prompt_sha256": self.prompt_sha256,
                "verifier_prompt_version": VERIFY_PROMPT_VERSION if self.verifier else None,
            },
        )


# --------------------------------------------------------------------------- resultado


class Posicao(BaseModel):
    """Onde o argumento está no texto: envelope das citações ancoradas (offsets do texto)."""

    char_start: int
    char_end: int


def posicao_of(citacoes: list[Citacao]) -> Posicao | None:
    spans = [(c.char_start, c.char_end) for c in citacoes if c.ancorada]
    if not spans:
        return None
    return Posicao(char_start=min(s for s, _ in spans), char_end=max(e for _, e in spans))


class ArgumentoExtraido(Argumento):
    posicao: Posicao | None = None  # None = nenhuma citação ancorada
    confianca: float = Field(ge=0, le=1)
    confianca_nivel: Literal["alta", "media", "baixa"]
    sinais: dict = Field(default_factory=dict)  # ancoragem, verificador, chunks de origem
    run_id: str


@dataclass
class Extraction:
    ruling_id: str
    run_id: str
    argumentos: list[ArgumentoExtraido]
    brutos: int  # argumentos devolvidos pelo LLM (antes de filtros)
    citacoes_brutas: int
    citacoes_ancoradas: int
    descartados_sem_ancora: int
    fundidos: int
    descartados_verificador: int
    n_chunks: int
    chunks_falhos: int
    chunks_com_retry: int
    latency_ms: float  # chunks em paralelo: max(chunk) + verificador
    cost_usd: float
    calls: list[CallRecord] = field(default_factory=list)
    sem_ancora: list[dict] = field(default_factory=list)  # o que foi descartado, para auditoria
    falhas: list[dict] = field(default_factory=list)  # chunks sem saída válida após retry


VERIFIER_FACTOR = {"sustentado": 1.0, "parcial": 0.6, "nao_sustentado": 0.2, None: 0.8}


def confidence(scores: list[float], check: Checagem | None) -> tuple[float, str]:
    """Confiança derivada de sinais verificáveis, não do auto-relato do modelo."""
    g = sum(scores) / len(scores) / 100 if scores else 0.0
    v = VERIFIER_FACTOR[check.suporte if check else None]
    if check is not None and not check.decisao_confere:
        v *= 0.7
    c = round(g * v, 3)
    return c, "alta" if c >= 0.85 else "media" if c >= 0.5 else "baixa"


# --------------------------------------------------------------------------- pipeline


@dataclass
class _Draft:
    arg: ArgumentoLLM
    citacoes: list[Citacao]
    chunk: int


def _user_full(text: str) -> str:
    return f"Sentença (texto integral):\n<<<\n{text}\n>>>"


def _user_chunk(head: str, tail: str, chunk: str, i: int, n: int) -> str:
    parts = []
    if head:
        parts.append(f"CONTEXTO - início da sentença (não extrair daqui):\n<<<\n{head}\n>>>")
    if tail:
        parts.append(f"CONTEXTO - dispositivo (não extrair daqui):\n<<<\n{tail}\n>>>")
    parts.append(f"TRECHO A ANALISAR (parte {i + 1} de {n}):\n<<<\n{chunk}\n>>>")
    return "\n\n".join(parts)


def extract(llm: LLM, cfg: Config, ruling_id: str, text: str) -> Extraction:
    run_id = cfg.spec().run_id
    sec = sections(text)
    system = PROMPTS[cfg.prompt]
    if cfg.mode == "full":
        jobs = [(0, system, _user_full(text))]
    elif cfg.mode == "truncate":
        jobs = [(0, system, _user_full(head_tokens(text, cfg.budget)))]
    else:
        chunks = chunk_range(text, sec.fund_start, sec.n, cfg.budget, cfg.overlap)
        head = head_tokens(text[: sec.fund_start], cfg.head) if sec.fund_start else ""
        disp = text[sec.disp_start :]
        jobs = []
        for ch in chunks:
            tail = tail_tokens(disp, cfg.tail) if disp.strip() and ch.end <= sec.disp_start else ""
            user = _user_chunk(head, tail, text[ch.start : ch.end], ch.idx, len(chunks))
            jobs.append((ch.idx, system + CHUNK_NOTE, user))

    # saída degenerada (JSON inválido, resposta incompleta): 1 nova tentativa com temperatura
    # diferente (outra chave de cache); se falhar de novo, o chunk é registrado como falho
    retry_temp = 0.3 if cfg.temperature is not None else None

    def call(job):
        idx, sys, user = job
        errors = []
        for temp in (cfg.temperature, retry_temp):
            try:
                out, rec = llm.structured(
                    SaidaLLM, system=sys, user=user, model=cfg.model,
                    temperature=temp, prompt_version=cfg.prompt_version,
                )  # fmt: skip
            except LLMError as e:
                errors.append(str(e)[:200])
                continue
            return idx, SaidaLLM.model_validate(fix_strings(out.model_dump())), rec, errors
        return idx, None, None, errors

    with ThreadPoolExecutor(max_workers=8) as ex:
        attempts = list(ex.map(call, jobs))
    results = [(i, out, rec) for i, out, rec, _ in attempts if out is not None]
    failed = [{"chunk": i, "erros": errs} for i, out, _, errs in attempts if out is None]
    retried = sum(1 for _, out, _, errs in attempts if out is not None and errs)
    calls = [r for _, _, r in results]

    src = Source(text)
    drafts: list[_Draft] = []
    dropped: list[dict] = []
    n_quotes = n_anchored = 0
    for idx, out, _ in results:
        for a in out.argumentos:
            cits = [anchor(q, src) for q in a.citacoes[: MAX_QUOTES + 2] if q.strip()]
            n_quotes += len(cits)
            ok = [c for c in cits if c.ancorada]
            n_anchored += len(ok)
            if cfg.ground_filter and not ok:
                dropped.append({"titulo": a.titulo, "citacoes": a.citacoes, "chunk": idx})
                continue
            drafts.append(_Draft(a, (ok if cfg.ground_filter else cits)[:MAX_QUOTES], idx))
    brutos = sum(len(out.argumentos) for _, out, _ in results)

    # argumentos provisórios (para merge e verificador)
    prov: list[ArgumentoExtraido] = []
    for i, d in enumerate(drafts, start=1):
        spans = [(c.char_start, c.char_end) for c in d.citacoes if c.ancorada]
        scores = [c.score for c in d.citacoes if c.ancorada]
        conf, level = confidence(scores, None)
        prov.append(
            ArgumentoExtraido(
                id=f"p{i:02d}",
                **d.arg.model_dump(exclude={"citacoes"}),
                citacoes=d.citacoes,
                secao=section_of(sec, spans) or "fundamentacao",
                confianca=conf,
                confianca_nivel=level,
                sinais={"chunks": [d.chunk], "ancoragem": scores},
                run_id=run_id,
            )
        )

    n_merged = 0
    if cfg.merge and len(prov) > 1:
        vecs = None
        if cfg.merge_threshold < 1.0:
            vecs, rec = llm.embed([f"{a.titulo}. {a.resumo}" for a in prov])
            calls.append(rec)
        prov, n_merged = merge(prov, vecs, cfg.merge_threshold)

    n_ver_drop = 0
    if cfg.verifier and prov:
        checks, rec = verify(llm, cfg.verifier, prov, text)
        calls.append(rec)
        kept = []
        for a in prov:
            ck = checks.get(a.id)
            scores = [c.score for c in a.citacoes if c.ancorada]
            a.confianca, a.confianca_nivel = confidence(scores, ck)
            if ck is not None:
                a.sinais["verificador"] = ck.model_dump()
            if cfg.verifier_drop and ck is not None and ck.suporte == "nao_sustentado":
                n_ver_drop += 1
                continue
            kept.append(a)
        prov = kept

    for i, a in enumerate(prov, start=1):
        a.id = f"p{i:02d}"
        a.posicao = posicao_of(a.citacoes)
    # chunks rodam em paralelo; embeddings do merge e verificador vêm depois, em série
    lat_extract = max((r.latency_ms for _, _, r in results), default=0.0)
    lat_after = sum(r.latency_ms for r in calls[len(results) :])
    return Extraction(
        ruling_id=ruling_id,
        run_id=run_id,
        argumentos=prov,
        brutos=brutos,
        citacoes_brutas=n_quotes,
        citacoes_ancoradas=n_anchored,
        descartados_sem_ancora=len(dropped),
        fundidos=n_merged,
        descartados_verificador=n_ver_drop,
        n_chunks=len(jobs),
        chunks_falhos=len(failed),
        chunks_com_retry=retried,
        latency_ms=lat_extract + lat_after,
        cost_usd=sum(r.cost_usd or 0.0 for r in calls),
        calls=calls,
        sem_ancora=dropped,
        falhas=failed,
    )
