import re

from common.llm import CallRecord
from common.text import fix_null_escapes
from q2_arguments.bench import Par, Pareamento, aggregate, match_llm, prf, report, score_ruling
from q2_arguments.extract import (
    ArgumentoExtraido,
    ArgumentoLLM,
    Config,
    SaidaLLM,
    confidence,
    extract,
)
from q2_arguments.gold import GoldRuling
from q2_arguments.merge import merge
from q2_arguments.schema import Argumento, Citacao
from q2_arguments.segment import chunk_range, sections
from q2_arguments.verify import Checagem, Checagens

PARAS = [
    f"Parágrafo {i} da fundamentação trata da questão número {i}, com texto bastante para "
    f"ocupar alguns tokens e simular o raciocínio do juiz sobre o ponto {i}."
    for i in range(8)
]
TEXT = (
    "SENTENÇA\n\nVistos.\n\nA autora alega atraso de voo.\n\nÉ o relatório. Fundamento e decido."
    "\n\n" + "\n\n".join(PARAS) + "\n\nAnte o exposto, JULGO PROCEDENTE o pedido.\n\nP.R.I."
)


def _rec(model: str = "gpt-4.1-mini", kind: str = "chat") -> CallRecord:
    return CallRecord(kind, model, model, "v", 100, 20, 0.001, 50.0, False, "k")


class FakeLLM:
    def __init__(self, args_fn=None, checks=None, pairs=None):
        self.calls: list[dict] = []
        self.args_fn = args_fn or (lambda user: [])
        self.checks = checks or {}
        self.pairs = pairs or []

    def structured(self, schema, *, system, user, model, temperature=None, prompt_version=None):
        self.calls.append({"schema": schema.__name__, "system": system, "user": user})
        if schema is SaidaLLM:
            return SaidaLLM(argumentos=self.args_fn(user)), _rec(model)
        if schema is Checagens:
            ids = re.findall(r"^\[(p\d+)\]", user, re.M)
            itens = [
                Checagem(id=i, suporte=self.checks.get(i, "sustentado"), decisao_confere=True,
                         motivo="ok")
                for i in ids
            ]  # fmt: skip
            return Checagens(itens=itens), _rec(model)
        if schema is Pareamento:
            return Pareamento(pares=[Par(gold=g, extraido=e) for g, e in self.pairs]), _rec(model)
        raise AssertionError(schema)

    def embed(self, texts):
        n = len(texts)
        vecs = [[1.0 if j == i else 0.0 for j in range(n)] for i in range(n)]  # ortogonais
        return vecs, _rec("text-embedding-3-small", "embed")


def _arg(titulo: str, quotes: list[str], tipo: str = "merito") -> ArgumentoLLM:
    return ArgumentoLLM(
        titulo=titulo, resumo="r", tipo=tipo, tema="t", suscitado_por="reu",
        decisao="rejeitado", fundamento_legal=[], citacoes=quotes,
    )  # fmt: skip


# --------------------------------------------------------------------------- segmentação


def test_sections_find_markers():
    s = sections(TEXT)
    assert TEXT[s.fund_start :].startswith("Parágrafo 0")
    assert TEXT[s.disp_start :].startswith("Ante o exposto")
    assert s.found == (True, True)
    assert s.name_at(0) == "relatorio" and s.name_at(s.disp_start) == "dispositivo"


def test_sections_without_markers_is_all_fundamentacao():
    t = "Texto sem marcador algum.\n\nOutro parágrafo qualquer."
    s = sections(t)
    assert (s.fund_start, s.disp_start, s.found) == (0, len(t), (False, False))


def test_chunks_cover_all_paragraphs_with_overlap():
    s = sections(TEXT)
    chunks = chunk_range(TEXT, s.fund_start, s.disp_start, budget=90, overlap=40)
    assert len(chunks) > 1
    assert chunks[0].start == s.fund_start
    assert chunks[-1].end == TEXT.index("\n\nAnte o exposto")
    assert chunks[1].start < chunks[0].end  # sobreposição
    for p in PARAS:
        i = TEXT.index(p)
        assert any(c.start <= i and i + len(p) <= c.end for c in chunks)


# --------------------------------------------------------------------------- extração


def test_extract_grounds_quotes_and_drops_invented():
    llm = FakeLLM(lambda u: [_arg("Real", [PARAS[2][:50]]), _arg("Inventado", ["frase inventada"])])
    cfg = Config(key="t", mode="full", merge=False)
    e = extract(llm, cfg, "r1", TEXT)
    [a] = e.argumentos
    assert a.titulo == "Real" and e.descartados_sem_ancora == 1 and e.brutos == 2
    c = a.citacoes[0]
    assert TEXT[c.char_start : c.char_end] == PARAS[2][:50]
    assert a.secao == "fundamentacao"
    assert (a.confianca, a.confianca_nivel) == (0.8, "media")  # sem verificador: teto 0,8
    assert a.run_id == e.run_id == cfg.spec().run_id
    assert e.sem_ancora[0]["titulo"] == "Inventado"


def test_extract_without_ground_filter_keeps_unanchored_with_low_confidence():
    llm = FakeLLM(lambda u: [_arg("Real", [PARAS[2][:50]]), _arg("Inventado", ["frase inventada"])])
    e = extract(llm, Config(key="t", mode="full", merge=False, ground_filter=False), "r1", TEXT)
    assert [(a.titulo, a.confianca_nivel) for a in e.argumentos] == [
        ("Real", "media"),
        ("Inventado", "baixa"),
    ]


def test_chunk_mode_sends_context_and_runs_chunks_in_parallel():
    llm = FakeLLM()
    e = extract(llm, Config(key="c", mode="chunk", budget=90, overlap=0, merge=False), "r", TEXT)
    chats = [c for c in llm.calls if c["schema"] == "SaidaLLM"]
    assert e.n_chunks == len(chats) > 1
    assert all("TRECHO A ANALISAR" in c["user"] and "UM TRECHO" in c["system"] for c in chats)
    assert "CONTEXTO - início" in chats[0]["user"] and "CONTEXTO - dispositivo" in chats[0]["user"]
    assert e.latency_ms == 50.0  # paralelo: max dos chunks, não a soma


def test_truncate_mode_only_sees_the_beginning():
    llm = FakeLLM()
    extract(llm, Config(key="t", mode="truncate", budget=60), "r", TEXT)
    [call] = llm.calls
    assert "Parágrafo 0" not in call["user"] or "Ante o exposto" not in call["user"]


def test_overlapping_chunks_produce_one_argument_after_merge():
    quote = PARAS[3][:60]
    llm = FakeLLM(
        lambda u: [_arg("Q3", [quote])] if quote in u.split("TRECHO A ANALISAR")[1] else []
    )
    cfg = Config(key="c", mode="chunk", budget=90, overlap=60, merge_threshold=1.01)
    e = extract(llm, cfg, "r", TEXT)
    assert e.brutos >= 2 and e.fundidos == e.brutos - 1
    [a] = e.argumentos
    assert len(a.sinais["chunks"]) >= 2


def test_verifier_sets_confidence_and_can_drop():
    def args(u):
        return [_arg("A", [PARAS[1][:40]]), _arg("B", [PARAS[5][:40]])]

    cfg = Config(key="v", mode="full", merge=False, verifier="gpt-4.1-mini")
    e = extract(FakeLLM(args, checks={"p02": "nao_sustentado"}), cfg, "r", TEXT)
    assert [(a.confianca, a.confianca_nivel) for a in e.argumentos] == [
        (1.0, "alta"),
        (0.2, "baixa"),
    ]
    assert e.argumentos[1].sinais["verificador"]["suporte"] == "nao_sustentado"
    assert e.latency_ms == 100.0  # extração + verificador em série
    drop = cfg.model_copy(update={"verifier_drop": True})
    e = extract(FakeLLM(args, checks={"p02": "nao_sustentado"}), drop, "r", TEXT)
    assert [a.titulo for a in e.argumentos] == ["A"] and e.descartados_verificador == 1


def test_confidence_rules():
    assert confidence([100, 90], None) == (0.76, "media")
    ck = Checagem(id="x", suporte="sustentado", decisao_confere=False, motivo="")
    assert confidence([100], ck) == (0.7, "media")
    assert confidence([], None) == (0.0, "baixa")


def test_llm_null_escapes_are_fixed():
    assert fix_null_escapes("rela\x00e7\x00e3o") == "relação"
    llm = FakeLLM(lambda u: [_arg("Rela\x00e7\x00e3o", [PARAS[0][:30]])])
    e = extract(llm, Config(key="t", mode="full", merge=False), "r", TEXT)
    assert e.argumentos[0].titulo == "Relação"


def test_run_id_tracks_config_but_not_key():
    assert Config(key="a").spec().run_id == Config(key="b").spec().run_id
    assert Config(key="a").spec().run_id != Config(key="a", model="gpt-4.1").spec().run_id
    assert Config(key="a").spec().run_id != Config(key="a", prompt="basico").spec().run_id


# --------------------------------------------------------------------------- merge


def _ext(i: int, tipo: str, spans: list[tuple[int, int]], conf: float = 0.5):
    return ArgumentoExtraido(
        id=f"p{i}", titulo=f"t{i}", resumo="r", tipo=tipo, tema="t", suscitado_por="reu",
        decisao="rejeitado", confianca=conf, confianca_nivel="media", run_id="r",
        citacoes=[Citacao(texto="x", char_start=s, char_end=e, score=100) for s, e in spans],
        sinais={"chunks": [i]},
    )  # fmt: skip


def test_merge_by_quote_overlap_same_type_only():
    args = [
        _ext(1, "merito", [(0, 10)], 0.4),
        _ext(2, "merito", [(5, 20)], 0.9),
        _ext(3, "quantum", [(5, 20)]),
        _ext(4, "merito", [(100, 120)]),
    ]
    out, n = merge(args, None, 0.88)
    assert n == 1 and [a.id for a in out] == ["p2", "p3", "p4"]
    assert len(out[0].citacoes) == 1  # citação sobreposta não é repetida
    assert out[0].sinais["chunks"] == [1, 2] and out[0].sinais["fundidos"] == 2


def test_merge_by_embedding_threshold():
    args = [
        _ext(1, "merito", [(0, 5)]),
        _ext(2, "merito", [(50, 60)]),
        _ext(3, "merito", [(90, 99)]),
    ]
    vecs = [[1.0, 0.0], [0.99, 0.1], [0.0, 1.0]]
    assert merge(args, vecs, 0.88)[1] == 1
    assert merge(args, vecs, 1.01)[1] == 0


# --------------------------------------------------------------------------- benchmark


def _gold_arg(i: int, spans=((0, 10),)) -> Argumento:
    return Argumento(
        id=f"a{i:02d}", titulo=f"g{i}", resumo="r", tipo="merito", tema="t",
        suscitado_por="reu", decisao="rejeitado",
        citacoes=[Citacao(texto="x", char_start=s, char_end=e, score=100) for s, e in spans],
    )  # fmt: skip


def test_match_llm_forces_one_to_one_and_ignores_bad_ids():
    llm = FakeLLM(pairs=[(1, 1), (1, 2), (2, 9), (0, 2), (2, 2)])
    pairs = match_llm(llm, [_gold_arg(1), _gold_arg(2)], [_gold_arg(1), _gold_arg(2)])
    assert pairs == [(0, 0), (1, 1)]


def test_prf():
    p, r, f1 = prf([{"tp": 2, "n_pred": 4, "n_gold": 2}])
    assert (p, r) == (0.5, 1.0) and abs(f1 - 2 / 3) < 1e-9


def test_score_aggregate_report_smoke():
    s = TEXT.index(PARAS[2])
    gold = GoldRuling(
        ruling_id="r1", text_sha256="", resultado="procedente",
        argumentos=[_gold_arg(1, [(s, s + 40)]), _gold_arg(2, [(0, 5)])],
    )  # fmt: skip
    llm = FakeLLM(lambda u: [_arg("Real", [PARAS[2][:50]])], pairs=[(1, 1)])
    cfg = Config(key="chunk_mini", mode="full", merge=False)
    e = extract(llm, cfg, "r1", TEXT)
    item = {"meta": {"ruling_id": "r1", "faixa": "curto", "dominio": "aereo"}, "text": TEXT,
            "gold": gold}  # fmt: skip
    row = score_ruling(llm, item, e)
    assert (row["tp"], row["n_gold"], row["n_pred"]) == (1, 2, 1)
    assert row["fields"]["citacao_confere"] == 1
    agg = aggregate([row])
    assert agg["precision"] == 1.0 and agg["recall"] == 0.5
    results = {"chunk_mini": {"config": cfg.model_dump(), "extraction_run_id": e.run_id,
                              "prompt_version": cfg.prompt_version, **agg}}  # fmt: skip
    meta = {"gold_status": {"rascunho": 1}, "provisorio": True, "n_dev": 0, "n_test": 1,
            "merge_limiar": 0.88, "merge_grid": [], "gold_modelo": ["gpt-5"]}  # fmt: skip
    md = report(results, meta, cfg.spec())
    assert "PROVISÓRIO" in md and "`chunk_mini`" in md


def test_failed_chunk_is_retried_then_recorded_not_fatal():
    from common.llm import LLMError

    def args(user):
        if "parte 1 de" in user:
            raise LLMError("saída não confere com o schema SaidaLLM")
        return []

    llm = FakeLLM(args)
    e = extract(llm, Config(key="c", mode="chunk", budget=90, overlap=0, merge=False), "r", TEXT)
    assert e.n_chunks > 1 and e.chunks_falhos == 1
    assert len(e.falhas[0]["erros"]) == 2  # 1 tentativa + 1 retry
    assert len([c for c in llm.calls if "parte 1 de" in c["user"]]) == 2
