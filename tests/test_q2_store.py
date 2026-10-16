"""Store da Q2: append-only, revisões, reextração ao lado, diff, herança, upcast, gold."""

import json
import sqlite3

import pytest

from common.text import sha256_text
from q2_arguments.extract import (
    EXTRACT_SCHEMA_VERSION,
    ArgumentoLLM,
    Config,
    SaidaLLM,
    extract,
)
from q2_arguments.gold import GoldRuling, load_gold, save_gold
from q2_arguments.store import Conflict, Store, diff_states
from q2_arguments.versions import upcast
from tests.test_q2_extract import PARAS, TEXT, FakeLLM

RID = "tjsp-teste"
META = {"ruling_id": RID, "tribunal": "TJSP", "vara": "1ª Vara", "magistrado": "Juiz"}


def _arg(titulo, quote, tipo="merito", decisao="rejeitado"):
    return ArgumentoLLM(
        titulo=titulo, resumo="r", tipo=tipo, tema="t", suscitado_por="reu", decisao=decisao,
        fundamento_legal=[], citacoes=[quote],
    )  # fmt: skip


def _extract(args, **cfg):
    cfg = Config(key=cfg.pop("key", "c1"), mode="full", merge=False, **cfg)
    return cfg, extract(FakeLLM(lambda u: args), cfg, RID, TEXT)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "q2.sqlite")
    yield s
    s.close()


@pytest.fixture
def run1(store):
    cfg, e = _extract([_arg("Falha do serviço", PARAS[1][:60]), _arg("Dano moral", PARAS[3][:60])])
    store.save_extraction(META, TEXT, cfg, e)
    return cfg, e


def test_save_is_idempotent_and_records_run_provenance(store, run1):
    cfg, e = run1
    _, new = store.save_extraction(META, TEXT, cfg, e)
    assert not new
    run = store.run(e.run_id)
    assert run["prompt_version"] == cfg.prompt_version
    assert run["prompt_sha256"] == cfg.prompt_sha256
    assert run["schema_version"] == EXTRACT_SCHEMA_VERSION
    assert run["model"] == cfg.model and run["model_snapshot"] == cfg.model  # FakeLLM devolve
    states = store.arguments(RID, e.run_id)
    assert [s.status for s in states] == ["pendente", "pendente"]
    assert states[0].arg.posicao.char_start == TEXT.index(PARAS[1])


def test_tables_are_append_only(store, run1):
    for sql in ("UPDATE arguments SET payload = '{}'", "DELETE FROM argument_revisions",
                "UPDATE extraction_runs SET model = 'x'"):  # fmt: skip
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            with store.conn:
                store.conn.execute(sql)


def test_accept_edit_reject_history(store, run1):
    _, e = run1
    a, b = store.arguments(RID, e.run_id)
    with pytest.raises(ValueError, match="revisor"):
        store.accept(a.arg_uid, " ")
    with pytest.raises(ValueError, match="motivo"):
        store.reject(b.arg_uid, "ana", "")
    store.accept(a.arg_uid, "ana")
    s = store.edit(a.arg_uid, {"titulo": "Falha na prestação", "decisao": "acolhido"}, "ana", "x")
    assert s.status == "editado" and s.arg.titulo == "Falha na prestação"
    assert s.original.titulo == "Falha do serviço"  # original preservado
    store.reject(b.arg_uid, "ana", "princípio geral, não questão decidida")
    hist = store.history(a.arg_uid)
    assert [h.status for h in hist] == ["pendente", "aceito", "editado"]
    assert hist[-1].diff == {"titulo": ["Falha do serviço", "Falha na prestação"],
                             "decisao": ["rejeitado", "acolhido"]}  # fmt: skip
    assert store.review_counts(e.run_id) == {"editado": 1, "rejeitado": 1}


def test_edit_reanchors_quotes_and_refuses_invented(store, run1):
    _, e = run1
    a = store.arguments(RID, e.run_id)[0]
    q = " ".join(PARAS[5][10:70].split())
    s = store.edit(a.arg_uid, {"citacoes": [q]}, "ana")
    c = s.arg.citacoes[0]
    assert TEXT[c.char_start : c.char_end] == PARAS[5][10:70]
    assert s.arg.posicao.char_start == c.char_start
    with pytest.raises(ValueError, match="não encontrada"):
        store.edit(a.arg_uid, {"citacoes": ["frase que o juiz nunca escreveu aqui"]}, "ana")
    with pytest.raises(ValueError, match="ao menos uma"):
        store.edit(a.arg_uid, {"citacoes": []}, "ana")


def test_stale_revision_conflicts(store, run1):
    _, e = run1
    a = store.arguments(RID, e.run_id)[0]
    store.accept(a.arg_uid, "ana", expected_seq=a.seq)
    with pytest.raises(Conflict):
        store.reject(a.arg_uid, "bia", "não", expected_seq=a.seq)  # bia viu a revisão antiga


def test_add_missed_argument(store, run1):
    _, e = run1
    s = store.add(RID, e.run_id, {"titulo": "Ônus da prova", "tipo": "preliminar",
                                  "citacoes": [PARAS[6][:50]]}, "ana", "perdido")  # fmt: skip
    assert s.origin == "humano" and s.status == "adicionado" and s.local_id == "h01"
    with pytest.raises(ValueError):
        store.add(RID, e.run_id, {"titulo": "X", "citacoes": ["inventada de ponta a ponta"]}, "ana")


def test_new_run_is_stored_alongside_and_reviews_survive(store, run1):
    _, e1 = run1
    a, b = store.arguments(RID, e1.run_id)
    store.edit(a.arg_uid, {"resumo": "editado pelo humano"}, "ana")
    store.reject(b.arg_uid, "ana", "não é argumento")
    cfg2, e2 = _extract(
        [_arg("Falha do serviço", PARAS[1][:60]),
         _arg("Dano moral", PARAS[3][:60], decisao="acolhido"),
         _arg("Prescrição", PARAS[7][:60], tipo="prejudicial")],
        key="c2", prompt="guia2",
    )  # fmt: skip
    assert e2.run_id != e1.run_id
    store.save_extraction(META, TEXT, cfg2, e2)
    # run antiga intacta
    old = {s.local_id: s for s in store.arguments(RID, e1.run_id)}
    assert old["p01"].status == "editado" and old["p01"].arg.resumo == "editado pelo humano"
    assert old["p02"].status == "rejeitado"
    assert len(store.runs_for(RID)) == 2

    d = store.diff_runs(RID, e1.run_id, e2.run_id)
    assert d.counts() == {"pareados": 2, "equivalentes": 1, "so_a": 0,
                          "so_a_revisados_mantidos": 0, "so_b": 1}  # fmt: skip
    changed = next(p for p in d.pairs if p.a.local_id == "p02")
    assert set(changed.changes) == {"decisao"} and not changed.equivalent

    out = store.carry_over(RID, e1.run_id, e2.run_id, "ana")
    assert out["herdados"] == 1 and out["nao_equivalentes"] == 1
    new = {s.local_id: s for s in store.arguments(RID, e2.run_id)}
    assert new["p01"].status == "editado" and new["p01"].arg.resumo == "editado pelo humano"
    assert "herdado de" in new["p01"].motivo
    assert new["p02"].status == "pendente"  # decisão mudou: humano decide
    assert new["p03"].status == "pendente"  # novo


def test_diff_pairs_by_title_when_quotes_differ(store, run1):
    _, e1 = run1
    cfg2, e2 = _extract([_arg("Falha do serviço da ré", PARAS[2][:60])], prompt="guia2")
    store.save_extraction(META, TEXT, cfg2, e2)
    d = store.diff_runs(RID, e1.run_id, e2.run_id)
    assert [(p.a.local_id, p.how) for p in d.pairs] == [("p01", "titulo")]
    assert [x.local_id for x in d.only_a] == ["p02"]


def test_upcast_v1_payload_and_unknown_version():
    cits = [
        {"texto": "a", "char_start": 10, "char_end": 20},
        {"texto": "b", "char_start": 5, "char_end": 8},
        {"texto": "c", "char_start": None, "char_end": None},
    ]
    v1 = {"id": "p01", "citacoes": cits}
    assert upcast(v1, "q2-extracted-v1")["posicao"] == {"char_start": 5, "char_end": 20}
    assert upcast({"citacoes": []}, "q2-extracted-v1")["posicao"] is None
    assert upcast(v1, EXTRACT_SCHEMA_VERSION) is v1
    with pytest.raises(ValueError, match="sem upcaster"):
        upcast(v1, "q2-extracted-v0")


def test_v1_rows_are_read_through_upcast(store, run1):
    """Linha gravada em v1 (sem posicao) continua legível; o payload original fica intocado."""
    _, e = run1
    s = store.arguments(RID, e.run_id)[0]
    v1 = {k: v for k, v in s.arg.model_dump().items() if k != "posicao"}
    with store.conn:
        store.conn.execute(
            "INSERT INTO arguments VALUES ('old', 1, ?, ?, 'p99', 'llm', 'q2-extracted-v1', ?, "
            "'sistema', 'x')", (RID, e.run_id, json.dumps(v1)),
        )  # fmt: skip
        store.conn.execute(
            "INSERT INTO argument_revisions (arg_uid, seq, status, schema_version, payload, diff,"
            " reviewer, created_at) VALUES ('old', 1, 'pendente', 'q2-extracted-v1', ?, '{}',"
            " 'sistema', 'x')", (json.dumps(v1),),
        )  # fmt: skip
    old = next(x for x in store.arguments(RID, e.run_id) if x.arg_uid == "old")
    assert old.stored_schema == "q2-extracted-v1"
    assert old.arg.posicao == s.arg.posicao
    raw = store.conn.execute("SELECT payload FROM arguments WHERE arg_uid='old'").fetchone()[0]
    assert "posicao" not in json.loads(raw)
    store.accept("old", "ana")  # revisão nova já sai na versão atual
    assert store.history("old")[-1].schema_version == EXTRACT_SCHEMA_VERSION


def test_export_gold_requires_full_review_and_never_overwrites(store, run1, tmp_path):
    _, e = run1
    a, b = store.arguments(RID, e.run_id)
    root = tmp_path / "gold"
    with pytest.raises(ValueError, match="pendentes"):
        store.export_gold(RID, e.run_id, "ana", "procedente", root=root)
    store.accept(a.arg_uid, "ana")
    store.reject(b.arg_uid, "ana", "repetido")
    store.add(RID, e.run_id, {"titulo": "Perdido", "citacoes": [PARAS[4][:50]]}, "ana")
    g = store.export_gold(RID, e.run_id, "ana", "procedente", root=root)
    assert g.status == "revisado" and g.revisor == "ana"
    assert [x.titulo for x in g.argumentos] == ["Falha do serviço", "Perdido"]
    assert [x.id for x in g.argumentos] == ["a01", "a02"]
    assert g.rascunho["revisoes"] == {"aceito": 1, "editado": 0, "rejeitado": 1, "adicionado": 1}
    assert load_gold(RID, root).status == "revisado"
    with pytest.raises(ValueError, match="já revisado"):
        store.export_gold(RID, e.run_id, "ana", "procedente", root=root)


def test_ruling_text_change_is_refused(store, run1):
    with pytest.raises(ValueError, match="texto mudou"):
        store.upsert_ruling(META, TEXT + " alterado")


def test_diff_states_empty():
    assert diff_states(RID, "a", "b", [], []).counts()["pareados"] == 0


def test_same_config_different_key_is_same_run(store, run1):
    _, e1 = run1
    cfg2, e2 = _extract([], key="outro_nome")
    assert e2.run_id == e1.run_id
    _, new = store.save_extraction(META, TEXT, cfg2, e2)
    assert not new and len(store.arguments(RID, e1.run_id)) == 2


def test_save_gold_roundtrip_unaffected(tmp_path):
    g = GoldRuling(ruling_id=RID, text_sha256=sha256_text(TEXT))
    save_gold(g, tmp_path)
    assert load_gold(RID, tmp_path) == g


def test_saida_llm_schema_unchanged():
    assert set(SaidaLLM.model_fields) == {"argumentos"}
