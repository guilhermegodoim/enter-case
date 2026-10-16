"""Revisão de argumentos (Streamlit) dirigida por AppTest, contra um store temporário."""

import pytest
from streamlit.testing.v1 import AppTest

from common.config import ROOT
from q2_arguments.corpus import load_rulings, ruling_text
from q2_arguments.extract import ArgumentoLLM, Config, extract
from q2_arguments.store import Store
from tests.test_q2_extract import FakeLLM

APP = str(ROOT / "q2_arguments" / "review_app.py")


@pytest.fixture
def seeded(monkeypatch, tmp_path):
    db = tmp_path / "q2.sqlite"
    monkeypatch.setenv("ENTER_Q2_DB", str(db))
    monkeypatch.setenv("ENTER_GOLD_DIR", str(tmp_path / "gold"))
    meta = load_rulings()[0]
    rid = meta["ruling_id"]
    text = ruling_text(rid)
    i = text.index("Vistos")
    quotes = [text[i : i + 40], text[len(text) // 2 : len(text) // 2 + 60]]
    args = [
        ArgumentoLLM(titulo=f"Tese {n}", resumo="r", tipo="merito", tema="t",
                     suscitado_por="reu", decisao="rejeitado", fundamento_legal=[],
                     citacoes=[q])
        for n, q in enumerate(quotes, 1)
    ]  # fmt: skip
    cfg = Config(key="app", mode="full", merge=False)
    e = extract(FakeLLM(lambda u: args), cfg, rid, text)
    store = Store(db)
    store.save_extraction(meta, text, cfg, e)
    states = store.arguments(rid, e.run_id)
    store.close()
    return db, rid, e.run_id, states, quotes


def _app():
    return AppTest.from_file(APP, default_timeout=30).run()


def test_accept_reject_edit_persist_as_revisions(seeded):
    db, rid, run_id, states, quotes = seeded
    a, b = states
    at = _app()
    assert not at.exception
    at.text_input(key="reviewer").set_value("revisor-teste").run()
    at.button(key=f"{a.arg_uid}:1:ok").click().run()
    assert not at.exception
    at.text_input(key=f"{b.arg_uid}:1:motivo").set_value("não é questão decidida")
    at.button(key=f"{b.arg_uid}:1:rej").click().run()
    assert not at.exception
    store = Store(db)
    got = {s.local_id: s for s in store.arguments(rid, run_id)}
    assert got["p01"].status == "aceito" and got["p01"].reviewer == "revisor-teste"
    assert got["p02"].status == "rejeitado" and got["p02"].motivo == "não é questão decidida"

    at = _app()
    at.text_input(key="reviewer").set_value("revisor-teste").run()
    at.text_input(key=f"{a.arg_uid}:2:f:titulo").set_value("Tese corrigida")
    next(b for b in at.button if b.label == "💾 Salvar edição").click().run()
    assert not at.exception
    s = store.state(a.arg_uid)
    assert s.status == "editado" and s.arg.titulo == "Tese corrigida"
    assert [h.status for h in store.history(a.arg_uid)] == ["pendente", "aceito", "editado"]


def test_reject_without_reason_shows_error(seeded):
    db, rid, run_id, states, _ = seeded
    at = _app()
    at.text_input(key="reviewer").set_value("revisor-teste").run()
    at.button(key=f"{states[0].arg_uid}:1:rej").click().run()
    assert any("motivo" in e.value for e in at.error)
    assert Store(db).state(states[0].arg_uid).status == "pendente"
