"""Editor do gold (Streamlit) dirigido por AppTest, contra um diretório de gold temporário."""

import pytest
from streamlit.testing.v1 import AppTest

from common.config import ROOT
from q2_arguments.corpus import load_rulings, ruling_text
from q2_arguments.gold import GoldRuling, load_gold, save_gold
from q2_arguments.schema import Argumento, Citacao

APP = str(ROOT / "q2_arguments" / "gold_app.py")


@pytest.fixture
def first(monkeypatch, tmp_path):
    monkeypatch.setenv("ENTER_GOLD_DIR", str(tmp_path))
    rid = load_rulings()[0]["ruling_id"]
    text = ruling_text(rid)
    quote = text[text.index("Vistos") : text.index("Vistos") + 40]
    g = GoldRuling(
        ruling_id=rid,
        text_sha256="",
        resultado="procedente",
        argumentos=[
            Argumento(
                id="a01",
                titulo="Velho",
                resumo="r",
                tipo="merito",
                tema="t",
                suscitado_por="reu",
                decisao="rejeitado",
                citacoes=[Citacao(texto=quote)],
            ),
            Argumento(
                id="a02",
                titulo="Inventado",
                resumo="r",
                tipo="merito",
                tema="t",
                suscitado_por="reu",
                decisao="rejeitado",
                citacoes=[Citacao(texto="trecho que não existe na sentença de jeito nenhum")],
            ),
        ],
    )
    save_gold(g)
    return rid, quote


def _app():
    return AppTest.from_file(APP, default_timeout=30).run()


def test_edit_delete_and_approve(first):
    rid, quote = first
    at = _app()
    assert not at.exception
    at.text_input(key="reviewer").set_value("revisor-teste")
    at.text_input(key=f"{rid}:a01:titulo").set_value("Título corrigido")
    at.checkbox(key=f"{rid}:a02:del").check()
    next(b for b in at.button if "revisado" in b.label).click()
    at.run()
    assert not at.exception
    g = load_gold(rid)
    assert g.status == "revisado" and g.revisor == "revisor-teste"
    assert [a.titulo for a in g.argumentos] == ["Título corrigido"]
    c = g.argumentos[0].citacoes[0]
    assert ruling_text(rid)[c.char_start : c.char_end] == quote


def test_approve_blocked_by_unanchored_quote(first):
    rid, _ = first
    at = _app()
    at.text_input(key="reviewer").set_value("revisor-teste")
    next(b for b in at.button if "revisado" in b.label).click()
    at.run()
    assert any("não ancoradas" in e.value for e in at.error)
    g = load_gold(rid)
    assert g.status == "rascunho"
    assert g.argumentos[0].citacoes[0].ancorada  # salvar reancora mesmo sem aprovar
