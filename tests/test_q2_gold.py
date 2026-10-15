import pytest

import q2_arguments.gold as gold
from common.llm import CallRecord
from q2_arguments.gold import (
    ArgumentoRascunho,
    GoldRuling,
    RascunhoSentenca,
    draft_all,
    load_gold,
    mark_reviewed,
    reanchor,
    save_gold,
)
from q2_arguments.schema import Argumento, Citacao

TEXT = (
    "Vistos.\n\nRejeito a preliminar de ilegitimidade passiva, "
    "pois a ré integra a cadeia de fornecimento."
)


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def structured(self, schema, **kw):
        self.calls += 1
        arg = ArgumentoRascunho(
            titulo="Ilegitimidade passiva",
            resumo="Rejeitada.",
            tipo="preliminar",
            tema="ilegitimidade passiva",
            suscitado_por="reu",
            decisao="rejeitado",
            fundamento_legal=[],
            secao="fundamentacao",
            citacoes=["a ré integra a cadeia de fornecimento", "texto que não existe na sentença"],
        )
        rec = CallRecord("chat", "gpt-5", "gpt-5-x", "v", 10, 5, 0.0, 1.0, False, "k")
        return RascunhoSentenca(resultado="procedente", argumentos=[arg]), rec


@pytest.fixture
def corpus(monkeypatch, tmp_path):
    monkeypatch.setattr(gold, "ruling_text", lambda rid: TEXT)
    monkeypatch.setattr(
        gold, "load_rulings", lambda: [{"ruling_id": "r1", "split": "test", "faixa": "curto"}]
    )
    return tmp_path


def test_draft_anchors_quotes_in_code(corpus):
    g = gold.draft_one(FakeLLM(), "r1")
    [a] = g.argumentos
    assert a.id == "a01"
    ok, bad = a.citacoes
    assert TEXT[ok.char_start : ok.char_end] == "a ré integra a cadeia de fornecimento"
    assert not bad.ancorada
    assert g.rascunho["citacoes_ancoradas"] == 1 and g.status == "rascunho"


def test_draft_all_never_overwrites_reviewed(corpus):
    llm = FakeLLM()
    reviewed = GoldRuling(ruling_id="r1", text_sha256="x", status="revisado", resultado="outro")
    save_gold(reviewed, corpus)
    out = draft_all(llm, root=corpus, force=True, workers=1)
    assert llm.calls == 0 and out["r1"] == "mantido (revisado)"
    assert load_gold("r1", corpus).resultado == "outro"


def test_mark_reviewed_requires_anchored_quotes_reviewer_and_result():
    g = GoldRuling(
        ruling_id="r1",
        text_sha256="",
        resultado="procedente",
        argumentos=[
            Argumento(
                id="a01",
                titulo="t",
                resumo="r",
                tipo="merito",
                tema="x",
                suscitado_por="reu",
                decisao="rejeitado",
                citacoes=[Citacao(texto="não existe")],
            )
        ],
    )
    reanchor(g, TEXT)
    with pytest.raises(ValueError, match="não ancoradas"):
        mark_reviewed(g, "rev")
    g.argumentos[0].citacoes = [Citacao(texto="Rejeito a preliminar de ilegitimidade passiva")]
    reanchor(g, TEXT)
    with pytest.raises(ValueError, match="revisor"):
        mark_reviewed(g, " ")
    assert mark_reviewed(g, "rev").status == "revisado"
