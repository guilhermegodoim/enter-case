"""Página de revisão hospedada: dados gerados e importação do JSON exportado pelo navegador."""

import json

import pytest

from common.text import sha256_text
from q2_arguments.corpus import load_rulings, ruling_text
from q2_arguments.gold import GoldRuling, load_gold, save_gold
from q2_arguments.review_page import EXPORT_FORMAT, build_data, import_review


def _export(tmp_path, sentencas, revisor="amiga"):
    p = tmp_path / "rev.txt"
    data = {"formato": EXPORT_FORMAT, "gold_fingerprint": "x", "revisor": revisor,
            "exportado_em": "2026-10-16T00:00:00Z", "sentencas": sentencas}  # fmt: skip
    p.write_text(json.dumps(data))
    return p


def _sent(rid, status="revisado", quote=None, **kw):
    text = ruling_text(rid)
    q = quote if quote is not None else " ".join(text[100:180].split())
    return {
        "ruling_id": rid, "text_sha256": sha256_text(text), "status": status,
        "resultado": "procedente", "notas": "",
        "argumentos": [{
            "titulo": "Tese", "resumo": "r", "tipo": "merito", "tema": "t",
            "suscitado_por": "reu", "decisao": "rejeitado", "secao": "fundamentacao",
            "fundamento_legal": [], "citacoes": [{"texto": q, "char_start": 1, "char_end": 2}],
        }],
        **kw,
    }  # fmt: skip


@pytest.fixture
def rids():
    return [r["ruling_id"] for r in load_rulings()[:3]]


def test_build_data_has_every_ruling_test_split_first(tmp_path):
    out = build_data(tmp_path / "data.json")
    d = json.loads(out.read_text())
    assert d["formato"] == EXPORT_FORMAT and len(d["rulings"]) == len(load_rulings())
    splits = [r["split"] for r in d["rulings"]]
    assert splits == sorted(splits, key=lambda s: s != "test")
    assert all(x["text_sha256"] == sha256_text(x["text"]) for x in d["rulings"])
    assert sum(len(x["rascunho"]["argumentos"]) for x in d["rulings"]) > 0


def test_import_reviewed_reanchors_in_code_ignoring_browser_offsets(tmp_path, rids):
    p = _export(tmp_path, [_sent(rids[0])])
    out = import_review(p, root=tmp_path)
    assert out[rids[0]].startswith("revisada")
    g = load_gold(rids[0], tmp_path)
    assert g.status == "revisado" and g.revisor == "amiga"
    c = g.argumentos[0].citacoes[0]
    assert (c.char_start, c.char_end) != (1, 2)  # offsets do navegador descartados
    assert " ".join(ruling_text(rids[0])[c.char_start : c.char_end].split()) == c.texto
    assert g.rascunho["revisao"]["origem"] == "pagina-revisao"


def test_import_bad_quote_kept_as_draft(tmp_path, rids):
    p = _export(tmp_path, [_sent(rids[0], quote="frase que nunca esteve nesta sentença judicial")])
    assert "gravada como rascunho" in import_review(p, root=tmp_path)[rids[0]]
    assert load_gold(rids[0], tmp_path).status == "rascunho"


def test_import_in_progress_text_change_unknown_and_bad_enum(tmp_path, rids):
    bad_enum = _sent(rids[2])
    bad_enum["argumentos"][0]["tipo"] = "inventado"
    stale = _sent(rids[1])
    stale["text_sha256"] = "0" * 64
    p = _export(tmp_path, [_sent(rids[0], status="andamento"), stale, bad_enum,
                           {**_sent(rids[0]), "ruling_id": "tjsp-nao-existe"}])  # fmt: skip
    out = import_review(p, root=tmp_path)
    assert out[rids[0]].startswith("rascunho em andamento")
    assert "texto da sentença mudou" in out[rids[1]]
    assert "campo inválido" in out[rids[2]] and load_gold(rids[2], tmp_path) is None
    assert "fora do corpus" in out["tjsp-nao-existe"]


def test_import_never_overwrites_reviewed_without_force(tmp_path, rids):
    rid = rids[0]
    save_gold(GoldRuling(ruling_id=rid, text_sha256="", status="revisado", revisor="outra"),
              tmp_path)  # fmt: skip
    p = _export(tmp_path, [_sent(rid)])
    assert "já revisado por outra" in import_review(p, root=tmp_path)[rid]
    assert import_review(p, root=tmp_path, force=True)[rid].startswith("revisada")
    assert load_gold(rid, tmp_path).revisor == "amiga"


def test_import_rejects_wrong_format_and_missing_reviewer(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text(json.dumps({"formato": "outro", "revisor": "a", "sentencas": []}))
    with pytest.raises(ValueError, match="formato"):
        import_review(p, root=tmp_path)
    with pytest.raises(ValueError, match="revisor"):
        import_review(_export(tmp_path, [], revisor=" "), root=tmp_path)
