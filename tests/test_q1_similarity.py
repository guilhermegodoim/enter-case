import random
from dataclasses import replace

import numpy as np
import pytest

from q1_similarity import spec
from q1_similarity.build import make_version
from q1_similarity.classify import LogReg, Rule, fit_prefilter
from q1_similarity.features import (
    align,
    boilerplate_shingles,
    containment,
    embedding_scores,
    jaccard,
    pair_features,
    shingles,
    signature,
)
from q1_similarity.judge import Judgement, render_prompt
from q1_similarity.noise import reformat, scan
from q1_similarity.pipeline import CaseIndex, Config
from q1_similarity.preprocess import prepare, strip_layout
from q1_similarity.render import DocSpec, base_peticao, case_entities, render


def _doc(tese="voo_atraso", firm="firm_b", seed=1) -> DocSpec:
    t, f = spec.TESES[tese], spec.FIRMS[firm]
    c = spec.clients_for(t)[0]
    ents = case_entities(random.Random(seed), t, c, f)
    return DocSpec(
        "peticao_inicial",
        spec.family_key(t.key, f.key),
        t.key,
        f.key,
        c.key,
        ents["AUTOR_CPF"],
        base_peticao(t, f, 13),
        ents,
    )


@pytest.fixture(scope="module")
def texts():
    base = _doc()
    other_ents = _doc(seed=2).ents
    return {
        "orig": render(base),
        "pdf": reformat(render(base), random.Random(1), header="Rocha Lima Advocacia"),
        "scan": scan(render(base), random.Random(2), "Rocha Lima Advocacia", "media"),
        "swap": render(replace(base, ents=other_ents)),
        "version": render(make_version(base, "media", random.Random(3))),
        "neighbor": render(_doc("voo_cancelamento")),
    }


def sig(text):
    return signature(prepare(text))


# --------------------------------------------------------------------------- preprocess


def test_strip_layout_removes_page_markers_and_running_header():
    lines = []
    for n in range(1, 5):
        if n > 1:
            lines += ["Rocha Lima Advocacia", ""]
        lines += [f"a) pedido curto número {n};", "", f"Página {n} de 4", ""]
    out = strip_layout("\n".join(lines))
    assert "Rocha Lima" not in out and "Página" not in out
    # linhas curtas parecidas do conteúdo não são confundidas com cabeçalho
    assert all(f"pedido curto número {n}" in out for n in range(1, 5))


def test_strip_layout_heals_paragraph_split_by_page_break():
    raw = "O autor comprou a\n\nPágina 1 de 2\n\npassagem aérea.\n\nPágina 2 de 2"
    assert prepare(raw).norm == "O autor comprou a passagem aérea."


def test_token_offsets_point_to_normalized_text(texts):
    p = prepare(texts["orig"])
    for k in range(0, len(p.tokens), 37):
        frag = p.norm[p.starts[k] : p.ends[k]]
        assert frag.strip()
        if not p.tokens[k].startswith("<"):
            assert frag.lower().isalnum() or "_" in frag


def test_skeleton_hash_ignores_entities_and_layout(texts):
    a, pdf, swap = (prepare(texts[k]) for k in ("orig", "pdf", "swap"))
    assert a.norm_hash == pdf.norm_hash  # reflow + paginação desfeitos por completo
    # foro/vara não são mascarados: troca de dados fica para o alinhamento, não para o hash
    assert prepare(texts["version"]).skeleton_hash != a.skeleton_hash
    assert swap.tokens.count("<nome>") >= 1


# --------------------------------------------------------------------------- features


def test_set_metrics():
    a, b = {1, 2, 3, 4}, {3, 4, 5}
    assert jaccard(a, b) == pytest.approx(2 / 5)
    assert containment(a, b) == pytest.approx(2 / 3)
    assert shingles(["a", "b"]) and len(shingles(list("abcdefg"))) == 3


def test_alignment_regions_are_traceable(texts):
    a, v = prepare(texts["orig"]), prepare(texts["version"])
    _, regs = align(a, v)
    content = [r for r in regs if r.kind == "conteudo"]
    assert content
    for r in content:
        if r.b_chr:
            assert v.norm[slice(*r.b_chr)] == v.norm[v.starts[r.b_tok[0]] : v.ends[r.b_tok[1] - 1]]


def test_copy_has_no_content_change_but_version_does(texts):
    o = sig(texts["orig"])
    f_scan, _ = pair_features(o, sig(texts["scan"]))
    f_swap, _ = pair_features(o, sig(texts["swap"]))
    f_ver, _ = pair_features(o, sig(texts["version"]))
    f_nb, _ = pair_features(o, sig(texts["neighbor"]))
    assert f_scan.content_ratio < 0.02 and f_swap.content_ratio < 0.02
    assert f_ver.content_ratio > 0.03
    assert f_scan.align_cov > 0.9 and f_ver.align_cov > 0.7
    assert f_nb.align_cov < 0.6  # mesma ré, mesmo boilerplate, tese vizinha


def test_boilerplate_counted_per_case_not_per_doc():
    common, rare = {1}, {2}
    cases = {"c1": [common | rare, common | rare], "c2": [common], "c3": [common]}
    assert boilerplate_shingles(cases, max_df=0.5) == frozenset({1})


def test_embedding_scores_identity():
    e = np.random.default_rng(0).normal(size=(4, 8))
    s = embedding_scores(e, e)
    assert s["emb_cos"] == pytest.approx(1.0) and s["emb_chunk_cov"] == 1.0


# --------------------------------------------------------------------------- classify


def _row(label, sim, change, exact=False):
    return {"label": label, "align_cov": sim, "content_ratio": change, "exact": exact,
            "skeleton": False}  # fmt: skip


def test_rule_fit_separates_three_classes():
    rows = [_row("diferente", 0.2 + i / 100, 0.6) for i in range(10)]
    rows += [_row("copia", 0.95, 0.001 * i) for i in range(10)]
    rows += [_row("versao", 0.85, 0.05 + 0.01 * i) for i in range(10)]
    r = Rule("t", "align_cov", "content_ratio", gray_factor=2.0).fit(rows)
    assert 0.29 < r.t_diff < 0.85 and 0.009 <= r.t_copy < 0.05
    assert r.predict(_row("?", 0.1, 0.0))[0] == "diferente"
    assert r.predict(_row("?", 0.95, 0.0))[0] == "copia"
    assert r.predict(_row("?", 0.9, 0.2)) == ("versao", False)
    assert r.predict(_row("?", 0.9, r.t_copy * 1.5))[1] is True  # zona cinzenta
    assert r.predict(_row("?", 0.0, 1.0, exact=True)) == ("copia", False)


def test_logreg_and_prefilter():
    rows = [_row("diferente", 0.1, 0.9)] * 5 + [_row("copia", 0.99, 0.0)] * 5
    rows += [_row("versao", 0.8, 0.2)] * 5
    m = LogReg("lr", ("align_cov", "content_ratio")).fit(rows)
    assert m.predict(_row("?", 0.99, 0.0))[0] == "copia"
    pre = [{"label": "copia", "minhash": 0.6}, {"label": "diferente", "minhash": 0.01}]
    assert fit_prefilter(pre) == pytest.approx(0.3)


# --------------------------------------------------------------------------- judge + pipeline


def test_judge_prompt_contains_only_diff_regions(texts):
    a, v = prepare(texts["orig"]), prepare(texts["version"])
    _, regs = align(a, v)
    prompt = render_prompt(a, v, [r for r in regs if r.kind == "conteudo"][:3])
    assert "⟦" in prompt and len(prompt) < len(v.norm) / 2


def test_pipeline_links_family_and_never_drops_documents(texts, monkeypatch):
    rule = Rule("r", "align_cov", "content_ratio", t_diff=0.65, t_copy=0.012, gray_factor=2.0)
    idx = CaseIndex(Config(rule, prefilter=0.05))
    d0 = idx.ingest("d0", texts["orig"])
    d1 = idx.ingest("d1", texts["pdf"])
    d2 = idx.ingest("d2", texts["version"])
    d3 = idx.ingest("d3", texts["neighbor"])
    assert (d0.action, d0.canonical) == ("novo", "d0")
    assert (d1.action, d1.match, d1.pairs[0].stage) == ("copia", "d0", "hash")
    assert (d2.action, d2.canonical) == ("versao", "d0") and d2.diff()
    assert d3.action == "novo" and d3.canonical == "d3"
    assert set(idx.sigs) == {"d0", "d1", "d2", "d3"}  # nada é descartado


def test_pipeline_gray_zone_uses_judge_verdict(texts, monkeypatch):
    calls = []

    def fake_judge(llm, a, b, regions, model):
        calls.append(len(regions))
        return Judgement("copia", [], None, None)

    monkeypatch.setattr("q1_similarity.pipeline.judge", fake_judge)
    rule = Rule("r", "align_cov", "content_ratio", t_diff=0.65, t_copy=1.0, gray_factor=1e6)
    idx = CaseIndex(Config(rule, use_llm=True))
    idx.ingest("d0", texts["orig"])
    d = idx.ingest("d1", texts["swap"])
    assert calls and d.action == "copia" and d.pairs[0].stage == "llm" and d.needs_review


def test_review_flag_only_when_chosen_match_is_gray(texts):
    rule = Rule("r", "align_cov", "content_ratio", t_diff=0.65, t_copy=0.012, gray_factor=2.0)
    idx = CaseIndex(Config(rule))
    idx.ingest("d0", texts["orig"])
    idx.ingest("d1", texts["version"])
    d = idx.ingest("d2", texts["pdf"])  # cópia certa por hash de d0
    assert d.action == "copia" and d.match == "d0" and not d.needs_review
