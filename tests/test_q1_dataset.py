import json
import random
import re
from collections import Counter
from dataclasses import replace

import pytest

from common.text import normalize
from q1_similarity import spec
from q1_similarity.build import build, build_case, label_pair, make_version, weighted_distance
from q1_similarity.noise import reformat, scan
from q1_similarity.render import (
    DocSpec,
    base_contestacao,
    base_peticao,
    case_entities,
    fill,
    render,
    strip_marker,
)
from q1_similarity.templates import PLACEHOLDER

MARKER_LINE = re.compile(r"^(?:Página \d+ de \d+|\d+/\d+|fls\. \d+)$")


def _petition(tese_key="voo_atraso", firm_key="firm_a", seed=1) -> DocSpec:
    tese, firm = spec.TESES[tese_key], spec.FIRMS[firm_key]
    client = spec.clients_for(tese)[0]
    ents = case_entities(random.Random(seed), tese, client, firm)
    return DocSpec(
        "peticao_inicial",
        spec.family_key(tese.key, firm.key),
        tese.key,
        firm.key,
        client.key,
        ents["AUTOR_CPF"],
        base_peticao(tese, firm, 13),
        ents,
    )


@pytest.mark.parametrize("tese", list(spec.TESES))
@pytest.mark.parametrize("firm", list(spec.FIRMS))
def test_every_template_renders_without_leftover_placeholders(tese, firm):
    doc = _petition(tese, firm)
    text = render(doc)
    assert not PLACEHOLDER.search(text)
    assert doc.ents["AUTOR_NOME"] in text and doc.ents["VALOR_CAUSA"] in text
    ct = replace(doc, doc_type="contestacao", blocks=base_contestacao(spec.TESES[tese]))
    assert not PLACEHOLDER.search(render(ct))


def test_fill_rejects_unknown_placeholder():
    with pytest.raises(KeyError, match="FOO"):
        fill("valor {FOO}", {})


def test_strip_marker():
    assert strip_marker("III - DA PROVA") == "DA PROVA"
    assert strip_marker("b) A condenação") == "A condenação"
    assert strip_marker("2. Do mérito") == "Do mérito"
    assert strip_marker("Artigo 14 do CDC") == "Artigo 14 do CDC"


def test_pdf_reformat_keeps_every_word_in_order():
    text = render(_petition())
    out = reformat(text, random.Random(3), header=None)
    words = [w for ln in out.split("\n") if not MARKER_LINE.match(ln) for w in ln.split()]
    assert words == text.split()


def test_scan_noise_is_deterministic_and_bounded():
    text = normalize(render(_petition()))
    a = scan(text, random.Random(5), None, "forte")
    b = scan(text, random.Random(5), None, "forte")
    assert a == b and a != text
    orig = Counter(text.split())
    for level, floor in (("leve", 0.9), ("forte", 0.6)):
        noisy = Counter(normalize(scan(text, random.Random(5), None, level)).split())
        kept = sum((orig & noisy).values()) / sum(orig.values())
        assert kept > floor, level  # ruído altera a superfície sem destruir o texto


@pytest.mark.parametrize("level", ["leve", "media", "forte"])
def test_version_changes_blocks_but_keeps_family(level):
    parent = _petition()
    child = make_version(parent, level, random.Random(7))
    assert child.family == parent.family
    assert child.block_ids != parent.block_ids
    assert 0 < weighted_distance(parent, child) < 0.7


def test_labels_follow_block_provenance():
    rng = random.Random(11)
    for i, tese in enumerate(spec.TESES.values()):
        docs = build_case(f"t{i}", tese, random.Random(rng.getrandbits(63)), 13)
        for j, b in enumerate(docs):
            for a in docs[:j]:
                p = label_pair(a, b)
                same_fam = a.spec.family == b.spec.family
                same_blocks = a.spec.block_ids == b.spec.block_ids
                if p["label"] == "copia":
                    assert same_fam and same_blocks and p["block_distance"] == 0
                elif p["label"] == "versao":
                    assert same_fam and not same_blocks and p["block_distance"] > 0
                else:
                    assert not same_fam


def test_case_doc_order_parent_before_child():
    docs = build_case("t", spec.TESES["negativacao_indevida"], random.Random(2), 13)
    seen = set()
    for d in docs:
        assert d.parent is None or d.parent in seen
        seen.add(d.doc_id)
    assert docs[0].op == "original"


def test_build_is_reproducible(tmp_path):
    m1 = build(out=tmp_path / "a", seed=3, n_cases=6)
    m2 = build(out=tmp_path / "b", seed=3, n_cases=6)
    m3 = build(out=tmp_path / "c", seed=4, n_cases=6)
    assert m1["fingerprint"] == m2["fingerprint"] != m3["fingerprint"]
    pairs = [json.loads(ln) for ln in (tmp_path / "a" / "pairs.jsonl").read_text().splitlines()]
    assert {p["label"] for p in pairs} <= {"copia", "versao", "diferente"}
    assert all(p["a"].split("-d")[0] == p["b"].split("-d")[0] for p in pairs)  # mesmo caso
