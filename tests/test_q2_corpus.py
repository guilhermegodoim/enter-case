from pathlib import Path

from q2_arguments.corpus import (
    Candidate,
    anonymize,
    clean,
    guess_result,
    header_parties,
    is_person_name,
    length_bucket,
    select,
)
from q2_arguments.scrape import Hit, parse_results

FIXTURE = Path(__file__).parent / "fixtures" / "cjpg_sample.html"


def test_parse_results_extracts_metadata_and_full_text():
    [h] = parse_results(FIXTURE.read_text("utf-8"), query_id="q", pagina=1)
    assert h.processo == "0000001-00.2026.8.26.0001"
    assert h.cd_documento == "00AAA0000-1--100"
    assert h.magistrado == "JUIZA FICTICIA"
    assert h.vara == "1ª Vara Cível"
    assert h.data_disponibilizacao == "2026-10-14"
    assert "Texto fictício da sentença" in h.texto  # <em> do destaque não quebra o texto
    assert "trecho" not in h.texto  # usa o inteiro teor, não o resumo


def test_clean_turns_lines_into_paragraphs():
    assert clean("  SENTENÇA\n\n\nRequerente:\tFulano\nVistos.\n") == (
        "SENTENÇA\n\nRequerente: Fulano\n\nVistos."
    )


def test_header_parties_keeps_people_and_drops_companies():
    text = (
        "Requerente: Maria José da Silva, registrada civilmente como Maria José da Silva\n\n"
        "Requerido: Banco Fictício S/A\n\n"
        "Requerido: JOÃO PEDRO SOUZA e outro\n\nVistos."
    )
    assert header_parties(text) == [
        ("autor", "Maria José da Silva"),
        ("reu", "JOÃO PEDRO SOUZA"),
    ]


def test_is_person_name():
    assert is_person_name("Ana Paula Lopes")
    assert not is_person_name("Gol Linhas Aéreas Inteligentes S.A")
    assert not is_person_name("Ana")
    assert not is_person_name("CLARO S/A")


def test_anonymize_replaces_name_variants_and_ids_keeps_judge():
    text = (
        "Requerente: Ana Paula Lopes Togni\n\nJuiz(a) de Direito: Dr(a). Vera Lucia Prado\n\n"
        "ANA PAULA LOPES TOGNI ajuizou ação. A autora Ana Togni, CPF 123.456.789-09, "
        "telefone (11) 98765-4321. A testemunha Carlos Mendes Rocha confirmou. Ana viajou."
    )
    out = anonymize(
        text,
        header_parties(text),
        ["Carlos Mendes Rocha", "Banco X S/A"],
        keep=["Vera Lucia Prado"],
    )
    assert "Togni" not in out.text and "Carlos" not in out.text
    assert out.text.count("[AUTOR]") == 3
    assert "[PESSOA_1] confirmou" in out.text
    assert "[CPF]" in out.text and "[TELEFONE]" in out.text
    assert "Vera Lucia Prado" in out.text  # magistrado mantido
    assert "Ana viajou" in out.text  # palavra isolada com < 4 letras não é trocada
    assert out.residuos == 0


def test_anonymize_single_words_inherit_label_with_place_and_public_guards():
    text = (
        "Requerente: Roberto Carlos Paulo\n\n"
        "ROBERTO CARLOS PAULO e MIGUEL AUGUSTO CORRÊA ajuizaram ação. O autor Roberto mora em "
        "São Paulo. Miguel viajou. Paulo desistiu. Conforme Carlos Roberto Gonçalves, há dano."
    )
    out = anonymize(
        text,
        header_parties(text),
        ["Miguel Augusto Corrêa", "Miguel"],
        keep=["Carlos Roberto Gonçalves"],
    )
    assert "[AUTOR] e [PESSOA_1] ajuizaram" in out.text  # "e" separa dois nomes
    assert "[PESSOA_1] viajou" in out.text  # prenome isolado herda o rótulo do nome completo
    assert "São Paulo" in out.text  # indicador de lugar protege
    assert "[AUTOR] desistiu" in out.text
    assert "O autor [AUTOR] mora" in out.text  # prenome isolado trocado mesmo sendo de nome público
    assert "Carlos Roberto Gonçalves" in out.text  # nome público mantido inteiro
    assert out.text.startswith("Requerente: [AUTOR]\n\n[AUTOR] e")  # não atravessa parágrafo
    # resíduos = palavras de nomes privados que ficaram: Paulo (lugar), Carlos e Roberto (público)
    assert out.residuos == 3


def test_anonymize_single_word_keep_protects_word_everywhere():
    text = "Requerente: Maria Santos Lima\n\nA autora Santos mora na comarca de Santos."
    out = anonymize(text, header_parties(text), [], keep=["SANTOS"])
    assert out.text.endswith("A autora Santos mora na comarca de Santos.")


def test_guess_result_order():
    assert guess_result("... JULGO PARCIALMENTE PROCEDENTE o pedido") == "parcial"
    assert guess_result("Ante o exposto, julgo improcedente") == "improcedente"
    assert guess_result("JULGO EXTINTO o processo") == "extinto"


def _cand(i: int, dom: str, bucket: str, res: str = "procedente", judge: str | None = None):
    hit = Hit(
        processo=f"{i:07d}-00.2026.8.26.0001",
        cd_documento="",
        classe="Procedimento Comum Cível",
        assunto="",
        magistrado=judge or f"J{i}",
        comarca="",
        foro="",
        vara="",
        data_disponibilizacao="2026-01-01",
        texto="",
        query_id="q",
        pagina=1,
    )
    return Candidate(hit, dom, "", 0, bucket, res)


def test_select_is_deterministic_balanced_and_caps_judges():
    cands = [_cand(i, ["aereo", "bancario"][i % 2], "curto") for i in range(20)]
    cands += [_cand(100 + i, "bancario", "muito_longo", judge="MESMO") for i in range(5)]
    cands += [_cand(200 + i, "aereo", "medio", res="extinto") for i in range(3)]
    a, b = select(cands), select(list(reversed(cands)))
    assert [c.hit.processo for c in a] == [c.hit.processo for c in b]
    curto = [c for c in a if c.bucket == "curto"]
    assert len(curto) == 6 and {c.dominio for c in curto} == {"aereo", "bancario"}
    assert sum(c.hit.magistrado == "MESMO" for c in a) == 2
    assert sum(c.resultado == "extinto" for c in a) == 2


def test_length_bucket():
    assert [length_bucket(t) for t in (10, 3000, 6500, 12000)] == [
        "curto",
        "medio",
        "longo",
        "muito_longo",
    ]
