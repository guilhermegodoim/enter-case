from common.text import mask_entities, normalize, paragraphs, skeleton

PETICAO_A = """EXCELENTÍSSIMO SENHOR DOUTOR JUIZ DE DIREITO DA 3ª VARA CÍVEL

MARIA APARECIDA DA SILVA, brasileira, casada, inscrita no CPF sob o nº 123.456.789-00,
residente na Rua das Flores, 100, CEP 01310-100, vem propor a presente AÇÃO em face de
AZUL LINHAS AÉREAS BRASILEIRAS S.A., inscrita no CNPJ sob o nº 09.296.295/0001-60.

Página 1 de 3

No dia 12/03/2024, a autora embarcaria no voo AD 4021, que foi cancelado sem aviso,
nos termos do art. 14 do CDC e da Lei 8.078/90. Requer indenização de R$ 10.000,00
(dez mil reais) a título de danos morais. Processo 0001234-56.2024.8.26.0100."""

PETICAO_B = (
    PETICAO_A.replace("MARIA APARECIDA DA SILVA", "JOSÉ CARLOS PEREIRA")
    .replace("123.456.789-00", "987.654.321-11")
    .replace("12/03/2024", "05/07/2024")
    .replace("AD 4021", "AD 2210")
    .replace("R$ 10.000,00", "R$ 8.500,00")
    .replace("(dez mil reais)", "(oito mil e quinhentos reais)")
)


def kinds(text):
    return {e.kind for e in mask_entities(normalize(text)).entities}


def test_normalize_removes_page_markers_and_joins_lines():
    norm = normalize(PETICAO_A)
    assert "Página 1" not in norm
    assert len(paragraphs(norm)) == 3
    assert "\n" not in paragraphs(norm)[1]


def test_normalize_dehyphenates_and_splits_ocr_paragraphs():
    raw = "O autor sofreu dano mo-\nral grave.\nA ré não respondeu."
    assert paragraphs(normalize(raw)) == ["O autor sofreu dano moral grave.", "A ré não respondeu."]


def test_mask_detects_core_entities():
    found = kinds(PETICAO_A)
    assert {"NOME", "CPF", "CNPJ", "CEP", "DATA", "VOO", "VALOR", "PROCESSO"} <= found


def test_mask_keeps_legal_citations():
    masked = mask_entities(normalize(PETICAO_A)).text
    assert "art. 14" in masked
    assert "Lei 8.078/90" in masked


def test_mask_offsets_point_to_source():
    norm = normalize(PETICAO_A)
    for e in mask_entities(norm).entities:
        assert norm[e.start : e.end] == e.value


def test_name_propagates_to_later_mentions():
    text = "JOÃO DOS SANTOS, brasileiro, solteiro. Ocorre que JOÃO DOS SANTOS comprou passagem."
    masked = mask_entities(text).text
    assert "JOÃO" not in masked
    assert masked.count("<NOME>") == 2


def test_skeleton_equal_for_entity_swapped_copy():
    assert skeleton(PETICAO_A) == skeleton(PETICAO_B)


def test_skeleton_differs_when_thesis_changes():
    other = PETICAO_A.replace("foi cancelado sem aviso", "atrasou mais de oito horas")
    assert skeleton(PETICAO_A) != skeleton(other)
