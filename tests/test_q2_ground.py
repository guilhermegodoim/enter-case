from q2_arguments.ground import MIN_SCORE, Source, anchor

TEXT = (
    "Vistos.\n\nA ré alega fortuito externo. Contudo, a reestruturação da malha aérea é "
    "risco inerente à atividade, configurando fortuito interno.\n\nJulgo procedente."
)


def _slice(c):
    return TEXT[c.char_start : c.char_end]


def test_exact_quote():
    q = "a reestruturação da malha aérea é risco inerente à atividade"
    c = anchor(q, TEXT)
    assert c.score == 100 and _slice(c) == q


def test_whitespace_case_and_accents_still_exact():
    c = anchor("A REESTRUTURACAO da   malha aerea\né risco inerente", TEXT)
    assert c.score == 100
    assert _slice(c) == "a reestruturação da malha aérea é risco inerente"


def test_fuzzy_quote_with_small_differences():
    c = anchor("a reestruturação da malha aérea é um risco inerente à atividade econômica", TEXT)
    assert c.ancorada and MIN_SCORE <= c.score < 100
    assert "malha aérea" in _slice(c)


def test_invented_quote_is_not_anchored():
    c = anchor("o dano moral é presumido em casos de atraso superior a quatro horas", TEXT)
    assert not c.ancorada and c.char_start is None


def test_quote_spanning_paragraphs():
    c = anchor("configurando fortuito interno. Julgo procedente", Source(TEXT))
    assert c.score == 100 and _slice(c).endswith("Julgo procedente")


def test_empty_quote():
    assert not anchor("   ", TEXT).ancorada
