"""Entidades fictícias e renderização de documentos a partir dos templates.

Um documento é um DocSpec: sequência de blocos lógicos (cada um com id estável) + valores das
entidades + estilo de formatação. O rótulo de um par (cópia/versão/diferente) é derivado dos
ids dos blocos, nunca do texto, então o gabarito é exato por construção.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from datetime import date, timedelta
from functools import cache

from faker import Faker
from num2words import num2words

from q1_similarity import spec
from q1_similarity.templates import (
    PLACEHOLDER,
    Boilerplate,
    Contestacao,
    Ementas,
    Family,
    paths,
)


def stable_seed(*parts: object) -> int:
    """Semente estável entre execuções (hash() do Python é aleatorizado por processo)."""
    digest = hashlib.sha256(":".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big")


_FAKER = Faker("pt_BR")


def _faker(seed: int) -> Faker:
    _FAKER.seed_instance(seed)
    return _FAKER


# --------------------------------------------------------------------------- templates


@cache
def boilerplate(firm: str, area: str) -> Boilerplate:
    return Boilerplate.model_validate_json(
        (paths()["boilerplate"] / f"{firm}__{area}.json").read_text()
    )


@cache
def family(tese: str, firm: str) -> Family:
    path = paths()["families"] / f"{spec.family_key(tese, firm)}.json"
    return Family.model_validate_json(path.read_text())


@cache
def ementas(tese: str) -> Ementas:
    return Ementas.model_validate_json((paths()["ementas"] / f"{tese}.json").read_text())


@cache
def contestacao(tese: str) -> Contestacao:
    return Contestacao.model_validate_json((paths()["contestacoes"] / f"{tese}.json").read_text())


# --------------------------------------------------------------------------- formatação de valores

_MONTHS = [
    "janeiro", "fevereiro", "março", "abril", "maio", "junho",
    "julho", "agosto", "setembro", "outubro", "novembro", "dezembro",
]  # fmt: skip


def brl(v: float) -> str:
    s = f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"


def extenso(v: float) -> str:
    return num2words(round(v, 2), lang="pt_BR", to="currency")


def data_curta(d: date) -> str:
    return d.strftime("%d/%m/%Y")


def data_longa(d: date) -> str:
    return f"{d.day} de {_MONTHS[d.month - 1]} de {d.year}"


# --------------------------------------------------------------------------- entidades

COMARCAS = [
    "São Paulo/SP", "Campinas/SP", "Guarulhos/SP", "Santos/SP",
    "Ribeirão Preto/SP", "São Bernardo do Campo/SP", "Sorocaba/SP",
]  # fmt: skip
AEROPORTOS = [
    "São Paulo (GRU)", "São Paulo (CGH)", "Rio de Janeiro (GIG)", "Brasília (BSB)",
    "Recife (REC)", "Salvador (SSA)", "Porto Alegre (POA)", "Belo Horizonte (CNF)",
    "Fortaleza (FOR)", "Curitiba (CWB)", "Manaus (MAO)", "Florianópolis (FLN)",
]  # fmt: skip
CARGOS = [
    "auxiliar administrativo", "operador de caixa", "agente de aeroporto",
    "analista de atendimento", "assistente comercial", "atendente de loja",
    "supervisor de vendas", "operador de telemarketing", "auxiliar de logística",
]  # fmt: skip
_CONN = ["da", "de", "", ""]


def _person_name(fk: Faker, rng: random.Random, female: bool) -> str:
    first = fk.first_name_female() if female else fk.first_name_male()
    middle = rng.choice(_CONN)
    parts = [first, fk.last_name()] + ([middle] if middle else []) + [fk.last_name()]
    return " ".join(parts)


def _oab(rng: random.Random) -> str:
    n = rng.randint(80_000, 499_999)
    return f"SP {n // 1000}.{n % 1000:03d}"


def _cep(fk: Faker) -> str:
    d = re.sub(r"\D", "", fk.postcode())
    return f"{d[:5]}-{d[5:]}"


def _address(fk: Faker) -> str:
    return (
        f"{fk.street_name()}, {fk.building_number()}, {fk.bairro()}, "
        f"{fk.city()}/{fk.estado_sigla()}, CEP {_cep(fk)}"
    )


@cache
def firm_lawyers(firm: str) -> tuple[tuple[str, str], ...]:
    rng = random.Random(stable_seed("advogados", firm))
    fk = _faker(stable_seed("advogados-faker", firm))
    return tuple((_person_name(fk, rng, rng.random() < 0.5), _oab(rng)) for _ in range(2))


@cache
def client_data(client: str) -> dict[str, str]:
    rng = random.Random(stable_seed("cliente", client))
    fk = _faker(stable_seed("cliente-faker", client))
    return {
        "REU_NOME": spec.CLIENTS[client].nome,
        "REU_CNPJ": fk.cnpj(),
        "REU_ENDERECO": _address(fk),
        "REU_ADVOGADO_NOME": _person_name(fk, rng, rng.random() < 0.5),
        "REU_ADVOGADO_OAB": _oab(rng),
    }


def _party(rng: random.Random, firm: spec.Firm) -> dict[str, str]:
    fk = _faker(rng.getrandbits(63))
    female = rng.random() < 0.5
    name = _person_name(fk, rng, female)
    if firm.key == "firm_a":  # estilo do escritório: nome da parte em caixa alta
        name = name.upper()
    nac = "brasileira" if female else "brasileiro"
    civil = rng.choice(
        ["casada", "solteira", "divorciada", "em união estável"]
        if female
        else ["casado", "solteiro", "divorciado", "em união estável"]
    )
    rg = re.sub(r"\D", "", fk.rg()).rjust(9, "0")
    return {
        "AUTOR_NOME": name,
        "AUTOR_QUALIFICACAO": f"{nac}, {civil}, {fk.job().lower()}",
        "AUTOR_CPF": fk.cpf(),
        "AUTOR_RG": f"{rg[:2]}.{rg[2:5]}.{rg[5:8]}-{rg[8]}",
        "AUTOR_ENDERECO": _address(fk),
        "AUTOR_EMAIL": fk.free_email(),
    }


def amounts(rng: random.Random, tese: spec.Tese) -> dict[str, str]:
    """Valores monetários (e extensos coerentes). Reamostrado em 'correção de dados'."""
    out: dict[str, float] = {}
    if tese.area == "consumidor":
        out["VALOR_DANO_MORAL"] = float(rng.choice([5000, 8000, 10000, 12000, 15000, 20000]))
        causa = out["VALOR_DANO_MORAL"]
        if tese.setor == "aereo":
            out["VALOR_DANO_MATERIAL"] = round(rng.uniform(250, 4800), 2)
            causa += out["VALOR_DANO_MATERIAL"]
        else:
            out["VALOR_DEBITO"] = round(rng.uniform(80, 3500), 2)
            causa += 2 * out["VALOR_DEBITO"]  # repetição em dobro
    else:
        out["SALARIO"] = round(rng.uniform(1600, 7800), 2)
        causa = round(out["SALARIO"] * rng.uniform(8, 30), 2)
    out["VALOR_CAUSA"] = causa
    res: dict[str, str] = {}
    for k, v in out.items():
        res[k] = brl(v)
        res[f"{k}_EXTENSO"] = extenso(v)
    return res


def _details(rng: random.Random, tese: spec.Tese, client: spec.Client) -> dict[str, str]:
    if tese.area == "trabalhista":
        adm = date(2016, 1, 4) + timedelta(days=rng.randint(0, 6 * 365))
        dem = min(adm + timedelta(days=rng.randint(365, 6 * 365)), date(2025, 6, 30))
        pet = dem + timedelta(days=rng.randint(30, 600))
        h1, h2 = rng.choice([7, 8, 9]), rng.choice([18, 19, 20])
        return {
            "CARGO": rng.choice(CARGOS),
            "DATA_ADMISSAO": data_curta(adm),
            "DATA_DEMISSAO": data_curta(dem),
            "JORNADA": f"{h1:02d}h às {h2:02d}h, de segunda a sábado",
            "HORAS_EXTRAS_SEMANAIS": str(rng.randint(6, 20)),
            "_pet_iso": pet.isoformat(),
        }
    fato = date(2023, 1, 2) + timedelta(days=rng.randint(0, 900))
    pet = fato + timedelta(days=rng.randint(20, 200))
    out = {"DATA_FATO": data_curta(fato), "_pet_iso": pet.isoformat()}
    if tese.setor == "aereo":
        origem, destino = rng.sample(AEROPORTOS, 2)
        out |= {
            "VOO_NUMERO": f"{client.iata} {rng.randint(1000, 9999)}",
            "VOO_ORIGEM": origem,
            "VOO_DESTINO": destino,
            "CODIGO_RESERVA": "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", k=6)),
            "HORAS_ATRASO": str(rng.randint(4, 14)),
        }
    else:
        out["CONTRATO_NUMERO"] = "-".join(f"{rng.randint(0, 9999):04d}" for _ in range(3))
    return out


def with_filing_date(ents: dict[str, str], d: date) -> dict[str, str]:
    return {**ents, "DATA_PETICAO": data_longa(d), "_doc_iso": d.isoformat()}


def doc_date(ents: dict[str, str]) -> date:
    return date.fromisoformat(ents["_doc_iso"])


def case_entities(
    rng: random.Random, tese: spec.Tese, client: spec.Client, firm: spec.Firm
) -> dict[str, str]:
    """Todos os valores de um caso (autor, réu, foro, fatos, valores)."""
    lawyer, oab = rng.choice(firm_lawyers(firm.key))
    e = _party(rng, firm) | client_data(client.key) | _details(rng, tese, client)
    e |= amounts(rng, tese)
    pet = date.fromisoformat(e["_pet_iso"])
    labor = tese.area == "trabalhista"
    j_tr = "5.02" if labor else "8.26"
    e |= {
        "NOME_ACAO": tese.nome_acao,
        "ADVOGADO_NOME": lawyer,
        "ADVOGADO_OAB": oab,
        "COMARCA": rng.choice(COMARCAS),
        "VARA": (
            f"{rng.randint(1, 90)}ª Vara do Trabalho"
            if labor
            else f"{rng.randint(1, 45)}ª Vara Cível"
        ),
        "PROCESSO_NUMERO": (
            f"{rng.randint(0, 9_999_999):07d}-{rng.randint(0, 99):02d}."
            f"{pet.year}.{j_tr}.{rng.randint(1, 999):04d}"
        ),
    }
    return with_filing_date(e, pet)


# --------------------------------------------------------------------------- blocos lógicos


@dataclass(frozen=True)
class Block:
    id: str
    kind: str  # title1 | title2 | para | item | line
    tpl: str
    section: str
    group: str = ""


@dataclass(frozen=True)
class Style:
    numbering: str  # roman | arabic | none
    items: str  # letter | number


@dataclass(frozen=True)
class DocSpec:
    doc_type: str  # peticao_inicial | contestacao
    family: str  # linhagem do template; mesma família = mesma origem
    tese: str
    firm: str | None
    client: str
    party: str  # identidade da parte autora
    blocks: tuple[Block, ...]
    ents: dict[str, str]
    style: Style = Style("none", "letter")

    @property
    def block_ids(self) -> list[str]:
        return [b.id for b in self.blocks]


FIRM_STYLE = {"firm_a": Style("roman", "letter"), "firm_b": Style("none", "letter")}
DEFENSE_STYLE = Style("arabic", "number")
_TITLES = {"fatos": "DOS FATOS", "direito": "DO DIREITO", "pedidos": "DOS PEDIDOS"}
_JURIS = {
    "firm_a": (
        "DA JURISPRUDÊNCIA",
        "Nesse sentido é a remansosa jurisprudência pátria, conforme se depreende dos "
        "seguintes julgados, cuja transcrição se faz necessária:",
    ),
    "firm_b": ("JURISPRUDÊNCIA", "Nesse sentido:"),
}
_PEDIDOS_INTRO = {
    "firm_a": "Ante todo o exposto, requer-se a Vossa Excelência:",
    "firm_b": "Diante do exposto, requer:",
}
_MARKER = re.compile(r"^\s*(?:[IVXLC]+|\d+(?:\.\d+)*|[a-zA-Z])\s*[-–—.)]\s+")
_INLINE_HEADING = re.compile(r"^(?:[IVXLC]+\s*[-–—.]\s*)?DAS? PARTES?\s*[.:-]\s*")


def strip_marker(text: str) -> str:
    """Remove numeração que o LLM pôs nos títulos/pedidos; o renderizador aplica a sua."""
    return _MARKER.sub("", text.strip(), count=1)


def base_peticao(tese: spec.Tese, firm: spec.Firm, seed: int) -> tuple[Block, ...]:
    bp = boilerplate(firm.key, tese.area)
    fam = family(tese.key, firm.key)
    fk = spec.family_key(tese.key, firm.key)
    b = f"bp:{firm.key}:{tese.area}"
    B: list[Block] = [
        Block(f"{b}:end", "line", bp.enderecamento, "cabecalho"),
        Block(f"{b}:qual", "para", _INLINE_HEADING.sub("", bp.qualificacao), "cabecalho"),
        Block(f"sec:{firm.key}:fatos", "title1", _TITLES["fatos"], "fatos"),
    ]
    B += [Block(f"fam:{fk}:f{i}", "para", p, "fatos") for i, p in enumerate(fam.especifico.fatos)]
    B.append(Block(f"{b}:grat:t", "title1", strip_marker(bp.gratuidade.titulo), "gratuidade"))
    B += [
        Block(f"{b}:grat:{i}", "para", p, "gratuidade")
        for i, p in enumerate(bp.gratuidade.paragrafos)
    ]
    B.append(Block(f"sec:{firm.key}:direito", "title1", _TITLES["direito"], "direito"))
    for s, sec in enumerate(bp.aplicabilidade):
        g = f"a{s}"
        B.append(Block(f"{b}:{g}:t", "title2", strip_marker(sec.titulo), "direito", g))
        B += [Block(f"{b}:{g}.{i}", "para", p, "direito", g) for i, p in enumerate(sec.paragrafos)]
    for s, sec in enumerate(fam.especifico.direito):
        g = f"d{s}"
        B.append(Block(f"fam:{fk}:{g}:t", "title2", strip_marker(sec.titulo), "direito", g))
        B += [
            Block(f"fam:{fk}:{g}.{i}", "para", p, "direito", g)
            for i, p in enumerate(sec.paragrafos)
        ]
    title, intro = _JURIS[firm.key]
    B += [
        Block(f"sec:{firm.key}:juris", "title1", title, "jurisprudencia"),
        Block(f"sec:{firm.key}:juris:intro", "line", intro, "jurisprudencia"),
    ]
    pool = ementas(tese.key).itens
    rng = random.Random(stable_seed(seed, "ementas", fk))  # escolha fixa por template
    for i in sorted(rng.sample(range(len(pool)), firm.n_ementas)):
        B += ementa_blocks(tese.key, i)
    B += [
        Block(f"sec:{firm.key}:pedidos", "title1", _TITLES["pedidos"], "pedidos"),
        Block(f"sec:{firm.key}:pedidos:intro", "line", _PEDIDOS_INTRO[firm.key], "pedidos"),
    ]
    B += [
        Block(f"fam:{fk}:p{i}", "item", strip_marker(p), "pedidos")
        for i, p in enumerate(fam.especifico.pedidos)
    ]
    B += [
        Block(f"{b}:pg{i}", "item", strip_marker(p), "pedidos")
        for i, p in enumerate(bp.pedidos_gerais)
    ]
    B += [Block(f"{b}:fecho{i}", "line", ln, "fecho") for i, ln in enumerate(bp.fecho)]
    return tuple(B)


def ementa_blocks(tese: str, i: int) -> list[Block]:
    em = ementas(tese).itens[i]
    return [
        Block(f"em:{tese}:{i}:ref", "line", em.referencia, "jurisprudencia"),
        Block(f"em:{tese}:{i}:txt", "para", em.texto, "jurisprudencia"),
    ]


def base_contestacao(tese: spec.Tese) -> tuple[Block, ...]:
    c = contestacao(tese.key)
    p = f"ct:{tese.key}"
    B: list[Block] = [
        Block(f"{p}:end", "line", c.enderecamento, "cabecalho"),
        Block(f"{p}:qual", "para", c.qualificacao, "cabecalho"),
        Block(f"{p}:sint:t", "title1", "SÍNTESE DA INICIAL", "sintese"),
    ]
    B += [Block(f"{p}:sint{i}", "para", t, "sintese") for i, t in enumerate(c.sintese)]
    for label, secs, key in (
        ("DAS PRELIMINARES", c.preliminares, "pre"),
        ("DO MÉRITO", c.merito, "mer"),
    ):
        B.append(Block(f"{p}:{key}:t", "title1", label, key))
        for s, sec in enumerate(secs):
            g = f"{key}{s}"
            B.append(Block(f"{p}:{g}:t", "title2", strip_marker(sec.titulo), key, g))
            B += [Block(f"{p}:{g}.{i}", "para", t, key, g) for i, t in enumerate(sec.paragrafos)]
    B.append(Block(f"{p}:ped:t", "title1", "DOS PEDIDOS", "pedidos"))
    B += [Block(f"{p}:ped{i}", "item", strip_marker(t), "pedidos") for i, t in enumerate(c.pedidos)]
    B += [Block(f"{p}:fecho{i}", "line", t, "fecho") for i, t in enumerate(c.fecho)]
    return tuple(B)


# --------------------------------------------------------------------------- renderização

_ROMAN = "I II III IV V VI VII VIII IX X XI XII XIII XIV XV".split()


def fill(tpl: str, ents: dict[str, str]) -> str:
    def sub(m: re.Match[str]) -> str:
        try:
            return ents[m.group(1)]
        except KeyError:
            raise KeyError(f"placeholder sem valor: {m.group(1)}") from None

    return PLACEHOLDER.sub(sub, tpl).replace("R$ R$", "R$")


def _title(text: str, n: int, numbering: str) -> str:
    if numbering == "roman":
        return f"{_ROMAN[n - 1]} - {text}"
    if numbering == "arabic":
        return f"{n}. {text}"
    return text


def _item(text: str, n: int, items: str) -> str:
    if items == "number":
        return f"{n}) {text}"
    label = ""
    while n:
        n, r = divmod(n - 1, 26)
        label = chr(ord("a") + r) + label
    return f"{label}) {text}"


def render(doc: DocSpec) -> str:
    out: list[str] = []
    top = item = 0
    section = None
    for b in doc.blocks:
        if b.section != section:
            section, item = b.section, 0
        text = fill(b.tpl, doc.ents)
        if b.kind == "title1":
            top += 1
            text = _title(text, top, doc.style.numbering)
        elif b.kind == "item":
            item += 1
            text = _item(text, item, doc.style.items)
        out.append(text)
    return "\n\n".join(out)
