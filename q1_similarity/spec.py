"""Especificação do dataset sintético da Q1: teses, escritórios, clientes e placeholders.

Tudo aqui é fictício. Nomes de empresas e escritórios foram inventados para o benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------- placeholders

COMMON = frozenset(
    {
        "NOME_ACAO",
        "AUTOR_NOME",
        "AUTOR_QUALIFICACAO",
        "AUTOR_CPF",
        "AUTOR_RG",
        "AUTOR_ENDERECO",
        "AUTOR_EMAIL",
        "REU_NOME",
        "REU_CNPJ",
        "REU_ENDERECO",
        "ADVOGADO_NOME",
        "ADVOGADO_OAB",
        "COMARCA",
        "VARA",
        "DATA_PETICAO",
        "VALOR_CAUSA",
        "VALOR_CAUSA_EXTENSO",
    }
)
CONSUMIDOR = frozenset({"DATA_FATO", "VALOR_DANO_MORAL", "VALOR_DANO_MORAL_EXTENSO"})
VOO = frozenset(
    {
        "VOO_NUMERO",
        "VOO_ORIGEM",
        "VOO_DESTINO",
        "CODIGO_RESERVA",
        "HORAS_ATRASO",
        "VALOR_DANO_MATERIAL",
        "VALOR_DANO_MATERIAL_EXTENSO",
    }
)
BANCO = frozenset({"CONTRATO_NUMERO", "VALOR_DEBITO", "VALOR_DEBITO_EXTENSO"})
TRABALHO = frozenset(
    {
        "CARGO",
        "SALARIO",
        "SALARIO_EXTENSO",
        "DATA_ADMISSAO",
        "DATA_DEMISSAO",
        "JORNADA",
        "HORAS_EXTRAS_SEMANAIS",
    }
)
# só a contestação usa (petição inicial ainda não tem número de processo)
DEFESA = frozenset({"PROCESSO_NUMERO", "REU_ADVOGADO_NOME", "REU_ADVOGADO_OAB"})

AREA_PLACEHOLDERS = {"consumidor": COMMON | CONSUMIDOR, "trabalhista": COMMON}


# --------------------------------------------------------------------------- teses


@dataclass(frozen=True)
class Tese:
    key: str
    area: str  # "consumidor" | "trabalhista"
    setor: str  # setor do réu: "aereo" | "banco" | "qualquer"
    nome_acao: str
    descricao: str
    extra_placeholders: frozenset[str]
    vizinha: str  # tese mais próxima (mesma área/setor), usada como negativo difícil

    @property
    def placeholders(self) -> frozenset[str]:
        return AREA_PLACEHOLDERS[self.area] | self.extra_placeholders

    @property
    def defense_placeholders(self) -> frozenset[str]:
        return self.placeholders | DEFESA


TESES: dict[str, Tese] = {
    t.key: t
    for t in [
        Tese(
            "voo_atraso",
            "consumidor",
            "aereo",
            "AÇÃO DE INDENIZAÇÃO POR DANOS MORAIS E MATERIAIS",
            "Atraso de voo de {HORAS_ATRASO} horas na chegada ao destino, sem assistência "
            "material adequada (Resolução ANAC 400/2016), com perda de compromisso e gastos "
            "extras. Responsabilidade objetiva do transportador (CDC art. 14, CC art. 737). "
            "Pede danos materiais e morais.",
            VOO,
            "voo_cancelamento",
        ),
        Tese(
            "voo_cancelamento",
            "consumidor",
            "aereo",
            "AÇÃO DE INDENIZAÇÃO POR DANOS MORAIS E MATERIAIS",
            "Cancelamento unilateral do voo sem aviso prévio mínimo de 72 horas, reacomodação "
            "apenas no dia seguinte, sem hospedagem nem alimentação, e recusa de reembolso "
            "integral (Resolução ANAC 400/2016, arts. 12 e 21). Pede danos materiais e morais.",
            VOO,
            "voo_atraso",
        ),
        Tese(
            "cobranca_indevida",
            "consumidor",
            "banco",
            "AÇÃO DECLARATÓRIA DE INEXISTÊNCIA DE DÉBITO C/C REPETIÇÃO DE INDÉBITO E "
            "INDENIZAÇÃO POR DANOS MORAIS",
            "Descontos mensais em conta corrente por seguro/pacote de serviços nunca "
            "contratado. Pede declaração de inexistência do débito, restituição em dobro "
            "(CDC art. 42, parágrafo único) e danos morais. Falha no dever de informação.",
            BANCO,
            "negativacao_indevida",
        ),
        Tese(
            "negativacao_indevida",
            "consumidor",
            "banco",
            "AÇÃO DECLARATÓRIA DE INEXISTÊNCIA DE DÉBITO C/C INDENIZAÇÃO POR DANOS MORAIS "
            "COM PEDIDO DE TUTELA DE URGÊNCIA",
            "Inscrição do nome da parte autora em cadastro de inadimplentes por dívida já "
            "quitada. Dano moral in re ipsa, inaplicabilidade da Súmula 385 do STJ (não há "
            "inscrições anteriores), tutela de urgência para exclusão imediata (CPC art. 300).",
            BANCO,
            "cobranca_indevida",
        ),
        Tese(
            "horas_extras",
            "trabalhista",
            "qualquer",
            "RECLAMAÇÃO TRABALHISTA",
            "Jornada habitual de {JORNADA}, com cerca de {HORAS_EXTRAS_SEMANAIS} horas extras "
            "por semana não pagas, supressão do intervalo intrajornada (CLT art. 71, § 4º) e "
            "controle de ponto britânico. Pede horas extras com adicional de 50% e reflexos.",
            TRABALHO,
            "verbas_rescisorias",
        ),
        Tese(
            "verbas_rescisorias",
            "trabalhista",
            "qualquer",
            "RECLAMAÇÃO TRABALHISTA",
            "Dispensa sem justa causa sem pagamento das verbas rescisórias no prazo legal: "
            "saldo de salário, aviso prévio indenizado, férias + 1/3, 13º proporcional, FGTS "
            "+ 40%. Pede multas dos arts. 467 e 477, § 8º, da CLT e entrega das guias.",
            TRABALHO,
            "horas_extras",
        ),
    ]
}


# --------------------------------------------------------------------------- escritórios


@dataclass(frozen=True)
class Firm:
    key: str
    nome: str
    estilo: str
    n_ementas: int  # quantas ementas colaciona por petição (estilo do escritório)


FIRMS: dict[str, Firm] = {
    f.key: f
    for f in [
        Firm(
            "firm_a",
            "Albuquerque Mendes & Tavares Advogados Associados",
            "Formal e prolixo. Parágrafos longos (120 a 200 palavras), períodos encadeados, "
            "expressões latinas (data venia, in casu, ab initio), cita doutrina pelo nome do "
            "autor, títulos numerados em romanos (I - DOS FATOS), adjetivação intensa.",
            5,
        ),
        Firm(
            "firm_b",
            "Rocha Lima Advocacia",
            "Direto e moderno. Parágrafos curtos (40 a 90 palavras), frases objetivas, sem "
            "latim, poucos adjetivos, títulos simples em caixa alta (DOS FATOS), pedidos em "
            "itens com letras.",
            2,
        ),
    ]
}

# escritório que defende os clientes nas contestações
DEFENSE_FIRM = (
    "Contestação escrita pelo jurídico externo da empresa ré: técnica, impessoal, "
    "parágrafos médios, preliminares antes do mérito, pedidos subsidiários."
)


# --------------------------------------------------------------------------- clientes (réus)


@dataclass(frozen=True)
class Client:
    key: str
    nome: str
    setor: str  # "aereo" | "banco" | "varejo"
    iata: str | None = None


CLIENTS: dict[str, Client] = {
    c.key: c
    for c in [
        Client("aerovia", "AEROVIA LINHAS AÉREAS S.A.", "aereo", "AV"),
        Client("condor", "CONDOR AZUL TRANSPORTES AÉREOS S.A.", "aereo", "CZ"),
        Client("paranoa", "BANCO PARANOÁ DIGITAL S.A.", "banco"),
        Client("taquari", "TAQUARI PAGAMENTOS S.A.", "banco"),
        Client("varanda", "LOJAS VARANDA COMÉRCIO VAREJISTA LTDA.", "varejo"),
    ]
}


def clients_for(tese: Tese) -> list[Client]:
    if tese.setor == "qualquer":
        return list(CLIENTS.values())
    return [c for c in CLIENTS.values() if c.setor == tese.setor]


def family_key(tese: str, firm: str) -> str:
    return f"{tese}__{firm}"
