"""Editor do gold da Q2 (Streamlit): `uv run q2 review`.

Esquerda: sentença com as citações destacadas por argumento. Direita: argumentos editáveis.
Salvar reancora todas as citações no texto; "Marcar revisado" exige 100% ancoradas.
"""

from __future__ import annotations

import html
import json

import streamlit as st

from common.text import sha256_text
from q2_arguments.corpus import load_rulings, ruling_text
from q2_arguments.gold import GoldRuling, load_gold, mark_reviewed, reanchor, save_gold
from q2_arguments.schema import (
    SCHEMA_VERSION,
    Argumento,
    Citacao,
    Decisao,
    Resultado,
    Secao,
    SuscitadoPor,
    TipoArgumento,
)

COLORS = ["#fde68a", "#bfdbfe", "#bbf7d0", "#fecaca", "#ddd6fe", "#fed7aa", "#a5f3fc", "#f5d0fe"]


def opts(lit) -> list[str]:
    return list(lit.__args__)


def highlight(text: str, g: GoldRuling) -> str:
    spans = []
    for i, a in enumerate(g.argumentos):
        for c in a.citacoes:
            if c.ancorada:
                spans.append((c.char_start, c.char_end, i, a.id))
    spans.sort()
    out, pos = [], 0
    for s, e, i, aid in spans:
        if s < pos:
            continue
        out.append(html.escape(text[pos:s]))
        color = COLORS[i % len(COLORS)]
        out.append(
            f'<mark style="background:{color};color:#111"><sup><b>{aid}</b></sup>'
            f"{html.escape(text[s:e])}</mark>"
        )
        pos = e
    out.append(html.escape(text[pos:]))
    return (
        '<div style="white-space:pre-wrap;font-family:Georgia,serif;font-size:0.95rem;'
        f'line-height:1.5">{"".join(out)}</div>'
    )


def next_id(g: GoldRuling) -> str:
    nums = [int(a.id[1:]) for a in g.argumentos if a.id[1:].isdigit()]
    return f"a{max(nums, default=0) + 1:02d}"


def main() -> None:
    st.set_page_config(page_title="Gold Q2", layout="wide")
    rulings = {r["ruling_id"]: r for r in load_rulings()}

    def label(rid: str) -> str:
        g = load_gold(rid)
        mark = {"revisado": "✅", "rascunho": "📝"}.get(g.status if g else "", "·")
        return f"{mark} {rid} ({rulings[rid]['faixa']})"

    with st.sidebar:
        st.header("Gold Q2")
        rid = st.selectbox("Sentença", list(rulings), format_func=label)
        reviewer = st.text_input("Revisor", key="reviewer")
        done = sum(1 for r in rulings if (g := load_gold(r)) and g.status == "revisado")
        st.metric("Revisadas", f"{done}/{len(rulings)}")
        meta = rulings[rid]
        st.caption(
            f"{meta['classe']} · {meta['assunto']}\n\n{meta['vara']} · {meta['comarca']}\n\n"
            f"{meta['n_tokens']} tokens · split {meta['split']}"
        )
        st.caption("Guia: data/gold/GUIDE.md")

    text = ruling_text(rid)
    g = load_gold(rid) or GoldRuling(ruling_id=rid, text_sha256=sha256_text(text))
    if g.text_sha256 != sha256_text(text):
        st.warning("O texto da sentença mudou desde o rascunho: salve para reancorar as citações.")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Sentença")
        with st.container(height=850):
            st.markdown(highlight(text, g), unsafe_allow_html=True)

    with right:
        st.subheader(f"Argumentos ({len(g.argumentos)}) · status: {g.status}")
        if g.rascunho:
            r = g.rascunho
            st.caption(
                f"Rascunho: {r.get('modelo')} · {r.get('prompt_version')} · "
                f"{r.get('citacoes_ancoradas')}/{r.get('citacoes')} citações ancoradas"
            )
        c1, c2 = st.columns(2)
        if c1.button("➕ Novo argumento"):
            g.argumentos.append(
                Argumento(
                    id=next_id(g),
                    titulo="",
                    resumo="",
                    tipo="merito",
                    tema="",
                    suscitado_por="reu",
                    decisao="rejeitado",
                    citacoes=[],
                )
            )
            g.status = "rascunho"
            save_gold(g)
            st.rerun()

        with st.form(f"form:{rid}"):
            res_opts = opts(Resultado)
            resultado = st.selectbox(
                "Resultado da sentença",
                res_opts,
                index=res_opts.index(g.resultado) if g.resultado in res_opts else 0,
                key=f"{rid}:resultado",
            )
            edited: list[Argumento] = []
            for a in g.argumentos:
                k = f"{rid}:{a.id}"
                bad = sum(not c.ancorada for c in a.citacoes)
                flag = (
                    f" ⚠️ {bad} não ancorada(s)" if bad else ("" if a.citacoes else " ⚠️ sem citação")
                )
                with st.expander(
                    f"{a.id} · {a.titulo or '(sem título)'}{flag}", expanded=bool(bad)
                ):
                    remove = st.checkbox("Excluir este argumento", key=f"{k}:del")
                    titulo = st.text_input("Título", a.titulo, key=f"{k}:titulo")
                    resumo = st.text_area("Resumo", a.resumo, key=f"{k}:resumo", height=90)
                    cols = st.columns(4)
                    tipo = cols[0].selectbox(
                        "Tipo",
                        opts(TipoArgumento),
                        opts(TipoArgumento).index(a.tipo),
                        key=f"{k}:tipo",
                    )
                    susc = cols[1].selectbox(
                        "Suscitado por",
                        opts(SuscitadoPor),
                        opts(SuscitadoPor).index(a.suscitado_por),
                        key=f"{k}:susc",
                    )
                    dec = cols[2].selectbox(
                        "Decisão", opts(Decisao), opts(Decisao).index(a.decisao), key=f"{k}:dec"
                    )
                    sec = cols[3].selectbox(
                        "Seção", opts(Secao), opts(Secao).index(a.secao), key=f"{k}:sec"
                    )
                    tema = st.text_input("Tema", a.tema, key=f"{k}:tema")
                    fund = st.text_area(
                        "Fundamento legal (um por linha)",
                        "\n".join(a.fundamento_legal),
                        key=f"{k}:fund",
                        height=70,
                    )
                    for c in a.citacoes:
                        if not c.ancorada:
                            st.error(f"Não encontrada no texto: “{c.texto[:160]}”")
                    # espaços colapsados: citação que atravessa parágrafo continua uma só;
                    # a ancoragem ignora diferenças de espaço
                    quotes = st.text_area(
                        "Citações literais (separe com uma linha em branco)",
                        "\n\n".join(" ".join(c.texto.split()) for c in a.citacoes),
                        key=f"{k}:cit",
                        height=150,
                    )
                    if not remove:
                        edited.append(
                            Argumento(
                                id=a.id,
                                titulo=titulo.strip(),
                                resumo=resumo.strip(),
                                tipo=tipo,
                                tema=tema.strip(),
                                suscitado_por=susc,
                                decisao=dec,
                                secao=sec,
                                fundamento_legal=[
                                    x.strip() for x in fund.splitlines() if x.strip()
                                ],
                                citacoes=[
                                    Citacao(texto=q.strip())
                                    for q in quotes.split("\n\n")
                                    if q.strip()
                                ],
                            )
                        )
            notas = st.text_area("Notas do revisor", g.notas, key=f"{rid}:notas", height=70)
            b1, b2 = st.columns(2)
            save = b1.form_submit_button("💾 Salvar (reancora)")
            approve = b2.form_submit_button("✅ Salvar e marcar revisado", type="primary")

        if save or approve:
            g.argumentos, g.resultado, g.notas = edited, resultado, notas
            g.schema_version = SCHEMA_VERSION
            g.status = "rascunho"
            reanchor(g, text)
            if approve:
                try:
                    mark_reviewed(g, reviewer)
                except ValueError as e:
                    save_gold(g)
                    st.error(f"Não marcado como revisado: {e}")
                    st.stop()
            save_gold(g)
            st.rerun()

        with st.expander("JSON"):
            st.code(json.dumps(g.model_dump(), indent=2, ensure_ascii=False), language="json")


main()
