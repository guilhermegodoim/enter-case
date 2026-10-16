"""Revisão dos argumentos extraídos (Streamlit): `uv run q2 review-args`.

Lê e grava no store (SQLite). Cada ação vira uma revisão append-only com revisor, motivo e
diff; nada é sobrescrito. Esquerda: sentença com as citações destacadas. Direita: argumentos
da run escolhida com aceitar / editar / rejeitar, histórico, argumento perdido (adicionar),
diff contra outra run da mesma sentença e exportação para o gold.
"""

from __future__ import annotations

import html
import json

import streamlit as st

from q2_arguments.extract import EXTRACT_SCHEMA_VERSION
from q2_arguments.gold import load_gold
from q2_arguments.schema import Decisao, Resultado, Secao, SuscitadoPor, TipoArgumento
from q2_arguments.store import ArgState, Conflict, Store

COLORS = ["#fde68a", "#bfdbfe", "#bbf7d0", "#fecaca", "#ddd6fe", "#fed7aa", "#a5f3fc", "#f5d0fe"]
BADGE = {"pendente": "⏳", "aceito": "✅", "editado": "✏️", "rejeitado": "❌", "adicionado": "➕"}


def opts(lit) -> list[str]:
    return list(lit.__args__)


def highlight(text: str, states: list[ArgState], focus: str | None) -> str:
    spans = []
    for i, s in enumerate(states):
        if s.status == "rejeitado":
            continue
        for c in s.arg.citacoes:
            if c.ancorada:
                spans.append((c.char_start, c.char_end, i, s.local_id))
    spans.sort()
    out, pos = [], 0
    for a, b, i, lid in spans:
        if a < pos:
            continue
        out.append(html.escape(text[pos:a]))
        style = f"background:{COLORS[i % len(COLORS)]};color:#111"
        if focus == lid:
            style += ";outline:2px solid #dc2626"
        out.append(f'<mark style="{style}"><sup><b>{lid}</b></sup>{html.escape(text[a:b])}</mark>')
        pos = b
    out.append(html.escape(text[pos:]))
    return (
        '<div style="white-space:pre-wrap;font-family:Georgia,serif;font-size:0.95rem;'
        f'line-height:1.5">{"".join(out)}</div>'
    )


def fields_form(prefix: str, a) -> dict:
    """Campos editáveis; a = ArgumentoExtraido ou None (novo)."""
    get = (lambda k, d: getattr(a, k)) if a is not None else (lambda k, d: d)
    titulo = st.text_input("Título", get("titulo", ""), key=f"{prefix}:titulo")
    resumo = st.text_area("Resumo", get("resumo", ""), key=f"{prefix}:resumo", height=90)
    cols = st.columns(4)

    def sel(col, label, lit, k, default):
        o = opts(lit)
        return col.selectbox(label, o, o.index(get(k, default)), key=f"{prefix}:{k}")

    tipo = sel(cols[0], "Tipo", TipoArgumento, "tipo", "merito")
    susc = sel(cols[1], "Suscitado por", SuscitadoPor, "suscitado_por", "reu")
    dec = sel(cols[2], "Decisão", Decisao, "decisao", "rejeitado")
    sec = sel(cols[3], "Seção", Secao, "secao", "fundamentacao")
    tema = st.text_input("Tema", get("tema", ""), key=f"{prefix}:tema")
    fund = st.text_area(
        "Fundamento legal (um por linha)",
        "\n".join(get("fundamento_legal", [])),
        key=f"{prefix}:fund",
        height=70,
    )
    quotes = st.text_area(
        "Citações literais (separe com uma linha em branco)",
        "\n\n".join(" ".join(c.texto.split()) for c in get("citacoes", [])),
        key=f"{prefix}:cit",
        height=130,
    )
    return {
        "titulo": titulo.strip(),
        "resumo": resumo.strip(),
        "tipo": tipo,
        "suscitado_por": susc,
        "decisao": dec,
        "secao": sec,
        "tema": tema.strip(),
        "fundamento_legal": [x.strip() for x in fund.splitlines() if x.strip()],
        "citacoes": [q.strip() for q in quotes.split("\n\n") if q.strip()],
    }


def act(fn, *args, **kw) -> None:
    try:
        fn(*args, **kw)
    except Conflict as e:
        st.session_state["flash"] = ("warning", f"Conflito: {e}. Recarregado; refaça a ação.")
    except ValueError as e:
        st.session_state["flash"] = ("error", str(e))
    st.rerun()


def argument_card(store: Store, s: ArgState, reviewer: str) -> None:
    a = s.arg
    k = f"{s.arg_uid}:{s.seq}"
    head = f"{BADGE[s.status]} {s.local_id} · {a.titulo or '(sem título)'}"
    with st.expander(head, expanded=s.status == "pendente"):
        st.caption(
            f"{a.tipo} · {a.decisao} · suscitado por {a.suscitado_por} · seção {a.secao} · "
            f"confiança {a.confianca} ({a.confianca_nivel}) · "
            + (f"posição {a.posicao.char_start}–{a.posicao.char_end}" if a.posicao else
               "posição não disponível")
            + f" · status **{s.status}**"
            + (f" por {s.reviewer}" if s.status != "pendente" else "")
        )  # fmt: skip
        st.write(a.resumo)
        if a.fundamento_legal:
            st.caption("Fundamento: " + "; ".join(a.fundamento_legal))
        for c in a.citacoes:
            quote = html.escape(" ".join(c.texto.split()))
            st.markdown(f"> {quote}  \n`{c.char_start}:{c.char_end}` score {c.score}")
        if s.stored_schema != EXTRACT_SCHEMA_VERSION:
            st.caption(f"Gravado em `{s.stored_schema}`; exibido após upcast.")
        motivo = st.text_input("Motivo / comentário", key=f"{k}:motivo")
        b1, b2, b3 = st.columns(3)
        if b1.button("✅ Aceitar", key=f"{k}:ok", disabled=s.status == "aceito"):
            act(store.accept, s.arg_uid, reviewer, motivo, expected_seq=s.seq)
        if b2.button("❌ Rejeitar", key=f"{k}:rej", disabled=s.status == "rejeitado"):
            act(store.reject, s.arg_uid, reviewer, motivo, expected_seq=s.seq)
        if s.status != "pendente" and b3.button("↩️ Reabrir", key=f"{k}:reopen"):
            act(store.reopen, s.arg_uid, reviewer, motivo, expected_seq=s.seq)
        with st.form(f"{k}:edit"):
            st.markdown("**Editar** (citações são reancoradas no texto ao salvar)")
            fields = fields_form(f"{k}:f", a)
            if st.form_submit_button("💾 Salvar edição"):
                act(store.edit, s.arg_uid, fields, reviewer, motivo, expected_seq=s.seq)
        hist = store.history(s.arg_uid)
        if len(hist) > 1:
            st.markdown("**Histórico**")
            for h in hist:
                diff = ", ".join(h.diff) if h.diff else "—"
                st.caption(f"#{h.seq} {h.created_at} · {h.status} · {h.reviewer} · "
                           f"campos: {diff} · {h.motivo}")  # fmt: skip
        if s.original != a:
            with st.popover("Original da extração"):
                st.json(s.original.model_dump(exclude={"sinais"}))


def run_label(r: dict) -> str:
    p = r["params"]
    return (f"{p.get('key', '?')} · {r['prompt_version']} · {r['model']} · "
            f"{r['revisados']}/{r['n_args']} revisados · {r['run_id']}")  # fmt: skip


def main() -> None:
    st.set_page_config(page_title="Revisão Q2", layout="wide")
    store = Store()
    rids = store.ruling_ids()
    with st.sidebar:
        st.header("Revisão de argumentos")
        if not rids:
            st.info("Store vazio. Rode `uv run q2 ingest` ou `uv run q2 import-bench`.")
            st.stop()
        rid = st.selectbox("Sentença", rids)
        runs = store.runs_for(rid)
        run = st.selectbox("Run de extração", runs, format_func=run_label, key="run")
        reviewer = st.text_input("Revisor", key="reviewer")
        st.caption(
            f"model snapshot `{run['model_snapshot'] or '—'}` · schema `{run['schema_version']}`"
            f" · prompt sha `{run['prompt_sha256']}` · código `{run['code_version']}`"
        )
        counts = store.review_counts(run["run_id"])
        st.caption(" · ".join(f"{BADGE[k]} {v}" for k, v in sorted(counts.items())))

    if "flash" in st.session_state:
        kind, msg = st.session_state.pop("flash")
        getattr(st, kind)(msg)
    if not reviewer.strip():
        st.warning("Informe o revisor na barra lateral para revisar.")

    text = store.text(rid)
    states = store.arguments(rid, run["run_id"])
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Sentença")
        focus = st.session_state.get("focus")
        with st.container(height=850):
            st.markdown(highlight(text, states, focus), unsafe_allow_html=True)

    with right:
        pend = sum(s.status == "pendente" for s in states)
        st.subheader(f"Argumentos ({len(states)}) · {pend} pendente(s)")
        tab_rev, tab_add, tab_diff, tab_gold = st.tabs(
            ["Revisar", "➕ Argumento perdido", "Comparar runs", "Exportar gold"]
        )
        with tab_rev:
            for s in states:
                argument_card(store, s, reviewer)
        with tab_add:
            with st.form(f"add:{rid}:{run['run_id']}", clear_on_submit=True):
                st.caption("Argumento que a extração não encontrou. Citação literal obrigatória.")
                fields = fields_form(f"add:{rid}", None)
                motivo = st.text_input("Motivo", key=f"add:{rid}:motivo")
                if st.form_submit_button("Adicionar"):
                    act(store.add, rid, run["run_id"], fields, reviewer, motivo)
        with tab_diff:
            others = [r for r in runs if r["run_id"] != run["run_id"]]
            if not others:
                st.info("Só há uma run para esta sentença.")
            else:
                other = st.selectbox("Comparar com", others, format_func=run_label, key="other")
                d = store.diff_runs(rid, other["run_id"], run["run_id"])
                st.caption(json.dumps(d.counts(), ensure_ascii=False))
                for p in d.pairs:
                    mark = "＝" if not p.changes else ("≈" if p.equivalent else "≠")
                    flow = f"{BADGE[p.a.status]}→{BADGE[p.b.status]}"
                    st.markdown(
                        f"{mark} **{p.a.local_id} → {p.b.local_id}** ({p.how}) {flow} "
                        f"{p.b.arg.titulo}"
                    )
                    for f, (x, y) in p.changes.items():
                        st.caption(f"{f}: {x} → {y}")
                for x in d.only_a:
                    st.markdown(
                        f"➖ só na outra run: {BADGE[x.status]} {x.local_id} {x.arg.titulo}"
                    )
                for y in d.only_b:
                    st.markdown(f"🆕 só nesta run: {y.local_id} {y.arg.titulo}")
                if st.button("Herdar revisões equivalentes da outra run", disabled=not reviewer):
                    out = store.carry_over(rid, other["run_id"], run["run_id"], reviewer)
                    st.session_state["flash"] = ("success", f"Herança: {out}")
                    st.rerun()
        with tab_gold:
            g = load_gold(rid)
            st.caption(f"Gold atual: {g.status if g else 'nenhum'}")
            res = opts(Resultado)
            resultado = st.selectbox(
                "Resultado da sentença", res,
                res.index(g.resultado) if g and g.resultado in res else 0, key=f"res:{rid}",
            )  # fmt: skip
            if st.button("Exportar revisão como gold revisado", disabled=bool(pend)):
                try:
                    out = store.export_gold(rid, run["run_id"], reviewer, resultado)
                    st.success(f"Gold salvo: {len(out.argumentos)} argumentos.")
                except ValueError as e:
                    st.error(str(e))
            if pend:
                st.caption("Revise todos os argumentos antes de exportar.")


main()
