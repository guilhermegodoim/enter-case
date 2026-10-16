"""Página de revisão do gold hospedada (claude.ai): dados de entrada e importação do retorno.

Fluxo: `q2 review-page-data` gera data.json (sentenças + rascunhos do gold) para publicar ao lado
de index.html. O revisor edita no navegador e exporta um JSON. `q2 import-review <arquivo>`
valida esse JSON (sentença existe, texto igual, enums, citações reancoradas em código) e grava
em data/gold. Nunca sobrescreve gold revisado sem --force.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from common.config import ROOT
from common.text import sha256_text
from q2_arguments.bench import gold_fingerprint, load_items
from q2_arguments.corpus import load_rulings, ruling_text
from q2_arguments.gold import GoldRuling, load_gold, mark_reviewed, reanchor, save_gold
from q2_arguments.schema import SCHEMA_VERSION, Argumento, Citacao, Resultado

PAGE_DIR = Path(__file__).parent
DATA_OUT = ROOT / ".cache" / "review_page" / "data.json"
EXPORT_FORMAT = "q2-revisao-v1"
BAND_ORDER = {"curto": 0, "medio": 1, "longo": 2, "muito_longo": 3}


def build_data(out: Path = DATA_OUT) -> Path:
    items = load_items("dev", "todos") + load_items("test", "todos")
    rows = sorted(
        load_rulings(),
        key=lambda r: (r["split"] != "test", BAND_ORDER[r["faixa"]], r["ruling_id"]),
    )
    rulings = []
    for r in rows:
        rid = r["ruling_id"]
        text = ruling_text(rid)
        g = load_gold(rid)
        rulings.append({
            "ruling_id": rid, "split": r["split"], "faixa": r["faixa"],
            "classe": r["classe"], "assunto": r["assunto"], "vara": r["vara"],
            "comarca": r["comarca"], "n_tokens": r["n_tokens"], "text": text,
            "text_sha256": sha256_text(text),
            "rascunho": {
                "status": g.status, "resultado": g.resultado, "notas": g.notas,
                "argumentos": [a.model_dump() for a in g.argumentos],
            } if g else None,
        })  # fmt: skip
    data = {
        "formato": EXPORT_FORMAT,
        "gold_fingerprint": gold_fingerprint(items),
        "gerado_em": datetime.now(UTC).isoformat(timespec="seconds"),
        "schema_version": SCHEMA_VERSION,
        "rulings": rulings,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False), "utf-8")
    return out


SITE_DIR = ROOT / "dist" / "review-page"
_HEAD = (
    '<!doctype html>\n<html lang="pt-BR">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
)


def build_site(out_dir: Path = SITE_DIR) -> Path:
    """Pasta estática para qualquer host (Netlify Drop, GitHub Pages): index.html + data.json.

    index.html é escrito sem esqueleto (o host do claude.ai o acrescenta); aqui ele vira um
    documento completo, com <title>/<style>/<link> no <head> e o resto no <body>.
    """
    src = (PAGE_DIR / "index.html").read_text("utf-8")
    cut = src.index("</style>") + len("</style>")
    page = f"{_HEAD}{src[:cut]}\n</head>\n<body>\n{src[cut:].lstrip()}\n</body>\n</html>\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(page, "utf-8")
    build_data(out_dir / "data.json")
    return out_dir


# --------------------------------------------------------------------------- importação


class CitacaoIn(BaseModel):
    texto: str
    char_start: int | None = None
    char_end: int | None = None


class ArgumentoIn(BaseModel):
    titulo: str
    resumo: str = ""
    tipo: str
    tema: str = ""
    suscitado_por: str
    decisao: str
    fundamento_legal: list[str] = Field(default_factory=list)
    citacoes: list[CitacaoIn]
    secao: str = "fundamentacao"


class SentencaIn(BaseModel):
    ruling_id: str
    text_sha256: str
    status: str
    resultado: Resultado | None = None
    notas: str = ""
    argumentos: list[ArgumentoIn]


class ExportIn(BaseModel):
    formato: str
    gold_fingerprint: str | None = None
    revisor: str
    exportado_em: str | None = None
    sentencas: list[SentencaIn]


def import_review(path: Path, *, force: bool = False, root: Path | None = None) -> dict[str, str]:
    """Grava o retorno da página no gold. -> {ruling_id: resultado da importação}."""
    data = ExportIn.model_validate_json(path.read_text("utf-8"))
    if data.formato != EXPORT_FORMAT:
        raise ValueError(f"formato {data.formato!r}; esperado {EXPORT_FORMAT!r}")
    if not data.revisor.strip():
        raise ValueError("arquivo sem nome do revisor")
    known = {r["ruling_id"] for r in load_rulings()}
    out: dict[str, str] = {}
    for s in data.sentencas:
        rid = s.ruling_id
        if rid not in known:
            out[rid] = "ignorada: sentença fora do corpus"
            continue
        text = ruling_text(rid)
        if s.text_sha256 != sha256_text(text):
            out[rid] = "ignorada: texto da sentença mudou desde a geração da página"
            continue
        old = load_gold(rid, root)
        if old is not None and old.status == "revisado" and not force:
            out[rid] = f"ignorada: gold já revisado por {old.revisor} (use --force)"
            continue
        try:
            args = [
                Argumento(
                    id=f"a{i:02d}",
                    **a.model_dump(exclude={"citacoes"}),
                    citacoes=[Citacao(texto=c.texto) for c in a.citacoes if c.texto.strip()],
                )  # fmt: skip
                for i, a in enumerate(s.argumentos, start=1)
            ]
        except ValidationError as e:
            out[rid] = f"ignorada: campo inválido ({e.errors()[0]['loc']}: {e.errors()[0]['msg']})"
            continue
        g = GoldRuling(
            ruling_id=rid,
            text_sha256=sha256_text(text),
            resultado=s.resultado,
            argumentos=args,
            rascunho={
                **(old.rascunho if old else {}),
                "revisao": {
                    "origem": "pagina-revisao",
                    "formato": data.formato,
                    "gold_fingerprint": data.gold_fingerprint,
                    "exportado_em": data.exportado_em,
                },
            },  # fmt: skip
            notas=s.notas,
        )
        reanchor(g, text)
        if s.status == "revisado":
            try:
                mark_reviewed(g, data.revisor)
            except ValueError as e:
                save_gold(g, root)
                out[rid] = f"gravada como rascunho: {e}"
                continue
            save_gold(g, root)
            out[rid] = f"revisada: {len(g.argumentos)} argumentos"
        else:
            save_gold(g, root)
            out[rid] = f"rascunho em andamento: {len(g.argumentos)} argumentos"
    return out
