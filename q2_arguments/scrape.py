"""Coleta de sentenças públicas no TJSP CJPG (Consulta de Julgados de 1º Grau).

- A lista de resultados do CJPG já traz o inteiro teor de cada sentença; não é preciso baixar PDF.
- Cortesia com o servidor: User-Agent identificado, 1 requisição a cada ~2 s, retry com backoff.
- HTML bruto fica em data/rulings/raw/ (fora do git). Reexecutar não refaz requisições já salvas.
- A paginação do CJPG depende da sessão (cookie): cada consulta usa um cliente próprio.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx
from selectolax.lexbor import LexborHTMLParser as HTMLParser

from common.config import ROOT

BASE = "https://esaj.tjsp.jus.br/cjpg"
RULINGS_DIR = ROOT / "data" / "rulings"
RAW_DIR = RULINGS_DIR / "raw"
QUERIES_PATH = RULINGS_DIR / "queries.json"
USER_AGENT = "enter-case-research/0.1 (estudo tecnico; coleta lenta de julgados publicos)"
DELAY_S = 2.0


@dataclass(frozen=True)
class Query:
    id: str
    dominio: str
    termo: str
    inicio: str  # ISO
    fim: str

    def params(self) -> dict[str, str]:
        br = lambda iso: date.fromisoformat(iso).strftime("%d/%m/%Y")  # noqa: E731
        return {
            "dadosConsulta.pesquisaLivre": self.termo,
            "dadosConsulta.dtInicio": br(self.inicio),
            "dadosConsulta.dtFim": br(self.fim),
            "dadosConsulta.ordenacao": "DESC",
        }


def load_queries(path: Path = QUERIES_PATH) -> tuple[list[Query], int]:
    cfg = json.loads(path.read_text("utf-8"))
    return [Query(**q) for q in cfg["consultas"]], int(cfg["paginas_por_consulta"])


@dataclass(frozen=True)
class Hit:
    """Um resultado da busca: metadados do CJPG + inteiro teor bruto."""

    processo: str
    cd_documento: str
    classe: str
    assunto: str
    magistrado: str
    comarca: str
    foro: str
    vara: str
    data_disponibilizacao: str  # ISO
    texto: str
    query_id: str
    pagina: int


_LABELS = {
    "Classe": "classe",
    "Assunto": "assunto",
    "Magistrado": "magistrado",
    "Comarca": "comarca",
    "Foro": "foro",
    "Vara": "vara",
    "Data de Disponibilização": "data_disponibilizacao",
}


def _br_to_iso(s: str) -> str:
    d, m, y = s.strip().split("/")
    return f"{y}-{m}-{d}"


def parse_results(html: str, *, query_id: str = "", pagina: int = 0) -> list[Hit]:
    tree = HTMLParser(html)
    hits = []
    for row in tree.css("tr.fundocinza1"):
        num = row.css_first("span.fonteNegrito")
        anchor = row.css_first("a[name]")
        full = row.css_first('div[style*="display: none"] span')
        if num is None or full is None:
            continue
        meta = {v: "" for v in _LABELS.values()}
        for td in row.css("tr.fonte td"):
            strong = td.css_first("strong")
            if strong is None:
                continue
            label = strong.text(strip=True).rstrip(":")
            if label in _LABELS:
                value = td.text(deep=True).split(":", 1)[1].strip()
                meta[_LABELS[label]] = " ".join(value.split())
        if meta["data_disponibilizacao"]:
            meta["data_disponibilizacao"] = _br_to_iso(meta["data_disponibilizacao"])
        hits.append(
            Hit(
                processo=num.text(strip=True),
                cd_documento=anchor.attributes.get("name", "") if anchor else "",
                texto=full.text(deep=True),
                query_id=query_id,
                pagina=pagina,
                **meta,
            )
        )
    return hits


class Fetcher:
    def __init__(self, raw_dir: Path = RAW_DIR, delay_s: float = DELAY_S):
        self.raw_dir = raw_dir
        self.delay_s = delay_s
        self._last = 0.0
        self.requests = 0

    def _wait(self) -> None:
        dt = time.monotonic() - self._last
        if dt < self.delay_s:
            time.sleep(self.delay_s - dt)
        self._last = time.monotonic()

    def _get(self, client: httpx.Client, url: str, params: dict | None = None) -> str:
        for attempt in range(4):
            self._wait()
            self.requests += 1
            try:
                r = client.get(url, params=params)
                # o CJPG às vezes devolve a página sem a tabela de resultados (erro transitório):
                # só aceita resposta com resultados; consultas da lista sempre têm resultados
                if r.status_code == 200 and "fundocinza1" in r.text:
                    return r.text
                err = f"HTTP {r.status_code}, sem tabela de resultados"
            except httpx.HTTPError as e:
                err = repr(e)
            time.sleep(2 ** (attempt + 1))
        raise RuntimeError(f"falha ao buscar {url}: {err}")

    def fetch_query(self, q: Query, pages: int) -> list[Path]:
        """Baixa (ou reaproveita do cache) as páginas 1..pages de uma consulta."""
        out = self.raw_dir / q.id
        paths = [out / f"p{n}.html" for n in range(1, pages + 1)]
        if all(p.exists() for p in paths):
            return paths
        out.mkdir(parents=True, exist_ok=True)
        headers = {"User-Agent": USER_AGENT, "Accept-Language": "pt-BR"}
        with httpx.Client(headers=headers, timeout=60, follow_redirects=True) as client:
            # a página 1 abre a sessão; as seguintes dependem do cookie dela
            for n, path in enumerate(paths, start=1):
                if n == 1:
                    html = self._get(client, f"{BASE}/pesquisar.do", q.params())
                else:
                    html = self._get(
                        client, f"{BASE}/trocarDePagina.do", {"pagina": n, "conversationId": ""}
                    )
                path.write_text(html, "utf-8")
        return paths


def scrape(queries: list[Query], pages: int, fetcher: Fetcher | None = None) -> dict[str, int]:
    fetcher = fetcher or Fetcher()
    counts = {}
    for q in queries:
        paths = fetcher.fetch_query(q, pages)
        counts[q.id] = sum(len(parse_results(p.read_text("utf-8"))) for p in paths)
    counts["_requisicoes"] = fetcher.requests
    return counts


def load_pool(raw_dir: Path = RAW_DIR, queries_path: Path = QUERIES_PATH) -> list[Hit]:
    """Todos os resultados em cache, na ordem das consultas e páginas."""
    queries, pages = load_queries(queries_path)
    hits = []
    for q in queries:
        for n in range(1, pages + 1):
            path = raw_dir / q.id / f"p{n}.html"
            if path.exists():
                hits += parse_results(path.read_text("utf-8"), query_id=q.id, pagina=n)
    return hits
