"""Armazenamento da Q2: sentenças, execuções de extração, argumentos e revisões humanas.

SQLite com DDL no formato do Postgres (tipos explícitos, FKs, JSON em TEXT = jsonb, timestamps
ISO 8601 em UTC). Regras:
- `extraction_runs` guarda tudo o que define uma execução: modelo pedido e snapshot devolvido
  pela API, versão + hash do prompt, versão do schema, versão do código e parâmetros.
- `arguments` e `argument_revisions` são append-only (triggers bloqueiam UPDATE/DELETE).
  O estado atual de um argumento é a última revisão (view `argument_current`).
- Reextrair com a mesma run não toca no que existe; uma run nova fica ao lado da antiga. O
  diff entre runs (`diff_runs`) mostra o que mudou e o que já estava revisado.
- Cada payload fica com o schema_version em que foi escrito; a leitura faz upcast (versions.py).
- Edição humana: citações reancoradas em código; citação que não está no texto é recusada.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from rapidfuzz import fuzz

from common.config import ROOT
from common.text import fold, sha256_text
from q2_arguments.extract import (
    EXTRACT_SCHEMA_VERSION,
    ArgumentoExtraido,
    Config,
    Extraction,
    posicao_of,
)
from q2_arguments.ground import Source, anchor
from q2_arguments.schema import Argumento, Citacao
from q2_arguments.versions import upcast

STORE_VERSION = 1  # PRAGMA user_version: migrações do próprio banco
DEFAULT_DB = ROOT / "data" / "store" / "q2.sqlite"

Status = Literal["pendente", "aceito", "editado", "rejeitado", "adicionado"]
DECIDED = ("aceito", "editado", "rejeitado", "adicionado")
KEPT = ("aceito", "editado", "adicionado")  # entram no gold
SYSTEM_USER = "sistema"
EDITABLE = (
    "titulo", "resumo", "tipo", "tema", "suscitado_por", "decisao", "fundamento_legal",
    "citacoes", "secao",
)  # fmt: skip

DDL = """
CREATE TABLE IF NOT EXISTS rulings (
    ruling_id TEXT PRIMARY KEY,
    text_sha256 TEXT NOT NULL,
    tribunal TEXT,
    processo TEXT,
    classe TEXT,
    assunto TEXT,
    vara TEXT,
    comarca TEXT,
    magistrado TEXT,
    data_disponibilizacao TEXT,
    text TEXT NOT NULL,
    n_chars INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS extraction_runs (
    run_id TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    model TEXT NOT NULL,
    model_snapshot TEXT,
    prompt_version TEXT NOT NULL,
    prompt_sha256 TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    code_version TEXT NOT NULL,
    params TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS extractions (
    extraction_id INTEGER PRIMARY KEY,
    ruling_id TEXT NOT NULL REFERENCES rulings(ruling_id),
    run_id TEXT NOT NULL REFERENCES extraction_runs(run_id),
    text_sha256 TEXT NOT NULL,
    cost_usd REAL,
    latency_ms REAL,
    stats TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (ruling_id, run_id)
);

CREATE TABLE IF NOT EXISTS arguments (
    arg_uid TEXT PRIMARY KEY,
    extraction_id INTEGER NOT NULL REFERENCES extractions(extraction_id),
    ruling_id TEXT NOT NULL REFERENCES rulings(ruling_id),
    run_id TEXT NOT NULL REFERENCES extraction_runs(run_id),
    local_id TEXT NOT NULL,
    origin TEXT NOT NULL CHECK (origin IN ('llm', 'humano')),
    schema_version TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (extraction_id, local_id)
);

CREATE TABLE IF NOT EXISTS argument_revisions (
    revision_id INTEGER PRIMARY KEY,
    arg_uid TEXT NOT NULL REFERENCES arguments(arg_uid),
    seq INTEGER NOT NULL,
    status TEXT NOT NULL
        CHECK (status IN ('pendente', 'aceito', 'editado', 'rejeitado', 'adicionado')),
    schema_version TEXT NOT NULL,
    payload TEXT NOT NULL,
    diff TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    motivo TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    UNIQUE (arg_uid, seq)
);

CREATE INDEX IF NOT EXISTS arguments_by_extraction ON arguments(ruling_id, run_id);

CREATE VIEW IF NOT EXISTS argument_current AS
SELECT r.* FROM argument_revisions r
JOIN (SELECT arg_uid, MAX(seq) AS seq FROM argument_revisions GROUP BY arg_uid) m
  ON m.arg_uid = r.arg_uid AND m.seq = r.seq;
"""

_IMMUTABLE = ("extraction_runs", "extractions", "arguments", "argument_revisions")
TRIGGERS = "\n".join(
    f"CREATE TRIGGER IF NOT EXISTS {t}_no_{op.lower()} BEFORE {op} ON {t} "
    f"BEGIN SELECT RAISE(ABORT, '{t} é append-only'); END;"
    for t in _IMMUTABLE
    for op in ("UPDATE", "DELETE")
)


class Conflict(RuntimeError):
    """Outra revisão foi gravada depois da que o revisor viu (concorrência otimista)."""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def default_db() -> Path:
    return Path(os.getenv("ENTER_Q2_DB") or DEFAULT_DB)


def _uid(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


@dataclass
class ArgState:
    """Argumento com seu estado de revisão atual (payload já em EXTRACT_SCHEMA_VERSION)."""

    arg_uid: str
    ruling_id: str
    run_id: str
    local_id: str
    origin: str
    status: Status
    seq: int
    reviewer: str
    motivo: str
    arg: ArgumentoExtraido  # estado atual
    original: ArgumentoExtraido  # como saiu da extração (ou como o humano adicionou)
    stored_schema: str  # schema_version em que o original foi gravado


@dataclass
class Revision:
    seq: int
    status: Status
    reviewer: str
    motivo: str
    diff: dict
    schema_version: str
    created_at: str


@dataclass
class DiffPair:
    a: ArgState
    b: ArgState
    how: Literal["citacao", "titulo"]
    changes: dict[str, tuple[Any, Any]]

    @property
    def equivalent(self) -> bool:
        """Mesma questão com o mesmo desfecho: decisão de revisão anterior se aplica."""
        return not {"tipo", "decisao", "suscitado_por"} & self.changes.keys()


@dataclass
class RunDiff:
    ruling_id: str
    run_a: str
    run_b: str
    pairs: list[DiffPair] = field(default_factory=list)
    only_a: list[ArgState] = field(default_factory=list)
    only_b: list[ArgState] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {
            "pareados": len(self.pairs),
            "equivalentes": sum(p.equivalent for p in self.pairs),
            "so_a": len(self.only_a),
            "so_a_revisados_mantidos": sum(s.status in KEPT for s in self.only_a),
            "so_b": len(self.only_b),
        }


# --------------------------------------------------------------------------- diff de campos


def _comparable(a: Argumento, k: str) -> Any:
    v = getattr(a, k)
    if k == "citacoes":
        return [" ".join(c.texto.split()) for c in v]
    return v


def field_diff(old: Argumento, new: Argumento) -> dict[str, tuple[Any, Any]]:
    return {
        k: (_comparable(old, k), _comparable(new, k))
        for k in EDITABLE
        if _comparable(old, k) != _comparable(new, k)
    }


def _spans(a: Argumento) -> list[tuple[int, int]]:
    return [(c.char_start, c.char_end) for c in a.citacoes if c.ancorada]


def _overlap(a: Argumento, b: Argumento) -> int:
    return sum(max(0, min(e1, e2) - max(s1, s2)) for s1, e1 in _spans(a) for s2, e2 in _spans(b))


TITLE_MATCH = 80  # rapidfuzz token_set_ratio mínimo para parear por título (mesmo tipo)


def diff_states(rid: str, run_a: str, run_b: str, a: list[ArgState], b: list[ArgState]) -> RunDiff:
    """Pareia argumentos de duas runs: citações sobrepostas no texto; senão título parecido."""
    cands = []
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            ov = _overlap(x.arg, y.arg)
            if ov:
                cands.append((2.0 + ov / 1e6, i, j, "citacao"))
            elif x.arg.tipo == y.arg.tipo:
                r = fuzz.token_set_ratio(fold(x.arg.titulo), fold(y.arg.titulo))
                if r >= TITLE_MATCH:
                    cands.append((r / 100, i, j, "titulo"))
    out = RunDiff(rid, run_a, run_b)
    used_a, used_b = set(), set()
    for _, i, j, how in sorted(cands, reverse=True):
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        out.pairs.append(DiffPair(a[i], b[j], how, field_diff(a[i].arg, b[j].arg)))
    out.pairs.sort(key=lambda p: p.b.local_id)
    out.only_a = [x for i, x in enumerate(a) if i not in used_a]
    out.only_b = [y for j, y in enumerate(b) if j not in used_b]
    return out


# --------------------------------------------------------------------------- store


class Store:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else default_db()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        with self.conn:
            self.conn.executescript(DDL + TRIGGERS)
            self.conn.execute(f"PRAGMA user_version = {STORE_VERSION}")

    def close(self) -> None:
        self.conn.close()

    def _one(self, sql: str, *args) -> sqlite3.Row | None:
        return self.conn.execute(sql, args).fetchone()

    def _all(self, sql: str, *args) -> list[sqlite3.Row]:
        return self.conn.execute(sql, args).fetchall()

    # ------------------------------------------------------------------ sentenças

    def upsert_ruling(self, meta: dict, text: str) -> None:
        sha = sha256_text(text)
        row = self._one("SELECT text_sha256 FROM rulings WHERE ruling_id = ?", meta["ruling_id"])
        if row is not None:
            if row["text_sha256"] != sha:
                raise ValueError(
                    f"{meta['ruling_id']}: texto mudou; offsets antigos ficariam inválidos"
                )
            return
        cols = ("tribunal", "processo", "classe", "assunto", "vara", "comarca", "magistrado",
                "data_disponibilizacao")  # fmt: skip
        with self.conn:
            self.conn.execute(
                f"INSERT INTO rulings (ruling_id, text_sha256, {', '.join(cols)}, text, n_chars, "
                f"created_at) VALUES ({', '.join('?' * (len(cols) + 5))})",
                (meta["ruling_id"], sha, *(meta.get(c) for c in cols), text, len(text), now()),
            )

    def ruling_ids(self) -> list[str]:
        rows = self._all("SELECT DISTINCT ruling_id FROM extractions ORDER BY ruling_id")
        return [r["ruling_id"] for r in rows]

    # ------------------------------------------------------------------ runs

    def register_run(self, cfg: Config, model_snapshot: str | None) -> str:
        spec = cfg.spec()
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO extraction_runs (run_id, task, model, model_snapshot, "
                "prompt_version, prompt_sha256, schema_version, code_version, params, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (spec.run_id, spec.task, spec.model, model_snapshot, spec.prompt_version,
                 cfg.prompt_sha256, spec.schema_version, spec.code_version,
                 _dumps({**spec.params, "key": cfg.key}), now()),
            )  # fmt: skip
        return spec.run_id

    def run(self, run_id: str) -> dict | None:
        row = self._one("SELECT * FROM extraction_runs WHERE run_id = ?", run_id)
        return {**dict(row), "params": json.loads(row["params"])} if row else None

    def runs_for(self, ruling_id: str) -> list[dict]:
        rows = self._all(
            "SELECT r.*, e.extraction_id, e.cost_usd, e.latency_ms, e.created_at AS extracted_at,"
            " (SELECT COUNT(*) FROM argument_current c JOIN arguments a USING (arg_uid)"
            "   WHERE a.extraction_id = e.extraction_id AND c.status != 'pendente') AS revisados,"
            " (SELECT COUNT(*) FROM arguments a WHERE a.extraction_id = e.extraction_id) AS n_args"
            " FROM extractions e JOIN extraction_runs r USING (run_id)"
            " WHERE e.ruling_id = ? ORDER BY e.extraction_id",
            ruling_id,
        )
        return [{**dict(r), "params": json.loads(r["params"])} for r in rows]

    # ------------------------------------------------------------------ extrações

    def save_extraction(
        self, meta: dict, text: str, cfg: Config, ext: Extraction
    ) -> tuple[int, bool]:
        """Grava a extração. Mesma (sentença, run) já gravada: nada muda. -> (id, nova?)"""
        if ext.run_id != cfg.spec().run_id:
            raise ValueError("extração não corresponde à configuração")
        self.upsert_ruling(meta, text)
        snapshot = next((r.model_snapshot for r in ext.calls if r.kind == "chat"), None)
        run_id = self.register_run(cfg, snapshot)
        row = self._one(
            "SELECT extraction_id FROM extractions WHERE ruling_id = ? AND run_id = ?",
            ext.ruling_id, run_id,
        )  # fmt: skip
        if row is not None:
            return row["extraction_id"], False
        stats = {
            k: getattr(ext, k)
            for k in ("brutos", "citacoes_brutas", "citacoes_ancoradas", "descartados_sem_ancora",
                      "fundidos", "descartados_verificador", "n_chunks", "chunks_falhos",
                      "chunks_com_retry", "sem_ancora", "falhas")
        }  # fmt: skip
        ts = now()
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO extractions (ruling_id, run_id, text_sha256, cost_usd, latency_ms, "
                "stats, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ext.ruling_id, run_id, sha256_text(text), ext.cost_usd, ext.latency_ms,
                 _dumps(stats), ts),
            )  # fmt: skip
            eid = cur.lastrowid
            for a in ext.argumentos:
                self._insert_arg(eid, ext.ruling_id, run_id, a, "llm", SYSTEM_USER, "pendente", ts)
        return eid, True

    def _insert_arg(
        self, eid: int, rid: str, run_id: str, a: ArgumentoExtraido, origin: str, who: str,
        status: Status, ts: str, motivo: str = "",
    ) -> str:  # fmt: skip
        uid = _uid(run_id, rid, a.id)
        payload = _dumps(a.model_dump())
        self.conn.execute(
            "INSERT INTO arguments (arg_uid, extraction_id, ruling_id, run_id, local_id, origin, "
            "schema_version, payload, created_by, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (uid, eid, rid, run_id, a.id, origin, EXTRACT_SCHEMA_VERSION, payload, who, ts),
        )
        self.conn.execute(
            "INSERT INTO argument_revisions (arg_uid, seq, status, schema_version, payload, diff, "
            "reviewer, motivo, created_at) VALUES (?, 1, ?, ?, ?, '{}', ?, ?, ?)",
            (uid, status, EXTRACT_SCHEMA_VERSION, payload, who, motivo, ts),
        )
        return uid

    def _extraction(self, ruling_id: str, run_id: str) -> int:
        row = self._one(
            "SELECT extraction_id FROM extractions WHERE ruling_id = ? AND run_id = ?",
            ruling_id, run_id,
        )  # fmt: skip
        if row is None:
            raise KeyError(f"sem extração de {ruling_id} na run {run_id}")
        return row["extraction_id"]

    def extraction(self, ruling_id: str, run_id: str) -> dict:
        row = self._one(
            "SELECT * FROM extractions WHERE extraction_id = ?", self._extraction(ruling_id, run_id)
        )
        return {**dict(row), "stats": json.loads(row["stats"])}

    # ------------------------------------------------------------------ leitura

    @staticmethod
    def _load(payload: str, version: str) -> ArgumentoExtraido:
        return ArgumentoExtraido.model_validate(upcast(json.loads(payload), version))

    def arguments(self, ruling_id: str, run_id: str) -> list[ArgState]:
        rows = self._all(
            "SELECT a.arg_uid, a.ruling_id, a.run_id, a.local_id, a.origin, "
            "a.schema_version AS a_schema, a.payload AS a_payload, c.seq, c.status, c.reviewer, "
            "c.motivo, c.schema_version AS c_schema, c.payload AS c_payload "
            "FROM arguments a JOIN argument_current c USING (arg_uid) "
            "WHERE a.ruling_id = ? AND a.run_id = ? ORDER BY a.origin DESC, a.local_id",
            ruling_id, run_id,
        )  # fmt: skip
        return [
            ArgState(
                arg_uid=r["arg_uid"],
                ruling_id=r["ruling_id"],
                run_id=r["run_id"],
                local_id=r["local_id"],
                origin=r["origin"],
                status=r["status"],
                seq=r["seq"],
                reviewer=r["reviewer"],
                motivo=r["motivo"],
                arg=self._load(r["c_payload"], r["c_schema"]),
                original=self._load(r["a_payload"], r["a_schema"]),
                stored_schema=r["a_schema"],
            )  # fmt: skip
            for r in rows
        ]

    def state(self, arg_uid: str) -> ArgState:
        row = self._one("SELECT ruling_id, run_id FROM arguments WHERE arg_uid = ?", arg_uid)
        if row is None:
            raise KeyError(arg_uid)
        states = self.arguments(row["ruling_id"], row["run_id"])
        return next(s for s in states if s.arg_uid == arg_uid)

    def history(self, arg_uid: str) -> list[Revision]:
        rows = self._all("SELECT * FROM argument_revisions WHERE arg_uid = ? ORDER BY seq", arg_uid)
        return [
            Revision(r["seq"], r["status"], r["reviewer"], r["motivo"], json.loads(r["diff"]),
                     r["schema_version"], r["created_at"])
            for r in rows
        ]  # fmt: skip

    # ------------------------------------------------------------------ revisão

    def _append(
        self, s: ArgState, status: Status, new: ArgumentoExtraido, reviewer: str, motivo: str,
        expected_seq: int | None,
    ) -> ArgState:  # fmt: skip
        if not reviewer.strip():
            raise ValueError("informe o revisor")
        diff = field_diff(s.arg, new)
        with self.conn:
            cur = self._one("SELECT MAX(seq) AS seq FROM argument_revisions WHERE arg_uid = ?",
                            s.arg_uid)["seq"]  # fmt: skip
            if expected_seq is not None and cur != expected_seq:
                raise Conflict(f"{s.local_id}: revisão {cur} gravada depois da {expected_seq}")
            try:
                self.conn.execute(
                    "INSERT INTO argument_revisions (arg_uid, seq, status, schema_version, "
                    "payload, diff, reviewer, motivo, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (s.arg_uid, cur + 1, status, EXTRACT_SCHEMA_VERSION,
                     _dumps(new.model_dump()), _dumps(diff), reviewer.strip(), motivo.strip(),
                     now()),
                )  # fmt: skip
            except sqlite3.IntegrityError as e:  # outra sessão gravou a mesma seq
                raise Conflict(f"{s.local_id}: revisão concorrente") from e
        return self.state(s.arg_uid)

    def accept(self, arg_uid: str, reviewer: str, motivo: str = "", expected_seq=None) -> ArgState:
        s = self.state(arg_uid)
        return self._append(s, "aceito", s.arg, reviewer, motivo, expected_seq)

    def reject(self, arg_uid: str, reviewer: str, motivo: str, expected_seq=None) -> ArgState:
        if not motivo.strip():
            raise ValueError("rejeição exige motivo (alimenta a análise de erros)")
        s = self.state(arg_uid)
        return self._append(s, "rejeitado", s.arg, reviewer, motivo, expected_seq)

    def reopen(self, arg_uid: str, reviewer: str, motivo: str = "", expected_seq=None) -> ArgState:
        s = self.state(arg_uid)
        return self._append(s, "pendente", s.arg, reviewer, motivo, expected_seq)

    def _ground(self, ruling_id: str, fields: dict, base: ArgumentoExtraido) -> ArgumentoExtraido:
        """Aplica campos editados, reancora citações no texto e recusa citação inexistente."""
        text = self._text(ruling_id)
        data = {**base.model_dump(), **{k: v for k, v in fields.items() if k in EDITABLE}}
        src = Source(text)
        quotes = [c.texto if isinstance(c, Citacao) else c["texto"] if isinstance(c, dict) else c
                  for c in data["citacoes"]]  # fmt: skip
        cits = [anchor(q, src) for q in quotes if q.strip()]
        bad = [c.texto[:80] for c in cits if not c.ancorada]
        if bad:
            raise ValueError(f"citação não encontrada no texto: {bad}")
        if not cits:
            raise ValueError("argumento precisa de ao menos uma citação literal")
        data["citacoes"] = [c.model_dump() for c in cits]
        new = ArgumentoExtraido.model_validate(data)
        new.posicao = posicao_of(new.citacoes)
        return new

    def edit(
        self, arg_uid: str, fields: dict, reviewer: str, motivo: str = "", expected_seq=None
    ) -> ArgState:
        s = self.state(arg_uid)
        new = self._ground(s.ruling_id, fields, s.arg)
        if not field_diff(s.arg, new):
            return self._append(s, "aceito", s.arg, reviewer, motivo or "sem alteração",
                                expected_seq)  # fmt: skip
        return self._append(s, "editado", new, reviewer, motivo, expected_seq)

    def add(
        self, ruling_id: str, run_id: str, fields: dict, reviewer: str, motivo: str = ""
    ) -> ArgState:
        """Argumento que a extração perdeu, adicionado pelo revisor (entra no gold)."""
        if not reviewer.strip():
            raise ValueError("informe o revisor")
        eid = self._extraction(ruling_id, run_id)
        n = self._one(
            "SELECT COUNT(*) AS n FROM arguments WHERE extraction_id = ? AND origin = 'humano'", eid
        )["n"]
        base = ArgumentoExtraido(
            id=f"h{n + 1:02d}", titulo="", resumo="", tipo="merito", tema="",
            suscitado_por="reu", decisao="rejeitado", citacoes=[], confianca=1.0,
            confianca_nivel="alta", sinais={"origem": "humano"}, run_id=run_id,
        )  # fmt: skip
        new = self._ground(ruling_id, fields, base)
        if not new.titulo.strip():
            raise ValueError("argumento precisa de título")
        with self.conn:
            uid = self._insert_arg(eid, ruling_id, run_id, new, "humano", reviewer.strip(),
                                   "adicionado", now(), motivo)  # fmt: skip
        return self.state(uid)

    def _text(self, ruling_id: str) -> str:
        row = self._one("SELECT text FROM rulings WHERE ruling_id = ?", ruling_id)
        if row is None:
            raise KeyError(ruling_id)
        return row["text"]

    def text(self, ruling_id: str) -> str:
        """Texto armazenado: os offsets de todas as citações são relativos a ele."""
        return self._text(ruling_id)

    # ------------------------------------------------------------------ comparação

    def diff_runs(self, ruling_id: str, run_a: str, run_b: str) -> RunDiff:
        a, b = self.arguments(ruling_id, run_a), self.arguments(ruling_id, run_b)
        return diff_states(ruling_id, run_a, run_b, a, b)

    def carry_over(self, ruling_id: str, run_a: str, run_b: str, reviewer: str) -> dict[str, int]:
        """Herda decisões de revisão da run A para argumentos pendentes equivalentes da run B.

        Só pares equivalentes (mesma questão por citação/título, mesmo tipo, decisão e quem
        suscitou). Aceito/editado -> mesmo conteúdo; rejeitado -> rejeitado com o mesmo
        motivo. Aceito em A com redação diferente em B: B recebe o conteúdo de A (o revisor
        nunca viu a redação de B). Cada herança é uma revisão nova na run B, com o arg_uid de
        origem no motivo.
        Pares não equivalentes e argumentos novos continuam pendentes para revisão humana.
        """
        out = {"herdados": 0, "nao_equivalentes": 0, "ja_revisados": 0, "origem_pendente": 0}
        for p in self.diff_runs(ruling_id, run_a, run_b).pairs:
            if p.b.status != "pendente":
                out["ja_revisados"] += 1
                continue
            if p.a.status == "pendente":
                out["origem_pendente"] += 1
                continue
            if not p.equivalent:
                out["nao_equivalentes"] += 1
                continue
            why = f"herdado de {p.a.arg_uid} ({run_a}): {p.a.motivo}".strip().rstrip(":")
            who = f"{reviewer.strip()} (herança)"
            if p.a.status == "rejeitado":
                self.reject(p.b.arg_uid, who, why)
            elif field_diff(p.b.arg, p.a.arg):
                # texto da run B nunca foi visto pelo revisor: grava o conteúdo revisado em A
                fields = {k: getattr(p.a.arg, k) for k in EDITABLE}
                self.edit(p.b.arg_uid, fields, who, why)
            else:
                self.accept(p.b.arg_uid, who, why)
            out["herdados"] += 1
        return out

    # ------------------------------------------------------------------ importação

    def import_bench(self, bench_dir: Path, key: str) -> dict[str, int]:
        """Importa as predições de uma config de um benchmark antigo, no schema em que foram
        escritas. Runs anteriores à fase 5 não gravavam o schema por config: eram todas
        q2-extracted-v1. Os payloads ficam como estão; a leitura faz upcast."""
        from q2_arguments.corpus import load_rulings, ruling_text

        metrics = json.loads((bench_dir / "metrics.json").read_text("utf-8"))
        bspec = json.loads((bench_dir / "spec.json").read_text("utf-8"))
        res = metrics["configs"][key]
        cfg = Config.model_validate(res["config"])
        run_id = res["extraction_run_id"]
        schema = res.get("schema_version", "q2-extracted-v1")
        meta = {r["ruling_id"]: r for r in load_rulings()}
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO extraction_runs (run_id, task, model, model_snapshot, "
                "prompt_version, prompt_sha256, schema_version, code_version, params, created_at) "
                "VALUES (?, 'q2-extract', ?, NULL, ?, ?, ?, ?, ?, ?)",
                (run_id, cfg.model, res["prompt_version"],
                 res.get("prompt_sha256") or cfg.prompt_sha256, schema, bspec["code_version"],
                 _dumps({**res["config"], "importado_de": bench_dir.name}), bspec["created_at"]),
            )  # fmt: skip
        out = {"importadas": 0, "ja_existiam": 0}
        for line in (bench_dir / "predictions.jsonl").read_text("utf-8").splitlines():
            p = json.loads(line)
            if p["config"] != key:
                continue
            rid = p["ruling_id"]
            self.upsert_ruling(meta[rid], ruling_text(rid))
            if self._one("SELECT 1 FROM extractions WHERE ruling_id = ? AND run_id = ?",
                         rid, run_id):  # fmt: skip
                out["ja_existiam"] += 1
                continue
            ts = bspec["created_at"]
            with self.conn:
                cur = self.conn.execute(
                    "INSERT INTO extractions (ruling_id, run_id, text_sha256, cost_usd, "
                    "latency_ms, stats, created_at) VALUES (?, ?, ?, NULL, NULL, ?, ?)",
                    (rid, run_id, sha256_text(ruling_text(rid)),
                     _dumps({"importado_de": bench_dir.name,
                             "sem_ancora": p.get("descartados_sem_ancora", [])}), ts),
                )  # fmt: skip
                for a in p["argumentos"]:
                    uid = _uid(run_id, rid, a["id"])
                    payload = _dumps(a)
                    self.conn.execute(
                        "INSERT INTO arguments (arg_uid, extraction_id, ruling_id, run_id, "
                        "local_id, origin, schema_version, payload, created_by, created_at) "
                        "VALUES (?, ?, ?, ?, ?, 'llm', ?, ?, ?, ?)",
                        (uid, cur.lastrowid, rid, run_id, a["id"], schema, payload, SYSTEM_USER,
                         ts),
                    )  # fmt: skip
                    self.conn.execute(
                        "INSERT INTO argument_revisions (arg_uid, seq, status, schema_version, "
                        "payload, diff, reviewer, motivo, created_at) "
                        "VALUES (?, 1, 'pendente', ?, ?, '{}', ?, '', ?)",
                        (uid, schema, payload, SYSTEM_USER, ts),
                    )
            out["importadas"] += 1
        return out

    def review_counts(self, run_id: str | None = None) -> dict[str, int]:
        sql = ("SELECT c.status, COUNT(*) AS n FROM argument_current c JOIN arguments a "
               "USING (arg_uid)")  # fmt: skip
        rows = (
            self._all(sql + " WHERE a.run_id = ? GROUP BY c.status", run_id)
            if run_id
            else (self._all(sql + " GROUP BY c.status"))
        )
        return {r["status"]: r["n"] for r in rows}

    # ------------------------------------------------------------------ gold

    def export_gold(self, ruling_id: str, run_id: str, reviewer: str, resultado: str,
                    root: Path | None = None):  # fmt: skip
        """Revisão completa de uma extração -> gold revisado. Nunca sobrescreve gold revisado."""
        from q2_arguments.gold import GoldRuling, load_gold, mark_reviewed, save_gold

        states = self.arguments(ruling_id, run_id)
        pending = [s.local_id for s in states if s.status == "pendente"]
        if pending:
            raise ValueError(f"argumentos ainda pendentes: {pending}")
        old = load_gold(ruling_id, root)
        if old is not None and old.status == "revisado":
            raise ValueError(f"{ruling_id}: gold já revisado; não sobrescrevo")
        kept = [s for s in states if s.status in KEPT]
        args = [
            Argumento(**s.arg.model_dump(include=set(Argumento.model_fields) - {"id"}),
                      id=f"a{i:02d}")
            for i, s in enumerate(kept, start=1)
        ]  # fmt: skip
        g = GoldRuling(
            ruling_id=ruling_id,
            text_sha256=sha256_text(self._text(ruling_id)),
            resultado=resultado,
            argumentos=args,
            rascunho={
                "origem": "review_app",
                "run_id": run_id,
                "extraction_id": self._extraction(ruling_id, run_id),
                "revisoes": {k: sum(s.status == k for s in states) for k in DECIDED},
                "argumentos_origem": [s.arg_uid for s in kept],
            },
            notas=old.notas if old else "",
        )
        mark_reviewed(g, reviewer)
        save_gold(g, root)
        return g
