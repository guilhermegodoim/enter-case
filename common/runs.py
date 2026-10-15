"""Identidade e artefatos de execuções de benchmark/extração.

run_id = hash(tarefa, modelo, versão do prompt, versão do schema, dataset, parâmetros,
versão do código). Mesma configuração gera o mesmo run_id, então resultados ficam
comparáveis entre versões e reexecuções sobrescrevem o mesmo diretório.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from common.config import ROOT


def git_version(root: Path = ROOT) -> str:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        sha = git("rev-parse", "--short=12", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")  # só arquivos versionados
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"
    return sha + ("+dirty" if dirty else "")


def fingerprint_files(paths: list[Path]) -> str:
    """Hash estável de um dataset (conteúdo + nomes relativos)."""
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


class RunSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    task: str
    model: str | None = None
    prompt_version: str | None = None
    schema_version: str | None = None
    dataset: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    code_version: str = Field(default_factory=git_version)

    @property
    def run_id(self) -> str:
        blob = json.dumps(self.model_dump(), sort_keys=True, default=str)
        return f"{self.task}-{hashlib.sha256(blob.encode()).hexdigest()[:10]}"


class RunWriter:
    def __init__(self, spec: RunSpec, root: Path = ROOT / "results"):
        self.spec = spec
        self.dir = root / spec.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "run_id": spec.run_id,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            **spec.model_dump(),
        }
        self.write_json("spec.json", meta)

    def write_json(self, name: str, obj: Any) -> Path:
        path = self.dir / name
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", "utf-8")
        return path

    def write_metrics(self, metrics: dict[str, Any]) -> Path:
        return self.write_json("metrics.json", metrics)

    def write_report(self, markdown: str) -> Path:
        path = self.dir / "report.md"
        path.write_text(markdown, "utf-8")
        return path
