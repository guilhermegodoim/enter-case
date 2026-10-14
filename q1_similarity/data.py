"""Leitura do dataset sintético da Q1 (data/synthetic/q1)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from common.config import ROOT

DATA_DIR = ROOT / "data" / "synthetic" / "q1"
LABELS = ("copia", "versao", "diferente")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text("utf-8").splitlines() if ln.strip()]


@dataclass
class Dataset:
    root: Path = DATA_DIR

    @cached_property
    def meta(self) -> dict:
        return json.loads((self.root / "meta.json").read_text())

    @cached_property
    def docs(self) -> dict[str, dict]:
        return {d["doc_id"]: d for d in load_jsonl(self.root / "docs.jsonl")}

    @cached_property
    def pairs(self) -> list[dict]:
        return load_jsonl(self.root / "pairs.jsonl")

    @cached_property
    def cases(self) -> dict[str, dict]:
        return {c["case_id"]: c for c in load_jsonl(self.root / "cases.jsonl")}

    def text(self, doc_id: str) -> str:
        return (self.root / "docs" / f"{doc_id}.txt").read_text("utf-8")

    def case_docs(self, case_id: str) -> list[dict]:
        """Metadados dos documentos do caso, em ordem de chegada."""
        return sorted(
            (d for d in self.docs.values() if d["case_id"] == case_id), key=lambda d: d["seq"]
        )

    def split_pairs(self, split: str) -> list[dict]:
        return [p for p in self.pairs if p["split"] == split]

    def split_cases(self, split: str) -> list[str]:
        return [c for c, v in self.cases.items() if v["split"] == split]
