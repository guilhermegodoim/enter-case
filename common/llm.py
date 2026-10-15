"""Wrapper OpenAI (Responses API): saídas estruturadas, cache em disco e ledger.

- Cache endereçado por sha256(tipo, modelo, mensagens, schema, parâmetros): mudar prompt,
  schema ou modelo invalida automaticamente; reexecutar benchmark custa zero.
- Cada chamada gera um CallRecord. Em cache hit, tokens/custo/latência são os da chamada
  original, então métricas de custo e latência ficam idênticas entre reexecuções.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from common.config import Settings
from common.pricing import cost_usd

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class CallRecord:
    kind: str  # "chat" | "embed"
    model: str
    model_snapshot: str | None  # snapshot datado devolvido pela API
    prompt_version: str | None
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    latency_ms: float  # latência da chamada original
    cache_hit: bool
    key: str


def cache_key(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, root: Path):
        self.root = root

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text("utf-8"))

    def put(self, key: str, value: dict[str, Any]) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False)
        os.replace(tmp, path)  # escrita atômica


class LLM:
    def __init__(
        self,
        *,
        client: Any = None,
        cache_dir: Path | None = None,
        ledger_path: Path | None = None,
        chat_model: str | None = None,
        embed_model: str | None = None,
    ):
        self.settings = Settings.load()
        self._client = client
        self.cache = DiskCache(cache_dir or self.settings.cache_dir)
        self.ledger_path = ledger_path
        self.chat_model = chat_model or self.settings.chat_model
        self.embed_model = embed_model or self.settings.embed_model
        self.records: list[CallRecord] = []
        self._lock = threading.Lock()  # chamadas concorrentes (ThreadPoolExecutor)

    @property
    def client(self) -> Any:
        if self._client is None:
            if not self.settings.openai_api_key:
                raise LLMError("OPENAI_API_KEY ausente (copie .env.example para .env)")
            from openai import OpenAI

            self._client = OpenAI(api_key=self.settings.openai_api_key)
        return self._client

    # ------------------------------------------------------------------ chat

    def structured(
        self,
        schema: type[T],
        *,
        system: str,
        user: str,
        model: str | None = None,
        temperature: float | None = 0.0,
        max_output_tokens: int | None = None,
        prompt_version: str | None = None,
    ) -> tuple[T, CallRecord]:
        """Chamada com saída estruturada validada por `schema` (Responses API, JSON schema estrito).

        Usa a Responses API: na Chat Completions, `response_format` passou a ser ignorado para
        esta conta (texto livre em vez de JSON). temperature=None omite o parâmetro (modelos de
        raciocínio não aceitam). store=False: nada fica guardado do lado da OpenAI.
        """
        model = model or self.chat_model
        key = cache_key(
            {
                "kind": "chat",
                "api": "responses",
                "model": model,
                "instructions": system,
                "input": user,
                "schema": schema.model_json_schema(),
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            }
        )
        entry = self.cache.get(key)
        hit = entry is not None
        if entry is None:
            kwargs: dict[str, Any] = {}
            if temperature is not None:
                kwargs["temperature"] = temperature
            if max_output_tokens is not None:
                kwargs["max_output_tokens"] = max_output_tokens
            t0 = time.perf_counter()
            try:
                resp = self.client.responses.parse(
                    model=model,
                    instructions=system,
                    input=[{"role": "user", "content": user}],
                    text_format=schema,
                    store=False,
                    **kwargs,
                )
            except ValidationError as e:  # JSON inválido (ex.: saída degenerada em loop)
                raise LLMError(f"saída não confere com o schema {schema.__name__}: {e}") from e
            latency = (time.perf_counter() - t0) * 1000
            if resp.status != "completed":
                raise LLMError(
                    f"resposta incompleta (status={resp.status}, {resp.incomplete_details})"
                )
            refusals = [
                c.refusal
                for item in resp.output
                if item.type == "message"
                for c in item.content
                if c.type == "refusal"
            ]
            if refusals:
                raise LLMError(f"recusa do modelo: {refusals[0]}")
            entry = {
                "content": resp.output_text,
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "latency_ms": latency,
                "model_snapshot": getattr(resp, "model", None),
            }
            self.cache.put(key, entry)
        parsed = schema.model_validate_json(entry["content"])
        rec = self._record("chat", model, prompt_version, entry, hit, key)
        return parsed, rec

    # ------------------------------------------------------------------ embeddings

    def embed(
        self, texts: list[str], *, model: str | None = None, batch_size: int = 256
    ) -> tuple[list[list[float]], CallRecord]:
        """Embeddings com cache por texto; só os textos ausentes vão para a API."""
        if any(not t.strip() for t in texts):
            raise ValueError("textos vazios não podem ser embedados")
        model = model or self.embed_model
        keys = [cache_key({"kind": "embed", "model": model, "input": t}) for t in texts]
        out: list[list[float] | None] = [None] * len(texts)
        tokens = latency = 0.0
        missing: list[int] = []
        for i, k in enumerate(keys):
            entry = self.cache.get(k)
            if entry is None:
                missing.append(i)
                continue
            out[i] = entry["embedding"]
            tokens += entry["tokens"]
            latency += entry["latency_ms"]
        snapshot = None
        for b in range(0, len(missing), batch_size):
            batch = missing[b : b + batch_size]
            t0 = time.perf_counter()
            resp = self.client.embeddings.create(model=model, input=[texts[i] for i in batch])
            lat = (time.perf_counter() - t0) * 1000
            snapshot = getattr(resp, "model", None)
            share_tok = resp.usage.prompt_tokens / len(batch)  # rateio aproximado por texto
            for i, d in zip(batch, sorted(resp.data, key=lambda d: d.index), strict=True):
                out[i] = d.embedding
                self.cache.put(
                    keys[i],
                    {"embedding": d.embedding, "tokens": share_tok, "latency_ms": lat / len(batch)},
                )
            tokens += resp.usage.prompt_tokens
            latency += lat
        entry = {
            "input_tokens": round(tokens),
            "output_tokens": 0,
            "latency_ms": latency,
            "model_snapshot": snapshot,
        }
        key = cache_key({"kind": "embed-batch", "keys": keys})
        rec = self._record("embed", model, None, entry, not missing, key)
        return out, rec  # type: ignore[return-value]

    # ------------------------------------------------------------------ ledger

    def _record(
        self,
        kind: str,
        model: str,
        prompt_version: str | None,
        entry: dict[str, Any],
        hit: bool,
        key: str,
    ) -> CallRecord:
        rec = CallRecord(
            kind=kind,
            model=model,
            model_snapshot=entry.get("model_snapshot"),
            prompt_version=prompt_version,
            input_tokens=entry["input_tokens"],
            output_tokens=entry["output_tokens"],
            cost_usd=cost_usd(model, entry["input_tokens"], entry["output_tokens"]),
            latency_ms=entry["latency_ms"],
            cache_hit=hit,
            key=key,
        )
        with self._lock:
            self.records.append(rec)
            if self.ledger_path:
                self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
                with self.ledger_path.open("a", encoding="utf-8") as f:
                    line = json.dumps({"ts": time.time(), **asdict(rec)}, ensure_ascii=False)
                    f.write(line + "\n")
        return rec

    def summary(self) -> dict[str, Any]:
        recs = self.records
        return {
            "calls": len(recs),
            "cache_hits": sum(r.cache_hit for r in recs),
            "input_tokens": sum(r.input_tokens for r in recs),
            "output_tokens": sum(r.output_tokens for r in recs),
            "cost_usd": sum(r.cost_usd or 0.0 for r in recs),
            "latency_ms": sum(r.latency_ms for r in recs),
            "unpriced_models": sorted({r.model for r in recs if r.cost_usd is None}),
        }

    def reset(self) -> None:
        self.records.clear()
