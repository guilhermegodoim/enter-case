from types import SimpleNamespace as NS

import pytest
from pydantic import BaseModel

from common.llm import LLM, LLMError


class Out(BaseModel):
    tese: str


class FakeClient:
    def __init__(self):
        self.chat_calls = 0
        self.embed_inputs: list[list[str]] = []
        self.status = "completed"
        self.refusal: str | None = None
        self.responses = NS(parse=self._parse)
        self.embeddings = NS(create=self._embed)

    def _parse(self, **kwargs):
        self.chat_calls += 1
        assert kwargs["store"] is False
        content = [NS(type="output_text", text='{"tese": "dano moral"}')]
        if self.refusal:
            content = [NS(type="refusal", refusal=self.refusal)]
        return NS(
            status=self.status,
            incomplete_details=None if self.status == "completed" else NS(reason="max_tokens"),
            output=[NS(type="message", content=content)],
            output_text='{"tese": "dano moral"}',
            usage=NS(input_tokens=100, output_tokens=10),
            model="gpt-4.1-mini-2025-04-14",
        )

    def _embed(self, model, input):
        self.embed_inputs.append(list(input))
        data = [NS(index=i, embedding=[float(len(t)), 1.0]) for i, t in enumerate(input)]
        return NS(data=data, usage=NS(prompt_tokens=5 * len(input)), model=model)


@pytest.fixture
def llm(tmp_path):
    fake = FakeClient()
    ledger = tmp_path / "ledger.jsonl"
    return LLM(client=fake, cache_dir=tmp_path / "cache", ledger_path=ledger), fake


def call(llm, user="petição"):
    return llm.structured(Out, system="s", user=user, model="gpt-4.1-mini", prompt_version="v1")


def test_cache_hit_skips_api_and_preserves_metrics(llm):
    client_llm, fake = llm
    out1, r1 = call(client_llm)
    out2, r2 = call(client_llm)
    assert fake.chat_calls == 1
    assert out1 == out2 == Out(tese="dano moral")
    assert (r1.cache_hit, r2.cache_hit) == (False, True)
    assert r1.cost_usd == r2.cost_usd == pytest.approx((100 * 0.40 + 10 * 1.60) / 1e6)
    assert r1.latency_ms == r2.latency_ms
    assert r2.model_snapshot == "gpt-4.1-mini-2025-04-14"


def test_prompt_change_misses_cache(llm):
    client_llm, fake = llm
    call(client_llm, "a")
    call(client_llm, "b")
    assert fake.chat_calls == 2


def test_cache_persists_across_instances(llm, tmp_path):
    _, fake = llm
    call(LLM(client=fake, cache_dir=tmp_path / "cache"))
    call(LLM(client=fake, cache_dir=tmp_path / "cache"))
    assert fake.chat_calls == 1


def test_ledger_and_summary(llm, tmp_path):
    client_llm, _ = llm
    call(client_llm)
    call(client_llm)
    lines = (tmp_path / "ledger.jsonl").read_text().splitlines()
    assert len(lines) == 2
    s = client_llm.summary()
    assert s["calls"] == 2 and s["cache_hits"] == 1 and s["unpriced_models"] == []


def test_embed_only_sends_uncached_texts(llm):
    client_llm, fake = llm
    v1, r1 = client_llm.embed(["voo", "banco"], model="text-embedding-3-small")
    v2, r2 = client_llm.embed(["banco", "trabalho", "voo"], model="text-embedding-3-small")
    assert fake.embed_inputs == [["voo", "banco"], ["trabalho"]]
    assert v2[0] == v1[1] and v2[2] == v1[0]
    assert not r1.cache_hit and not r2.cache_hit
    _, r3 = client_llm.embed(["voo"], model="text-embedding-3-small")
    assert r3.cache_hit


def test_incomplete_and_refusal_raise_and_are_not_cached(llm):
    client_llm, fake = llm
    fake.status = "incomplete"
    with pytest.raises(LLMError, match="incompleta"):
        call(client_llm)
    fake.status, fake.refusal = "completed", "não posso"
    with pytest.raises(LLMError, match="recusa"):
        call(client_llm)
    fake.refusal = None
    _, rec = call(client_llm)
    assert not rec.cache_hit and fake.chat_calls == 3


def test_missing_key_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(LLMError):
        _ = LLM(cache_dir=tmp_path).client


@pytest.mark.live
def test_live_smoke(tmp_path):
    llm = LLM(cache_dir=tmp_path)
    if not llm.settings.openai_api_key:
        pytest.skip("OPENAI_API_KEY ausente")
    out, rec = llm.structured(Out, system="Extraia a tese em 3 palavras.", user="Voo cancelado.")
    assert out.tese and rec.input_tokens > 0


def test_invalid_json_from_sdk_becomes_llm_error_and_is_not_cached(tmp_path):
    from pydantic import ValidationError

    try:
        Out.model_validate_json('{"tese": "x" } } } }')
    except ValidationError as e:
        err = e

    def parse(**kwargs):
        raise err

    llm = LLM(client=NS(responses=NS(parse=parse)), cache_dir=tmp_path)
    with pytest.raises(LLMError, match="schema Out"):
        llm.structured(Out, system="s", user="u")
    assert not any(tmp_path.rglob("*.json"))
