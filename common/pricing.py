"""Preços OpenAI em USD por 1M tokens: (input, output).

Valores de referência; conferir em https://openai.com/api/pricing antes de citar
custos nas respostas escritas e atualizar PRICES_CHECKED_AT.
"""

from __future__ import annotations

PRICES_CHECKED_AT = "2026-10-16"

PRICES: dict[str, tuple[float, float]] = {
    "gpt-5": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}


def price_for(model: str) -> tuple[float, float] | None:
    """Casa nome exato ou snapshot datado (ex.: gpt-4.1-mini-2025-04-14) pelo prefixo mais longo."""
    candidates = [k for k in PRICES if model == k or model.startswith(k + "-")]
    if not candidates:
        return None
    return PRICES[max(candidates, key=len)]


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    price = price_for(model)
    if price is None:
        return None
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000
