"""Method-neutral online accounting for protected routing baselines."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Literal

ReceiptRole = Literal[
    "router",
    "cheap_agent",
    "strong_agent",
    "verifier",
    "recovery",
    "user_simulator",
    "evaluator",
]

AGENT_SIDE_ROLES = frozenset(
    {"router", "cheap_agent", "strong_agent", "verifier", "recovery"}
)
HARNESS_ROLES = frozenset({"user_simulator", "evaluator"})


@dataclass(frozen=True)
class ModelPrice:
    input_per_million_usd: float
    output_per_million_usd: float

    def __post_init__(self) -> None:
        for field_name in (
            "input_per_million_usd",
            "output_per_million_usd",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{field_name}_invalid")


FROZEN_CACHE_NEUTRAL_PRICES: Mapping[str, ModelPrice] = MappingProxyType(
    {
        "gpt-5.4-mini": ModelPrice(0.75, 4.5),
        "gpt-5.5": ModelPrice(5.0, 30.0),
        "text-embedding-3-small": ModelPrice(0.02, 0.0),
    }
)
EMBEDDING_PRICE_SOURCE = (
    "https://developers.openai.com/api/docs/guides/embeddings"
    "#embedding-models"
)
EMBEDDING_PRICE_SNAPSHOT_DATE = "2026-07-30"
EMBEDDING_PRICE_DERIVATION = (
    "62500_pages_per_usd_times_800_tokens_per_page_equals_"
    "50000000_input_tokens_per_usd"
)


@dataclass(frozen=True)
class CallReceipt:
    role: ReceiptRole
    phase: str
    model: str
    reasoning_effort: str
    provider_calls: int
    input_tokens: int | None
    cached_input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    retry_count: int
    latency_seconds: float
    provider_reported_cost_usd: float | None = None

    def __post_init__(self) -> None:
        if self.role not in AGENT_SIDE_ROLES | HARNESS_ROLES:
            raise ValueError("router_receipt_role_invalid")
        for field_name in ("phase", "model", "reasoning_effort"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"router_receipt_{field_name}_invalid")
        for field_name in ("provider_calls", "retry_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"router_receipt_{field_name}_invalid")
        for field_name in (
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_tokens",
        ):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"router_receipt_{field_name}_invalid")
        if (
            self.input_tokens is not None
            and self.cached_input_tokens is not None
            and self.cached_input_tokens > self.input_tokens
        ):
            raise ValueError("router_receipt_cached_tokens_exceed_input")
        if (
            isinstance(self.latency_seconds, bool)
            or not isinstance(self.latency_seconds, (int, float))
            or not math.isfinite(float(self.latency_seconds))
            or float(self.latency_seconds) < 0.0
        ):
            raise ValueError("router_receipt_latency_invalid")
        if self.provider_reported_cost_usd is not None and (
            isinstance(self.provider_reported_cost_usd, bool)
            or not isinstance(self.provider_reported_cost_usd, (int, float))
            or not math.isfinite(float(self.provider_reported_cost_usd))
            or float(self.provider_reported_cost_usd) < 0.0
        ):
            raise ValueError("router_receipt_provider_cost_invalid")
        if self.provider_calls == 0 and any(
            value not in {None, 0}
            for value in (
                self.input_tokens,
                self.cached_input_tokens,
                self.output_tokens,
                self.reasoning_tokens,
            )
        ):
            raise ValueError("local_router_receipt_has_provider_tokens")

    @classmethod
    def local_router(
        cls,
        *,
        method: str,
        latency_seconds: float,
    ) -> CallReceipt:
        return cls(
            role="router",
            phase="route",
            model=method,
            reasoning_effort="local",
            provider_calls=0,
            input_tokens=0,
            cached_input_tokens=0,
            output_tokens=0,
            reasoning_tokens=0,
            retry_count=0,
            latency_seconds=latency_seconds,
            provider_reported_cost_usd=0.0,
        )

    def public_record(self) -> dict[str, Any]:
        return asdict(self)


def aggregate_online_accounting(
    receipts: Sequence[CallReceipt],
    *,
    prices: Mapping[str, ModelPrice],
) -> dict[str, Any]:
    resolved = tuple(receipts)
    if any(not isinstance(receipt, CallReceipt) for receipt in resolved):
        raise TypeError("router_accounting_receipt_invalid")
    agent_receipts = tuple(
        receipt for receipt in resolved if receipt.role in AGENT_SIDE_ROLES
    )
    harness_receipts = tuple(
        receipt for receipt in resolved if receipt.role in HARNESS_ROLES
    )
    agent_costs = tuple(
        _cache_neutral_cost(receipt, prices=prices)
        for receipt in agent_receipts
    )
    harness_costs = tuple(
        _cache_neutral_cost(receipt, prices=prices)
        for receipt in harness_receipts
    )
    return {
        "receipt_count": len(resolved),
        "agent_side": _role_summary(agent_receipts, agent_costs),
        "excluded_harness": _role_summary(harness_receipts, harness_costs),
        "cache_neutral_pricing": True,
        "cost_scope": "provider_api_usd",
        "local_router_provider_cost_usd_is_zero": True,
    }


def _role_summary(
    receipts: tuple[CallReceipt, ...],
    costs: tuple[float | None, ...],
) -> dict[str, Any]:
    usage_complete = all(cost is not None for cost in costs)
    return {
        "receipt_count": len(receipts),
        "provider_calls": sum(receipt.provider_calls for receipt in receipts),
        "retry_count": sum(receipt.retry_count for receipt in receipts),
        "latency_seconds": round(
            sum(float(receipt.latency_seconds) for receipt in receipts),
            6,
        ),
        "input_tokens": _sum_optional(
            receipt.input_tokens for receipt in receipts
        ),
        "cached_input_tokens": _sum_optional(
            receipt.cached_input_tokens for receipt in receipts
        ),
        "output_tokens": _sum_optional(
            receipt.output_tokens for receipt in receipts
        ),
        "reasoning_tokens": _sum_optional(
            receipt.reasoning_tokens for receipt in receipts
        ),
        "usage_complete": usage_complete,
        "cost_usd": (
            round(sum(float(cost) for cost in costs if cost is not None), 9)
            if usage_complete
            else None
        ),
    }


def _cache_neutral_cost(
    receipt: CallReceipt,
    *,
    prices: Mapping[str, ModelPrice],
) -> float | None:
    if receipt.provider_calls == 0:
        return 0.0
    price = prices.get(receipt.model)
    if (
        price is None
        or receipt.input_tokens is None
        or receipt.output_tokens is None
    ):
        return None
    return (
        receipt.input_tokens * price.input_per_million_usd
        + receipt.output_tokens * price.output_per_million_usd
    ) / 1_000_000


def _sum_optional(values: Sequence[int | None] | Any) -> int | None:
    resolved = tuple(values)
    if any(value is None for value in resolved):
        return None
    return sum(int(value) for value in resolved)
