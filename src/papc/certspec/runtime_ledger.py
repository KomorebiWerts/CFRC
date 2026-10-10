"""Public-safe runtime usage and latency ledger for PAPC method arms.

The ledger is benchmark-independent. It accepts only numeric accounting fields
that are already safe to publish: token counts, elapsed seconds, costs, and a
coarse runtime role. Raw prompts, messages, tool payloads, provider JSON, task
ids, and state diffs must stay outside this module.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from papc.certspec.provider import UsageAccumulator, as_mapping

SCHEMA = "papc.runtime_ledger.v1"
ROLES = (
    "cheap",
    "strong",
    "router",
    "output_only_strong",
    "evaluator",
    "user_simulator",
)


@dataclass(frozen=True)
class RoleCall:
    role: str
    call_count: int = 1
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    latency_seconds: float | None = None
    cost_usd: float | None = None
    token_usage_available: bool = True


def role_call_from_usage(
    role: str,
    usage: Mapping[str, Any] | None,
    *,
    latency_seconds: float | None = None,
    cost_usd: float | None = None,
    call_count: int = 1,
) -> RoleCall:
    """Normalize one provider usage record, retaining missing-total status."""

    usage_map = as_mapping(usage)
    token_totals = UsageAccumulator().add_usage(usage_map).as_dict()
    reasoning_tokens = token_totals["reasoning_tokens"] or int(
        usage_map.get("reasoning_output_tokens")
        or usage_map.get("reasoning_tokens")
        or 0
    )
    return RoleCall(
        role=role,
        call_count=call_count,
        input_tokens=token_totals["input_tokens"],
        cached_input_tokens=cached_input_tokens_from_usage(usage_map),
        output_tokens=token_totals["output_tokens"],
        reasoning_tokens=reasoning_tokens,
        latency_seconds=latency_seconds,
        cost_usd=cost_usd,
        token_usage_available=_usage_available(usage_map),
    )


@dataclass
class _RoleTotals:
    calls: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cached_input_usage_call_count: int = 0
    cached_input_usage_unavailable_count: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    latency_seconds: float = 0.0
    cost_usd: float = 0.0
    token_usage_call_count: int = 0
    token_usage_unavailable_count: int = 0
    latency_call_count: int = 0
    latency_unavailable_count: int = 0
    cost_call_count: int = 0
    cost_unavailable_count: int = 0


class RuntimeLedger:
    """Accumulate public-safe runtime accounting by model role."""

    def __init__(self, roles: tuple[str, ...] = ROLES) -> None:
        self._roles = roles
        self._totals = {role: _RoleTotals() for role in roles}
        self._private_call_records: list[dict[str, Any]] = []

    def add(self, call: RoleCall) -> "RuntimeLedger":
        role = self._role(call.role)
        totals = self._totals[role]
        call_count = max(0, int(call.call_count))
        if call_count == 0:
            return self
        totals.calls += call_count
        if (
            call.token_usage_available
            and call.input_tokens is not None
            and call.output_tokens is not None
        ):
            totals.token_usage_call_count += call_count
            totals.input_tokens += _nonnegative_int(call.input_tokens)
            if call.cached_input_tokens is None:
                totals.cached_input_usage_unavailable_count += call_count
            else:
                totals.cached_input_usage_call_count += call_count
                totals.cached_input_tokens += min(
                    _nonnegative_int(call.cached_input_tokens),
                    _nonnegative_int(call.input_tokens),
                )
            totals.output_tokens += _nonnegative_int(call.output_tokens)
            totals.reasoning_tokens += _nonnegative_int(call.reasoning_tokens)
        else:
            totals.token_usage_unavailable_count += call_count
            totals.cached_input_usage_unavailable_count += call_count
        if call.latency_seconds is None:
            totals.latency_unavailable_count += call_count
        else:
            totals.latency_call_count += call_count
            totals.latency_seconds += max(0.0, float(call.latency_seconds))
        if call.cost_usd is None:
            totals.cost_unavailable_count += call_count
        else:
            totals.cost_call_count += call_count
            totals.cost_usd += max(0.0, float(call.cost_usd))
        return self

    def add_usage(
        self,
        role: str,
        usage: Mapping[str, Any] | None,
        *,
        latency_seconds: float | None = None,
        cost_usd: float | None = None,
        call_count: int = 1,
        provider_bound_request_bytes: int | None = None,
    ) -> "RuntimeLedger":
        usage_map = as_mapping(usage)
        usage_available = _usage_available(usage_map)
        private_record: dict[str, Any] = {
            "role": self._role(role),
            "call_count": max(1, int(call_count)),
            "usage_missing": not usage_available,
        }
        if provider_bound_request_bytes is not None:
            private_record["provider_bound_request_bytes"] = _nonnegative_int(
                provider_bound_request_bytes
            )
        self._private_call_records.append(private_record)
        return self.add(
            role_call_from_usage(
                role,
                usage_map,
                call_count=call_count,
                latency_seconds=latency_seconds,
                cost_usd=cost_usd,
            )
        )

    def private_call_records(self) -> list[dict[str, Any]]:
        """Return private per-call completeness metadata; never enters public summary."""

        return copy.deepcopy(self._private_call_records)

    def merge_public_summary(
        self, summary: Mapping[str, Any] | None
    ) -> "RuntimeLedger":
        summary_map = as_mapping(summary)
        per_role = as_mapping(summary_map.get("per_role"))
        for role in self._roles:
            role_summary = as_mapping(per_role.get(role))
            calls = int(role_summary.get("calls") or 0)
            if calls <= 0:
                continue
            totals = self._totals[role]
            totals.calls += calls
            totals.input_tokens += int(role_summary.get("input_tokens") or 0)
            totals.cached_input_tokens += int(
                role_summary.get("cached_input_tokens") or 0
            )
            totals.output_tokens += int(role_summary.get("output_tokens") or 0)
            totals.reasoning_tokens += int(role_summary.get("reasoning_tokens") or 0)
            totals.latency_seconds += float(role_summary.get("latency_seconds") or 0.0)
            totals.cost_usd += float(role_summary.get("cost_usd") or 0.0)
            totals.token_usage_call_count += int(
                role_summary.get("token_usage_call_count") or 0
            )
            totals.token_usage_unavailable_count += int(
                role_summary.get("token_usage_unavailable_count") or 0
            )
            if "cached_input_usage_call_count" in role_summary:
                totals.cached_input_usage_call_count += int(
                    role_summary.get("cached_input_usage_call_count") or 0
                )
                totals.cached_input_usage_unavailable_count += int(
                    role_summary.get("cached_input_usage_unavailable_count") or 0
                )
            elif role_summary.get("cached_input_tokens") is None or (
                role_summary.get("cached_input_tokens") == 0
            ):
                totals.cached_input_usage_unavailable_count += calls
            else:
                totals.cached_input_usage_call_count += int(
                    role_summary.get("token_usage_call_count") or 0
                )
                totals.cached_input_usage_unavailable_count += int(
                    role_summary.get("token_usage_unavailable_count") or 0
                )
            totals.latency_call_count += int(
                role_summary.get("latency_call_count") or 0
            )
            totals.latency_unavailable_count += int(
                role_summary.get("latency_unavailable_count") or 0
            )
            totals.cost_call_count += int(role_summary.get("cost_call_count") or 0)
            totals.cost_unavailable_count += int(
                role_summary.get("cost_unavailable_count") or 0
            )
        return self

    def public_summary(self) -> dict[str, Any]:
        per_role = {role: self._public_role_summary(role) for role in self._roles}
        totals = _RoleTotals()
        for role_totals in self._totals.values():
            totals.calls += role_totals.calls
            totals.input_tokens += role_totals.input_tokens
            totals.cached_input_tokens += role_totals.cached_input_tokens
            totals.cached_input_usage_call_count += (
                role_totals.cached_input_usage_call_count
            )
            totals.cached_input_usage_unavailable_count += (
                role_totals.cached_input_usage_unavailable_count
            )
            totals.output_tokens += role_totals.output_tokens
            totals.reasoning_tokens += role_totals.reasoning_tokens
            totals.latency_seconds += role_totals.latency_seconds
            totals.cost_usd += role_totals.cost_usd
            totals.token_usage_call_count += role_totals.token_usage_call_count
            totals.token_usage_unavailable_count += (
                role_totals.token_usage_unavailable_count
            )
            totals.latency_call_count += role_totals.latency_call_count
            totals.latency_unavailable_count += role_totals.latency_unavailable_count
            totals.cost_call_count += role_totals.cost_call_count
            totals.cost_unavailable_count += role_totals.cost_unavailable_count
        return {
            "schema": SCHEMA,
            "roles": list(self._roles),
            "public_safe": True,
            "raw_provider_payloads_excluded": True,
            "per_role": per_role,
            "totals": _summary_from_totals(totals),
        }

    def _public_role_summary(self, role: str) -> dict[str, Any]:
        return _summary_from_totals(self._totals[role])

    def _role(self, role: str) -> str:
        normalized = str(role or "").strip()
        if normalized not in self._totals:
            raise ValueError(f"unsupported runtime role: {role!r}")
        return normalized


def _summary_from_totals(totals: _RoleTotals) -> dict[str, Any]:
    has_token_usage = totals.token_usage_call_count > 0
    has_cached_input_usage = totals.cached_input_usage_call_count > 0
    has_latency = totals.latency_call_count > 0
    has_cost = totals.cost_call_count > 0
    token_usage_available = (
        totals.calls > 0
        and has_token_usage
        and totals.token_usage_unavailable_count == 0
    )
    cached_input_usage_available = (
        totals.calls > 0
        and has_cached_input_usage
        and totals.cached_input_usage_unavailable_count == 0
    )
    latency_available = (
        totals.calls > 0 and has_latency and totals.latency_unavailable_count == 0
    )
    cost_available = (
        totals.calls > 0 and has_cost and totals.cost_unavailable_count == 0
    )
    return {
        "observed": totals.calls > 0,
        "calls": totals.calls,
        "input_tokens": totals.input_tokens if has_token_usage else None,
        "cached_input_tokens": (
            totals.cached_input_tokens if cached_input_usage_available else None
        ),
        "billable_input_tokens": (
            max(0, totals.input_tokens - totals.cached_input_tokens)
            if has_token_usage and cached_input_usage_available
            else None
        ),
        "output_tokens": totals.output_tokens if has_token_usage else None,
        "reasoning_tokens": totals.reasoning_tokens if has_token_usage else None,
        "total_tokens": (
            totals.input_tokens + totals.output_tokens if has_token_usage else None
        ),
        "latency_seconds": round(totals.latency_seconds, 6) if has_latency else None,
        "cost_usd": round(totals.cost_usd, 6) if has_cost else None,
        "token_usage_available": token_usage_available,
        "token_usage_call_count": totals.token_usage_call_count,
        "token_usage_unavailable_count": totals.token_usage_unavailable_count,
        "cached_input_usage_available": cached_input_usage_available,
        "cached_input_usage_call_count": totals.cached_input_usage_call_count,
        "cached_input_usage_unavailable_count": (
            totals.cached_input_usage_unavailable_count
        ),
        "latency_available": latency_available,
        "latency_call_count": totals.latency_call_count,
        "latency_unavailable_count": totals.latency_unavailable_count,
        "cost_available": cost_available,
        "cost_call_count": totals.cost_call_count,
        "cost_unavailable_count": totals.cost_unavailable_count,
    }


def _usage_available(usage: Mapping[str, Any]) -> bool:
    # Cache/reasoning details do not establish the billable input/output totals.
    # Explicit zero is known usage; a missing or null total is not.
    return any(
        usage.get(key) is not None for key in ("input_tokens", "prompt_tokens")
    ) and any(
        usage.get(key) is not None for key in ("output_tokens", "completion_tokens")
    )


def cached_input_tokens_from_usage(usage: Mapping[str, Any]) -> int | None:
    """Return explicit provider cache usage; missing cache metadata stays None."""

    direct = usage.get("cached_input_tokens")
    if direct is not None:
        return _nonnegative_int(direct)
    input_details = as_mapping(usage.get("input_tokens_details"))
    prompt_details = as_mapping(usage.get("prompt_tokens_details"))
    if input_details.get("cached_tokens") is not None:
        return _nonnegative_int(input_details["cached_tokens"])
    if prompt_details.get("cached_tokens") is not None:
        return _nonnegative_int(prompt_details["cached_tokens"])
    return None


def _nonnegative_int(value: int | None) -> int:
    return max(0, int(value or 0))
