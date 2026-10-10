"""ToolSandbox public-safe runtime accounting helpers."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from papc.certspec.runtime_ledger import RuntimeLedger


class ToolSandboxRuntimeAccountingRecorder:
    """Record model-call usage and latency from ToolSandbox role objects."""

    def __init__(self, ledger: RuntimeLedger | None = None) -> None:
        self._ledger = ledger or RuntimeLedger()

    def wrap_role_model_inference(
        self,
        role: Any,
        *,
        role_name: str,
        runtime_role: str,
    ) -> None:
        """Wrap ``role.model_inference`` and account for one provider call per call."""

        original = getattr(role, "model_inference", None)
        if original is None:
            raise AttributeError(f"{role_name} role has no model_inference")
        if getattr(original, "_papc_runtime_accounting_wrapped", False):
            return

        def accounted_model_inference(*args: Any, **kwargs: Any) -> Any:
            started = time.monotonic()
            try:
                response = original(*args, **kwargs)
            except Exception:
                self._ledger.add_usage(
                    runtime_role,
                    None,
                    latency_seconds=round(time.monotonic() - started, 6),
                )
                raise
            self._ledger.add_usage(
                runtime_role,
                _usage_from_response(response),
                latency_seconds=round(time.monotonic() - started, 6),
            )
            return response

        accounted_model_inference._papc_runtime_accounting_wrapped = True  # type: ignore[attr-defined]
        role.model_inference = accounted_model_inference

    def public_summary(self) -> dict[str, Any]:
        return self._ledger.public_summary()


def _usage_from_response(response: Any) -> Mapping[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, Mapping):
        usage = response.get("usage")
    if hasattr(usage, "model_dump"):
        dumped = usage.model_dump()
        return dict(dumped) if isinstance(dumped, Mapping) else None
    if isinstance(usage, Mapping):
        return dict(usage)
    return None


__all__ = ["ToolSandboxRuntimeAccountingRecorder"]
