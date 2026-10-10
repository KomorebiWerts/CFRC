"""ToolSandbox reached-state PAPC residual runtime seam.

This module is intentionally ToolSandbox-local.  It restores a private cheap
reached-state context, executes only the code attached to projected residual
suffix units, and then delegates scoring to ToolSandbox's official scenario
evaluator.  It does not call a provider; the strong proposal is supplied by the
caller after the one-B-call proposer step.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from papc.certspec.toolsandbox.adapter import ToolSandboxOfflineAdapter

_NAMED_SCENARIOS_CACHE: dict[tuple[str, str], Mapping[str, Any]] = {}


def execute_projected_suffix_and_score(
    *,
    checkout: Path,
    scenario_name: str,
    reached_context_json: Path,
    digest: Mapping[str, Any],
    proposal: Mapping[str, Any],
    output_directory: Path,
) -> dict[str, Any]:
    """Execute a projected suffix from a cheap reached state and official-score it.

    The returned object is private-run material.  Public scripts must reduce it
    before publication.
    """

    _prepare_toolsandbox_imports(checkout)
    _patch_numpy_compat()

    from tool_sandbox.common.execution_context import (  # type: ignore
        ExecutionContext,
        set_current_context,
    )
    from tool_sandbox.common.tool_discovery import ToolBackend  # type: ignore

    context = ExecutionContext.from_dict(json.loads(reached_context_json.read_text(encoding="utf-8")))
    set_current_context(context)
    _replay_execution_environment_setup(context)

    adapter = ToolSandboxOfflineAdapter()
    projection = adapter.project(digest, proposal)
    executed_calls = []
    runtime_errors = []
    if projection.get("execute") is True:
        for call in projected_code_calls(projection=projection, proposal=proposal):
            result = _execute_code_call(context, str(call["code"]))
            executed_calls.append(
                {
                    "effect_units": list(call.get("effect_units") or []),
                    "code_sha256": call.get("code_sha256"),
                    "tool_call_exception": result.get("tool_call_exception"),
                }
            )
            if result.get("tool_call_exception"):
                runtime_errors.append(result["tool_call_exception"])

    scenarios = named_scenarios_for_checkout(
        checkout,
        preferred_tool_backend=ToolBackend.DEFAULT,
    )
    scenario = scenarios[scenario_name]
    evaluation_result = scenario.evaluation.evaluate(
        execution_context=context,
        max_turn_count=scenario.max_messages,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    ending_context_path = output_directory / "execution_context_after_projected_suffix.json"
    ending_context_path.write_text(
        json.dumps(context.to_dict(serialize_console=False), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "schema": "papc.toolsandbox.method_runtime_private_receipt.v1",
        "scenario_name": scenario_name,
        "projection": projection,
        "executed_projected_call_count": len(executed_calls),
        "executed_projected_calls": executed_calls,
        "runtime_error_count": len(runtime_errors),
        "runtime_errors": runtime_errors,
        "evaluation": official_evaluation_summary(evaluation_result),
        "ending_context_path": str(ending_context_path),
        "provider_calls": 0,
    }


def projected_code_calls(*, projection: Mapping[str, Any], proposal: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return proposal calls whose effect units survived projection."""

    if projection.get("execute") is not True:
        return []
    projected = set(_sequence(nested(projection, "action_projection", "projected_footprint")))
    if not projected:
        return []
    selected: list[dict[str, Any]] = []
    for call in _sequence(proposal.get("tool_calls")):
        call_map = call if isinstance(call, Mapping) else {}
        effect_units = [str(unit) for unit in _sequence(call_map.get("effect_units"))]
        code = str(call_map.get("code") or "").strip()
        if not code or not projected.intersection(effect_units):
            continue
        selected.append(
            {
                "effect_units": effect_units,
                "code": code,
                "code_sha256": stable_code_sha256(code),
            }
        )
    return selected


def official_evaluation_summary(evaluation_result: Any) -> dict[str, Any]:
    """Reduce ToolSandbox's official EvaluationResult without scorer-side labels."""

    return {
        "milestone_similarity": round(float(getattr(evaluation_result, "milestone_similarity", 0.0) or 0.0), 6),
        "minefield_similarity": round(float(getattr(evaluation_result, "minefield_similarity", 0.0) or 0.0), 6),
        "similarity": round(float(getattr(evaluation_result, "similarity", 0.0) or 0.0), 6),
        "turn_count": int(getattr(evaluation_result, "turn_count", 0) or 0),
    }


def nested(mapping_: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping_
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def stable_code_sha256(code: str) -> str:
    import hashlib

    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _execute_code_call(context: Any, code: str) -> dict[str, Any]:
    from tool_sandbox.common.execution_context import (  # type: ignore
        DatabaseNamespace,
        RoleType,
        set_current_context,
    )
    from tool_sandbox.common.message_conversion import Message  # type: ignore
    from tool_sandbox.roles.execution_environment import (
        ExecutionEnvironment,  # type: ignore
    )

    set_current_context(context)
    ExecutionEnvironment.add_messages(
        [
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content=code,
                visible_to=[RoleType.AGENT, RoleType.EXECUTION_ENVIRONMENT],
            )
        ]
    )
    ExecutionEnvironment().respond()
    sandbox = context.get_database(
        namespace=DatabaseNamespace.SANDBOX,
        get_all_history_snapshots=True,
        drop_sandbox_message_index=False,
    )
    last = sandbox.to_dicts()[-1] if not sandbox.is_empty() else {}
    return {
        "tool_call_exception": last.get("tool_call_exception"),
        "response_present": bool(last.get("content")),
    }


def _replay_execution_environment_setup(context: Any) -> None:
    from tool_sandbox.common.execution_context import (  # type: ignore
        DatabaseNamespace,
        RoleType,
        set_current_context,
    )

    set_current_context(context)
    sandbox = context.get_database(
        namespace=DatabaseNamespace.SANDBOX,
        get_all_history_snapshots=True,
        drop_sandbox_message_index=False,
    )
    for row in sandbox.to_dicts():
        if row.get("sender") == RoleType.SYSTEM and row.get("recipient") == RoleType.EXECUTION_ENVIRONMENT:
            context.interactive_console.runcode(compile(str(row.get("content") or ""), "<toolsandbox-setup>", "exec"))


def _prepare_toolsandbox_imports(checkout: Path) -> None:
    checkout_path = str(checkout)
    if checkout_path not in sys.path:
        sys.path.insert(0, checkout_path)
    insertion_index = sys.path.index(checkout_path) + 1
    for site_packages in _embedded_toolsandbox_site_packages(checkout):
        site_path = str(site_packages)
        if site_path not in sys.path:
            sys.path.insert(insertion_index, site_path)
            insertion_index += 1


def named_scenarios_for_checkout(
    checkout: Path,
    *,
    preferred_tool_backend: Any,
) -> Mapping[str, Any]:
    """Load and cache ToolSandbox's expensive named scenario table."""

    _prepare_toolsandbox_imports(checkout)
    _patch_numpy_compat()
    from tool_sandbox.scenarios import named_scenarios  # type: ignore

    cache_key = (
        str(checkout.resolve()),
        f"{type(preferred_tool_backend).__module__}."
        f"{type(preferred_tool_backend).__qualname__}:{preferred_tool_backend}",
    )
    if cache_key not in _NAMED_SCENARIOS_CACHE:
        _NAMED_SCENARIOS_CACHE[cache_key] = named_scenarios(
            preferred_tool_backend=preferred_tool_backend
        )
    return _NAMED_SCENARIOS_CACHE[cache_key]


def _embedded_toolsandbox_site_packages(checkout: Path) -> list[Path]:
    venv_lib = checkout / ".venv" / "lib"
    if not venv_lib.exists():
        return []
    return sorted(path for path in venv_lib.glob("python*/site-packages") if path.is_dir())


def _patch_numpy_compat() -> None:
    try:
        import numpy as np  # type: ignore
    except Exception:  # noqa: BLE001 - ToolSandbox will report import failures later.
        return
    if not hasattr(np, "cumproduct") and hasattr(np, "cumprod"):
        np.cumproduct = np.cumprod  # type: ignore[attr-defined]


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


__all__ = [
    "execute_projected_suffix_and_score",
    "named_scenarios_for_checkout",
    "official_evaluation_summary",
    "projected_code_calls",
    "stable_code_sha256",
]
