"""Read agent-side latency out of a run artifact, on the same caliber as cost.

Cost and latency do not move together, so a bill alone can recommend the wrong
operating point. The Retail zero-obligation arm is the cheapest one measured
(1.5573x) and also the slowest (131.2 seconds per episode against the registered
arm's 114.1), because the work it takes off the successor lands on a cheap model
that needs more turns to finish it. Reporting one axis without the other hides
that entirely.

The caliber matches ``papc.certspec.cost_readout``: agent roles only. The user
simulator and the evaluator are harness roles, and every artifact this reads
already reports their latency as unavailable rather than zero, so the exclusion
is the artifact's own and not one this script imposes.

Two numbers are reported because they answer different questions:

``model_latency_seconds``
    The summed latency of the agent-side provider calls. This is the number to
    compare between arms, and it is what ``latency_accounting`` in the artifact
    declares primary. It is independent of how many episodes ran in parallel.

``mean_episode_wall_seconds``
    Wall clock for one episode end to end, including tool execution and the
    harness roles. This is what a user would feel. It is not the sum of model
    latencies and must not be added across episodes.

Wall clock for the whole run is deliberately not reported as a comparison: it is
a function of the concurrency the operator chose, not of the method.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _external_time_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """tau2 and Agent-Diff shape: one entry per arm under external_time_accounting."""

    rows: list[dict[str, Any]] = []
    accounting = payload.get("external_time_accounting")
    if not isinstance(accounting, Mapping):
        return rows
    for arm, value in sorted(accounting.items()):
        if not isinstance(value, Mapping):
            continue
        rows.append(
            {
                "arm": str(arm),
                "model_latency_seconds": value.get("aggregate_model_latency_seconds"),
                "mean_episode_wall_seconds": value.get("mean_episode_wall_seconds"),
                "aggregate_episode_wall_seconds": value.get(
                    "aggregate_episode_wall_seconds"
                ),
                "latency_unavailable_roles": list(
                    value.get("latency_unavailable_roles") or ()
                ),
                "model_latency_is_partial": bool(
                    value.get("model_latency_is_partial_when_roles_unavailable")
                ),
            }
        )
    return rows


def _per_role_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """STATE shape: latency lives per role in the runtime ledger.

    Only the two agent roles are summed. A role whose latency the provider never
    returned is reported by name rather than counted as zero, because a missing
    latency and a zero latency are different claims.
    """

    ledger = payload.get("runtime_ledger")
    if not isinstance(ledger, Mapping):
        return []
    per_role = ledger.get("per_role")
    if not isinstance(per_role, Mapping):
        return []
    total = 0.0
    observed: list[str] = []
    missing: list[str] = []
    for role in ("cheap", "strong"):
        value = per_role.get(role)
        if not isinstance(value, Mapping):
            continue
        seconds = value.get("latency_seconds")
        if seconds is None:
            missing.append(role)
            continue
        total += float(seconds)
        observed.append(role)
    if not observed:
        return []
    return [
        {
            "arm": "agent_roles",
            "model_latency_seconds": total,
            "mean_episode_wall_seconds": None,
            "aggregate_episode_wall_seconds": None,
            "latency_unavailable_roles": missing,
            "model_latency_is_partial": bool(missing),
        }
    ]


def _agent_resource_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Agent-Diff shape: one agent_resource block with the two roles split out."""

    resource = payload.get("agent_resource")
    if not isinstance(resource, Mapping):
        return []
    total = resource.get("model_latency_seconds")
    if total is None:
        return []
    return [
        {
            "arm": "agent_resource",
            "model_latency_seconds": total,
            "mean_episode_wall_seconds": None,
            "aggregate_episode_wall_seconds": resource.get("elapsed_seconds"),
            "cheap_latency_seconds": resource.get("cheap_model_latency_seconds"),
            "strong_latency_seconds": resource.get("strong_model_latency_seconds"),
            "latency_unavailable_roles": [],
            "model_latency_is_partial": False,
        }
    ]


def _resource_accounting_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """ToolSandbox shape, and it is sharded, so one artifact is a partial run.

    ``agent_side`` is the caliber that matches the cost readout. ``full_runtime``
    additionally carries the harness roles and is reported so nobody reads the
    smaller number as the wall a user would feel.
    """

    accounting = payload.get("resource_accounting")
    if not isinstance(accounting, Mapping):
        return []
    method = accounting.get("method_resources")
    if not isinstance(method, Mapping):
        return []
    breakdown = method.get("runtime_breakdown")
    if not isinstance(breakdown, Mapping):
        return []
    agent_side = breakdown.get("agent_side")
    if not isinstance(agent_side, Mapping):
        return []
    total = agent_side.get("latency_seconds")
    if total is None:
        return []
    full = breakdown.get("full_runtime")
    return [
        {
            "arm": "agent_side (this shard only)",
            "model_latency_seconds": total,
            "mean_episode_wall_seconds": None,
            "aggregate_episode_wall_seconds": (
                full.get("latency_seconds") if isinstance(full, Mapping) else None
            ),
            "latency_unavailable_roles": ["evaluator"],
            "model_latency_is_partial": True,
        }
    ]


def _official_row_directory_rows(path: Path) -> list[dict[str, Any]]:
    """STATE official-script baseline: one file per task, latency on the ledger."""

    files = sorted(path.rglob("*.json"))
    total = 0.0
    counted = 0
    for candidate in files:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if not isinstance(payload, Mapping):
            continue
        ledger = payload.get("papc_runtime_ledger")
        if not isinstance(ledger, Mapping):
            continue
        seconds = ledger.get("provider_latency_seconds")
        if seconds is None:
            continue
        total += float(seconds)
        counted += 1
    if not counted:
        return []
    return [
        {
            "arm": f"official baseline over {counted} rows",
            "model_latency_seconds": total,
            "mean_episode_wall_seconds": None,
            "aggregate_episode_wall_seconds": None,
            "mean_row_model_latency_seconds": total / counted,
            "latency_unavailable_roles": [],
            "model_latency_is_partial": False,
        }
    ]


def _receipt_directory_rows(path: Path) -> list[dict[str, Any]]:
    """STATE writes one receipt per row, so the run's latency is their sum.

    A row whose provider never returned a latency is named rather than counted
    as zero, and the per-row mean is reported over the rows that did return one.
    """

    receipts = sorted(path.glob("receipt_*.json"))
    if not receipts:
        return []
    total = 0.0
    rows_with_latency = 0
    missing: set[str] = set()
    for receipt in receipts:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        per_role = ((payload.get("runtime_ledger") or {}).get("per_role")) or {}
        row_total = 0.0
        row_observed = False
        for role in ("cheap", "strong"):
            value = per_role.get(role)
            if not isinstance(value, Mapping):
                continue
            seconds = value.get("latency_seconds")
            if seconds is None:
                if value.get("calls"):
                    missing.add(role)
                continue
            row_total += float(seconds)
            row_observed = True
        if row_observed:
            total += row_total
            rows_with_latency += 1
    if not rows_with_latency:
        return []
    return [
        {
            "arm": f"agent_roles over {rows_with_latency} receipts",
            "model_latency_seconds": total,
            "mean_episode_wall_seconds": None,
            "aggregate_episode_wall_seconds": None,
            "mean_row_model_latency_seconds": total / rows_with_latency,
            "latency_unavailable_roles": sorted(missing),
            "model_latency_is_partial": bool(missing),
        }
    ]


def read_latency(path: Path) -> dict[str, Any]:
    if path.is_dir():
        rows = _receipt_directory_rows(path) or _official_row_directory_rows(path)
        return {"artifact": str(path), "rows": rows, "latency_available": bool(rows)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"artifact is not a JSON object: {path}")
    rows = (
        _external_time_rows(payload)
        or _agent_resource_rows(payload)
        or _resource_accounting_rows(payload)
        or _per_role_rows(payload)
    )
    return {
        "artifact": str(path),
        "rows": rows,
        "latency_available": bool(rows),
    }


def _format(report: Mapping[str, Any]) -> str:
    lines = [str(report["artifact"])]
    if not report["rows"]:
        lines.append("  no agent-side latency recorded in this artifact")
        return "\n".join(lines)
    for row in report["rows"]:
        latency = row["model_latency_seconds"]
        wall = row["mean_episode_wall_seconds"]
        parts = [f"  {row['arm']}"]
        parts.append(
            "model_latency=unavailable"
            if latency is None
            else f"model_latency={float(latency):.1f}s"
        )
        parts.append(
            "mean_episode_wall=unavailable"
            if wall is None
            else f"mean_episode_wall={float(wall):.1f}s"
        )
        per_row = row.get("mean_row_model_latency_seconds")
        if per_row is not None:
            parts.append(f"mean_row_model_latency={float(per_row):.1f}s")
        if row["latency_unavailable_roles"]:
            parts.append(
                "excluded_roles=" + ",".join(row["latency_unavailable_roles"])
            )
        lines.append("  ".join(parts))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Agent-side latency for one or more run artifacts, on the same "
            "caliber as the cache-neutral cost readout."
        )
    )
    parser.add_argument("artifact", nargs="+", type=Path)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the reports as JSON instead of a table.",
    )
    args = parser.parse_args()
    reports = [read_latency(path) for path in args.artifact]
    if args.json:
        print(json.dumps(reports, indent=2, sort_keys=True))
        return 0
    for report in reports:
        print(_format(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
