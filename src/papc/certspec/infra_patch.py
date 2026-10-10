"""Re-running the episodes a provider failure took, and only those.

A bounded run that loses an episode to a provider failure has an incomplete
denominator, and a denominator short of the official one is not a measurement.
Re-running the whole split to recover one episode spends the other thirty-nine
again for nothing, so the repair is to re-run the failed episodes alone and
substitute them into the base results.

Two rules keep that repair from selecting on outcome. Only episodes whose
termination reason was an infrastructure error may be re-run, so an episode
that produced a reward is never re-rolled. And the failed attempt's own decision
rows are dropped rather than added, so the merged cost is the cost of the
episodes the merged result reports and not of the attempts behind them.

The tau2 airline runner carried these functions privately. They are here
because the retail runner needs the same repair on the same artifact shape,
and a second copy is a second thing to keep correct.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

INFRASTRUCTURE_TERMINATION = "infrastructure_error"


def base_results_payload(
    base_root: Path, *, method_dir: str, results_name: str
) -> dict[str, Any] | None:
    """The private results this repo wrote for a base run, if they are there."""

    path = Path(base_root) / method_dir / results_name
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def infrastructure_task_ids(payload: dict[str, Any]) -> list[str]:
    """Episodes the base run never scored because the provider failed.

    These are the only episodes a patch may re-run. An episode that produced a
    reward was measured, and re-rolling it would select on outcome.
    """

    return sorted(
        {
            str(simulation.get("task_id"))
            for simulation in payload.get("simulations") or []
            if str(simulation.get("termination_reason")) == INFRASTRUCTURE_TERMINATION
        }
    )


def merge_patch_results(
    *,
    base_payload: dict[str, Any],
    patch_payload: dict[str, Any],
    patched_task_ids: Sequence[str],
) -> dict[str, Any]:
    """Return the base run with its infrastructure episodes substituted."""

    patched = {str(task_id) for task_id in patched_task_ids}
    replacement = {
        str(simulation.get("task_id")): simulation
        for simulation in patch_payload.get("simulations") or []
    }
    merged = [
        replacement.get(str(simulation.get("task_id")), simulation)
        if str(simulation.get("task_id")) in patched
        else simulation
        for simulation in base_payload.get("simulations") or []
    ]
    payload = dict(patch_payload)
    payload["simulations"] = merged
    return payload


def merge_patch_sidecar(
    *,
    base_rows: list[dict[str, Any]],
    patch_rows: list[dict[str, Any]],
    patched_task_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Decision rows for the merged episode set.

    The base attempt at a patched episode failed on infrastructure, and this
    repo does not carry infrastructure attempts into quality or cost. Its rows
    are therefore dropped rather than added to the patch rows, so the merged
    cost is the cost of the episodes the merged result actually reports.
    """

    patched_hashes = {
        hashlib.sha256(str(task_id).encode("utf-8")).hexdigest()[:16]
        for task_id in patched_task_ids
    }
    kept = [
        row for row in base_rows if str(row.get("task_hash")) not in patched_hashes
    ]
    return [*kept, *patch_rows]
