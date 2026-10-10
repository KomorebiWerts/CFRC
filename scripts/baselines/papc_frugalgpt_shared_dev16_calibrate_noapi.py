#!/usr/bin/env python3
"""Calibrate a Retail-trained FrugalGPT scorer on paired dev16 endpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from papc.baselines.frugalgpt_upstream import (  # noqa: E402
    FRUGALGPT_UPSTREAM_REVISION,
    FrugalGPTDevExample,
    FrugalGPTDistilBertScorer,
    FrugalGPTScorerArtifact,
    select_frugalgpt_operating_point,
)
from papc.baselines.router_accounting import (  # noqa: E402
    FROZEN_CACHE_NEUTRAL_PRICES,
    CallReceipt,
    aggregate_online_accounting,
)
from papc.baselines.router_contract import (  # noqa: E402
    AGENT_DIFF,
    CHEAP_MODEL,
    STATE_BENCH,
    STRONG_MODEL,
    CheapCompletionRouterView,
    SplitManifest,
)
from papc.paths import DATA_ROOT

SCHEMA = "papc.frugalgpt.shared_dev16_calibration.v1"
SURFACES = (STATE_BENCH, AGENT_DIFF)
DEFAULT_PRIVATE_ROOT = Path(
    f"{DATA_ROOT}/papc/shared_endpoint_dev16_calibration_v1"
)
DEFAULT_RETAIL_ARTIFACT = Path(
    f"{DATA_ROOT}/papc/frugalgpt_retail_v1/artifact.json"
)
DEFAULT_RETAIL_CHECKPOINT = Path(
    f"{DATA_ROOT}/papc/frugalgpt_retail_v1/checkpoint"
)
DEFAULT_SPLIT_GATE = (
    REPO_ROOT / "outputs/papc/external_router_split_gate_noapi_20260730_v2.json"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", choices=SURFACES, required=True)
    parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE_ROOT)
    parser.add_argument(
        "--retail-artifact", type=Path, default=DEFAULT_RETAIL_ARTIFACT
    )
    parser.add_argument(
        "--retail-checkpoint", type=Path, default=DEFAULT_RETAIL_CHECKPOINT
    )
    parser.add_argument("--split-gate-json", type=Path, default=DEFAULT_SPLIT_GATE)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"json_object_required:{path}")
    return dict(value)


def _read_rows(path: Path) -> tuple[dict[str, Any], ...]:
    rows = tuple(
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(rows) != 16 or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError(f"dev16_endpoint_rows_invalid:{path}")
    return tuple(dict(row) for row in rows)


def _agent_cost(row: Mapping[str, Any]) -> float:
    raw = row.get("receipts")
    if not isinstance(raw, list):
        raise ValueError("dev16_receipts_missing")
    receipts = tuple(CallReceipt(**item) for item in raw)
    accounting = aggregate_online_accounting(
        receipts, prices=FROZEN_CACHE_NEUTRAL_PRICES
    )
    value = accounting["agent_side"]["cost_usd"]
    if value is None:
        raise ValueError("dev16_agent_cost_incomplete")
    return float(value)


def _endpoint_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    root = args.private_root.resolve() / args.surface
    return root / "cheap/dev16_rows.jsonl", root / "strong/dev16_rows.jsonl"


def _validate_pair(
    *,
    cheap: Mapping[str, Any],
    strong: Mapping[str, Any],
    expected: Mapping[str, Any],
    selection_sha256: str,
) -> None:
    expected_ordinal = expected["ordinal"]
    expected_example = expected["example_sha256"]
    expected_payload = expected["payload_sha256"]
    for row, endpoint, model in (
        (cheap, "cheap", CHEAP_MODEL),
        (strong, "strong", STRONG_MODEL),
    ):
        if (
            row.get("ordinal") != expected_ordinal
            or row.get("surface") != expected["surface"]
            or row.get("endpoint_role") != endpoint
            or row.get("model") != model
            or row.get("example_sha256") != expected_example
            or row.get("payload_sha256") != expected_payload
            or row.get("selection_sha256") != selection_sha256
            or row.get("model_role") != endpoint
            or not isinstance(row.get("official_score"), (int, float))
            or not isinstance(row.get("original_request"), str)
            or not isinstance(row.get("generated_answer"), str)
        ):
            raise ValueError("dev16_endpoint_row_binding_mismatch")
    if cheap["original_request"] != strong["original_request"]:
        raise ValueError("dev16_endpoint_original_request_mismatch")


def run(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    gate_path = (
        REPO_ROOT
        / f"outputs/papc/shared_dev16_{args.surface}_gate_noapi_20260802_v1.json"
    )
    gate = _read_mapping(gate_path)
    if (
        gate.get("decision") != "ready_for_explicit_execute"
        or gate.get("selected_count") != 16
        or gate.get("official_test_outcomes_read") is not False
        or gate.get("runtime_routing_uses_task_or_example_id") is not False
    ):
        raise ValueError("dev16_gate_not_qualified")
    selected = gate.get("selected_rows")
    if not isinstance(selected, list) or len(selected) != 16:
        raise ValueError("dev16_gate_membership_invalid")
    selection_sha = gate["source_authority"]["selection_sha256"]
    cheap_path, strong_path = _endpoint_paths(args)
    cheap_rows = _read_rows(cheap_path)
    strong_rows = _read_rows(strong_path)
    for cheap, strong, expected in zip(
        cheap_rows, strong_rows, selected, strict=True
    ):
        _validate_pair(
            cheap=cheap,
            strong=strong,
            expected=expected,
            selection_sha256=selection_sha,
        )

    retail_record = _read_mapping(args.retail_artifact)
    retail_artifact = FrugalGPTScorerArtifact.from_public_record(retail_record)
    if retail_artifact.surface != "tau2_retail":
        raise ValueError("frugalgpt_scorer_not_retail_trained")
    scorer = FrugalGPTDistilBertScorer.load(
        artifact=retail_artifact,
        checkpoint_dir=args.retail_checkpoint.resolve(),
    )
    examples = []
    score_rows = []
    for cheap, strong in zip(cheap_rows, strong_rows, strict=True):
        score = scorer(
            CheapCompletionRouterView(
                original_request=cheap["original_request"],
                cheap_generated_answer=cheap["generated_answer"],
            )
        )
        example = FrugalGPTDevExample(
            example_sha256=cheap["example_sha256"],
            predicted_cheap_quality=score,
            cheap_official_quality=float(cheap["official_score"]),
            strong_official_quality=float(strong["official_score"]),
            cheap_episode_cost_usd=_agent_cost(cheap),
            strong_episode_cost_usd=_agent_cost(strong),
        )
        examples.append(example)
        score_rows.append(
            {
                "ordinal": cheap["ordinal"],
                "request_sha256": hashlib.sha256(
                    cheap["original_request"].encode()
                ).hexdigest(),
                "predicted_cheap_quality": score,
                "cheap_official_quality": cheap["official_score"],
                "strong_official_quality": strong["official_score"],
                "cheap_episode_cost_usd": example.cheap_episode_cost_usd,
                "strong_episode_cost_usd": example.strong_episode_cost_usd,
            }
        )
    manifest = SplitManifest(
        role="router_dev",
        registry_sha256=selection_sha,
        example_sha256s=tuple(row["example_sha256"] for row in selected),
    )
    operating = select_frugalgpt_operating_point(examples, dev_manifest=manifest)
    split_gate = _read_mapping(args.split_gate_json)
    test_registry_sha = split_gate["surface_splits"][args.surface][
        "test_registry_sha256"
    ]
    policy = {
        "schema": SCHEMA,
        "decision": "frozen_before_official_test_outcomes",
        "method": "FrugalGPT (2-model)",
        "surface": args.surface,
        "cheap_model": CHEAP_MODEL,
        "strong_model": STRONG_MODEL,
        "execution_semantics": (
            "complete_cheap_then_accept_or_restart_strong_from_pristine_state"
        ),
        "scorer_training_surface": "tau2_retail",
        "scorer_upstream_revision": FRUGALGPT_UPSTREAM_REVISION,
        "retail_scorer_artifact_sha256": retail_artifact.sha256,
        "retail_scorer_artifact_file_sha256": _file_sha256(args.retail_artifact),
        "retail_checkpoint_bundle_sha256": (
            retail_artifact.checkpoint_bundle_sha256
        ),
        "calibration": operating.public_record(),
        "calibration_sha256": operating.sha256,
        "dev_score_rows": score_rows,
        "dev_score_rows_sha256": _digest(score_rows),
        "dev_selection_sha256": selection_sha,
        "dev_gate_file_sha256": _file_sha256(gate_path),
        "cheap_dev_rows_file_sha256": _file_sha256(cheap_path),
        "strong_dev_rows_file_sha256": _file_sha256(strong_path),
        "test_registry_sha256": test_registry_sha,
        "acceptance_rule": "predicted_cheap_quality > accept_threshold",
        "runtime_tie_break_uses_task_or_example_id": False,
        "dev_labels_use": "threshold_calibration_only",
        "official_test_rows_read": False,
        "official_test_outcomes_read": False,
        "provider_calls_added_by_calibration": 0,
        "secret_reads": 0,
        "api_spend_performed": False,
    }
    policy["policy_sha256"] = _digest(policy)
    target = args.output_json.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_bytes(policy))
    return policy


def main(argv: list[str] | None = None) -> int:
    try:
        report = run(argv)
        print(json.dumps(report, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - provider-zero exact blocker report.
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "decision": "blocked",
                    "error_type": type(exc).__name__,
                    "error": str(exc) or type(exc).__name__,
                    "provider_calls": 0,
                    "secret_reads": 0,
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
