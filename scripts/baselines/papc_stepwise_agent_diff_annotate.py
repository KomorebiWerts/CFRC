#!/usr/bin/env python3
"""Annotate the bounded Agent-Diff fit prefix and compile StepWise datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from papc.baselines.router_accounting import (  # noqa: E402
    FROZEN_CACHE_NEUTRAL_PRICES,
    CallReceipt,
    aggregate_online_accounting,
)
from papc.baselines.router_contract import (  # noqa: E402
    REASONING_EFFORT,
    STRONG_MODEL,
)
from papc.baselines.stepwise_agent_diff_fit import (  # noqa: E402
    AnnotationKind,
    StepWiseAnnotation,
    StepWiseFitTrajectory,
    build_annotation_request,
    build_training_datasets,
    parse_annotation,
)
from papc.paths import DATA_ROOT

SCHEMA = "papc.stepwise_agent_diff_annotation_run.v1"
STATE_SCHEMA = "papc.stepwise_state_bench_annotation_run.v1"
ANNOTATION_CHECKPOINT_SCHEMA = "papc.stepwise_annotation_checkpoint.v1"
EXPECTED_FIT_ROWS = 12
EXPECTED_ANNOTATION_CALLS = 24
ANNOTATION_SPEND_STOP_USD = 10.0
DEFAULT_GATE = (
    REPO_ROOT
    / "outputs/papc/stepwise_agent_diff_pilot_gate_noapi_20260731_v2.json"
)
DEFAULT_PRIVATE_ROOT = Path(
    f"{DATA_ROOT}/papc/stepwise_agent_diff_pilot_v1"
)
DEFAULT_ANNOTATIONS = DEFAULT_PRIVATE_ROOT / "annotations.jsonl"
DEFAULT_DATASET_DIR = DEFAULT_PRIVATE_ROOT / "datasets"

ClientFactory = Callable[..., Any]


class AnnotationResponseError(ValueError):
    """A provider response was consumed but its annotation was unusable."""

    def __init__(
        self,
        message: str,
        *,
        receipt: CallReceipt,
        response_sha256: str,
    ) -> None:
        super().__init__(message)
        self.receipt = receipt
        self.response_sha256 = response_sha256


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-json", type=Path, default=DEFAULT_GATE)
    parser.add_argument(
        "--annotations-jsonl", type=Path, default=DEFAULT_ANNOTATIONS
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--model", default=STRONG_MODEL)
    parser.add_argument("--reasoning-effort", default=REASONING_EFFORT)
    parser.add_argument(
        "--max-annotation-calls", type=int, default=EXPECTED_ANNOTATION_CALLS
    )
    parser.add_argument(
        "--spend-stop-usd", type=float, default=ANNOTATION_SPEND_STOP_USD
    )
    parser.add_argument("--provider-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def run(
    argv: list[str] | None = None,
    *,
    env: Mapping[str, str] = os.environ,
    client_factory: ClientFactory | None = None,
) -> tuple[int, dict[str, Any]]:
    args = parse_args(argv)
    secret_reads = 0
    report_schema = SCHEMA
    prior_receipts: tuple[CallReceipt, ...] = ()
    new_receipts: list[CallReceipt] = []
    try:
        if not isinstance(args.api_base, str) or not args.api_base.strip():
            raise ValueError("stepwise_annotation_api_base_empty")
        if (
            args.model != STRONG_MODEL
            or args.reasoning_effort != REASONING_EFFORT
            or args.max_annotation_calls != EXPECTED_ANNOTATION_CALLS
            or args.spend_stop_usd != ANNOTATION_SPEND_STOP_USD
        ):
            raise ValueError("stepwise_annotation_execution_contract_drifted")
        gate, trajectories, source = _load_ready_inputs(args.gate_json)
        report_schema = _report_schema(source)
        existing, annotation_header = _load_or_prepare_annotations(
            args.annotations_jsonl,
            gate=gate,
            gate_path=args.gate_json,
            source=source,
            trajectories=trajectories,
            create=args.execute,
        )
        pending = _pending_requests(trajectories, existing)
        if not args.execute:
            report = _report(
                schema=report_schema,
                decision="ready_for_explicit_execute",
                execute=False,
                gate_path=args.gate_json,
                source=source,
                annotation_path=args.annotations_jsonl,
                dataset_dir=args.dataset_dir,
                existing=existing,
                pending=pending,
                receipts=(),
                dataset_stats=None,
                blockers=["execute_flag_missing"],
                secret_reads=0,
            )
            _write_json_no_replace(args.report_json, report)
            return 0, report

        if args.api_key_env != "OPENAI_API_KEY":
            raise ValueError("stepwise_annotation_api_key_env_not_frozen")
        api_key = env.get(args.api_key_env)
        secret_reads = 1
        if not isinstance(api_key, str) or not api_key.strip():
            raise RuntimeError("stepwise_annotation_api_key_missing")
        factory = client_factory or _default_client_factory
        client = factory(
            base_url=args.api_base.rstrip("/"),
            api_key=api_key,
            timeout=args.provider_timeout_seconds,
            max_retries=0,
        )
        prior_receipts = _annotation_receipts(args.annotations_jsonl)
        observed_cost = sum(
            float(receipt.provider_reported_cost_usd or 0.0)
            for receipt in prior_receipts
        )
        for trajectory, kind in pending:
            if len(prior_receipts) + len(new_receipts) >= args.max_annotation_calls:
                raise RuntimeError("stepwise_annotation_call_cap_reached")
            if observed_cost >= args.spend_stop_usd:
                raise RuntimeError("stepwise_annotation_spend_stop_reached")
            request = build_annotation_request(trajectory, kind=kind)
            try:
                annotation, receipt = _call_annotation(
                    client,
                    trajectory=trajectory,
                    kind=kind,
                )
            except AnnotationResponseError as exc:
                new_receipts.append(exc.receipt)
                observed_cost += float(
                    exc.receipt.provider_reported_cost_usd or 0.0
                )
                _append_jsonl(
                    args.annotations_jsonl,
                    {
                        "record_type": "annotation_error",
                        "example_sha256": trajectory.example_sha256,
                        "kind": kind,
                        "request_sha256": request.sha256,
                        "response_sha256": exc.response_sha256,
                        "error": str(exc),
                        "receipt": exc.receipt.public_record(),
                    },
                )
                raise
            new_receipts.append(receipt)
            observed_cost += float(receipt.provider_reported_cost_usd or 0.0)
            record = {
                "record_type": "annotation",
                "example_sha256": trajectory.example_sha256,
                "kind": kind,
                "request_sha256": request.sha256,
                "positive_steps": list(annotation.positive_steps),
                "payload": dict(annotation.payload),
                "receipt": receipt.public_record(),
            }
            _append_jsonl(args.annotations_jsonl, record)
            existing[(trajectory.example_sha256, kind)] = annotation

        if len(existing) != EXPECTED_ANNOTATION_CALLS:
            raise RuntimeError("stepwise_annotation_checkpoint_incomplete")
        datasets, dataset_stats = build_training_datasets(
            trajectories,
            existing,
        )
        _publish_datasets(
            args.dataset_dir,
            datasets=datasets,
            stats=dataset_stats,
            annotation_header=annotation_header,
        )
        all_receipts = _annotation_receipts(args.annotations_jsonl)
        report = _report(
            schema=report_schema,
            decision=(
                "complete_ready_for_upstream_training"
                if dataset_stats["ready_for_upstream_training"]
                else "label_degenerate_extend_to_frozen_16_row_prefix"
            ),
            execute=True,
            gate_path=args.gate_json,
            source=source,
            annotation_path=args.annotations_jsonl,
            dataset_dir=args.dataset_dir,
            existing=existing,
            pending=(),
            receipts=all_receipts,
            dataset_stats=dataset_stats,
            blockers=(
                []
                if dataset_stats["ready_for_upstream_training"]
                else ["detector_training_labels_degenerate"]
            ),
            secret_reads=secret_reads,
        )
        _write_json_no_replace(args.report_json, report)
        return (
            0 if dataset_stats["ready_for_upstream_training"] else 2,
            report,
        )
    except Exception as exc:  # noqa: BLE001 - fail-closed public preflight.
        accounting = aggregate_online_accounting(
            (*prior_receipts, *new_receipts),
            prices=FROZEN_CACHE_NEUTRAL_PRICES,
        )
        report = {
            "schema": report_schema,
            "decision": "blocked",
            "execute": bool(args.execute),
            "provider_calls": accounting["agent_side"]["provider_calls"],
            "api_spend_performed": bool(
                accounting["agent_side"]["provider_calls"]
            ),
            "secret_reads": secret_reads,
            "annotation_accounting": accounting["agent_side"],
            "blockers": [f"{type(exc).__name__}:{exc}"],
            "public_safe": True,
        }
        _write_json_no_replace(args.report_json, report)
        return 2, report


def _load_ready_inputs(
    gate_path: Path,
) -> tuple[dict[str, Any], tuple[StepWiseFitTrajectory, ...], dict[str, Any]]:
    gate_raw = Path(gate_path).read_bytes()
    gate = json.loads(gate_raw)
    gate_schema = gate.get("schema")
    if (
        gate_schema
        not in {
            "papc.stepwise_agent_diff_pilot_gate_noapi.v1",
            "papc.stepwise_state_bench_pilot_gate_noapi.v1",
        }
        or gate.get("decision") != "READY_FOR_ANNOTATION"
        or gate.get("provider_calls") != 0
        or gate.get("secret_reads") != 0
    ):
        raise ValueError("stepwise_annotation_gate_not_ready")
    fit = gate.get("fit_plan")
    checkpoint = fit.get("checkpoint") if isinstance(fit, dict) else None
    selected = fit.get("selected_example_sha256s") if isinstance(fit, dict) else None
    if not isinstance(checkpoint, dict) or not isinstance(selected, list):
        raise ValueError("stepwise_annotation_fit_prefix_not_ready")
    surface = (
        "state_bench"
        if gate_schema == "papc.stepwise_state_bench_pilot_gate_noapi.v1"
        else "agent_diff"
    )
    if len(selected) != EXPECTED_FIT_ROWS:
        raise ValueError("stepwise_annotation_fit_prefix_not_ready")
    if surface == "agent_diff":
        ready = (
            checkpoint.get("status") == "SELECTED_PREFIX_COMPLETE"
            and checkpoint.get("selected_fit_count") == EXPECTED_FIT_ROWS
        )
    else:
        ready = (
            checkpoint.get("status") == "SELECTED_COMPLETE"
            and checkpoint.get("selected_observed_count") == EXPECTED_FIT_ROWS
            and checkpoint.get("selected_complete") is True
        )
    if not ready:
        raise ValueError("stepwise_annotation_fit_prefix_not_ready")
    checkpoint_path = Path(str(checkpoint["path"]))
    records = tuple(
        json.loads(line)
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if surface == "agent_diff":
        items = records[1 : EXPECTED_FIT_ROWS + 1]
        selected_prefix_sha256 = _digest(records[: EXPECTED_FIT_ROWS + 1])
    else:
        by_sha = {
            item.get("example_sha256"): item
            for item in records[1:]
            if item.get("record_type") == "item"
        }
        items = tuple(by_sha.get(expected) for expected in selected)
        if any(item is None for item in items):
            raise ValueError("stepwise_annotation_fit_items_missing")
        selected_prefix_sha256 = _digest(items)
    if len(items) != EXPECTED_FIT_ROWS:
        raise ValueError("stepwise_annotation_fit_items_missing")
    if selected_prefix_sha256 != checkpoint.get("selected_prefix_sha256"):
        raise ValueError("stepwise_annotation_fit_prefix_drifted")
    trajectories: list[StepWiseFitTrajectory] = []
    for selected_ordinal, (expected, item) in enumerate(
        zip(selected, items, strict=True),
        1,
    ):
        assert isinstance(item, dict)
        item_ordinal = item.get("ordinal")
        if (
            item.get("example_sha256") != expected
            or isinstance(item_ordinal, bool)
            or not isinstance(item_ordinal, int)
            or item_ordinal < 1
            or (surface == "agent_diff" and item_ordinal != selected_ordinal)
        ):
            raise ValueError("stepwise_annotation_fit_membership_mismatch")
        output = item.get("output")
        if not isinstance(output, dict):
            raise TypeError("stepwise_annotation_fit_output_invalid")
        trajectories.append(
            StepWiseFitTrajectory(
                example_sha256=expected,
                task_name=f"{surface}_fit_{item_ordinal:03d}",
                original_request=output["original_request"],
                public_episode=tuple(output["public_episode"]),
            )
        )
    source = {
        "surface": surface,
        "gate_path": str(Path(gate_path)),
        "gate_sha256": hashlib.sha256(gate_raw).hexdigest(),
        "fit_checkpoint_path": str(checkpoint_path),
        "fit_checkpoint_sha256": selected_prefix_sha256,
        "fit_checkpoint_snapshot_sha256": checkpoint["file_sha256"],
        "fit_selected_prefix_sha256": selected_prefix_sha256,
        "selected_example_sha256s": selected,
    }
    return gate, tuple(trajectories), source


def _load_or_prepare_annotations(
    path: Path,
    *,
    gate: Mapping[str, Any],
    gate_path: Path,
    source: Mapping[str, Any],
    trajectories: Sequence[StepWiseFitTrajectory],
    create: bool,
) -> tuple[dict[tuple[str, AnnotationKind], StepWiseAnnotation], dict[str, Any]]:
    header = {
        "schema": ANNOTATION_CHECKPOINT_SCHEMA,
        "record_type": "checkpoint_header",
        "gate_path": str(Path(gate_path)),
        "gate_sha256": source["gate_sha256"],
        "fit_checkpoint_sha256": source["fit_checkpoint_sha256"],
        "selected_example_sha256s": source["selected_example_sha256s"],
        "trajectory_count": len(trajectories),
        "annotation_count": EXPECTED_ANNOTATION_CALLS,
        "model": STRONG_MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "upstream_revision": gate["upstream"]["revision"],
    }
    if source.get("surface") != "agent_diff":
        header["surface"] = source["surface"]
    target = Path(path)
    if not target.exists():
        if create:
            target.parent.mkdir(parents=True, exist_ok=True)
            _append_jsonl(target, header, create_only=True)
        return {}, header
    records = tuple(
        json.loads(line)
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not records or records[0] != header:
        raise ValueError("stepwise_annotation_checkpoint_header_mismatch")
    trajectories_by_sha = {item.example_sha256: item for item in trajectories}
    annotations: dict[tuple[str, AnnotationKind], StepWiseAnnotation] = {}
    for record in records[1:]:
        example = record.get("example_sha256")
        kind = record.get("kind")
        key = (example, kind)
        if record.get("record_type") == "annotation_error":
            if (
                example not in trajectories_by_sha
                or kind not in {"stuck", "milestones"}
                or record.get("request_sha256")
                != build_annotation_request(
                    trajectories_by_sha[example],
                    kind=kind,
                ).sha256
            ):
                raise ValueError("stepwise_annotation_error_record_invalid")
            CallReceipt(**record.get("receipt"))
            raise RuntimeError(
                "stepwise_annotation_failed_attempt_requires_fresh_run"
            )
        if (
            record.get("record_type") != "annotation"
            or example not in trajectories_by_sha
            or kind not in {"stuck", "milestones"}
            or key in annotations
        ):
            raise ValueError("stepwise_annotation_checkpoint_record_invalid")
        trajectory = trajectories_by_sha[example]
        request = build_annotation_request(trajectory, kind=kind)
        if record.get("request_sha256") != request.sha256:
            raise ValueError("stepwise_annotation_request_sha256_mismatch")
        annotation = parse_annotation(
            kind=kind,
            payload=record.get("payload"),
            step_count=len(trajectory.steps),
        )
        if list(annotation.positive_steps) != record.get("positive_steps"):
            raise ValueError("stepwise_annotation_positive_steps_mismatch")
        CallReceipt(**record.get("receipt"))
        annotations[key] = annotation
    return annotations, header


def _pending_requests(
    trajectories: Sequence[StepWiseFitTrajectory],
    existing: Mapping[tuple[str, AnnotationKind], StepWiseAnnotation],
) -> tuple[tuple[StepWiseFitTrajectory, AnnotationKind], ...]:
    pending: list[tuple[StepWiseFitTrajectory, AnnotationKind]] = []
    for trajectory in trajectories:
        for kind in ("stuck", "milestones"):
            if (trajectory.example_sha256, kind) not in existing:
                pending.append((trajectory, kind))
    return tuple(pending)


def _call_annotation(
    client: Any,
    *,
    trajectory: StepWiseFitTrajectory,
    kind: AnnotationKind,
) -> tuple[StepWiseAnnotation, CallReceipt]:
    request = build_annotation_request(trajectory, kind=kind)
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=STRONG_MODEL,
        messages=[dict(item) for item in request.messages],
        response_format=dict(request.response_format),
        reasoning_effort=REASONING_EFFORT,
        max_tokens=request.max_tokens,
        temperature=0.0,
    )
    latency = max(0.0, time.perf_counter() - started)
    content = _member(_member(response, "choices")[0], "message")
    text = _member(content, "content")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("stepwise_annotation_provider_content_invalid")
    usage = _member(response, "usage")
    input_tokens = _usage_int(usage, "input_tokens", fallback="prompt_tokens")
    output_tokens = _usage_int(
        usage,
        "output_tokens",
        fallback="completion_tokens",
    )
    if input_tokens is None or output_tokens is None:
        raise ValueError("stepwise_annotation_provider_usage_missing")
    input_details = _member(
        usage,
        "input_tokens_details",
        fallback="prompt_tokens_details",
    )
    output_details = _member(
        usage,
        "output_tokens_details",
        fallback="completion_tokens_details",
    )
    cached = _usage_int(input_details, "cached_tokens") or 0
    reasoning = _usage_int(output_details, "reasoning_tokens") or 0
    price = FROZEN_CACHE_NEUTRAL_PRICES[STRONG_MODEL]
    cost = (
        input_tokens * price.input_per_million_usd
        + output_tokens * price.output_per_million_usd
    ) / 1_000_000
    receipt = CallReceipt(
        role="verifier",
        phase=f"fit_annotation_{kind}",
        model=STRONG_MODEL,
        reasoning_effort=REASONING_EFFORT,
        provider_calls=1,
        input_tokens=input_tokens,
        cached_input_tokens=cached,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning,
        retry_count=0,
        latency_seconds=latency,
        provider_reported_cost_usd=cost,
    )
    try:
        annotation = parse_annotation(
            kind=kind,
            payload=json.loads(text),
            step_count=len(trajectory.steps),
        )
    except Exception as exc:
        raise AnnotationResponseError(
            f"{type(exc).__name__}:{exc}",
            receipt=receipt,
            response_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        ) from exc
    return annotation, receipt


def _publish_datasets(
    path: Path,
    *,
    datasets: Mapping[str, Sequence[Mapping[str, Any]]],
    stats: Mapping[str, Any],
    annotation_header: Mapping[str, Any],
) -> None:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    files = {
        "stuck_dataset.json": list(datasets["stuck"]),
        "milestone_dataset.json": list(datasets["milestones"]),
        "dataset_stats.json": {
            **dict(stats),
            "annotation_checkpoint_header_sha256": _digest(annotation_header),
        },
    }
    for name, payload in files.items():
        output = target / name
        serialized = (
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )
        if output.exists():
            if output.read_text(encoding="utf-8") != serialized:
                raise FileExistsError(f"stepwise_dataset_exists_with_drift:{name}")
            continue
        _write_text_no_replace(output, serialized)


def _annotation_receipts(path: Path) -> tuple[CallReceipt, ...]:
    records = tuple(
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()[1:]
        if line.strip()
    )
    return tuple(CallReceipt(**record["receipt"]) for record in records)


def _report(
    *,
    schema: str,
    decision: str,
    execute: bool,
    gate_path: Path,
    source: Mapping[str, Any],
    annotation_path: Path,
    dataset_dir: Path,
    existing: Mapping[tuple[str, AnnotationKind], StepWiseAnnotation],
    pending: Sequence[Any],
    receipts: Sequence[CallReceipt],
    dataset_stats: Mapping[str, Any] | None,
    blockers: Sequence[str],
    secret_reads: int,
) -> dict[str, Any]:
    accounting = aggregate_online_accounting(
        receipts,
        prices=FROZEN_CACHE_NEUTRAL_PRICES,
    )
    dataset_artifacts = None
    if dataset_stats is not None:
        dataset_artifacts = {
            name: {
                "path": str(Path(dataset_dir) / name),
                "sha256": _file_sha256(Path(dataset_dir) / name),
            }
            for name in (
                "stuck_dataset.json",
                "milestone_dataset.json",
                "dataset_stats.json",
            )
        }
    return {
        "schema": schema,
        "decision": decision,
        "execute": execute,
        "gate_path": str(Path(gate_path)),
        "gate_sha256": source["gate_sha256"],
        "fit_checkpoint_sha256": source["fit_checkpoint_sha256"],
        "selected_fit_rows": EXPECTED_FIT_ROWS,
        "annotation_checkpoint": str(Path(annotation_path)),
        "dataset_dir": str(Path(dataset_dir)),
        "completed_annotations": len(existing),
        "pending_annotations": len(pending),
        "provider_call_hard_cap": EXPECTED_ANNOTATION_CALLS,
        "observed_spend_stop_usd": ANNOTATION_SPEND_STOP_USD,
        "provider_calls": accounting["agent_side"]["provider_calls"],
        "api_spend_performed": bool(accounting["agent_side"]["provider_calls"]),
        "secret_reads": secret_reads,
        "annotation_accounting": accounting["agent_side"],
        "dataset_stats": dict(dataset_stats) if dataset_stats else None,
        "dataset_artifacts": dataset_artifacts,
        "blockers": list(blockers),
        "official_test_outcomes_read": False,
        "public_contains_requests_or_expected_answers": False,
        "public_safe": True,
    }


def _report_schema(source: Mapping[str, Any]) -> str:
    return STATE_SCHEMA if source.get("surface") == "state_bench" else SCHEMA


def _default_client_factory(**kwargs: Any) -> Any:
    from openai import OpenAI

    return OpenAI(**kwargs)


def _usage_int(
    value: Any,
    name: str,
    *,
    fallback: str | None = None,
) -> int | None:
    raw = _member(value, name, fallback=fallback)
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise TypeError(f"stepwise_annotation_usage_invalid:{name}")
    return raw


def _member(value: Any, name: str, *, fallback: str | None = None) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        result = value.get(name)
        if result is None and fallback is not None:
            result = value.get(fallback)
        return result
    result = getattr(value, name, None)
    if result is None and fallback is not None:
        result = getattr(value, fallback, None)
    return result


def _append_jsonl(
    path: Path,
    payload: Mapping[str, Any],
    *,
    create_only: bool = False,
) -> None:
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if create_only else os.O_APPEND)
    with os.fdopen(os.open(path, flags, 0o600), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_no_replace(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _write_text_no_replace(path, serialized)


def _write_text_no_replace(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    with os.fdopen(os.open(path, flags, 0o600), "w", encoding="utf-8") as handle:
        handle.write(text)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def main(argv: list[str] | None = None) -> int:
    code, report = run(argv)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "provider_calls": report["provider_calls"],
                "completed_annotations": report.get("completed_annotations", 0),
                "pending_annotations": report.get("pending_annotations", 0),
            },
            sort_keys=True,
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
