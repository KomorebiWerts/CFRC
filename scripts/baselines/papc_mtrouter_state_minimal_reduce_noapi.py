#!/usr/bin/env python3
"""Reduce completed MTRouter STATE fit/dev evidence without provider calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from papc.paths import DATA_ROOT

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "papc.mtrouter.state_bench_minimal_result.v1"
EXPECTED_PRIVATE_EVIDENCE_SHA256 = (
    "8eb78a45989507df27cab6b41cc66f5f4d7c20f2fd3e640c973c292e80cb5b30"
)
EXPECTED_GATE_SHA256 = (
    "eb02e0d52e02f41d96fdfb7444e13af5bb2d977d51c20f9d26e48e6950323c5f"
)
EXPECTED_ALL_STRONG_REPLAY_SHA256 = (
    "ffe94abbd9dbcb3aa79597229a369f7837388a9026d85dd7534e71e93af1cfe0"
)
DEFAULT_PRIVATE_EVIDENCE = Path(
    f"{DATA_ROOT}/papc/mtrouter_state_minimal_20260801_v1/"
    "mtrouter_state_minimal_fit_dev_evidence.json"
)
DEFAULT_GATE = (
    REPO_ROOT
    / "outputs/papc/mtrouter_state_minimal_fit_dev_gate_noapi_20260801_v1.json"
)
DEFAULT_ALL_STRONG_REPLAY = (
    REPO_ROOT / "outputs/papc/mtrouter_state_all_strong_replay_noapi_20260801_v1.json"
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_append_only(path: Path, record: Mapping[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(dict(record)) + b"\n")
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--private-evidence-json", type=Path, default=DEFAULT_PRIVATE_EVIDENCE
    )
    parser.add_argument("--gate-json", type=Path, default=DEFAULT_GATE)
    parser.add_argument(
        "--all-strong-replay-json", type=Path, default=DEFAULT_ALL_STRONG_REPLAY
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def _load_exact(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    if _file_sha256(path) != expected_sha256:
        raise ValueError(f"mtrouter_state_reduce_{label}_sha256_mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"mtrouter_state_reduce_{label}_mapping_required")
    return dict(value)


def _receipts(episodes: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    return tuple(
        dict(receipt) for episode in episodes for receipt in episode.get("receipts", [])
    )


def run(argv: list[str] | None = None) -> tuple[int, dict[str, Any]]:
    args = parse_args(argv)
    evidence = _load_exact(
        args.private_evidence_json,
        EXPECTED_PRIVATE_EVIDENCE_SHA256,
        "private_evidence",
    )
    gate = _load_exact(args.gate_json, EXPECTED_GATE_SHA256, "gate")
    all_strong = _load_exact(
        args.all_strong_replay_json,
        EXPECTED_ALL_STRONG_REPLAY_SHA256,
        "all_strong_replay",
    )
    blockers: list[str] = []
    fit = tuple(evidence.get("fit_evidence", ()))
    dev = tuple(evidence.get("dev_evidence", ()))
    if len(fit) != 8 or len(dev) != 6:
        blockers.append("online_episode_count_mismatch")
    if CounterLike(item.get("candidate") for item in dev) != {
        "learned": 3,
        "all_strong": 3,
    }:
        blockers.append("dev_candidate_membership_mismatch")
    episode_scores = tuple(
        [
            float(item["trajectory"]["rows"][0]["terminal_official_reward"])
            for item in fit
        ]
        + [float(item["official_score"]) for item in dev]
    )
    if len(episode_scores) != 14 or any(
        score not in {0.0, 1.0} for score in episode_scores
    ):
        blockers.append("official_episode_score_incomplete")
    receipts = _receipts(fit) + _receipts(dev)
    if len(receipts) != 161:
        blockers.append("settled_receipt_count_mismatch")
    if any(
        receipt.get("provider_calls") != 1
        or receipt.get("retry_count") != 0
        or receipt.get("provider_reported_cost_usd") is None
        for receipt in receipts
    ):
        blockers.append("provider_receipt_not_atomic_complete")
    ledger = evidence.get("ledger", {})
    global_ledger = ledger.get("global", {})
    if (
        ledger.get("blocked") is not False
        or ledger.get("open_reservation_count") != 0
        or global_ledger.get("attempts_reserved") != 161
        or global_ledger.get("receipts_settled") != 161
    ):
        blockers.append("provider_ledger_not_complete_and_quiescent")
    dev_aggregates = evidence.get("dev_aggregates", {})
    learned = dev_aggregates.get("learned", {})
    strong = dev_aggregates.get("all_strong", {})
    if (
        learned.get("correct") != 2.0
        or learned.get("denominator") != 3
        or strong.get("correct") != 3.0
        or strong.get("denominator") != 3
        or evidence.get("selected_candidate") != "all_strong"
    ):
        blockers.append("dev_aggregate_or_selection_mismatch")
    if all_strong.get("decision") != "qualified_all_strong_test_candidate":
        blockers.append("selected_all_strong_test_replay_not_qualified")
    if gate.get("frozen", {}).get("test_registry_sha256") != all_strong.get(
        "official_test", {}
    ).get("registry_sha256"):
        blockers.append("fit_dev_test_replay_registry_mismatch")
    training = evidence.get("training", {})
    checkpoint = Path(str(training.get("checkpoint_path", "")))
    if not checkpoint.is_file() or _file_sha256(checkpoint) != training.get(
        "checkpoint_sha256"
    ):
        blockers.append("learned_checkpoint_missing_or_mismatched")

    table_result = {
        "accuracy": all_strong.get("official_test", {}).get("accuracy"),
        "correct": all_strong.get("official_test", {}).get("correct"),
        "denominator": all_strong.get("official_test", {}).get("denominator"),
        "cache_neutral_agent_side_cost_usd": all_strong.get("endpoint", {}).get(
            "cache_neutral_agent_side_cost_usd"
        ),
        "agent_calls": all_strong.get("endpoint", {}).get("calls"),
    }
    report = {
        "schema": SCHEMA,
        "decision": "publication_result_ready" if not blockers else "blocked",
        "blockers": blockers,
        "method": "MTRouter",
        "implementation": (
            "published_frozen_Qwen3_Embedding_0.6B_history_encoder_plus_single_head_Q;"
            "bounded_in_domain_fit8_dev3"
        ),
        "surface": "state_bench",
        "cheap_model": gate.get("frozen", {}).get("cheap_model", "gpt-5.4-mini"),
        "strong_model": gate.get("frozen", {}).get("strong_model", "gpt-5.5"),
        "reasoning_effort": "high",
        "fit": {
            "official_train_tasks": len(fit),
            "roulette_trajectories": len(fit),
            "turn_rows": training.get("fit_turn_row_count"),
            "terminal_rewards": [
                item["trajectory"]["rows"][0]["terminal_official_reward"]
                for item in fit
            ],
            "checkpoint_sha256": training.get("checkpoint_sha256"),
            "checkpoint_epoch": training.get("checkpoint_epoch"),
            "qwen_source_sha256": training.get("qwen_source_sha256"),
        },
        "held_out_dev": {
            "official_train_tasks": 3,
            "learned": learned,
            "all_strong": strong,
            "selected_candidate": evidence.get("selected_candidate"),
            "selection_rule": gate.get("frozen", {}).get("selection_rule"),
            "learned_candidate_actually_evaluated": True,
        },
        "official_test": {
            **table_result,
            "selected_policy": "all_strong_native_shortcut",
            "registry_sha256": all_strong.get("official_test", {}).get(
                "registry_sha256"
            ),
            "metric": "task_completion_pass@1",
            "test_source_model": all_strong.get("endpoint", {}).get("model"),
            "test_source_reasoning_effort": all_strong.get("endpoint", {}).get(
                "reasoning_effort"
            ),
            "source_equality": all_strong.get("source_equality"),
            "quality_anchor_policy": all_strong.get("quality_anchor_policy"),
            "test_source_provider_calls": all_strong.get("endpoint", {}).get("calls"),
            "offline_replay_provider_calls": 0,
        },
        "online_fit_dev_execution": {
            "episode_count": 14,
            "provider_receipts": len(receipts),
            "provider_error_count": 0,
            "retry_count": 0,
            "ledger_blocked": ledger.get("blocked"),
            "open_reservation_count": ledger.get("open_reservation_count"),
            "agent_side_provider_api_cost_usd": global_ledger.get(
                "agent_side_provider_api_cost_usd"
            ),
            "all_role_observed_provider_api_cost_usd": global_ledger.get(
                "all_role_observed_provider_api_cost_usd"
            ),
        },
        "evidence": {
            "gate_file_sha256": EXPECTED_GATE_SHA256,
            "private_fit_dev_evidence_sha256": EXPECTED_PRIVATE_EVIDENCE_SHA256,
            "all_strong_replay_file_sha256": EXPECTED_ALL_STRONG_REPLAY_SHA256,
            "fit_dev_frozen_sha256": evidence.get("frozen_sha256"),
            "source_binding_sha256": gate.get("frozen", {}).get(
                "source_binding_sha256"
            ),
        },
        "provider_calls": 0,
        "secret_reads": 0,
        "api_spend_performed": False,
        "reducer_only": True,
        "no_online_rerun": True,
    }
    report["semantic_sha256"] = _digest(report)
    return (0 if not blockers else 2), report


def CounterLike(values: Sequence[Any]) -> dict[Any, int]:
    counts: dict[Any, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    code, report = run(argv)
    _write_json_append_only(args.output_json, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
