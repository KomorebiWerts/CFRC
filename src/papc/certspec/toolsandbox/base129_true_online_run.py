"""ToolSandbox base129 true-online CFRC method."""

from __future__ import annotations

import json
import multiprocessing
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from papc.certspec.cfrc_compiler import CFRCCompileConfig
from papc.certspec.cfrc_ir import AuthorityLattice
from papc.certspec.costing import (
    public_openai_price_for_model,
    public_pricing_metadata_for_model,
)
from papc.certspec.runtime_ledger import RuntimeLedger
from papc.certspec.toolsandbox.online_cfrc_controller import (
    StrongProposer,
    ToolSandboxOnlineCFRCConfig,
    ToolSandboxOnlineCFRCController,
    ToolSandboxOnlineRuntime,
)
from papc.certspec.toolsandbox.online_frontier_monitor import (
    ToolSandboxReminderAddFrontierMonitor,
)
from papc.certspec.toolsandbox.public_hygiene import public_leakage_audit

SCHEMA = "papc.toolsandbox.base129_true_online_run.v1"
PRIVATE_SCHEMA = "papc.toolsandbox.base129_true_online_run.private.v1"
NOAPI_READY_DECISION = "toolsandbox_base129_true_online_run_ready_noapi"
EXECUTE_READY_DECISION = "toolsandbox_base129_true_online_run_ready"
BLOCKED_DECISION = "blocked_toolsandbox_base129_true_online_run"

APPROVAL_SCOPE = "toolsandbox_base129_true_online_cfrc_method_arm"
METHOD_PROFILE = "full_cfrc"
EXPECTED_SURFACE = "toolsandbox"
EXPECTED_SCENARIO_SCOPE = "base129"
EXPECTED_TASK_COUNT = 129
EXPECTED_CHEAP_COMPLETE = 29
EXPECTED_STRONG_COMPLETE = 37
EXPECTED_CHEAP_AVG_SIMILARITY_SCORE = 0.515558
EXPECTED_STRONG_AVG_SIMILARITY_SCORE = 0.582825
TARGET_COMPLETE = 36
EXPECTED_CHEAP_MODEL = "gpt-5.4-mini"
EXPECTED_STRONG_MODEL = "gpt-5.5"
EXPECTED_USER_MODEL = "gpt-5.4"
EXPECTED_REASONING = "high"
OPERATING_POINT_REFERENCES: dict[tuple[str, str, str], dict[str, Any]] = {
    (
        EXPECTED_CHEAP_MODEL,
        EXPECTED_STRONG_MODEL,
        EXPECTED_USER_MODEL,
    ): {
        "reference_id": "gpt54mini_gpt55_user54",
        "cheap_complete_count": EXPECTED_CHEAP_COMPLETE,
        "cheap_avg_similarity_score": EXPECTED_CHEAP_AVG_SIMILARITY_SCORE,
        "strong_complete_count": EXPECTED_STRONG_COMPLETE,
        "strong_avg_similarity_score": EXPECTED_STRONG_AVG_SIMILARITY_SCORE,
        "target_complete_count": TARGET_COMPLETE,
    },
    (
        EXPECTED_CHEAP_MODEL,
        EXPECTED_STRONG_MODEL,
        "gpt-5.5",
    ): {
        "reference_id": "gpt54mini_gpt55_user55",
        # The cheap and strong bounds below were measured under the gpt-5.4 user
        # simulator, so they are NOT matched to this operating point. They gate
        # nothing: the method arm scores itself from its own receipts, and only
        # the delta-versus-bound columns read them. Recompute those columns from
        # the gpt-5.5-user baselines before any of them is quoted.
        "cheap_complete_count": EXPECTED_CHEAP_COMPLETE,
        "cheap_avg_similarity_score": EXPECTED_CHEAP_AVG_SIMILARITY_SCORE,
        "strong_complete_count": EXPECTED_STRONG_COMPLETE,
        "strong_avg_similarity_score": EXPECTED_STRONG_AVG_SIMILARITY_SCORE,
        "target_complete_count": TARGET_COMPLETE,
        "reference_bounds_matched_user_model": False,
    },
    ("gpt-5.6-luna", "gpt-5.6-sol", EXPECTED_USER_MODEL): {
        "reference_id": "gpt56_luna_sol_user54",
        "cheap_complete_count": 34,
        "cheap_avg_similarity_score": 0.6664454961240309,
        "strong_complete_count": 39,
        "strong_avg_similarity_score": 0.6865741317829455,
        "target_complete_count": 39,
        "reference_bounds_matched_user_model": True,
    },
    ("gemini-3.6-flash-high", "gpt-5.6-sol", EXPECTED_USER_MODEL): {
        "reference_id": "gemini36flashhigh_gpt56sol_user54",
        # The matching Gemini baseline is part of this run pair and is not yet
        # measured. Keep its bound absent instead of borrowing another cheap
        # model's score. The Sol-only reference is already measured under the
        # same user simulator, so only that side and the target are populated.
        "cheap_complete_count": None,
        "cheap_avg_similarity_score": None,
        "strong_complete_count": 39,
        "strong_avg_similarity_score": 0.6865741317829455,
        "target_complete_count": 39,
        "reference_bounds_matched_user_model": True,
        "reference_bounds_complete": False,
    },
    ("gemini-3.5-flash-high", "gpt-5.6-sol", EXPECTED_USER_MODEL): {
        "reference_id": "gemini35flashhigh_gpt56sol_user54",
        # The matching Gemini baseline must be measured on this run pair.
        # Preserve the matched Sol-only bound without borrowing a cheap-model
        # result from another operating point.
        "cheap_complete_count": None,
        "cheap_avg_similarity_score": None,
        "strong_complete_count": 39,
        "strong_avg_similarity_score": 0.6865741317829455,
        "target_complete_count": 39,
        "reference_bounds_matched_user_model": True,
        "reference_bounds_complete": False,
    },
    ("gemini-3.5-flash-high", "gpt-5.5", EXPECTED_USER_MODEL): {
        "reference_id": "gemini35flashhigh_gpt55_user54",
        # The matching Gemini and GPT-5.5 bounds are established by the
        # source-bound full-domain baseline/method run; leave them unset until
        # those receipts exist rather than borrowing another operating point.
        "cheap_complete_count": None,
        "cheap_avg_similarity_score": None,
        "strong_complete_count": None,
        "strong_avg_similarity_score": None,
        "target_complete_count": None,
        "reference_bounds_matched_user_model": True,
        "reference_bounds_complete": False,
    },
    ("gemini-3.5-flash-high", "gpt-5.5", "gpt-5.5"): {
        "reference_id": "gemini35flashhigh_gpt55_user55",
        "cheap_complete_count": None,
        "cheap_avg_similarity_score": None,
        "strong_complete_count": None,
        "strong_avg_similarity_score": None,
        "target_complete_count": None,
        "reference_bounds_matched_user_model": True,
        "reference_bounds_complete": False,
    },
    ("gpt-5.4-mini", "gemini-3.5-flash-high", EXPECTED_USER_MODEL): {
        "reference_id": "gpt54mini_gemini35flashhigh_user54",
        "cheap_complete_count": None,
        "cheap_avg_similarity_score": None,
        "strong_complete_count": None,
        "strong_avg_similarity_score": None,
        "target_complete_count": None,
        "reference_bounds_matched_user_model": True,
        "reference_bounds_complete": False,
    },
    (
        "gpt-5.6-luna",
        "gemini-3.7-flash-high",
        "gpt-5.4",
    ): {
        "reference_id": "gpt56luna_gemini37flashhigh_user54",
        "cheap_complete_count": None,
        "cheap_avg_similarity_score": None,
        "strong_complete_count": None,
        "strong_avg_similarity_score": None,
        "target_complete_count": None,
        "reference_bounds_matched_user_model": True,
        "reference_bounds_complete": False,
    },
    # Two ids for one model. Surfaces that reach the provider through a client
    # library are given the prefixed form, which the library strips; this one
    # sends the id verbatim over raw HTTP and must be given the plain form. The
    # bounds are the same measurements either way.
    ("gpt-5.6-luna", "claude-sonnet-5", EXPECTED_USER_MODEL): {
        "reference_id": "gpt56luna_claude_sonnet5_user54",
        "cheap_complete_count": 34,
        "cheap_avg_similarity_score": 0.6664454961240309,
        "strong_complete_count": 40,
        "strong_avg_similarity_score": None,
        "target_complete_count": 40,
        "reference_bounds_matched_user_model": False,
    },
    ("gpt-5.6-luna", "openai/claude-sonnet-5", EXPECTED_USER_MODEL): {
        "reference_id": "gpt56luna_claude_sonnet5_user54",
        # The cheap bound is the one measured for this prefix under this user
        # simulator, so it is matched. The strong bound is not: claude-sonnet-5
        # was measured alone on this surface at forty of one hundred and
        # twenty-nine, but under a gpt-5.5 user simulator, and that run recorded
        # no average similarity at all. Both facts are stated here rather than
        # papered over with the numbers from a different pool, because these
        # values reach the report only through the delta-versus-bound columns
        # and a borrowed bound would make those columns read as measured.
        "cheap_complete_count": 34,
        "cheap_avg_similarity_score": 0.6664454961240309,
        "strong_complete_count": 40,
        "strong_avg_similarity_score": None,
        "target_complete_count": 40,
        "reference_bounds_matched_user_model": False,
    },
    ("gpt-5.6-luna", "claude-opus-5", EXPECTED_USER_MODEL): {
        "reference_id": "gpt56luna_claude_opus5_user54",
        # Same prefix as the two entries above, so the cheap bound is the one
        # measured for it under this user simulator and is matched. The strong
        # bound is left absent on purpose: Opus 5 alone on this surface is being
        # measured in the same batch as this arm and has no recorded number yet.
        # Borrowing Sonnet's would make the delta-versus-bound columns read as
        # measured when nothing had been.
        "cheap_complete_count": 34,
        "cheap_avg_similarity_score": 0.6664454961240309,
        "strong_complete_count": None,
        "strong_avg_similarity_score": None,
        "target_complete_count": 40,
        "reference_bounds_matched_user_model": True,
    },
    # The two Haiku-prefix pools. Their cheap bound is one measurement: Haiku
    # alone on this surface under this user simulator, strict forty-four of one
    # hundred and twenty-nine. Each strong bound is the same model measured
    # alone here, so Sonnet's is populated and Sol's is the registered new-pool
    # strong count; neither is borrowed across pools.
    ("claude-haiku-4-5-20251001", "claude-sonnet-5", EXPECTED_USER_MODEL): {
        "reference_id": "claudehaiku45_claude_sonnet5_user54",
        "cheap_complete_count": 44,
        "cheap_avg_similarity_score": 0.856873,
        "strong_complete_count": 47,
        "strong_avg_similarity_score": 0.827559,
        "target_complete_count": 44,
        "reference_bounds_matched_user_model": True,
    },
    ("claude-haiku-4-5-20251001", "openai/claude-sonnet-5", EXPECTED_USER_MODEL): {
        "reference_id": "claudehaiku45_claude_sonnet5_user54",
        "cheap_complete_count": 44,
        "cheap_avg_similarity_score": 0.856873,
        "strong_complete_count": 47,
        "strong_avg_similarity_score": 0.827559,
        "target_complete_count": 44,
        "reference_bounds_matched_user_model": True,
    },
    ("claude-haiku-4-5-20251001", "gpt-5.6-sol", EXPECTED_USER_MODEL): {
        "reference_id": "claudehaiku45_gpt56sol_user54",
        "cheap_complete_count": 44,
        "cheap_avg_similarity_score": 0.856873,
        "strong_complete_count": 39,
        "strong_avg_similarity_score": 0.6865741317829455,
        "target_complete_count": 44,
        "reference_bounds_matched_user_model": True,
    },
    ("claude-sonnet-5", "gpt-5.5", "gpt-5.5"): {
        "reference_id": "sonnet5_gpt55_user55",
        "cheap_complete_count": 38,
        "cheap_avg_similarity_score": 0.761699,
        "strong_complete_count": 39,
        "strong_avg_similarity_score": 0.691812,
        "target_complete_count": 49,
        "paper_avg_similarity_floor": 0.809,
        "paper_agent_side_cost_ceiling_usd": 5.05,
    },
}
EXPECTED_OFFICIAL_ENGINE = "tool_sandbox Scenario.play_and_evaluate"
REQUIRED_TOOL_ENV_VARS = ("RAPID_API_KEY",)
DEFAULT_STRONG_RESIDUAL_CALL_CEILING = 7
PROTECTED_ROW_STRONG_RESIDUAL_CALL_CEILING = 8
PROTECTED_ROW_CAP8_APPROVAL_REVISION = "protected_cheap_pass_regression_cap8"
PROTECTED_ROW_CAP8_APPROVAL_BASIS = "cap_exhausted_online_frontier_protected_row"
CAPABILITY_STRONG_RESIDUAL_CALL_CEILING = 129
CAPABILITY_CAP129_APPROVAL_REVISION = "frontier_starvation_capability_cap129"
CAPABILITY_CAP129_APPROVAL_BASIS = "cap8_fcfs_starvation_diagnostic_20260628"
# Receipt-grounded authoring spends one provider call per native step rather than
# one per episode, so a full run needs a budget an order of magnitude above the
# per-episode ceilings above. The ceiling is stated per episode and scaled by the
# cap, so a sharded run carries the same per-row bound as a full run. It stays
# opt-in behind its own approval revision; nothing here changes the default cap.
GROUNDED_AUTHORING_PER_EPISODE_CALL_CEILING = 16
GROUNDED_AUTHORING_STRONG_RESIDUAL_CALL_CEILING = (
    GROUNDED_AUTHORING_PER_EPISODE_CALL_CEILING * EXPECTED_TASK_COUNT
)
GROUNDED_AUTHORING_APPROVAL_REVISION = "grounded_authoring_receipt_bounded"
GROUNDED_AUTHORING_APPROVAL_BASIS = "receipt_grounded_iterative_authoring_20260729"
DEFAULT_MAX_AUTHORING_ROUNDS = 14
DEFAULT_MAX_REPROPOSAL_ROUNDS = 2
BUDGET_POLICY_GLOBAL_FCFS = "global_fcfs"
PROVIDER_ACCOUNTING_SCOPE = "cheap_strong_user_simulator_model_calls_and_wall_clock"

RuntimeFactory = Callable[[], ToolSandboxOnlineRuntime]
StrongProposerFactory = Callable[[], StrongProposer]
EpisodeReceiptCallback = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class Base129ScenarioSpec:
    public_ordinal: int
    scenario_name: str


def build_base129_true_online_run(
    *,
    scenario_specs: Sequence[Mapping[str, Any] | Base129ScenarioSpec],
    approval_receipt: Mapping[str, Any],
    execute_requested: bool,
    runtime_factory: RuntimeFactory | None = None,
    strong_proposer_factory: StrongProposerFactory | None = None,
    cap: int = EXPECTED_TASK_COUNT,
    max_strong_residual_calls: int = 7,
    max_turn_steps: int = 60,
    max_authoring_rounds: int = DEFAULT_MAX_AUTHORING_ROUNDS,
    max_reproposal_rounds: int = DEFAULT_MAX_REPROPOSAL_ROUNDS,
    cheap_model: str = EXPECTED_CHEAP_MODEL,
    cheap_reasoning_effort: str = EXPECTED_REASONING,
    strong_model: str = EXPECTED_STRONG_MODEL,
    strong_reasoning_effort: str = EXPECTED_REASONING,
    user_model: str = EXPECTED_USER_MODEL,
    user_reasoning_effort: str = EXPECTED_REASONING,
    execution_environment_blockers: Sequence[str] = (),
    existing_receipts: Sequence[Mapping[str, Any]] = (),
    episode_receipt_callback: EpisodeReceiptCallback | None = None,
    episode_wall_timeout_seconds: float | None = None,
    terminal_with_last_call: bool = False,
    now: datetime | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run or preflight the base129 true-online ToolSandbox method."""

    specs = _scenario_specs(scenario_specs, cap=cap)
    blockers = _blockers(
        specs=specs,
        approval_receipt=approval_receipt,
        execute_requested=execute_requested,
        runtime_factory=runtime_factory,
        strong_proposer_factory=strong_proposer_factory,
        cap=cap,
        max_strong_residual_calls=max_strong_residual_calls,
        cheap_model=cheap_model,
        cheap_reasoning_effort=cheap_reasoning_effort,
        strong_model=strong_model,
        strong_reasoning_effort=strong_reasoning_effort,
        user_model=user_model,
        user_reasoning_effort=user_reasoning_effort,
        terminal_with_last_call=terminal_with_last_call,
        execution_environment_blockers=execution_environment_blockers,
    )
    receipts: list[dict[str, Any]] = [
        dict(receipt) for receipt in existing_receipts if isinstance(receipt, Mapping)
    ]
    completed_ordinals = {_int(receipt.get("public_ordinal")) for receipt in receipts}
    strong_calls_used = sum(_int(receipt.get("strong_call_count")) for receipt in receipts)
    runtime_failures = sum(
        1 for receipt in receipts if receipt.get("decision") == "runtime_failure"
    )
    if execute_requested and not blockers:
        for spec in specs:
            if spec.public_ordinal in completed_ordinals:
                continue
            remaining_strong_calls = max(0, int(max_strong_residual_calls) - strong_calls_used)
            runtime = runtime_factory() if runtime_factory is not None else None
            proposer = (
                strong_proposer_factory()
                if strong_proposer_factory is not None
                else None
            )
            if runtime is None or proposer is None:
                raise RuntimeError("execute path requires runtime and proposer factories")
            try:
                private_receipt, public_summary = _run_controller_episode(
                    spec=spec,
                    runtime=runtime,
                    proposer=proposer,
                    max_turn_steps=max_turn_steps,
                    max_strong_calls=remaining_strong_calls,
                    max_authoring_rounds=max_authoring_rounds,
                    max_reproposal_rounds=max_reproposal_rounds,
                    terminal_with_last_call=terminal_with_last_call,
                    episode_wall_timeout_seconds=episode_wall_timeout_seconds,
                )
            except Exception as exc:  # noqa: BLE001 - receipt records runtime failure.
                runtime_failures += 1
                private_receipt = {
                    "schema": "papc.toolsandbox.base129_true_online_episode_error.v1",
                    "private": {"scenario_name": spec.scenario_name},
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
                public_summary = {
                    "decision": "runtime_failure",
                    "strong_call_count": 0,
                    "official_evaluator_used": False,
                    "evaluation": {},
                }
            strong_calls_used += _int(public_summary.get("strong_call_count"))
            receipts.append(
                _episode_receipt(
                    spec=spec,
                    private_receipt=private_receipt,
                    public_summary=public_summary,
                )
            )
            completed_ordinals.add(spec.public_ordinal)
            if episode_receipt_callback is not None:
                episode_receipt_callback(receipts[-1])
    generated_at = (now or datetime.now(timezone.utc)).isoformat()
    private_report = {
        "schema": PRIVATE_SCHEMA,
        "surface": EXPECTED_SURFACE,
        "mode": "toolsandbox_base129_true_online_cfrc_private",
        "generated_at": generated_at,
        "execute_requested": bool(execute_requested),
        "provider_accounting_scope": PROVIDER_ACCOUNTING_SCOPE,
        "strong_residual_provider_calls": strong_calls_used,
        "receipt_count": len(receipts),
        "receipts": receipts,
        "blockers": blockers,
    }
    if terminal_with_last_call:
        private_report["method_variant"] = "terminal_with_last_call"
    public_report = _public_report(
        generated_at=generated_at,
        execute_requested=execute_requested,
        specs=specs,
        receipts=receipts,
        blockers=blockers,
        strong_calls_used=strong_calls_used,
        runtime_failures=runtime_failures,
        cap=cap,
        max_strong_residual_calls=max_strong_residual_calls,
        cheap_model=cheap_model,
        cheap_reasoning_effort=cheap_reasoning_effort,
        strong_model=strong_model,
        strong_reasoning_effort=strong_reasoning_effort,
        user_model=user_model,
        user_reasoning_effort=user_reasoning_effort,
        terminal_with_last_call=terminal_with_last_call,
    )
    return private_report, public_report


def _run_controller_episode(
    *,
    spec: Base129ScenarioSpec,
    runtime: ToolSandboxOnlineRuntime,
    proposer: StrongProposer,
    max_turn_steps: int,
    max_strong_calls: int,
    max_authoring_rounds: int,
    max_reproposal_rounds: int,
    terminal_with_last_call: bool,
    episode_wall_timeout_seconds: float | None,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if episode_wall_timeout_seconds is not None and episode_wall_timeout_seconds > 0:
        return _run_controller_episode_subprocess(
            spec=spec,
            runtime=runtime,
            proposer=proposer,
            max_turn_steps=max_turn_steps,
            max_strong_calls=max_strong_calls,
            max_authoring_rounds=max_authoring_rounds,
            max_reproposal_rounds=max_reproposal_rounds,
            terminal_with_last_call=terminal_with_last_call,
            episode_wall_timeout_seconds=float(episode_wall_timeout_seconds),
        )
    return _run_controller_episode_inline(
        spec=spec,
        runtime=runtime,
        proposer=proposer,
        max_turn_steps=max_turn_steps,
        max_strong_calls=max_strong_calls,
        max_authoring_rounds=max_authoring_rounds,
        max_reproposal_rounds=max_reproposal_rounds,
        terminal_with_last_call=terminal_with_last_call,
    )


def _run_controller_episode_inline(
    *,
    spec: Base129ScenarioSpec,
    runtime: ToolSandboxOnlineRuntime,
    proposer: StrongProposer,
    max_turn_steps: int,
    max_strong_calls: int,
    max_authoring_rounds: int,
    max_reproposal_rounds: int,
    terminal_with_last_call: bool,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    controller = ToolSandboxOnlineCFRCController(
        runtime=runtime,
        frontier_monitor=ToolSandboxReminderAddFrontierMonitor(
            public_ordinal=spec.public_ordinal
        ),
        strong_proposer=proposer,
        compile_config=_compile_config(),
    )
    return controller.run_episode(
        scenario_name=spec.scenario_name,
        config=ToolSandboxOnlineCFRCConfig(
            max_turn_steps=max_turn_steps,
            max_strong_calls=max_strong_calls,
            max_authoring_rounds=max_authoring_rounds,
            max_reproposal_rounds=max_reproposal_rounds,
            terminal_with_last_call=terminal_with_last_call,
        ),
    )


def _run_controller_episode_subprocess(
    *,
    spec: Base129ScenarioSpec,
    runtime: ToolSandboxOnlineRuntime,
    proposer: StrongProposer,
    max_turn_steps: int,
    max_strong_calls: int,
    max_authoring_rounds: int,
    max_reproposal_rounds: int,
    terminal_with_last_call: bool,
    episode_wall_timeout_seconds: float,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    context = multiprocessing.get_context("fork")
    with tempfile.TemporaryDirectory(prefix="papc_toolsandbox_episode_") as tmpdir:
        result_path = Path(tmpdir) / "result.json"
        process = context.Process(
            target=_episode_subprocess_entry,
            kwargs={
                "result_path": result_path,
                "spec": spec,
                "runtime": runtime,
                "proposer": proposer,
                "max_turn_steps": max_turn_steps,
                "max_strong_calls": max_strong_calls,
                "max_authoring_rounds": max_authoring_rounds,
                "max_reproposal_rounds": max_reproposal_rounds,
                "terminal_with_last_call": terminal_with_last_call,
            },
        )
        process.start()
        process.join(float(episode_wall_timeout_seconds))
        if process.is_alive():
            process.terminate()
            process.join(5.0)
            if process.is_alive():
                process.kill()
                process.join(5.0)
            raise TimeoutError(
                f"ToolSandbox episode public_ordinal={spec.public_ordinal} "
                f"timed out after {episode_wall_timeout_seconds:g}s"
            )
        if not result_path.exists():
            raise RuntimeError(
                f"ToolSandbox episode public_ordinal={spec.public_ordinal} "
                f"subprocess exited without result (exitcode={process.exitcode})"
            )
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError("ToolSandbox episode subprocess returned non-object result")
    if payload.get("ok") is True:
        private_receipt = payload.get("private_receipt")
        public_summary = payload.get("public_summary")
        if not isinstance(private_receipt, Mapping) or not isinstance(
            public_summary, Mapping
        ):
            raise RuntimeError("ToolSandbox episode subprocess returned malformed receipt")
        return private_receipt, public_summary
    error_type = str(payload.get("error_type") or "RuntimeError")
    error_message = str(payload.get("error_message") or "episode subprocess failed")
    raise RuntimeError(f"{error_type}: {error_message}")


def _episode_subprocess_entry(
    *,
    result_path: Path,
    spec: Base129ScenarioSpec,
    runtime: ToolSandboxOnlineRuntime,
    proposer: StrongProposer,
    max_turn_steps: int,
    max_strong_calls: int,
    max_authoring_rounds: int,
    max_reproposal_rounds: int,
    terminal_with_last_call: bool,
) -> None:
    try:
        private_receipt, public_summary = _run_controller_episode_inline(
            spec=spec,
            runtime=runtime,
            proposer=proposer,
            max_turn_steps=max_turn_steps,
            max_strong_calls=max_strong_calls,
            max_authoring_rounds=max_authoring_rounds,
            max_reproposal_rounds=max_reproposal_rounds,
            terminal_with_last_call=terminal_with_last_call,
        )
        payload = {
            "ok": True,
            "private_receipt": private_receipt,
            "public_summary": public_summary,
        }
    except BaseException as exc:  # noqa: BLE001 - child reports error to parent.
        payload = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def scenario_specs_from_lower_receipts(receipt_root: Path) -> list[Base129ScenarioSpec]:
    import json

    specs: list[Base129ScenarioSpec] = []
    for path in sorted(Path(receipt_root).glob("case_*.json")):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        private = receipt.get("private") if isinstance(receipt, Mapping) else {}
        scenario_name = str(private.get("scenario_name") or "")
        if not scenario_name:
            continue
        specs.append(
            Base129ScenarioSpec(
                public_ordinal=_int(receipt.get("public_ordinal")),
                scenario_name=scenario_name,
            )
        )
    return specs


def _public_report(
    *,
    generated_at: str,
    execute_requested: bool,
    specs: Sequence[Base129ScenarioSpec],
    receipts: Sequence[Mapping[str, Any]],
    blockers: Sequence[str],
    strong_calls_used: int,
    runtime_failures: int,
    cap: int,
    max_strong_residual_calls: int,
    cheap_model: str,
    cheap_reasoning_effort: str,
    strong_model: str,
    strong_reasoning_effort: str,
    user_model: str,
    user_reasoning_effort: str,
    terminal_with_last_call: bool,
) -> dict[str, Any]:
    operating_point_reference = _operating_point_reference(
        cheap_model=cheap_model,
        strong_model=strong_model,
        user_model=user_model,
    )
    if operating_point_reference is None:
        operating_point_reference = {
            "reference_id": "unregistered",
            "cheap_complete_count": EXPECTED_CHEAP_COMPLETE,
            "cheap_avg_similarity_score": EXPECTED_CHEAP_AVG_SIMILARITY_SCORE,
            "strong_complete_count": EXPECTED_STRONG_COMPLETE,
            "strong_avg_similarity_score": EXPECTED_STRONG_AVG_SIMILARITY_SCORE,
        }
    cheap_complete_reference = _optional_int(
        operating_point_reference.get("cheap_complete_count")
    )
    strong_complete_reference = _optional_int(
        operating_point_reference.get("strong_complete_count")
    )
    official_scored = sum(
        1 for receipt in receipts if receipt.get("official_evaluator_used") is True
    )
    method_complete = sum(1 for receipt in receipts if _milestone_complete(receipt))
    official_score_summary = _official_score_summary(
        receipts,
        strict_complete_count=method_complete,
        cheap_avg_similarity_reference=_optional_float(
            operating_point_reference.get("cheap_avg_similarity_score")
        ),
        strong_avg_similarity_reference=_optional_float(
            operating_point_reference.get("strong_avg_similarity_score")
        ),
    )
    minefield_hits = sum(1 for receipt in receipts if _minefield_hit(receipt))
    total_minefield_hits = sum(1 for receipt in receipts if _any_minefield_hit(receipt))
    inherited_or_cheap_minefield_hits = total_minefield_hits - minefield_hits
    frontier_open = sum(1 for receipt in receipts if receipt.get("online_frontier_generated") is True)
    suppression_counts = _suppression_counts(
        receipts,
        max_strong_residual_calls=max_strong_residual_calls,
    )
    strong_cap_exhausted_frontier = suppression_counts["global_cap_exhausted"]
    per_episode_frontier_suppressed = suppression_counts["per_episode_limit"]
    total_frontier_suppressed = suppression_counts["total"]
    # Either ceiling claims the same thing: the budget no longer starves any
    # frontier. Only the number of calls one frontier needs differs. A shard
    # carries the same per-row budget but cannot witness capability on its own.
    capability_witness_requested = _is_capability_request(
        cap=cap,
        max_strong_residual_calls=max_strong_residual_calls,
    ) or (
        int(cap) == EXPECTED_TASK_COUNT
        and _is_grounded_authoring_request(
            cap=cap,
            max_strong_residual_calls=max_strong_residual_calls,
        )
    )
    capability_witness_valid = bool(
        capability_witness_requested
        and execute_requested
        and len(receipts) == EXPECTED_TASK_COUNT
        and strong_cap_exhausted_frontier == 0
    )
    residual_execute = sum(1 for receipt in receipts if receipt.get("decision") == "execute")
    residual_block_or_abstain = sum(
        1
        for receipt in receipts
        if receipt.get("decision") == "cheap_continued_after_strong_block"
    )
    runtime_failure_count = sum(
        1 for receipt in receipts if receipt.get("decision") == "runtime_failure"
    )
    residual_runtime_error_count = sum(
        _residual_runtime_error_count(receipt) for receipt in receipts
    )
    checked_execution_attempt_count = sum(
        1 for receipt in receipts if receipt.get("checked_execution_attempted") is True
    )
    checked_execution_admitted_count = sum(
        1 for receipt in receipts if receipt.get("checked_execution_admitted") is True
    )
    checked_execution_block_count = (
        checked_execution_attempt_count - checked_execution_admitted_count
    )
    method_runtime_ledger = _aggregate_runtime_ledger(receipts)
    method_runtime_breakdown = _method_runtime_breakdown(
        method_runtime_ledger,
        per_row_wall_seconds=_method_wall_clock_seconds(receipts),
        cheap_model=cheap_model,
        strong_model=strong_model,
    )
    method_resource_available = _runtime_accounting_available(method_runtime_ledger)
    provider_calls_observed = (
        _runtime_provider_calls(method_runtime_ledger) if execute_requested else 0
    )
    report_blockers = set(blockers)
    report_findings: set[str] = set()
    target_complete = _int(
        operating_point_reference.get("target_complete_count")
    ) or TARGET_COMPLETE
    paper_avg_floor = _optional_float(
        operating_point_reference.get("paper_avg_similarity_floor")
    )
    paper_cost_ceiling = _optional_float(
        operating_point_reference.get("paper_agent_side_cost_ceiling_usd")
    )
    agent_cache_neutral_cost = _optional_float(
        _mapping(
            method_runtime_breakdown.get("agent_cache_neutral_cost")
        ).get("agent_side_total_cost_usd")
    )
    paper_gate_evaluated = bool(
        execute_requested
        and len(receipts) == EXPECTED_TASK_COUNT
        and official_scored == EXPECTED_TASK_COUNT
        and paper_avg_floor is not None
        and paper_cost_ceiling is not None
    )
    bounded_canary_evaluated = bool(
        execute_requested
        and 0 < len(receipts) < EXPECTED_TASK_COUNT
        and official_scored == len(receipts)
        and checked_execution_attempt_count == len(receipts)
    )
    bounded_canary_ready = bool(
        bounded_canary_evaluated
        and checked_execution_admitted_count == len(receipts)
        and residual_runtime_error_count == 0
        and runtime_failure_count == 0
        and minefield_hits == 0
    )
    if execute_requested:
        if runtime_failure_count:
            report_blockers.add("runtime_failure_present")
        if residual_runtime_error_count:
            report_blockers.add("residual_runtime_error_present")
        if len(receipts) != min(int(cap), EXPECTED_TASK_COUNT):
            report_blockers.add("episode_count_mismatch")
        if receipts and official_scored != len(receipts):
            report_blockers.add("official_evaluator_missing")
        if strong_calls_used > int(max_strong_residual_calls):
            report_blockers.add("strong_residual_call_cap_exceeded")
        if capability_witness_requested and strong_cap_exhausted_frontier:
            report_blockers.add("capability_witness_cap_still_binding")
        if len(receipts) == EXPECTED_TASK_COUNT and official_scored == EXPECTED_TASK_COUNT:
            if paper_gate_evaluated:
                avg_similarity = _optional_float(
                    official_score_summary.get("official_avg_similarity_score")
                )
                if avg_similarity is None or avg_similarity < float(paper_avg_floor):
                    report_blockers.add("paper_avg_similarity_floor_missed")
                if method_complete < target_complete:
                    report_blockers.add("paper_strict_complete_floor_missed")
                if agent_cache_neutral_cost is None:
                    report_blockers.add("paper_agent_cost_unavailable")
                elif agent_cache_neutral_cost > float(paper_cost_ceiling):
                    report_blockers.add("paper_agent_cost_ceiling_exceeded")
            elif method_complete < target_complete:
                report_findings.add("strict_complete_below_secondary_target")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "surface": EXPECTED_SURFACE,
        "mode": "toolsandbox_base129_true_online_cfrc",
        "generated_at": generated_at,
        "execute_requested": bool(execute_requested),
        "provider_calls": provider_calls_observed,
        "provider_accounting_scope": PROVIDER_ACCOUNTING_SCOPE,
        "api_spend_performed": bool(execute_requested),
        "official_engine": EXPECTED_OFFICIAL_ENGINE,
        "official_evaluator_used": bool(receipts)
        and official_scored == len(receipts)
        and len(receipts) == min(int(cap), EXPECTED_TASK_COUNT),
        "models": {
            "cheap_model": cheap_model,
            "cheap_reasoning_effort": cheap_reasoning_effort,
            "strong_model": strong_model,
            "strong_reasoning_effort": strong_reasoning_effort,
            "user_model": user_model,
            "user_reasoning_effort": user_reasoning_effort,
        },
        "budget": {
            "scenario_scope": EXPECTED_SCENARIO_SCOPE,
            "cap": int(cap),
            "max_cheap_episodes": min(int(cap), EXPECTED_TASK_COUNT),
            "max_strong_residual_calls": int(max_strong_residual_calls),
            "budget_policy": BUDGET_POLICY_GLOBAL_FCFS,
            "budget_policy_enum": [
                BUDGET_POLICY_GLOBAL_FCFS,
                "recoverability_priority",
            ],
            "full_agent_fallback_allowed": False,
        },
        "summary": {
            "itt_task_count": EXPECTED_TASK_COUNT,
            "attempted_episode_count": len(receipts),
            "scenario_spec_count": len(specs),
            **official_score_summary,
            "operating_point_reference_id": operating_point_reference["reference_id"],
            # False means the cheap/strong bounds came from a different user
            # simulator, so every delta-versus-bound column below is provisional.
            "reference_bounds_matched_user_model": bool(
                operating_point_reference.get(
                    "reference_bounds_matched_user_model", True
                )
            ),
            "reference_bounds_complete": bool(
                operating_point_reference.get(
                    "reference_bounds_complete",
                    cheap_complete_reference is not None
                    and strong_complete_reference is not None,
                )
            ),
            "cheap_complete_floor_count": cheap_complete_reference,
            "strong_complete_bound_count": strong_complete_reference,
            "target_complete_count": target_complete,
            "cheap_to_strong_gap_count": (
                strong_complete_reference - cheap_complete_reference
                if cheap_complete_reference is not None
                and strong_complete_reference is not None
                else None
            ),
            "online_frontier_open_count": frontier_open,
            "frontier_suppressed_count": total_frontier_suppressed,
            "strong_cap_exhausted_frontier_count": strong_cap_exhausted_frontier,
            "per_episode_frontier_suppressed_count": per_episode_frontier_suppressed,
            "capability_witness_requested": capability_witness_requested,
            "capability_witness_valid": capability_witness_valid,
            "residual_execute_count": residual_execute,
            "residual_block_or_abstain_count": residual_block_or_abstain,
            "method_complete_count": method_complete,
            "method_target_complete_met": method_complete >= target_complete,
            "method_target_shortfall_count": max(0, target_complete - method_complete),
            "recovered_cheap_to_strong_gap_count": max(
                0, method_complete - cheap_complete_reference
            )
            if cheap_complete_reference is not None
            else None,
            "method_delta_vs_cheap_floor": (
                method_complete - cheap_complete_reference
                if cheap_complete_reference is not None
                else None
            ),
            "method_delta_vs_strong_bound": (
                method_complete - strong_complete_reference
                if strong_complete_reference is not None
                else None
            ),
            "strong_residual_call_count": strong_calls_used,
            "method_resource_accounting_available": method_resource_available,
            "official_scored_episode_count": official_scored,
            "runtime_failure_count": runtime_failure_count,
            "residual_runtime_error_count": residual_runtime_error_count,
            "checked_execution_attempt_count": checked_execution_attempt_count,
            "checked_execution_admitted_count": checked_execution_admitted_count,
            "checked_execution_block_count": checked_execution_block_count,
            "method_introduced_minefield_hit_count": minefield_hits,
            "inherited_or_cheap_minefield_hit_count": inherited_or_cheap_minefield_hits,
            "total_minefield_hit_count": total_minefield_hits,
            "single_trial_caveat": True,
        },
        "paper_acceptance_gate": {
            "evaluated": paper_gate_evaluated,
            "requires_full_base129": True,
            "official_avg_similarity_floor": paper_avg_floor,
            "strict_complete_floor": target_complete,
            "agent_side_cache_neutral_cost_ceiling_usd": paper_cost_ceiling,
            "agent_side_cache_neutral_cost_usd": agent_cache_neutral_cost,
            "method_introduced_minefield_hits_allowed": 0,
            "runtime_errors_allowed": 0,
        },
        "bounded_canary_gate": {
            "evaluated": bounded_canary_evaluated,
            "ready_for_fresh_full_gate": bounded_canary_ready,
            "requires_every_selected_row_checked": True,
            "selected_row_count": len(receipts),
            "checked_execution_attempt_count": checked_execution_attempt_count,
            "checked_execution_admitted_count": checked_execution_admitted_count,
            "runtime_error_count": residual_runtime_error_count,
            "runtime_failure_count": runtime_failure_count,
            "method_introduced_minefield_hit_count": minefield_hits,
        },
        "public_rows": _public_rows(
            receipts,
            max_strong_residual_calls=max_strong_residual_calls,
        ),
        "claim_boundary": [
            "This is a true-online ToolSandbox base129 CFRC report.",
            "ToolSandbox's primary official score is average similarity; strict complete count is secondary.",
            "It reports recovery of the cheap-to-strong gap, not parity with strong.",
            "n=129 and this is a single trial; one task is within run noise.",
        ],
        "resource_accounting": {
            "schema": "papc.toolsandbox.base129_method_resource_accounting.v1",
            "missing_fields_are_unavailable_not_zero": True,
            "raw_provider_payloads_excluded": True,
            "baseline_resources": {
                "status": "pending_fresh_accounting_run",
                "operating_point_reference_id": operating_point_reference["reference_id"],
                "cheap_complete_floor_count": cheap_complete_reference,
                "strong_complete_bound_count": strong_complete_reference,
                "token_latency_accounting_available": False,
            },
            "method_resources": {
                "runtime_ledger": method_runtime_ledger,
                "runtime_breakdown": method_runtime_breakdown,
            },
        },
        "authorizations": {
            "records_explicit_approval": bool(
                execute_requested
                and not any(str(blocker).startswith("approval_") for blocker in blockers)
            ),
            "authorizes_method_result_claim": bool(
                execute_requested and not report_blockers
            ),
            "authorizes_strong_equivalence_claim": False,
            "authorizes_parity_claim": False,
            "authorizes_full_domain_expansion": False,
        },
        "blockers": sorted(report_blockers),
        "non_blocking_findings": sorted(report_findings),
    }
    if terminal_with_last_call:
        report["method_variant"] = "terminal_with_last_call"
        rejection_reason_counts: dict[str, int] = {}
        for receipt in receipts:
            for reason in receipt.get(
                "terminal_with_last_call_rejection_reasons"
            ) or ():
                key = str(reason)
                rejection_reason_counts[key] = (
                    int(rejection_reason_counts.get(key, 0)) + 1
                )
        report["summary"]["terminal_with_last_call"] = {
            "offered_count": sum(
                _int(item.get("terminal_with_last_call_offered_count"))
                for item in receipts
            ),
            "accepted_count": sum(
                _int(item.get("terminal_with_last_call_accepted_count"))
                for item in receipts
            ),
            "rejected_count": sum(
                _int(item.get("terminal_with_last_call_rejected_count"))
                for item in receipts
            ),
            "episode_count_by_rejection_reason": dict(
                sorted(rejection_reason_counts.items())
            ),
        }
    if "strong_residual_call_cap8_revision_missing" in report_blockers:
        report["approval_revision_required"] = {
            "approval_revision": PROTECTED_ROW_CAP8_APPROVAL_REVISION,
            "approval_revision_basis": PROTECTED_ROW_CAP8_APPROVAL_BASIS,
            "max_strong_residual_calls": PROTECTED_ROW_STRONG_RESIDUAL_CALL_CEILING,
        }
    if "strong_residual_capability_revision_missing" in report_blockers:
        report["approval_revision_required"] = {
            "approval_revision": CAPABILITY_CAP129_APPROVAL_REVISION,
            "approval_revision_basis": CAPABILITY_CAP129_APPROVAL_BASIS,
            "max_strong_residual_calls": CAPABILITY_STRONG_RESIDUAL_CALL_CEILING,
        }
    if execute_requested and minefield_hits:
        report["blockers"] = sorted(
            {*report["blockers"], "method_introduced_minefield_hit"}
        )
        report["authorizations"]["authorizes_method_result_claim"] = False
    report["decision"] = (
        EXECUTE_READY_DECISION
        if execute_requested and not report["blockers"]
        else NOAPI_READY_DECISION
        if not execute_requested and not report["blockers"]
        else BLOCKED_DECISION
    )
    leakage = public_leakage_audit(report)
    report["public_leakage_audit"] = leakage
    if leakage["pass"] is not True:
        report["blockers"] = sorted({*report["blockers"], "public_leakage_marker_present"})
        report["decision"] = BLOCKED_DECISION
        report["authorizations"]["authorizes_method_result_claim"] = False
    return report


def _blockers(
    *,
    specs: Sequence[Base129ScenarioSpec],
    approval_receipt: Mapping[str, Any],
    execute_requested: bool,
    runtime_factory: RuntimeFactory | None,
    strong_proposer_factory: StrongProposerFactory | None,
    cap: int,
    max_strong_residual_calls: int,
    cheap_model: str,
    cheap_reasoning_effort: str,
    strong_model: str,
    strong_reasoning_effort: str,
    user_model: str,
    user_reasoning_effort: str,
    terminal_with_last_call: bool,
    execution_environment_blockers: Sequence[str],
) -> list[str]:
    blockers: list[str] = []
    blockers.extend(str(blocker) for blocker in execution_environment_blockers if blocker)
    if cap < 1 or cap > EXPECTED_TASK_COUNT:
        blockers.append("cap_outside_base129_bounds")
    if not specs:
        blockers.append("scenario_specs_missing")
    if len(specs) != min(int(cap), EXPECTED_TASK_COUNT):
        blockers.append("scenario_spec_count_mismatch")
    if max_strong_residual_calls < 0:
        blockers.append("strong_residual_call_cap_negative")
    elif _is_grounded_authoring_request(
        cap=cap,
        max_strong_residual_calls=max_strong_residual_calls,
    ):
        if not _has_grounded_authoring_revision(approval_receipt):
            blockers.append("strong_residual_grounded_authoring_revision_missing")
    elif max_strong_residual_calls == GROUNDED_AUTHORING_STRONG_RESIDUAL_CALL_CEILING:
        blockers.append("strong_residual_grounded_authoring_requires_all129")
    elif _is_capability_request(
        cap=cap,
        max_strong_residual_calls=max_strong_residual_calls,
    ):
        if not _has_capability_revision(approval_receipt):
            blockers.append("strong_residual_capability_revision_missing")
    elif max_strong_residual_calls == CAPABILITY_STRONG_RESIDUAL_CALL_CEILING:
        blockers.append("strong_residual_capability_requires_all129")
    elif max_strong_residual_calls > PROTECTED_ROW_STRONG_RESIDUAL_CALL_CEILING:
        blockers.append("strong_residual_call_cap_exceeds_protected_row_ceiling")
    elif max_strong_residual_calls > DEFAULT_STRONG_RESIDUAL_CALL_CEILING and not _has_cap8_revision(
        approval_receipt
    ):
        blockers.append("strong_residual_call_cap8_revision_missing")
    if (
        _operating_point_reference(
            cheap_model=cheap_model,
            strong_model=strong_model,
            user_model=user_model,
        )
        is None
    ):
        blockers.append("operating_point_reference_missing")
    if cheap_reasoning_effort != EXPECTED_REASONING:
        blockers.append("cheap_reasoning_effort_mismatch")
    if strong_reasoning_effort != EXPECTED_REASONING:
        blockers.append("strong_reasoning_effort_mismatch")
    if user_reasoning_effort != EXPECTED_REASONING:
        blockers.append("user_reasoning_effort_mismatch")
    if execute_requested:
        if runtime_factory is None:
            blockers.append("runtime_factory_missing")
        if strong_proposer_factory is None:
            blockers.append("strong_proposer_factory_missing")
        blockers.extend(
            _approval_blockers(
                approval_receipt=approval_receipt,
                cap=cap,
                max_strong_residual_calls=max_strong_residual_calls,
                cheap_model=cheap_model,
                cheap_reasoning_effort=cheap_reasoning_effort,
                strong_model=strong_model,
                strong_reasoning_effort=strong_reasoning_effort,
                user_model=user_model,
                user_reasoning_effort=user_reasoning_effort,
                terminal_with_last_call=terminal_with_last_call,
            )
        )
    return sorted(set(blockers))


def _has_cap8_revision(approval_receipt: Mapping[str, Any]) -> bool:
    return (
        approval_receipt.get("approval_revision")
        == PROTECTED_ROW_CAP8_APPROVAL_REVISION
        and approval_receipt.get("approval_revision_basis")
        == PROTECTED_ROW_CAP8_APPROVAL_BASIS
    )


def _has_capability_revision(approval_receipt: Mapping[str, Any]) -> bool:
    return (
        approval_receipt.get("approval_revision")
        == CAPABILITY_CAP129_APPROVAL_REVISION
        and approval_receipt.get("approval_revision_basis")
        == CAPABILITY_CAP129_APPROVAL_BASIS
    )


def _is_capability_request(*, cap: int, max_strong_residual_calls: int) -> bool:
    return (
        int(cap) == EXPECTED_TASK_COUNT
        and int(max_strong_residual_calls) == CAPABILITY_STRONG_RESIDUAL_CALL_CEILING
    )


def _has_grounded_authoring_revision(approval_receipt: Mapping[str, Any]) -> bool:
    return (
        approval_receipt.get("approval_revision")
        == GROUNDED_AUTHORING_APPROVAL_REVISION
        and approval_receipt.get("approval_revision_basis")
        == GROUNDED_AUTHORING_APPROVAL_BASIS
    )


def grounded_authoring_call_ceiling(cap: int) -> int:
    """Return the receipt-grounded call budget for a cap of this size."""

    return GROUNDED_AUTHORING_PER_EPISODE_CALL_CEILING * min(
        int(cap), EXPECTED_TASK_COUNT
    )


def _is_grounded_authoring_request(
    *,
    cap: int,
    max_strong_residual_calls: int,
) -> bool:
    return int(max_strong_residual_calls) == grounded_authoring_call_ceiling(cap)


def _approval_blockers(
    *,
    approval_receipt: Mapping[str, Any],
    cap: int,
    max_strong_residual_calls: int,
    cheap_model: str,
    cheap_reasoning_effort: str,
    strong_model: str,
    strong_reasoning_effort: str,
    user_model: str,
    user_reasoning_effort: str,
    terminal_with_last_call: bool,
) -> list[str]:
    blockers: list[str] = []
    if approval_receipt.get("approval_scope") != APPROVAL_SCOPE:
        blockers.append("approval_scope_mismatch")
    if approval_receipt.get("arm") != METHOD_PROFILE:
        blockers.append("approval_arm_mismatch")
    if approval_receipt.get("surface") != EXPECTED_SURFACE:
        blockers.append("approval_surface_mismatch")
    if approval_receipt.get("scenario_scope") != EXPECTED_SCENARIO_SCOPE:
        blockers.append("approval_scenario_scope_mismatch")
    if _int(approval_receipt.get("max_cheap_episodes")) != min(int(cap), EXPECTED_TASK_COUNT):
        blockers.append("approval_max_cheap_episodes_mismatch")
    if _int(approval_receipt.get("max_strong_residual_calls")) != int(max_strong_residual_calls):
        blockers.append("approval_max_strong_residual_calls_mismatch")
    if approval_receipt.get("cheap_model") != cheap_model:
        blockers.append("approval_cheap_model_mismatch")
    if approval_receipt.get("cheap_reasoning_effort") != cheap_reasoning_effort:
        blockers.append("approval_cheap_reasoning_effort_mismatch")
    if approval_receipt.get("strong_model") != strong_model:
        blockers.append("approval_strong_model_mismatch")
    if approval_receipt.get("strong_reasoning_effort") != strong_reasoning_effort:
        blockers.append("approval_strong_reasoning_effort_mismatch")
    if approval_receipt.get("user_model") != user_model:
        blockers.append("approval_user_model_mismatch")
    if approval_receipt.get("user_reasoning_effort") != user_reasoning_effort:
        blockers.append("approval_user_reasoning_effort_mismatch")
    if approval_receipt.get("full_agent_fallback_allowed") is not False:
        blockers.append("approval_full_agent_fallback_mismatch")
    if approval_receipt.get("resource_accounting_required") is not True:
        blockers.append("approval_resource_accounting_required_missing")
    if (
        terminal_with_last_call
        and approval_receipt.get("method_variant") != "terminal_with_last_call"
    ):
        blockers.append("approval_method_variant_mismatch")
    return blockers


def _episode_receipt(
    *,
    spec: Base129ScenarioSpec,
    private_receipt: Mapping[str, Any],
    public_summary: Mapping[str, Any],
) -> dict[str, Any]:
    evaluation = public_summary.get("evaluation")
    receipt = {
        "schema": "papc.toolsandbox.base129_true_online_episode_receipt.v1",
        "surface": EXPECTED_SURFACE,
        "public_ordinal": spec.public_ordinal,
        "decision": str(public_summary.get("decision") or ""),
        "online_frontier_generated": public_summary.get("online_frontier_generated") is True,
        "frontier_suppressed_by_strong_cap": public_summary.get(
            "frontier_suppressed_by_strong_cap"
        )
        is True,
        "checked_execution_attempted": public_summary.get(
            "checked_execution_attempted"
        )
        is True,
        "checked_execution_admitted": public_summary.get(
            "checked_execution_admitted"
        )
        is True,
        "checked_execution_replay_match": public_summary.get(
            "checked_execution_replay_match"
        )
        is True,
        "method_refused": public_summary.get("method_refused") is True,
        "strong_call_count": _int(public_summary.get("strong_call_count")),
        "official_evaluator_used": public_summary.get("official_evaluator_used") is True,
        "official_engine": EXPECTED_OFFICIAL_ENGINE,
        "evaluation": dict(evaluation) if isinstance(evaluation, Mapping) else {},
        "private": {
            "scenario_name": spec.scenario_name,
            "controller_receipt": dict(private_receipt),
        },
    }
    authoring = public_summary.get("grounded_authoring")
    if (
        public_summary.get("method_variant") == "terminal_with_last_call"
        and isinstance(authoring, Mapping)
    ):
        receipt.update(
            {
                "method_variant": "terminal_with_last_call",
                "terminal_with_last_call_offered_count": _int(
                    authoring.get("terminal_with_last_call_offered_total")
                ),
                "terminal_with_last_call_accepted_count": _int(
                    authoring.get("terminal_with_last_call_accepted_total")
                ),
                "terminal_with_last_call_rejected_count": _int(
                    authoring.get("terminal_with_last_call_rejected_total")
                ),
                "terminal_with_last_call_rejection_reasons": sorted(
                    str(item)
                    for item in authoring.get(
                        "terminal_with_last_call_rejection_reasons"
                    )
                    or ()
                ),
            }
        )
    return receipt


def _scenario_specs(
    scenario_specs: Sequence[Mapping[str, Any] | Base129ScenarioSpec],
    *,
    cap: int,
) -> tuple[Base129ScenarioSpec, ...]:
    specs: list[Base129ScenarioSpec] = []
    for item in scenario_specs[: max(0, int(cap))]:
        if isinstance(item, Base129ScenarioSpec):
            specs.append(item)
            continue
        if not isinstance(item, Mapping):
            continue
        specs.append(
            Base129ScenarioSpec(
                public_ordinal=_int(item.get("public_ordinal")),
                scenario_name=str(item.get("scenario_name") or ""),
            )
        )
    return tuple(spec for spec in specs if spec.scenario_name)


def _compile_config() -> CFRCCompileConfig:
    return CFRCCompileConfig(
        mode="strict_core",
        authority_lattice=AuthorityLattice.from_ordered_labels(("ell0", "ell1")),
        strict_allowed_authorities=("ell0", "ell1"),
        allowed_authorities=("ell0", "ell1"),
        suffix_tool_call_budget=12,
        # Every open obligation must be discharged by a named action whose
        # replica receipt succeeded. Without this the coverage claims the
        # grounded bridge emits would be recorded but never checked.
        require_action_receipt_witness=True,
    )


def _milestone_complete(receipt: Mapping[str, Any]) -> bool:
    evaluation = receipt.get("evaluation")
    if not isinstance(evaluation, Mapping):
        return False
    return _float(evaluation.get("milestone_similarity")) >= 1.0


def _official_score_summary(
    receipts: Sequence[Mapping[str, Any]],
    *,
    strict_complete_count: int,
    cheap_avg_similarity_reference: float | None,
    strong_avg_similarity_reference: float | None,
) -> dict[str, Any]:
    similarity_values = _evaluation_values(receipts, "similarity")
    milestone_values = _evaluation_values(receipts, "milestone_similarity")
    minefield_values = _evaluation_values(receipts, "minefield_similarity")
    avg_similarity = _mean_or_none(similarity_values)
    cheap_gap = (
        round(
            strong_avg_similarity_reference - cheap_avg_similarity_reference,
            6,
        )
        if cheap_avg_similarity_reference is not None
        and strong_avg_similarity_reference is not None
        else None
    )
    recovered_gap = (
        round(max(0.0, avg_similarity - cheap_avg_similarity_reference), 6)
        if avg_similarity is not None and cheap_avg_similarity_reference is not None
        else None
    )
    recovered_gap_fraction = (
        round(recovered_gap / cheap_gap, 6)
        if recovered_gap is not None and cheap_gap is not None and cheap_gap > 0
        else None
    )
    return {
        "official_primary_metric": "average_similarity_score",
        "official_avg_similarity_score": _round_optional(avg_similarity),
        "official_avg_similarity_percent": _percent_optional(avg_similarity),
        "official_avg_milestone_similarity_score": _round_optional(
            _mean_or_none(milestone_values)
        ),
        "official_avg_milestone_similarity_percent": _percent_optional(
            _mean_or_none(milestone_values)
        ),
        "official_avg_minefield_similarity_score": _round_optional(
            _mean_or_none(minefield_values)
        ),
        "official_avg_minefield_similarity_percent": _percent_optional(
            _mean_or_none(minefield_values)
        ),
        "official_similarity_scored_episode_count": len(similarity_values),
        "strict_complete_count": int(strict_complete_count),
        "strict_complete_rate": (
            round(int(strict_complete_count) / len(receipts), 6) if receipts else None
        ),
        "strict_complete_is_secondary_metric": True,
        "pinned_cheap_avg_similarity_bound_score": cheap_avg_similarity_reference,
        "pinned_strong_avg_similarity_bound_score": strong_avg_similarity_reference,
        "cheap_to_strong_similarity_gap_score": cheap_gap,
        "recovered_cheap_to_strong_similarity_gap_score": recovered_gap,
        "recovered_cheap_to_strong_similarity_gap_fraction": recovered_gap_fraction,
        "method_delta_vs_cheap_avg_similarity_bound_score": (
            round(avg_similarity - cheap_avg_similarity_reference, 6)
            if avg_similarity is not None
            and cheap_avg_similarity_reference is not None
            else None
        ),
        "method_delta_vs_strong_avg_similarity_bound_score": (
            round(avg_similarity - strong_avg_similarity_reference, 6)
            if avg_similarity is not None
            and strong_avg_similarity_reference is not None
            else None
        ),
    }


def _operating_point_reference(
    *,
    cheap_model: str,
    strong_model: str,
    user_model: str,
) -> dict[str, Any] | None:
    reference = OPERATING_POINT_REFERENCES.get(
        (str(cheap_model), str(strong_model), str(user_model))
    )
    return dict(reference) if reference is not None else None


def _evaluation_values(
    receipts: Sequence[Mapping[str, Any]],
    key: str,
) -> list[float]:
    values: list[float] = []
    for receipt in receipts:
        if receipt.get("official_evaluator_used") is not True:
            continue
        evaluation = receipt.get("evaluation")
        if not isinstance(evaluation, Mapping):
            continue
        value = _optional_float(evaluation.get(key))
        if value is not None:
            values.append(float(value))
    return values


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(float(value) for value in values) / len(values)


def _round_optional(value: float | None) -> float | None:
    return round(float(value), 6) if value is not None else None


def _percent_optional(value: float | None) -> float | None:
    return round(float(value) * 100.0, 6) if value is not None else None


def _minefield_hit(receipt: Mapping[str, Any]) -> bool:
    evaluation = receipt.get("evaluation")
    if not isinstance(evaluation, Mapping):
        return False
    return _float(evaluation.get("minefield_similarity")) > 0.0 and _int(
        receipt.get("strong_call_count")
    ) > 0


def _any_minefield_hit(receipt: Mapping[str, Any]) -> bool:
    evaluation = receipt.get("evaluation")
    if not isinstance(evaluation, Mapping):
        return False
    return _float(evaluation.get("minefield_similarity")) > 0.0


def _budget_cell(receipt: Mapping[str, Any]) -> str:
    frontier = receipt.get("online_frontier_generated") is True
    served = _int(receipt.get("strong_call_count")) > 0
    suppressed = receipt.get("frontier_suppressed_by_strong_cap") is True
    complete = _milestone_complete(receipt)
    if frontier and served:
        return "served_complete" if complete else "served_incomplete"
    if frontier and suppressed:
        return "suppressed_complete" if complete else "suppressed_incomplete"
    if frontier:
        return "unserved_complete" if complete else "unserved_incomplete"
    return "no_frontier_complete" if complete else "no_frontier_incomplete"


def _suppression_counts(
    receipts: Sequence[Mapping[str, Any]],
    *,
    max_strong_residual_calls: int,
) -> dict[str, int]:
    counts = {
        "global_cap_exhausted": 0,
        "per_episode_limit": 0,
        "total": 0,
    }
    strong_calls_seen = 0
    for receipt in receipts:
        suppressed = receipt.get("frontier_suppressed_by_strong_cap") is True
        if suppressed:
            counts["total"] += 1
            if strong_calls_seen >= int(max_strong_residual_calls):
                counts["global_cap_exhausted"] += 1
            else:
                counts["per_episode_limit"] += 1
        strong_calls_seen += _int(receipt.get("strong_call_count"))
    return counts


def _public_rows(
    receipts: Sequence[Mapping[str, Any]],
    *,
    max_strong_residual_calls: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    strong_calls_seen = 0
    for receipt in receipts:
        suppression_reason = _suppression_reason(
            receipt,
            strong_calls_seen=strong_calls_seen,
            max_strong_residual_calls=max_strong_residual_calls,
        )
        row = {
                "public_ordinal": _int(receipt.get("public_ordinal")),
                "decision": str(receipt.get("decision") or ""),
                "online_frontier_generated": receipt.get("online_frontier_generated") is True,
                "frontier_opened": receipt.get("online_frontier_generated") is True,
                "frontier_suppressed_by_strong_cap": receipt.get(
                    "frontier_suppressed_by_strong_cap"
                )
                is True,
                "suppressed_by_cap": suppression_reason == "global_strong_cap_exhausted",
                "suppression_reason": suppression_reason,
                "strong_call_count": _int(receipt.get("strong_call_count")),
                "strong_served": _int(receipt.get("strong_call_count")) > 0,
                "checked_execution_attempted": receipt.get(
                    "checked_execution_attempted"
                )
                is True,
                "checked_execution_admitted": receipt.get(
                    "checked_execution_admitted"
                )
                is True,
                "checked_execution_replay_match": receipt.get(
                    "checked_execution_replay_match"
                )
                is True,
                "method_refused": receipt.get("method_refused") is True,
                "official_evaluator_used": receipt.get("official_evaluator_used") is True,
                "milestone_complete": _milestone_complete(receipt),
                "minefield_hit": _minefield_hit(receipt),
                "method_introduced_minefield_hit": _minefield_hit(receipt),
                "budget_cell": _budget_cell(receipt),
            }
        if receipt.get("method_variant") == "terminal_with_last_call":
            row.update(
                {
                    "terminal_with_last_call_offered_count": _int(
                        receipt.get("terminal_with_last_call_offered_count")
                    ),
                    "terminal_with_last_call_accepted_count": _int(
                        receipt.get("terminal_with_last_call_accepted_count")
                    ),
                    "terminal_with_last_call_rejected_count": _int(
                        receipt.get("terminal_with_last_call_rejected_count")
                    ),
                }
            )
        rows.append(row)
        strong_calls_seen += _int(receipt.get("strong_call_count"))
    return rows


def _suppression_reason(
    receipt: Mapping[str, Any],
    *,
    strong_calls_seen: int,
    max_strong_residual_calls: int,
) -> str | None:
    if receipt.get("frontier_suppressed_by_strong_cap") is not True:
        return None
    if strong_calls_seen >= int(max_strong_residual_calls):
        return "global_strong_cap_exhausted"
    return "per_episode_strong_call_limit"


def _residual_runtime_error_count(receipt: Mapping[str, Any]) -> int:
    private = receipt.get("private")
    if not isinstance(private, Mapping):
        return 0
    controller = private.get("controller_receipt")
    if not isinstance(controller, Mapping):
        return 0
    attempts = controller.get("cfrc_attempts")
    if not isinstance(attempts, Sequence) or isinstance(
        attempts, (str, bytes, bytearray)
    ):
        return 0
    total = 0
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            continue
        public_summary = attempt.get("public_summary")
        if not isinstance(public_summary, Mapping):
            continue
        execution = public_summary.get("execution")
        if not isinstance(execution, Mapping):
            continue
        total += _int(execution.get("runtime_error_count"))
    return total


def _aggregate_runtime_ledger(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ledger = RuntimeLedger()
    for receipt in receipts:
        controller = _controller_receipt(receipt)
        runtime_ledger = controller.get("runtime_ledger")
        if isinstance(runtime_ledger, Mapping):
            ledger.merge_public_summary(runtime_ledger)
    return ledger.public_summary()


def _method_wall_clock_seconds(receipts: Sequence[Mapping[str, Any]]) -> list[float]:
    values: list[float] = []
    for receipt in receipts:
        value = _controller_receipt(receipt).get("method_wall_clock_seconds")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(max(0.0, float(value)))
    return values


def _method_runtime_breakdown(
    runtime_ledger: Mapping[str, Any],
    *,
    per_row_wall_seconds: Sequence[float],
    cheap_model: str,
    strong_model: str,
) -> dict[str, Any]:
    per_role = _mapping(runtime_ledger.get("per_role"))
    cheap = _role_breakdown(per_role.get("cheap"))
    strong = _role_breakdown(per_role.get("strong"))
    user_simulator = _role_breakdown(per_role.get("user_simulator"))
    evaluator = _role_breakdown(per_role.get("evaluator"))
    agent_side = _sum_role_breakdowns(cheap, strong)
    totals = _role_breakdown(runtime_ledger.get("totals"))
    if not totals["observed"]:
        totals = _sum_role_breakdowns(cheap, strong, user_simulator, evaluator)
    wall_sum = (
        round(sum(max(0.0, float(value)) for value in per_row_wall_seconds), 6)
        if per_row_wall_seconds
        else None
    )
    return {
        "schema": "papc.toolsandbox.method_runtime_breakdown.v1",
        "cheap": cheap,
        "strong": strong,
        "user_simulator": user_simulator,
        "evaluator": evaluator,
        "agent_side": agent_side,
        "full_runtime": totals,
        "model_provider_latency": {
            "cheap_seconds": cheap["latency_seconds"],
            "strong_seconds": strong["latency_seconds"],
            "agent_side_seconds": agent_side["latency_seconds"],
            "user_simulator_seconds": user_simulator["latency_seconds"],
            "full_runtime_seconds": totals["latency_seconds"],
        },
        "method_wall_clock": {
            "row_count": len(per_row_wall_seconds),
            "per_row_wall_seconds_sum": wall_sum,
            "total_run_latency_seconds": wall_sum,
            "total_run_latency_source": (
                "per_row_wall_seconds_sum" if wall_sum is not None else "unavailable"
            ),
        },
        "agent_cache_neutral_cost": _agent_cache_neutral_cost(
            runtime_ledger,
            cheap_model=cheap_model,
            strong_model=strong_model,
        ),
        "claim_boundary": [
            "Agent-side model accounting is cheap plus residual strong only.",
            "Full runtime accounting additionally includes user simulator when observed.",
            "Model-provider latency is not the same as elapsed method wall-clock.",
        ],
    }


def _agent_cache_neutral_cost(
    runtime_ledger: Mapping[str, Any],
    *,
    cheap_model: str,
    strong_model: str,
) -> dict[str, Any]:
    per_role = _mapping(runtime_ledger.get("per_role"))
    cheap = _cache_neutral_role_cost(
        _mapping(per_role.get("cheap")),
        model=cheap_model,
    )
    strong = _cache_neutral_role_cost(
        _mapping(per_role.get("strong")),
        model=strong_model,
    )
    totals = [
        row.get("total_cost_usd")
        for row in (cheap, strong)
        if isinstance(row, Mapping)
    ]
    return {
        "basis": "all_input_tokens_at_ordinary_input_rate",
        "cheap": cheap,
        "residual_strong": strong,
        "agent_side_total_cost_usd": (
            round(sum(float(value) for value in totals), 6)
            if len(totals) == 2 and all(value is not None for value in totals)
            else None
        ),
    }


def _cache_neutral_role_cost(
    role: Mapping[str, Any],
    *,
    model: str,
) -> dict[str, Any] | None:
    prices = public_openai_price_for_model(model)
    pricing_source, pricing_snapshot_date = public_pricing_metadata_for_model(
        model
    )
    input_tokens = _optional_int(role.get("input_tokens"))
    output_tokens = _optional_int(role.get("output_tokens"))
    if prices is None or input_tokens is None or output_tokens is None:
        return None
    input_price = float(prices["input_price_per_1m"])
    output_price = float(prices["output_price_per_1m"])
    input_cost = input_tokens * input_price / 1_000_000
    output_cost = output_tokens * output_price / 1_000_000
    return {
        "model": model,
        "basis": "all_input_tokens_at_ordinary_input_rate",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_price_per_1m": input_price,
        "output_price_per_1m": output_price,
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(input_cost + output_cost, 6),
        "pricing_source": pricing_source,
        "pricing_snapshot_date": pricing_snapshot_date,
    }


def _runtime_accounting_available(runtime_ledger: Mapping[str, Any]) -> bool:
    totals = _mapping(runtime_ledger.get("totals"))
    return _int(totals.get("calls")) > 0


def _runtime_provider_calls(runtime_ledger: Mapping[str, Any]) -> int:
    totals = _mapping(runtime_ledger.get("totals"))
    return _int(totals.get("calls"))


def _controller_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    private = receipt.get("private")
    if not isinstance(private, Mapping):
        return {}
    controller = private.get("controller_receipt")
    return dict(controller) if isinstance(controller, Mapping) else {}


def _role_breakdown(value: Any) -> dict[str, Any]:
    role = _mapping(value)
    if _int(role.get("calls")) <= 0:
        return _empty_role_breakdown()
    return {
        "observed": True,
        "calls": _int(role.get("calls")),
        "input_tokens": _optional_int(role.get("input_tokens")),
        "output_tokens": _optional_int(role.get("output_tokens")),
        "reasoning_tokens": _optional_int(role.get("reasoning_tokens")),
        "total_tokens": _optional_int(role.get("total_tokens")),
        "latency_seconds": _optional_float(role.get("latency_seconds")),
        "cost_usd": _optional_float(role.get("cost_usd")),
        "token_usage_available": role.get("token_usage_available") is True,
        "latency_available": role.get("latency_available") is True,
    }


def _sum_role_breakdowns(*roles: Mapping[str, Any]) -> dict[str, Any]:
    observed = [role for role in roles if role.get("observed") is True]
    if not observed:
        return _empty_role_breakdown()
    latency_values = [
        role.get("latency_seconds")
        for role in observed
        if isinstance(role.get("latency_seconds"), (int, float))
        and not isinstance(role.get("latency_seconds"), bool)
    ]
    cost_values = [
        role.get("cost_usd")
        for role in observed
        if isinstance(role.get("cost_usd"), (int, float))
        and not isinstance(role.get("cost_usd"), bool)
    ]
    return {
        "observed": True,
        "calls": sum(_int(role.get("calls")) for role in observed),
        "input_tokens": sum(_int(role.get("input_tokens")) for role in observed),
        "output_tokens": sum(_int(role.get("output_tokens")) for role in observed),
        "reasoning_tokens": sum(_int(role.get("reasoning_tokens")) for role in observed),
        "total_tokens": sum(_int(role.get("total_tokens")) for role in observed),
        "latency_seconds": round(sum(float(value) for value in latency_values), 6)
        if latency_values
        else None,
        "cost_usd": round(sum(float(value) for value in cost_values), 6)
        if len(cost_values) == len(observed)
        else None,
        "token_usage_available": all(
            role.get("token_usage_available") is True for role in observed
        ),
        "latency_available": all(role.get("latency_available") is True for role in observed),
    }


def _empty_role_breakdown() -> dict[str, Any]:
    return {
        "observed": False,
        "calls": 0,
        "input_tokens": None,
        "output_tokens": None,
        "reasoning_tokens": None,
        "total_tokens": None,
        "latency_seconds": None,
        "cost_usd": None,
        "token_usage_available": False,
        "latency_available": False,
    }


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


__all__ = [
    "APPROVAL_SCOPE",
    "BLOCKED_DECISION",
    "CAPABILITY_CAP129_APPROVAL_BASIS",
    "CAPABILITY_CAP129_APPROVAL_REVISION",
    "CAPABILITY_STRONG_RESIDUAL_CALL_CEILING",
    "EXECUTE_READY_DECISION",
    "NOAPI_READY_DECISION",
    "PRIVATE_SCHEMA",
    "SCHEMA",
    "Base129ScenarioSpec",
    "build_base129_true_online_run",
    "scenario_specs_from_lower_receipts",
]
