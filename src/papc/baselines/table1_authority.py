"""Provider-zero authority construction for the frozen PAPC Table 1.

The authority manifest binds datasets, scorers, and native source bundles.  It
does not load credentials, start benchmark environments, or authorize a run.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from papc.baselines.agent_diff_router_driver import (
    AGENT_DIFF_SCORER_NAME,
    FROZEN_AGENT_DIFF_REPO_REVISION,
    load_agent_diff_official_registry,
)
from papc.baselines.state_bench_native_runtime import (
    FROZEN_STATE_BENCH_RUNTIME_REVISION,
    STATE_BENCH_OFFICIAL_SCORER,
)
from papc.baselines.state_bench_router_driver import (
    load_state_bench_official_test_registry,
)
from papc.baselines.table1_results import (
    AGENT_DIFF,
    STATE_BENCH,
    SURFACE_DENOMINATORS,
    SURFACE_ORDER,
    TAU2_AIRLINE,
    TAU2_RETAIL,
    TOOL_SANDBOX,
    SurfaceAuthority,
    Table1AuthorityManifest,
)
from papc.baselines.tau2_native_runtime import (
    FROZEN_TAU2_RUNTIME_REVISION,
    TAU2_OFFICIAL_EVALUATION,
)
from papc.baselines.tau2_router_driver import (
    load_tau2_official_test_registry,
)

SPLIT_GATE_SCHEMA = "papc.external_router_split_gate_noapi.v2"
TRIAL_POLICY_SCHEMA = "papc.table1.single_replicate_trial_policy.v1"
FROZEN_TOOLSANDBOX_REVISION = "165848b9a78cead7ca7fe7c89c688b58e6501219"
READY_DECISION = "ready_authority_manifest_only"
BLOCKED_DECISION = "blocked_authority_manifest"
EXPECTED_SPLIT_COUNTS = {
    STATE_BENCH: (300, 240, 60, 150),
    TAU2_RETAIL: (74, 59, 15, 40),
    TAU2_AIRLINE: (30, 24, 6, 20),
    AGENT_DIFF: (179, 141, 38, 45),
}


@dataclass(frozen=True)
class AuthorityGateInputs:
    repo_root: Path
    split_gate_json: Path
    state_repo: Path
    tau2_data_repo: Path
    tau2_runtime_repo: Path
    agent_diff_repo: Path
    agent_diff_dataset: Path
    toolsandbox_checkout: Path
    toolsandbox_lower_receipt_root: Path


@dataclass(frozen=True)
class SurfaceSourceAuthority:
    surface: str
    test_registry_sha256: str
    official_scorer: str
    official_metric: str
    scorer_sha256: str
    runtime_name: str
    runtime_sha256: str

    def __post_init__(self) -> None:
        if self.surface not in SURFACE_ORDER:
            raise ValueError("table1_source_authority_surface_invalid")
        for field_name in (
            "test_registry_sha256",
            "scorer_sha256",
            "runtime_sha256",
        ):
            _required_sha256(getattr(self, field_name), field_name)
        for field_name in (
            "official_scorer",
            "official_metric",
            "runtime_name",
        ):
            _required_text(getattr(self, field_name), field_name)


@dataclass(frozen=True)
class AuthorityGateResult:
    manifest: Table1AuthorityManifest | None
    blockers: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return self.manifest is not None and not self.blockers

    def public_summary(self) -> dict[str, Any]:
        return {
            "decision": READY_DECISION if self.ready else BLOCKED_DECISION,
            "provider_calls": 0,
            "api_spend_performed": False,
            "secret_reads": 0,
            "authority_manifest_sha256": (
                self.manifest.sha256 if self.manifest is not None else None
            ),
            "surface_count": (
                len(self.manifest.authorities)
                if self.manifest is not None
                else 0
            ),
            "blockers": list(self.blockers),
            "authorizations": {
                "authorizes_paid_run": False,
                "authorizes_test_execution": False,
                "authorizes_result_claim": False,
            },
        }


def build_table1_authority_gate(
    inputs: AuthorityGateInputs,
) -> AuthorityGateResult:
    """Construct the authority manifest or return exact provider-zero blockers."""

    if not isinstance(inputs, AuthorityGateInputs):
        raise TypeError("table1_authority_gate_inputs_invalid")
    try:
        split_entries = _validated_split_entries(inputs.split_gate_json)
    except Exception as exc:  # noqa: BLE001 - an authority gate must fail closed.
        return AuthorityGateResult(
            manifest=None,
            blockers=(_blocker("split_gate", exc),),
        )

    collectors = (
        (
            STATE_BENCH,
            lambda: _state_authority(
                inputs=inputs,
                split_entry=split_entries[STATE_BENCH],
            ),
        ),
        (
            TAU2_RETAIL,
            lambda: _tau2_authority(
                inputs=inputs,
                split_entry=split_entries[TAU2_RETAIL],
                domain="retail",
            ),
        ),
        (
            TAU2_AIRLINE,
            lambda: _tau2_authority(
                inputs=inputs,
                split_entry=split_entries[TAU2_AIRLINE],
                domain="airline",
            ),
        ),
        (
            TOOL_SANDBOX,
            lambda: _toolsandbox_authority(
                inputs=inputs,
                split_entry=split_entries[TOOL_SANDBOX],
            ),
        ),
        (
            AGENT_DIFF,
            lambda: _agent_diff_authority(
                inputs=inputs,
                split_entry=split_entries[AGENT_DIFF],
            ),
        ),
    )
    bindings: dict[str, SurfaceSourceAuthority] = {}
    blockers: list[str] = []
    for surface, collect in collectors:
        try:
            bindings[surface] = collect()
        except Exception as exc:  # noqa: BLE001 - collect every surface blocker.
            blockers.append(_blocker(surface, exc))
    if blockers:
        return AuthorityGateResult(
            manifest=None,
            blockers=tuple(blockers),
        )
    try:
        manifest = _assemble_manifest(
            split_entries=split_entries,
            bindings=bindings,
        )
    except Exception as exc:  # noqa: BLE001 - never emit a partial manifest.
        return AuthorityGateResult(
            manifest=None,
            blockers=(_blocker("manifest", exc),),
        )
    return AuthorityGateResult(manifest=manifest, blockers=())


def trial_policy_record(
    *,
    surface: str,
    test_registry_sha256: str,
) -> dict[str, Any]:
    """Return the deterministic, method-agnostic one-replicate policy."""

    if surface not in SURFACE_ORDER:
        raise ValueError("table1_trial_policy_surface_invalid")
    registry = _required_sha256(
        test_registry_sha256,
        "trial_policy_test_registry_sha256",
    )
    return {
        "schema": TRIAL_POLICY_SCHEMA,
        "surface": surface,
        "test_registry_sha256": registry,
        "replicates_per_task": 1,
        "trial_slot": "table1_replicate_0",
        "same_trial_slot_for_every_row": True,
        "task_order": "frozen_registry_order",
        "result_conditioned_retry_or_replacement": False,
        "infra_or_missing_trial_blocks_cell": True,
        "denominator_reduction_allowed": False,
        "state_bench_official_five_run_protocol_claimed": False,
    }


def trial_policy_sha256(
    *,
    surface: str,
    test_registry_sha256: str,
) -> str:
    return _digest(
        trial_policy_record(
            surface=surface,
            test_registry_sha256=test_registry_sha256,
        )
    )


def build_toolsandbox_test_registry(
    *,
    receipt_root: Path,
    checkout: Path,
) -> tuple[str, str]:
    """Build an outcome-free base129 source binding and test registry."""

    root = Path(receipt_root).resolve()
    source_checkout = Path(checkout).resolve()
    revision = _git_revision(
        source_checkout,
        expected=FROZEN_TOOLSANDBOX_REVISION,
        label="toolsandbox",
    )
    scenario_root = source_checkout / "tool_sandbox" / "scenarios"
    scenario_paths = _required_glob(
        scenario_root,
        "*.py",
        "toolsandbox_scenario_sources",
    )
    scenario_literals: set[str] = set()
    for path in scenario_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        scenario_literals.update(
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
        )

    receipt_paths = tuple(sorted(root.glob("case_*.json")))
    if len(receipt_paths) != SURFACE_DENOMINATORS[TOOL_SANDBOX]:
        raise ValueError("toolsandbox_base129_receipt_count_mismatch")
    selection: list[dict[str, Any]] = []
    for path in receipt_paths:
        payload = _read_json(path)
        if not isinstance(payload, Mapping):
            raise TypeError("toolsandbox_base129_receipt_not_mapping")
        if (
            payload.get("schema")
            != "papc.toolsandbox.base129.official_private_receipt.v1"
            or payload.get("surface") != TOOL_SANDBOX
            or payload.get("arm") != "lower"
        ):
            raise ValueError("toolsandbox_base129_receipt_identity_mismatch")
        ordinal = payload.get("public_ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int):
            raise TypeError("toolsandbox_base129_public_ordinal_invalid")
        private = payload.get("private")
        if not isinstance(private, Mapping):
            raise TypeError("toolsandbox_base129_private_record_invalid")
        scenario_name = _required_text(
            private.get("scenario_name"),
            "toolsandbox_scenario_name",
        )
        source_family = _required_text(
            payload.get("source_family"),
            "toolsandbox_source_family",
        )
        if scenario_name not in scenario_literals:
            raise ValueError(
                "toolsandbox_base129_scenario_not_in_source:"
                f"{scenario_name}"
            )
        selection.append(
            {
                "public_ordinal": ordinal,
                "scenario_name": scenario_name,
                "source_family": source_family,
            }
        )
    selection.sort(key=lambda item: item["public_ordinal"])
    ordinals = [item["public_ordinal"] for item in selection]
    if ordinals != list(range(SURFACE_DENOMINATORS[TOOL_SANDBOX])):
        raise ValueError("toolsandbox_base129_public_ordinals_not_exact")
    scenario_names = [item["scenario_name"] for item in selection]
    if len(scenario_names) != len(set(scenario_names)):
        raise ValueError("toolsandbox_base129_scenario_duplicate")

    scenario_bundle_sha256 = _source_bundle_sha256(
        root=scenario_root,
        paths=scenario_paths,
    )
    source_binding_sha256 = _digest(
        {
            "schema": "papc.table1.toolsandbox_base129_source.v1",
            "surface": TOOL_SANDBOX,
            "toolsandbox_revision": revision,
            "scenario_source_bundle_sha256": scenario_bundle_sha256,
            "selection": selection,
        }
    )
    example_sha256s = [
        _digest(
            {
                "surface": TOOL_SANDBOX,
                "stratum": item["source_family"],
                "source_identity": (
                    f"{item['public_ordinal']}:{item['scenario_name']}"
                ),
                "payload_sha256": _digest(item),
            }
        )
        for item in selection
    ]
    test_registry_sha256 = _digest(
        {
            "role": "official_test",
            "surface": TOOL_SANDBOX,
            "source_binding_sha256": source_binding_sha256,
            "example_sha256s": example_sha256s,
        }
    )
    return source_binding_sha256, test_registry_sha256


def _validated_split_entries(path: Path) -> dict[str, Mapping[str, Any]]:
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        raise TypeError("external_router_split_gate_not_mapping")
    if payload.get("schema") != SPLIT_GATE_SCHEMA:
        raise ValueError("external_router_split_gate_schema_mismatch")
    if payload.get("decision") != "ready_for_training_collection_plan":
        raise ValueError("external_router_split_gate_not_ready")
    if (
        payload.get("provider_calls") != 0
        or payload.get("api_spend_performed") is not False
        or payload.get("secret_reads") != 0
        or payload.get("execute") is not False
        or payload.get("public_safe") is not True
    ):
        raise ValueError("external_router_split_gate_not_provider_zero")
    authorizations = payload.get("authorizations")
    if not isinstance(authorizations, Mapping) or any(
        authorizations.get(key) is not False
        for key in (
            "authorizes_paid_run",
            "authorizes_baseline_result_claim",
            "authorizes_test_execution",
        )
    ):
        raise ValueError("external_router_split_gate_authorizes_execution")
    checks = payload.get("checks")
    if not isinstance(checks, Mapping) or not checks or any(
        value is not True for value in checks.values()
    ):
        raise ValueError("external_router_split_gate_checks_failed")
    raw_entries = payload.get("surface_splits")
    if not isinstance(raw_entries, Mapping):
        raise TypeError("external_router_split_gate_surfaces_invalid")
    if set(raw_entries) != set(SURFACE_ORDER):
        raise ValueError("external_router_split_gate_surface_set_mismatch")

    entries: dict[str, Mapping[str, Any]] = {}
    for surface in SURFACE_ORDER:
        raw = raw_entries.get(surface)
        if not isinstance(raw, Mapping):
            raise TypeError(f"{surface}_split_entry_invalid")
        if raw.get("official_test_count") != SURFACE_DENOMINATORS[surface]:
            raise ValueError(f"{surface}_split_test_count_mismatch")
        if surface == TOOL_SANDBOX:
            if (
                raw.get("official_train_count") != 0
                or raw.get("eligible_for_in_domain_router") is not False
                or any(
                    raw.get(key) is not None
                    for key in (
                        "fit_registry_sha256",
                        "dev_registry_sha256",
                        "test_registry_sha256",
                    )
                )
            ):
                raise ValueError("toolsandbox_split_data_boundary_invalid")
        else:
            observed_counts = tuple(
                raw.get(key)
                for key in (
                    "official_train_count",
                    "router_fit_count",
                    "router_dev_count",
                    "official_test_count",
                )
            )
            if observed_counts != EXPECTED_SPLIT_COUNTS[surface]:
                raise ValueError(f"{surface}_split_counts_mismatch")
            for key in (
                "source_binding_sha256",
                "fit_registry_sha256",
                "dev_registry_sha256",
                "test_registry_sha256",
            ):
                _required_sha256(raw.get(key), f"{surface}_{key}")
            if len(
                {
                    raw["fit_registry_sha256"],
                    raw["dev_registry_sha256"],
                    raw["test_registry_sha256"],
                }
            ) != 3:
                raise ValueError(f"{surface}_split_registry_reused")
        entries[surface] = raw
    return entries


def _state_authority(
    *,
    inputs: AuthorityGateInputs,
    split_entry: Mapping[str, Any],
) -> SurfaceSourceAuthority:
    registry = load_state_bench_official_test_registry(
        inputs.state_repo,
        split_gate_json=inputs.split_gate_json,
    )
    _match_registry(
        surface=STATE_BENCH,
        split_entry=split_entry,
        source_binding_sha256=registry.source_binding_sha256,
        test_registry_sha256=registry.test_registry_sha256,
    )
    runtime_repo = Path(inputs.state_repo).resolve()
    revision = _git_revision(
        runtime_repo,
        expected=FROZEN_STATE_BENCH_RUNTIME_REVISION,
        label=STATE_BENCH,
    )
    runtime_paths = (
        *_required_glob(
            runtime_repo / "state_bench",
            "**/*.py",
            "state_bench_runtime_sources",
        ),
        *_optional_paths(
            runtime_repo,
            (
                "pyproject.toml",
                "uv.lock",
                "agents/papc_openai_compatible.py",
                "clients/papc_openai_compatible.py",
            ),
        ),
        *_required_glob(
            runtime_repo / "state_bench" / "configs",
            "**/*.yaml",
            "state_bench_runtime_configs",
        ),
        *_required_glob(
            runtime_repo / "state_bench" / "configs",
            "**/*.json",
            "state_bench_protocol_configs",
        ),
    )
    runtime_sha256 = _runtime_digest(
        surface=STATE_BENCH,
        revision=revision,
        source_binding_sha256=registry.source_binding_sha256,
        source_tree_sha256=_source_bundle_sha256(
            root=runtime_repo,
            paths=runtime_paths,
        ),
    )
    scorer_paths = (
        runtime_repo / "state_bench" / "scoring.py",
        runtime_repo / "state_bench" / "scripts" / "score.py",
        runtime_repo / "state_bench" / "protocol.py",
        runtime_repo
        / "state_bench"
        / "configs"
        / "eval_protocols"
        / "gpt54.json",
    )
    return SurfaceSourceAuthority(
        surface=STATE_BENCH,
        test_registry_sha256=registry.test_registry_sha256,
        official_scorer=STATE_BENCH_OFFICIAL_SCORER,
        official_metric="task_completion_pass",
        scorer_sha256=_source_bundle_sha256(
            root=runtime_repo,
            paths=scorer_paths,
        ),
        runtime_name="state_bench.native_single_task_run",
        runtime_sha256=runtime_sha256,
    )


def _tau2_authority(
    *,
    inputs: AuthorityGateInputs,
    split_entry: Mapping[str, Any],
    domain: str,
) -> SurfaceSourceAuthority:
    data_registry = load_tau2_official_test_registry(
        inputs.tau2_data_repo,
        domain=domain,
    )
    runtime_registry = load_tau2_official_test_registry(
        inputs.tau2_runtime_repo,
        domain=domain,
    )
    surface = data_registry.surface
    _match_registry(
        surface=surface,
        split_entry=split_entry,
        source_binding_sha256=data_registry.source_binding_sha256,
        test_registry_sha256=data_registry.test_registry_sha256,
    )
    if (
        runtime_registry.source_binding_sha256
        != data_registry.source_binding_sha256
        or runtime_registry.test_registry_sha256
        != data_registry.test_registry_sha256
    ):
        raise ValueError("tau2_runtime_and_data_registry_mismatch")
    runtime_repo = Path(inputs.tau2_runtime_repo).resolve()
    revision = _git_revision(
        runtime_repo,
        expected=FROZEN_TAU2_RUNTIME_REVISION,
        label=surface,
    )
    runtime_paths = (
        *_required_glob(
            runtime_repo / "src" / "tau2",
            "**/*.py",
            f"{surface}_runtime_sources",
        ),
        *_optional_paths(runtime_repo, ("pyproject.toml", "uv.lock")),
    )
    runtime_sha256 = _runtime_digest(
        surface=surface,
        revision=revision,
        source_binding_sha256=data_registry.source_binding_sha256,
        source_tree_sha256=_source_bundle_sha256(
            root=runtime_repo,
            paths=runtime_paths,
        ),
    )
    evaluator_root = runtime_repo / "src" / "tau2" / "evaluator"
    scorer_paths = (
        *_required_glob(
            evaluator_root,
            "**/*.py",
            "tau2_evaluator_sources",
        ),
        runtime_repo / "src" / "tau2" / "data_model" / "simulation.py",
        runtime_repo / "src" / "tau2" / "runner" / "batch.py",
    )
    return SurfaceSourceAuthority(
        surface=surface,
        test_registry_sha256=data_registry.test_registry_sha256,
        official_scorer=(
            "tau2.reward_info.reward:"
            f"{TAU2_OFFICIAL_EVALUATION}"
        ),
        official_metric="official_task_reward",
        scorer_sha256=_source_bundle_sha256(
            root=runtime_repo,
            paths=scorer_paths,
        ),
        runtime_name=f"tau2.native_text_run:{domain}",
        runtime_sha256=runtime_sha256,
    )


def _agent_diff_authority(
    *,
    inputs: AuthorityGateInputs,
    split_entry: Mapping[str, Any],
) -> SurfaceSourceAuthority:
    registry = load_agent_diff_official_registry(
        inputs.agent_diff_dataset,
        split_gate_json=inputs.split_gate_json,
    )
    split = registry.split
    _match_registry(
        surface=AGENT_DIFF,
        split_entry=split_entry,
        source_binding_sha256=split.source_binding_sha256,
        test_registry_sha256=split.test_registry_sha256,
    )
    for key in ("fit_registry_sha256", "dev_registry_sha256"):
        if split_entry.get(key) != getattr(split, key):
            raise ValueError(f"agent_diff_{key}_mismatch")
    runtime_repo = Path(inputs.agent_diff_repo).resolve()
    dataset = Path(inputs.agent_diff_dataset).resolve()
    if not dataset.is_relative_to(runtime_repo):
        raise ValueError("agent_diff_dataset_outside_runtime_repo")
    revision = _git_revision(
        runtime_repo,
        expected=FROZEN_AGENT_DIFF_REPO_REVISION,
        label=AGENT_DIFF,
    )
    runtime_paths = (
        *_required_glob(
            runtime_repo / "backend" / "src",
            "**/*.py",
            "agent_diff_backend_sources",
        ),
        *_required_glob(
            runtime_repo / "sdk" / "agent-diff-python" / "agent_diff",
            "**/*.py",
            "agent_diff_sdk_sources",
        ),
        *_optional_paths(
            runtime_repo,
            (
                "backend/pyproject.toml",
                "backend/uv.lock",
                "sdk/agent-diff-python/pyproject.toml",
                "sdk/agent-diff-python/uv.lock",
            ),
        ),
    )
    runtime_sha256 = _runtime_digest(
        surface=AGENT_DIFF,
        revision=revision,
        source_binding_sha256=split.source_binding_sha256,
        source_tree_sha256=_source_bundle_sha256(
            root=runtime_repo,
            paths=runtime_paths,
        ),
    )
    scorer_paths = (
        *_required_glob(
            runtime_repo
            / "backend"
            / "src"
            / "platform"
            / "evaluationEngine",
            "**/*.py",
            "agent_diff_evaluator_sources",
        ),
        runtime_repo / "backend" / "src" / "platform" / "api" / "routes.py",
    )
    upstream_scorer_sha256 = _source_bundle_sha256(
        root=runtime_repo,
        paths=scorer_paths,
    )
    registered_correction_sha256 = _file_sha256(
        Path(inputs.repo_root)
        / "scripts"
        / "agent_diff"
        / "papc_agent_diff_scoring.py"
    )
    return SurfaceSourceAuthority(
        surface=AGENT_DIFF,
        test_registry_sha256=split.test_registry_sha256,
        official_scorer=AGENT_DIFF_SCORER_NAME,
        official_metric="passed",
        scorer_sha256=_digest(
            {
                "schema": "papc.table1.agent_diff_scorer_bundle.v1",
                "upstream_scorer_sha256": upstream_scorer_sha256,
                "registered_correction_sha256": (
                    registered_correction_sha256
                ),
            }
        ),
        runtime_name="agent_diff.react_bash_executor_proxy",
        runtime_sha256=runtime_sha256,
    )


def _toolsandbox_authority(
    *,
    inputs: AuthorityGateInputs,
    split_entry: Mapping[str, Any],
) -> SurfaceSourceAuthority:
    if (
        split_entry.get("official_train_count") != 0
        or split_entry.get("official_test_count")
        != SURFACE_DENOMINATORS[TOOL_SANDBOX]
    ):
        raise ValueError("toolsandbox_split_boundary_mismatch")
    source_binding, test_registry = build_toolsandbox_test_registry(
        receipt_root=inputs.toolsandbox_lower_receipt_root,
        checkout=inputs.toolsandbox_checkout,
    )
    checkout = Path(inputs.toolsandbox_checkout).resolve()
    revision = _git_revision(
        checkout,
        expected=FROZEN_TOOLSANDBOX_REVISION,
        label=TOOL_SANDBOX,
    )
    runtime_paths = (
        *_required_glob(
            checkout / "tool_sandbox",
            "**/*.py",
            "toolsandbox_runtime_sources",
        ),
        *_optional_paths(checkout, ("pyproject.toml", "uv.lock")),
    )
    runtime_sha256 = _runtime_digest(
        surface=TOOL_SANDBOX,
        revision=revision,
        source_binding_sha256=source_binding,
        source_tree_sha256=_source_bundle_sha256(
            root=checkout,
            paths=runtime_paths,
        ),
    )
    scorer_paths = (
        checkout / "tool_sandbox" / "common" / "evaluation.py",
        checkout / "tool_sandbox" / "common" / "scenario.py",
        *_required_glob(
            checkout / "tool_sandbox" / "scenarios",
            "**/*.py",
            "toolsandbox_scorer_scenario_sources",
        ),
    )
    return SurfaceSourceAuthority(
        surface=TOOL_SANDBOX,
        test_registry_sha256=test_registry,
        official_scorer=(
            "tool_sandbox.common.scenario.Scenario.play_and_evaluate"
        ),
        official_metric="average_similarity_score",
        scorer_sha256=_source_bundle_sha256(
            root=checkout,
            paths=scorer_paths,
        ),
        runtime_name="tool_sandbox.native_scenario_runtime",
        runtime_sha256=runtime_sha256,
    )


def _assemble_manifest(
    *,
    split_entries: Mapping[str, Mapping[str, Any]],
    bindings: Mapping[str, SurfaceSourceAuthority],
) -> Table1AuthorityManifest:
    if tuple(bindings) != SURFACE_ORDER:
        raise ValueError("table1_source_authority_surface_order_mismatch")
    authorities: list[SurfaceAuthority] = []
    for surface in SURFACE_ORDER:
        binding = bindings[surface]
        split = split_entries[surface]
        if surface != TOOL_SANDBOX and (
            split.get("test_registry_sha256")
            != binding.test_registry_sha256
        ):
            raise ValueError(f"{surface}_authority_test_registry_mismatch")
        authorities.append(
            SurfaceAuthority(
                surface=surface,
                denominator=SURFACE_DENOMINATORS[surface],
                replicates_per_task=1,
                trial_policy_sha256=trial_policy_sha256(
                    surface=surface,
                    test_registry_sha256=binding.test_registry_sha256,
                ),
                test_registry_sha256=binding.test_registry_sha256,
                official_scorer=binding.official_scorer,
                official_metric=binding.official_metric,
                scorer_sha256=binding.scorer_sha256,
                runtime_name=binding.runtime_name,
                runtime_sha256=binding.runtime_sha256,
                fit_registry_sha256=(
                    None
                    if surface == TOOL_SANDBOX
                    else _required_sha256(
                        split.get("fit_registry_sha256"),
                        f"{surface}_fit_registry_sha256",
                    )
                ),
                dev_registry_sha256=(
                    None
                    if surface == TOOL_SANDBOX
                    else _required_sha256(
                        split.get("dev_registry_sha256"),
                        f"{surface}_dev_registry_sha256",
                    )
                ),
            )
        )
    return Table1AuthorityManifest(authorities=tuple(authorities))


def _match_registry(
    *,
    surface: str,
    split_entry: Mapping[str, Any],
    source_binding_sha256: str,
    test_registry_sha256: str,
) -> None:
    if (
        split_entry.get("source_binding_sha256")
        != source_binding_sha256
    ):
        raise ValueError(f"{surface}_source_binding_mismatch")
    if split_entry.get("test_registry_sha256") != test_registry_sha256:
        raise ValueError(f"{surface}_test_registry_mismatch")


def _runtime_digest(
    *,
    surface: str,
    revision: str,
    source_binding_sha256: str,
    source_tree_sha256: str,
) -> str:
    return _digest(
        {
            "schema": "papc.table1.native_runtime_source_bundle.v1",
            "surface": surface,
            "revision": _required_text(revision, "runtime_revision"),
            "source_binding_sha256": _required_sha256(
                source_binding_sha256,
                "runtime_source_binding_sha256",
            ),
            "source_tree_sha256": _required_sha256(
                source_tree_sha256,
                "runtime_source_tree_sha256",
            ),
        }
    )


def _source_bundle_sha256(
    *,
    root: Path,
    paths: Sequence[Path],
) -> str:
    source_root = Path(root).resolve()
    resolved = tuple(sorted({Path(path).resolve() for path in paths}))
    if not resolved:
        raise ValueError("source_bundle_empty")
    manifest: list[dict[str, str]] = []
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(f"source_bundle_file_missing:{path}")
        try:
            label = path.relative_to(source_root).as_posix()
        except ValueError as exc:
            raise ValueError("source_bundle_path_outside_root") from exc
        manifest.append({"path": label, "sha256": _file_sha256(path)})
    return _digest(manifest)


def _required_glob(
    root: Path,
    pattern: str,
    label: str,
) -> tuple[Path, ...]:
    paths = tuple(sorted(path for path in Path(root).glob(pattern) if path.is_file()))
    if not paths:
        raise FileNotFoundError(f"{label}_missing")
    return paths


def _optional_paths(
    root: Path,
    relatives: Sequence[str],
) -> tuple[Path, ...]:
    return tuple(
        path
        for relative in relatives
        if (path := Path(root) / relative).is_file()
    )


def _git_revision(
    repo: Path,
    *,
    expected: str,
    label: str,
) -> str:
    root = Path(repo).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"{label}_runtime_repo_missing")
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip()
    if revision != expected:
        raise ValueError(f"{label}_runtime_revision_mismatch")
    return revision


def _blocker(scope: str, exc: BaseException) -> str:
    detail = str(exc).strip() or type(exc).__name__
    return f"{scope}:{type(exc).__name__}:{detail}"


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name}_must_be_str")
    resolved = value.strip()
    if not resolved:
        raise ValueError(f"{field_name}_empty")
    return resolved


def _required_sha256(value: Any, field_name: str) -> str:
    resolved = _required_text(value, field_name)
    if len(resolved) != 64 or any(
        char not in "0123456789abcdef" for char in resolved
    ):
        raise ValueError(f"{field_name}_invalid_sha256")
    return resolved


__all__ = [
    "AuthorityGateInputs",
    "AuthorityGateResult",
    "BLOCKED_DECISION",
    "FROZEN_TOOLSANDBOX_REVISION",
    "READY_DECISION",
    "SPLIT_GATE_SCHEMA",
    "SurfaceSourceAuthority",
    "TRIAL_POLICY_SCHEMA",
    "build_table1_authority_gate",
    "build_toolsandbox_test_registry",
    "trial_policy_record",
    "trial_policy_sha256",
]
