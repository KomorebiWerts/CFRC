"""Receipt-bound fit/dev compiler inputs for the FrugalGPT baseline."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from papc.baselines.external_router_execution import (
    SHARED_STAGE,
    ExecutionContractError,
    ExecutionJob,
    ExecutionPlan,
)
from papc.baselines.external_router_fit_dev import read_fit_dev_job_output
from papc.baselines.frugalgpt_upstream import (
    FrugalGPTDevExample,
    FrugalGPTTrainingExample,
)
from papc.baselines.router_accounting import AGENT_SIDE_ROLES, CallReceipt
from papc.baselines.router_contract import (
    CheapCompletionRouterView,
    SplitManifest,
)


@dataclass(frozen=True)
class FrugalGPTEndpointEpisode:
    example_sha256: str
    original_request: str
    generated_answer: str
    official_quality: float
    agent_side_cost_usd: float


ScoreCheapAnswer = Callable[[CheapCompletionRouterView], float]


def find_shared_endpoint_job(
    plan: ExecutionPlan,
    *,
    surface: str,
    split_role: str,
    model_role: str,
) -> ExecutionJob:
    if split_role not in {"router_fit", "router_dev"}:
        raise ValueError("frugalgpt_split_role_invalid")
    if model_role not in {"cheap", "strong"}:
        raise ValueError("frugalgpt_model_role_invalid")
    matches = tuple(
        job
        for job in plan.jobs
        if job.stage == SHARED_STAGE
        and job.kind == "shared_endpoint_episode"
        and job.method == f"shared_{model_role}_endpoint"
        and job.surface == surface
        and job.split_role == split_role
        and dict(job.details).get("model_role") == model_role
    )
    if len(matches) != 1:
        raise ExecutionContractError(
            "frugalgpt_shared_endpoint_job_not_unique"
        )
    return matches[0]


def load_bound_endpoint_episodes(
    path: Path,
    *,
    plan: ExecutionPlan,
    job: ExecutionJob,
    manifest: SplitManifest,
    model_role: str,
) -> tuple[FrugalGPTEndpointEpisode, ...]:
    _validate_job_binding(job, manifest=manifest, model_role=model_role)
    records = read_fit_dev_job_output(path)
    header = records[0]
    expected_header = {
        "plan_sha256": plan.sha256,
        "plan_protocol_semantic_sha256": plan.protocol_semantic_sha256,
        "job_id": job.job_id,
        "stage": job.stage,
        "kind": job.kind,
        "method": job.method,
        "surface": job.surface,
        "split_role": job.split_role,
        "item_count": job.item_count,
        "registry_sha256": job.registry_sha256,
        "source_binding_sha256": job.source_binding_sha256,
        "split_gate_sha256": job.split_gate_sha256,
        "official_test_outcomes_read": False,
        "partial_resume_or_cross_run_merge_allowed": False,
    }
    if any(header.get(key) != value for key, value in expected_header.items()):
        raise ExecutionContractError(
            "frugalgpt_endpoint_header_binding_mismatch"
        )
    episodes = tuple(
        _episode_from_record(
            record,
            ordinal=ordinal,
            example_sha256=example_sha256,
            job=job,
            model_role=model_role,
        )
        for ordinal, (example_sha256, record) in enumerate(
            zip(manifest.example_sha256s, records[1:], strict=True),
            start=1,
        )
    )
    if len(episodes) != len(manifest.example_sha256s):
        raise ExecutionContractError(
            "frugalgpt_endpoint_membership_count_mismatch"
        )
    return episodes


def build_frugalgpt_training_examples(
    episodes: Sequence[FrugalGPTEndpointEpisode],
    *,
    fit_manifest: SplitManifest,
) -> tuple[FrugalGPTTrainingExample, ...]:
    resolved = _ordered_episodes(
        episodes,
        manifest=fit_manifest,
        expected_role="router_fit",
    )
    examples = []
    for episode in resolved:
        if episode.official_quality not in {0.0, 1.0}:
            raise ValueError("frugalgpt_fit_official_quality_not_binary")
        examples.append(
            FrugalGPTTrainingExample(
                example_sha256=episode.example_sha256,
                original_request=episode.original_request,
                cheap_generated_answer=episode.generated_answer,
                cheap_official_quality=int(episode.official_quality),
            )
        )
    return tuple(examples)


def build_frugalgpt_dev_examples(
    cheap_episodes: Sequence[FrugalGPTEndpointEpisode],
    strong_episodes: Sequence[FrugalGPTEndpointEpisode],
    *,
    dev_manifest: SplitManifest,
    score_cheap_answer: ScoreCheapAnswer,
) -> tuple[FrugalGPTDevExample, ...]:
    if not callable(score_cheap_answer):
        raise TypeError("frugalgpt_dev_scorer_not_callable")
    cheap = _ordered_episodes(
        cheap_episodes,
        manifest=dev_manifest,
        expected_role="router_dev",
    )
    strong = _ordered_episodes(
        strong_episodes,
        manifest=dev_manifest,
        expected_role="router_dev",
    )
    examples = []
    for cheap_episode, strong_episode in zip(cheap, strong, strict=True):
        if (
            cheap_episode.example_sha256 != strong_episode.example_sha256
            or cheap_episode.original_request
            != strong_episode.original_request
        ):
            raise ExecutionContractError(
                "frugalgpt_dev_endpoint_pairing_mismatch"
            )
        predicted = score_cheap_answer(
            CheapCompletionRouterView(
                original_request=cheap_episode.original_request,
                cheap_generated_answer=cheap_episode.generated_answer,
            )
        )
        if (
            isinstance(predicted, bool)
            or not isinstance(predicted, (int, float))
            or not math.isfinite(float(predicted))
            or not 0.0 <= float(predicted) <= 1.0
        ):
            raise ValueError("frugalgpt_dev_predicted_quality_invalid")
        examples.append(
            FrugalGPTDevExample(
                example_sha256=cheap_episode.example_sha256,
                predicted_cheap_quality=float(predicted),
                cheap_official_quality=cheap_episode.official_quality,
                strong_official_quality=strong_episode.official_quality,
                cheap_episode_cost_usd=cheap_episode.agent_side_cost_usd,
                strong_episode_cost_usd=strong_episode.agent_side_cost_usd,
            )
        )
    return tuple(examples)


def _validate_job_binding(
    job: ExecutionJob,
    *,
    manifest: SplitManifest,
    model_role: str,
) -> None:
    if (
        job.stage != SHARED_STAGE
        or job.kind != "shared_endpoint_episode"
        or job.method != f"shared_{model_role}_endpoint"
        or dict(job.details).get("model_role") != model_role
        or job.split_role != manifest.role
        or job.registry_sha256 != manifest.registry_sha256
        or job.item_count != len(manifest.example_sha256s)
    ):
        raise ExecutionContractError(
            "frugalgpt_endpoint_job_binding_mismatch"
        )


def _episode_from_record(
    record: Mapping[str, Any],
    *,
    ordinal: int,
    example_sha256: str,
    job: ExecutionJob,
    model_role: str,
) -> FrugalGPTEndpointEpisode:
    if (
        record.get("ordinal") != ordinal
        or record.get("job_id") != job.job_id
        or record.get("example_sha256") != example_sha256
    ):
        raise ExecutionContractError(
            "frugalgpt_endpoint_row_membership_mismatch"
        )
    output = record.get("output")
    receipts = record.get("receipts")
    if not isinstance(output, Mapping) or not isinstance(receipts, list):
        raise ExecutionContractError("frugalgpt_endpoint_row_invalid")
    original_request = output.get("original_request")
    generated_answer = output.get("generated_answer")
    quality = output.get("official_score")
    if (
        output.get("model_role") != model_role
        or not isinstance(original_request, str)
        or not original_request.strip()
        or not isinstance(generated_answer, str)
        or not generated_answer.strip()
        or isinstance(quality, bool)
        or not isinstance(quality, (int, float))
        or not math.isfinite(float(quality))
        or not 0.0 <= float(quality) <= 1.0
    ):
        raise ExecutionContractError(
            "frugalgpt_endpoint_output_invalid"
        )
    parsed_receipts = tuple(
        _receipt_from_record(item) for item in receipts
    )
    required_role = f"{model_role}_agent"
    forbidden_roles = {
        "strong_agent" if model_role == "cheap" else "cheap_agent",
        "router",
        "verifier",
    }
    roles = {receipt.role for receipt in parsed_receipts}
    if required_role not in roles or roles & forbidden_roles:
        raise ExecutionContractError(
            "frugalgpt_endpoint_receipt_role_mismatch"
        )
    agent_cost = sum(
        float(receipt.provider_reported_cost_usd)
        for receipt in parsed_receipts
        if receipt.role in AGENT_SIDE_ROLES
    )
    return FrugalGPTEndpointEpisode(
        example_sha256=example_sha256,
        original_request=original_request,
        generated_answer=generated_answer,
        official_quality=float(quality),
        agent_side_cost_usd=agent_cost,
    )


def _receipt_from_record(record: Any) -> CallReceipt:
    if not isinstance(record, Mapping):
        raise ExecutionContractError(
            "frugalgpt_endpoint_receipt_not_mapping"
        )
    receipt = CallReceipt(**dict(record))
    if (
        receipt.provider_calls != 1
        or receipt.retry_count != 0
        or receipt.provider_reported_cost_usd is None
    ):
        raise ExecutionContractError(
            "frugalgpt_endpoint_receipt_incomplete"
        )
    return receipt


def _ordered_episodes(
    episodes: Sequence[FrugalGPTEndpointEpisode],
    *,
    manifest: SplitManifest,
    expected_role: str,
) -> tuple[FrugalGPTEndpointEpisode, ...]:
    if manifest.role != expected_role:
        raise ValueError("frugalgpt_manifest_role_invalid")
    resolved = tuple(episodes)
    if not resolved or any(
        not isinstance(item, FrugalGPTEndpointEpisode)
        for item in resolved
    ):
        raise ValueError("frugalgpt_endpoint_episodes_invalid")
    by_id = {item.example_sha256: item for item in resolved}
    if (
        len(by_id) != len(resolved)
        or set(by_id) != set(manifest.example_sha256s)
    ):
        raise ValueError(
            "frugalgpt_endpoint_registry_membership_mismatch"
        )
    return tuple(by_id[value] for value in manifest.example_sha256s)


__all__ = [
    "FrugalGPTEndpointEpisode",
    "ScoreCheapAnswer",
    "build_frugalgpt_dev_examples",
    "build_frugalgpt_training_examples",
    "find_shared_endpoint_job",
    "load_bound_endpoint_episodes",
]
