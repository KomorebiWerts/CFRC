"""Provider-neutral dispatch for the early external-router paid stages.

The module reconstructs only the frozen official-train fit/dev rows, binds
them to the final execution-plan jobs, and runs an injected native executor
through :class:`BatchCoordinator`.  It never imports CFRC/certspec code and
does not construct a provider client.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from papc.baselines.automix_upstream import (
    VERIFIER_MAX_OUTPUT_TOKENS,
    VERIFIER_PROMPT_TEMPLATE_SHA256,
    VERIFIER_PROVIDER_BATCH_SIZE,
    VERIFIER_SAMPLES,
    VERIFIER_STOP,
    VERIFIER_TEMPERATURE,
    VERIFIER_TOKENIZER_BUNDLE_SHA256,
    VerifierTokenizer,
    build_automix_verifier_request,
    parse_automix_verifier_outputs,
    serialize_automix_agent_context,
    validate_automix_verifier_tokenizer_root,
)
from papc.baselines.external_router_execution import (
    AUTOMIX_STAGE,
    SHARED_STAGE,
    BatchCoordinator,
    ExecutionContractError,
    ExecutionJob,
    ExecutionPlan,
    JobExecutionResult,
    ProviderAttemptBroker,
    StageExecutionAuthorization,
)
from papc.baselines.router_accounting import (
    FROZEN_CACHE_NEUTRAL_PRICES,
    CallReceipt,
)
from papc.baselines.router_contract import (
    AGENT_DIFF,
    AUTOMIX,
    CHEAP_MODEL,
    REASONING_EFFORT,
    ROUTELLM_EMBEDDING_MODEL,
    STATE_BENCH,
    TAU2_AIRLINE,
    TAU2_RETAIL,
    AutoMixRouterView,
)

FIT_DEV_DISPATCH_SCHEMA = "papc.external_router.fit_dev_dispatch.v1"
FIT_DEV_OUTPUT_SCHEMA = "papc.external_router.fit_dev_job_output.v1"
FIT_DEV_CHECKPOINT_SCHEMA = (
    "papc.external_router.fit_dev_item_checkpoint.v1"
)
DEV_SPLIT_SEED = "papc.external_router.dev_split.v1"
SUPPORTED_KINDS = frozenset(
    {
        "shared_endpoint_episode",
        "route_embedding",
        "automix_n32_verifier",
    }
)
PRODUCTION_WIRED_KINDS = frozenset(
    {
        "shared_endpoint_episode",
        "route_embedding",
        "automix_n32_verifier",
    }
)
ROUTE_EMBEDDING_REASONING_EFFORT = "none"
ROUTE_EMBEDDING_PHASE = "route_embedding"
AUTOMIX_VERIFIER_PHASE = "self_verification_n32"
FIT_DEV_ROLES = frozenset({"router_fit", "router_dev"})
EXPECTED_TRAIN_COUNTS = {
    STATE_BENCH: 300,
    TAU2_RETAIL: 74,
    TAU2_AIRLINE: 30,
    AGENT_DIFF: 179,
}


@dataclass(frozen=True)
class FitDevSourceRow:
    """One exact fit/dev source row with a harness-only native locator."""

    surface: str
    stratum: str
    source_identity: str = field(repr=False)
    original_request: str
    payload_sha256: str
    example_sha256: str
    native_locator: Mapping[str, str] = field(repr=False)

    def __post_init__(self) -> None:
        for field_name in (
            "surface",
            "stratum",
            "source_identity",
            "original_request",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"fit_dev_{field_name}_invalid")
        for field_name in ("payload_sha256", "example_sha256"):
            _require_sha256(getattr(self, field_name), field_name)
        if self.example_sha256 != _digest(
            {
                "surface": self.surface,
                "stratum": self.stratum,
                "source_identity": self.source_identity,
                "payload_sha256": self.payload_sha256,
            }
        ):
            raise ValueError("fit_dev_example_sha256_mismatch")
        locator = dict(self.native_locator)
        if not locator or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in locator.items()
        ):
            raise ValueError("fit_dev_native_locator_invalid")
        object.__setattr__(self, "native_locator", locator)

    def private_record(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "stratum": self.stratum,
            "source_identity": self.source_identity,
            "original_request": self.original_request,
            "payload_sha256": self.payload_sha256,
            "example_sha256": self.example_sha256,
            "native_locator": dict(self.native_locator),
        }


@dataclass(frozen=True)
class FitDevRegistry:
    surface: str
    split_role: str
    source_binding_sha256: str
    registry_sha256: str
    rows: tuple[FitDevSourceRow, ...]

    def __post_init__(self) -> None:
        if self.surface not in EXPECTED_TRAIN_COUNTS:
            raise ValueError("fit_dev_registry_surface_invalid")
        if self.split_role not in FIT_DEV_ROLES:
            raise ValueError("fit_dev_registry_role_invalid")
        _require_sha256(
            self.source_binding_sha256,
            "source_binding_sha256",
        )
        _require_sha256(self.registry_sha256, "registry_sha256")
        rows = tuple(self.rows)
        if not rows or any(row.surface != self.surface for row in rows):
            raise ValueError("fit_dev_registry_rows_invalid")
        if len({row.example_sha256 for row in rows}) != len(rows):
            raise ValueError("fit_dev_registry_example_duplicate")
        expected_order = tuple(
            sorted(rows, key=lambda row: row.example_sha256)
        )
        if rows != expected_order:
            raise ValueError("fit_dev_registry_order_invalid")
        expected_registry = _registry_sha256(
            role=self.split_role,
            surface=self.surface,
            source_binding_sha256=self.source_binding_sha256,
            rows=rows,
        )
        if self.registry_sha256 != expected_registry:
            raise ValueError("fit_dev_registry_sha256_mismatch")
        object.__setattr__(self, "rows", rows)


@dataclass(frozen=True)
class StageItemInput:
    """Method-neutral input passed to one injected native execution."""

    ordinal: int
    source: FitDevSourceRow | None
    upstream_payload: Mapping[str, Any] | None = None
    request_payload: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.ordinal, bool)
            or not isinstance(self.ordinal, int)
            or self.ordinal <= 0
        ):
            raise ValueError("fit_dev_item_ordinal_invalid")
        if self.source is not None and not isinstance(
            self.source,
            FitDevSourceRow,
        ):
            raise TypeError("fit_dev_item_source_invalid")
        if self.upstream_payload is not None:
            object.__setattr__(
                self,
                "upstream_payload",
                dict(self.upstream_payload),
            )
        if self.request_payload is not None:
            object.__setattr__(
                self,
                "request_payload",
                dict(self.request_payload),
            )


@dataclass(frozen=True)
class StageItemResult:
    """Normalized native result plus every broker-settled raw receipt."""

    output: Mapping[str, Any]
    receipts: tuple[CallReceipt, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.output, Mapping):
            raise TypeError("fit_dev_item_output_invalid")
        receipts = tuple(self.receipts)
        if not receipts or any(
            not isinstance(receipt, CallReceipt) for receipt in receipts
        ):
            raise TypeError("fit_dev_item_receipts_invalid")
        if any(
            receipt.provider_calls != 1 or receipt.retry_count != 0
            for receipt in receipts
        ):
            raise ValueError("fit_dev_item_receipt_not_atomic")
        object.__setattr__(self, "output", dict(self.output))
        object.__setattr__(self, "receipts", receipts)


class RawAttemptBroker(Protocol):
    def perform(
        self,
        raw_attempt: Callable[[], CallReceipt],
    ) -> CallReceipt: ...


StageItemExecutor = Callable[
    [ExecutionJob, StageItemInput, RawAttemptBroker],
    StageItemResult,
]


@dataclass(frozen=True)
class BoundAutoMixTokenizer:
    """Tokenizer tied to the exact frozen local byte bundle."""

    tokenizer: VerifierTokenizer
    bundle_sha256: str

    def __post_init__(self) -> None:
        if self.tokenizer is None:
            raise TypeError("automix_bound_tokenizer_missing")
        if self.bundle_sha256 != VERIFIER_TOKENIZER_BUNDLE_SHA256:
            raise ValueError("automix_bound_tokenizer_sha256_mismatch")


def load_bound_automix_tokenizer(root: Path) -> BoundAutoMixTokenizer:
    """Validate frozen tokenizer bytes before a local-only load."""

    resolved = Path(root).resolve()
    binding = validate_automix_verifier_tokenizer_root(resolved)
    if binding.get("bundle_sha256") != VERIFIER_TOKENIZER_BUNDLE_SHA256:
        raise ValueError("automix_bound_tokenizer_record_mismatch")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(resolved),
        local_files_only=True,
        use_fast=False,
    )
    return BoundAutoMixTokenizer(
        tokenizer=tokenizer,
        bundle_sha256=VERIFIER_TOKENIZER_BUNDLE_SHA256,
    )


def openai_compatible_executor_factory(
    api_base: str,
    api_key: str,
    max_retries: int,
    *,
    client_factory: Callable[..., Any] | None = None,
) -> StageItemExecutor:
    """Construct the production executor without making a provider call."""

    if not isinstance(api_base, str) or not api_base.strip():
        raise ValueError("fit_dev_api_base_invalid")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("fit_dev_api_key_invalid")
    if isinstance(max_retries, bool) or max_retries != 0:
        raise ValueError("fit_dev_provider_max_retries_must_be_zero")
    factory = client_factory
    if factory is None:
        from openai import OpenAI

        factory = OpenAI
    client = factory(
        api_key=api_key,
        base_url=api_base.rstrip("/"),
        max_retries=0,
    )
    return build_openai_compatible_stage_executor(client)


def build_openai_compatible_stage_executor(
    provider_client: Any,
) -> StageItemExecutor:
    """Bind the two production-wired fit/dev provider operations."""

    if provider_client is None:
        raise TypeError("fit_dev_provider_client_missing")

    def execute(
        job: ExecutionJob,
        item: StageItemInput,
        broker: RawAttemptBroker,
    ) -> StageItemResult:
        if job.kind == "route_embedding":
            return _execute_route_embedding(
                job=job,
                item=item,
                broker=broker,
                provider_client=provider_client,
            )
        if job.kind == "automix_n32_verifier":
            return _execute_automix_verifier(
                job=job,
                item=item,
                broker=broker,
                provider_client=provider_client,
            )
        raise ExecutionContractError(
            f"fit_dev_production_kind_not_wired:{job.kind}"
        )

    return execute


def _execute_route_embedding(
    *,
    job: ExecutionJob,
    item: StageItemInput,
    broker: RawAttemptBroker,
    provider_client: Any,
) -> StageItemResult:
    if job.kind != "route_embedding" or job.method != "routellm_task":
        raise ExecutionContractError("route_embedding_job_binding_invalid")
    if item.source is None or item.upstream_payload is None:
        raise ExecutionContractError("route_embedding_input_missing")
    prompt = item.upstream_payload.get("original_request")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ExecutionContractError("route_embedding_prompt_invalid")
    returned: dict[str, Any] = {}

    def raw_attempt() -> CallReceipt:
        started = time.perf_counter()
        response = provider_client.embeddings.create(
            model=ROUTELLM_EMBEDDING_MODEL,
            input=prompt,
            encoding_format="float",
        )
        vector = _embedding_vector(response)
        receipt = _complete_response_receipt(
            response=response,
            role="router",
            phase=ROUTE_EMBEDDING_PHASE,
            model=ROUTELLM_EMBEDDING_MODEL,
            reasoning_effort=ROUTE_EMBEDDING_REASONING_EFFORT,
            latency_seconds=max(0.0, time.perf_counter() - started),
            embedding=True,
        )
        returned["embedding"] = vector
        return receipt

    receipt = broker.perform(raw_attempt)
    embedding = returned.get("embedding")
    if not isinstance(embedding, list):
        raise RuntimeError("route_embedding_result_missing_after_settle")
    return StageItemResult(
        output={
            "input_sha256": hashlib.sha256(
                prompt.encode("utf-8")
            ).hexdigest(),
            "embedding": embedding,
        },
        receipts=(receipt,),
    )


def _execute_automix_verifier(
    *,
    job: ExecutionJob,
    item: StageItemInput,
    broker: RawAttemptBroker,
    provider_client: Any,
) -> StageItemResult:
    if job.kind != "automix_n32_verifier" or job.method != AUTOMIX:
        raise ExecutionContractError("automix_verifier_job_binding_invalid")
    request = _validated_automix_request_payload(item)
    config = request["generation_config"]
    returned: dict[str, Any] = {}

    def raw_attempt() -> CallReceipt:
        started = time.perf_counter()
        response = provider_client.chat.completions.create(
            model=CHEAP_MODEL,
            messages=[{"role": "user", "content": request["prompt"]}],
            n=config["n"],
            temperature=config["temperature"],
            stop=config["stop"],
            max_tokens=config["max_tokens"],
            reasoning_effort=REASONING_EFFORT,
        )
        outputs = _chat_outputs(response, expected_count=VERIFIER_SAMPLES)
        receipt = _complete_response_receipt(
            response=response,
            role="verifier",
            phase=AUTOMIX_VERIFIER_PHASE,
            model=CHEAP_MODEL,
            reasoning_effort=REASONING_EFFORT,
            latency_seconds=max(0.0, time.perf_counter() - started),
            embedding=False,
        )
        returned["outputs"] = outputs
        return receipt

    receipt = broker.perform(raw_attempt)
    outputs = returned.get("outputs")
    if not isinstance(outputs, list):
        raise RuntimeError("automix_verifier_result_missing_after_settle")
    score = parse_automix_verifier_outputs(outputs)
    return StageItemResult(
        output={
            "prompt_sha256": request["prompt_sha256"],
            "request_sha256": request["request_sha256"],
            "outputs": outputs,
            "sample_count": score.sample_count,
            "valid_count": score.valid_count,
            "correct_count": score.correct_count,
            "confidence": score.confidence,
        },
        receipts=(receipt,),
    )


def _validated_automix_request_payload(
    item: StageItemInput,
) -> dict[str, Any]:
    if item.upstream_payload is None or item.request_payload is None:
        raise ExecutionContractError("automix_verifier_request_missing")
    request = dict(item.request_payload)
    request_sha = request.pop("request_sha256", None)
    if request_sha != _digest(request):
        raise ExecutionContractError("automix_request_artifact_sha256_mismatch")
    request["request_sha256"] = request_sha
    prompt = request.get("prompt")
    if (
        not isinstance(prompt, str)
        or not prompt
        or request.get("prompt_sha256")
        != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    ):
        raise ExecutionContractError("automix_request_prompt_mismatch")
    if (
        request.get("upstream_payload_sha256")
        != _digest(item.upstream_payload)
        or request.get("model") != CHEAP_MODEL
        or request.get("reasoning_effort") != REASONING_EFFORT
        or request.get("provider_endpoint") != "chat.completions.create"
        or request.get("provider_batch_count") != 1
        or request.get("tokenizer_bundle_sha256")
        != VERIFIER_TOKENIZER_BUNDLE_SHA256
        or request.get("prompt_template_sha256")
        != VERIFIER_PROMPT_TEMPLATE_SHA256
    ):
        raise ExecutionContractError("automix_request_authority_mismatch")
    expected_config = {
        "n": VERIFIER_SAMPLES,
        "temperature": VERIFIER_TEMPERATURE,
        "stop": VERIFIER_STOP,
        "max_tokens": VERIFIER_MAX_OUTPUT_TOKENS,
        "max_provider_batch_size": VERIFIER_PROVIDER_BATCH_SIZE,
    }
    if request.get("generation_config") != expected_config:
        raise ExecutionContractError("automix_request_config_mismatch")
    return request


def _embedding_vector(response: Any) -> list[float]:
    data = _member(response, "data")
    if (
        isinstance(data, (str, bytes))
        or not isinstance(data, Sequence)
        or len(data) != 1
    ):
        raise ValueError("route_embedding_response_count_invalid")
    vector = _member(data[0], "embedding")
    if (
        isinstance(vector, (str, bytes))
        or not isinstance(vector, Sequence)
        or not vector
    ):
        raise ValueError("route_embedding_response_vector_invalid")
    resolved = [float(value) for value in vector]
    if any(not math.isfinite(value) for value in resolved):
        raise ValueError("route_embedding_response_vector_nonfinite")
    return resolved


def _chat_outputs(response: Any, *, expected_count: int) -> list[str]:
    choices = _member(response, "choices")
    if (
        isinstance(choices, (str, bytes))
        or not isinstance(choices, Sequence)
        or len(choices) != expected_count
    ):
        raise ValueError("automix_verifier_response_count_invalid")
    outputs: list[str] = []
    for choice in choices:
        message = _member(choice, "message")
        content = _member(message, "content")
        if not isinstance(content, str):
            raise TypeError("automix_verifier_response_content_invalid")
        outputs.append(content)
    return outputs


def _complete_response_receipt(
    *,
    response: Any,
    role: str,
    phase: str,
    model: str,
    reasoning_effort: str,
    latency_seconds: float,
    embedding: bool,
) -> CallReceipt:
    usage = _member(response, "usage")
    input_tokens = _usage_int(
        usage,
        "input_tokens",
        fallback="prompt_tokens",
    )
    if input_tokens is None:
        raise ValueError("fit_dev_provider_input_usage_missing")
    if embedding:
        output_tokens = 0
        cached_tokens = 0
        reasoning_tokens = 0
    else:
        output_tokens = _usage_int(
            usage,
            "output_tokens",
            fallback="completion_tokens",
        )
        if output_tokens is None:
            raise ValueError("fit_dev_provider_output_usage_missing")
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
        cached_tokens = _usage_int(input_details, "cached_tokens")
        reasoning_tokens = _usage_int(output_details, "reasoning_tokens")
        if cached_tokens is None:
            cached_tokens = 0
        if reasoning_tokens is None:
            reasoning_tokens = 0
    return CallReceipt(
        role=role,  # type: ignore[arg-type]
        phase=phase,
        model=model,
        reasoning_effort=reasoning_effort,
        provider_calls=1,
        input_tokens=input_tokens,
        cached_input_tokens=cached_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        retry_count=0,
        latency_seconds=latency_seconds,
        provider_reported_cost_usd=_cache_neutral_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


def _cache_neutral_cost(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    price = FROZEN_CACHE_NEUTRAL_PRICES.get(model)
    if price is None:
        raise ValueError("fit_dev_provider_model_price_missing")
    return (
        input_tokens * price.input_per_million_usd
        + output_tokens * price.output_per_million_usd
    ) / 1_000_000


def _usage_int(
    usage: Any,
    name: str,
    *,
    fallback: str | None = None,
) -> int | None:
    value = _member(usage, name, fallback=fallback)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"fit_dev_provider_usage_invalid:{name}")
    return value


def _member(
    value: Any,
    name: str,
    *,
    fallback: str | None = None,
) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        resolved = value.get(name)
        if resolved is None and fallback is not None:
            resolved = value.get(fallback)
        return resolved
    resolved = getattr(value, name, None)
    if resolved is None and fallback is not None:
        resolved = getattr(value, fallback, None)
    return resolved


@dataclass(frozen=True)
class PreparedFitDevJob:
    plan: ExecutionPlan
    job: ExecutionJob
    authorization: StageExecutionAuthorization
    items: tuple[StageItemInput, ...]

    def public_record(self) -> dict[str, Any]:
        return {
            "schema": FIT_DEV_DISPATCH_SCHEMA,
            "plan_sha256": self.plan.sha256,
            "plan_protocol_semantic_sha256": (
                self.plan.protocol_semantic_sha256
            ),
            "job": self.job.public_record(),
            "item_count": len(self.items),
            "source_rows_reconstructed": True,
            "official_test_outcomes_read": False,
            "provider_calls": 0,
            "secret_reads": 0,
            "api_spend_performed": False,
        }


def prepare_fit_dev_job(
    *,
    plan: ExecutionPlan,
    job_id: str,
    authorization: StageExecutionAuthorization,
    split_gate_json: Path,
    state_repo: Path,
    tau2_repo: Path,
    agent_diff_dataset: Path,
    automix_tokenizer: BoundAutoMixTokenizer | None = None,
) -> PreparedFitDevJob:
    """Reconstruct and bind one early-stage job without provider access."""

    try:
        job = plan.job(job_id)
    except KeyError as exc:
        raise ExecutionContractError(
            "fit_dev_job_not_in_execution_plan"
        ) from exc
    if job.kind not in SUPPORTED_KINDS:
        raise ExecutionContractError("fit_dev_job_kind_not_supported")
    expected_stage = {
        "shared_endpoint_episode": SHARED_STAGE,
        "route_embedding": SHARED_STAGE,
        "automix_n32_verifier": AUTOMIX_STAGE,
    }[job.kind]
    if job.stage != expected_stage:
        raise ExecutionContractError("fit_dev_job_stage_mismatch")
    if authorization.stage != job.stage:
        raise ExecutionContractError(
            "fit_dev_authorization_stage_mismatch"
        )
    # Construction binds exact final-plan SHA, stage attempt cap, and USD stop.
    BatchCoordinator(plan=plan, authorization=authorization)
    if job.output_path.exists():
        raise FileExistsError("job_output_exists_no_overwrite")

    registry = load_fit_dev_registry(
        job=job,
        split_gate_json=split_gate_json,
        state_repo=state_repo,
        tau2_repo=tau2_repo,
        agent_diff_dataset=agent_diff_dataset,
    )
    details = dict(job.details)
    needs_shared_cheap = (
        job.kind in {"route_embedding", "automix_n32_verifier"}
        or (
            job.kind == "shared_endpoint_episode"
            and details.get("model_role") == "strong"
        )
    )
    upstream = (
        _load_shared_cheap_payloads(
            plan=plan,
            job=job,
            registry=registry,
        )
        if needs_shared_cheap
        else {}
    )
    if job.kind == "automix_n32_verifier" and automix_tokenizer is None:
        raise ExecutionContractError(
            "automix_bound_tokenizer_required"
        )
    resolved_items: list[StageItemInput] = []
    for ordinal, row in enumerate(registry.rows, start=1):
        upstream_payload = upstream.get(row.example_sha256)
        request_payload = (
            _build_automix_request_payload(
                source=row,
                upstream_payload=upstream_payload,
                bound_tokenizer=automix_tokenizer,
            )
            if job.kind == "automix_n32_verifier"
            else None
        )
        resolved_items.append(
            StageItemInput(
                ordinal=ordinal,
                source=row,
                upstream_payload=upstream_payload,
                request_payload=request_payload,
            )
        )
    items = tuple(resolved_items)
    if len(items) != job.item_count:
        raise ExecutionContractError("fit_dev_job_item_count_mismatch")
    return PreparedFitDevJob(
        plan=plan,
        job=job,
        authorization=authorization,
        items=items,
    )


def load_fit_dev_registry(
    *,
    job: ExecutionJob,
    split_gate_json: Path,
    state_repo: Path,
    tau2_repo: Path,
    agent_diff_dataset: Path,
) -> FitDevRegistry:
    """Reconstruct only official-train rows and prove the frozen registry."""

    if job.surface not in EXPECTED_TRAIN_COUNTS:
        raise ExecutionContractError("fit_dev_surface_not_supported")
    if job.split_role not in FIT_DEV_ROLES:
        raise ExecutionContractError("fit_dev_split_role_invalid")
    split_path = Path(split_gate_json).resolve()
    split_raw = split_path.read_bytes()
    if hashlib.sha256(split_raw).hexdigest() != job.split_gate_sha256:
        raise ExecutionContractError("fit_dev_split_gate_sha256_mismatch")
    split = json.loads(split_raw)
    entry = split["surface_splits"][job.surface]
    registry_key = (
        "fit_registry_sha256"
        if job.split_role == "router_fit"
        else "dev_registry_sha256"
    )
    if (
        entry.get(registry_key) != job.registry_sha256
        or entry.get("source_binding_sha256")
        != job.source_binding_sha256
    ):
        raise ExecutionContractError("fit_dev_job_split_binding_mismatch")

    if job.surface == STATE_BENCH:
        train_rows = _load_state_train(Path(state_repo).resolve())
    elif job.surface in {TAU2_RETAIL, TAU2_AIRLINE}:
        domain = "retail" if job.surface == TAU2_RETAIL else "airline"
        train_rows = _load_tau2_train(
            Path(tau2_repo).resolve(),
            domain=domain,
        )
    else:
        train_rows = _load_agent_diff_train(
            Path(agent_diff_dataset).resolve()
        )
    if len(train_rows) != EXPECTED_TRAIN_COUNTS[job.surface]:
        raise ExecutionContractError("fit_dev_official_train_count_mismatch")
    fit, dev = deterministic_fit_dev(train_rows)
    rows = fit if job.split_role == "router_fit" else dev
    return FitDevRegistry(
        surface=job.surface,
        split_role=job.split_role,
        source_binding_sha256=job.source_binding_sha256,
        registry_sha256=job.registry_sha256,
        rows=rows,
    )


def deterministic_fit_dev(
    rows: Sequence[FitDevSourceRow],
) -> tuple[tuple[FitDevSourceRow, ...], tuple[FitDevSourceRow, ...]]:
    """Apply the frozen outcome-blind, stratum-preserving split."""

    resolved = tuple(rows)
    if not resolved:
        raise ValueError("fit_dev_official_train_rows_empty")
    by_stratum: dict[str, list[FitDevSourceRow]] = defaultdict(list)
    for row in resolved:
        if not isinstance(row, FitDevSourceRow):
            raise TypeError("fit_dev_source_row_invalid")
        by_stratum[row.stratum].append(row)
    fit: list[FitDevSourceRow] = []
    dev: list[FitDevSourceRow] = []
    for stratum, stratum_rows in sorted(by_stratum.items()):
        ordered = sorted(
            stratum_rows,
            key=lambda row: _digest(
                {
                    "seed": DEV_SPLIT_SEED,
                    "stratum": stratum,
                    "example_sha256": row.example_sha256,
                }
            ),
        )
        dev_count = max(1, (len(ordered) + 4) // 5)
        if dev_count >= len(ordered):
            raise ValueError("fit_dev_stratum_too_small")
        dev.extend(ordered[:dev_count])
        fit.extend(ordered[dev_count:])
    return (
        tuple(sorted(fit, key=lambda row: row.example_sha256)),
        tuple(sorted(dev, key=lambda row: row.example_sha256)),
    )


def execute_prepared_fit_dev_job(
    prepared: PreparedFitDevJob,
    executor: StageItemExecutor,
    *,
    coordinator: BatchCoordinator | None = None,
    max_concurrency: int = 1,
) -> tuple[JobExecutionResult, dict[str, Any]]:
    """Execute a whole job and publish its complete private JSONL atomically."""

    if not isinstance(prepared, PreparedFitDevJob):
        raise TypeError("prepared_fit_dev_job_invalid")
    if not callable(executor):
        raise TypeError("fit_dev_executor_not_callable")
    if (
        isinstance(max_concurrency, bool)
        or not isinstance(max_concurrency, int)
        or not 1 <= max_concurrency <= 5
    ):
        raise ValueError("fit_dev_max_concurrency_invalid")
    if (
        prepared.job.kind == "shared_endpoint_episode"
        and max_concurrency != 1
    ):
        raise ValueError(
            "fit_dev_native_episode_concurrency_must_be_one"
        )
    active_coordinator = coordinator or BatchCoordinator(
        plan=prepared.plan,
        authorization=prepared.authorization,
    )
    if (
        active_coordinator.plan.sha256 != prepared.plan.sha256
        or active_coordinator.authorization != prepared.authorization
    ):
        raise ExecutionContractError(
            "fit_dev_shared_coordinator_binding_mismatch"
        )
    serialized: bytes | None = None
    checkpoint_path = fit_dev_checkpoint_path(
        prepared.job.output_path
    )
    checkpoint_records = list(
        _load_or_initialize_item_checkpoint(
            prepared,
            checkpoint_path=checkpoint_path,
        )
    )
    resumed_item_count = len(checkpoint_records)

    if resumed_item_count == len(prepared.items):
        serialized = _serialize_job_output(
            plan=prepared.plan,
            job=prepared.job,
            item_records=checkpoint_records,
        )
        result = JobExecutionResult(
            completed_item_count=len(checkpoint_records),
            registry_sha256=prepared.job.registry_sha256,
            output_payload_sha256=hashlib.sha256(
                serialized
            ).hexdigest(),
        )
        _publish_append_only_atomic(
            prepared.job.output_path,
            serialized,
        )
        return result, {
            "schema": FIT_DEV_DISPATCH_SCHEMA,
            "decision": "complete_from_bound_item_checkpoint",
            "job_id": prepared.job.job_id,
            "private_output": str(prepared.job.output_path),
            "checkpoint_path": str(checkpoint_path),
            "resumed_item_count": resumed_item_count,
            "executed_item_count": 0,
            "output_payload_sha256": result.output_payload_sha256,
            "provider_calls": 0,
            "coordinator": active_coordinator.public_report(),
        }

    def execute(
        job: ExecutionJob,
        broker: ProviderAttemptBroker,
    ) -> JobExecutionResult:
        nonlocal serialized

        def execute_item(item: StageItemInput) -> dict[str, Any]:
            recording_broker = _RecordingBroker(broker)
            result = executor(job, item, recording_broker)
            _validate_item_result(
                job=job,
                item=item,
                result=result,
                brokered_attempt_count=len(recording_broker.receipts),
                brokered_receipts=recording_broker.receipts,
            )
            return _item_record(
                job=job,
                item=item,
                result=result,
            )

        if max_concurrency == 1:
            item_records = list(checkpoint_records)
            for item in prepared.items[resumed_item_count:]:
                record = execute_item(item)
                _append_item_checkpoint(checkpoint_path, record)
                item_records.append(record)
        else:
            with ThreadPoolExecutor(
                max_workers=max_concurrency,
                thread_name_prefix="papc-fit-dev",
            ) as pool:
                new_records = list(
                    pool.map(
                        execute_item,
                        prepared.items[resumed_item_count:],
                    )
                )
            item_records = [*checkpoint_records, *new_records]
            for record in new_records:
                _append_item_checkpoint(checkpoint_path, record)
        serialized = _serialize_job_output(
            plan=prepared.plan,
            job=job,
            item_records=item_records,
        )
        return JobExecutionResult(
            completed_item_count=len(item_records),
            registry_sha256=job.registry_sha256,
            output_payload_sha256=hashlib.sha256(serialized).hexdigest(),
        )

    result = active_coordinator.execute_job(prepared.job.job_id, execute)
    if serialized is None:
        raise RuntimeError("fit_dev_job_output_not_materialized")
    _publish_append_only_atomic(prepared.job.output_path, serialized)
    report = active_coordinator.public_report()
    return result, {
        "schema": FIT_DEV_DISPATCH_SCHEMA,
        "decision": "complete",
        "job_id": prepared.job.job_id,
        "private_output": str(prepared.job.output_path),
        "checkpoint_path": str(checkpoint_path),
        "resumed_item_count": resumed_item_count,
        "executed_item_count": (
            len(prepared.items) - resumed_item_count
        ),
        "output_payload_sha256": result.output_payload_sha256,
        "provider_calls": report["ledger"]["jobs"][prepared.job.job_id][
            "attempts_reserved"
        ],
        "coordinator": report,
    }


def fit_dev_checkpoint_path(output_path: Path) -> Path:
    target = Path(output_path).resolve()
    return target.parent / f".{target.name}.items.jsonl"


def inspect_fit_dev_item_checkpoint(
    prepared: PreparedFitDevJob,
) -> dict[str, Any]:
    if not isinstance(prepared, PreparedFitDevJob):
        raise TypeError("prepared_fit_dev_job_invalid")
    path = fit_dev_checkpoint_path(prepared.job.output_path)
    records = _load_or_initialize_item_checkpoint(
        prepared,
        checkpoint_path=path,
    )
    return {
        "checkpoint_path": str(path),
        "resumed_item_count": len(records),
        "remaining_item_count": len(prepared.items) - len(records),
        "checkpoint_complete": len(records) == len(prepared.items),
    }


def read_fit_dev_job_output(path: Path) -> tuple[dict[str, Any], ...]:
    """Read one atomically completed job output."""

    records = tuple(
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not records or any(not isinstance(item, dict) for item in records):
        raise ValueError("fit_dev_output_records_invalid")
    header = records[0]
    if (
        header.get("schema") != FIT_DEV_OUTPUT_SCHEMA
        or header.get("record_type") != "job_header"
    ):
        raise ValueError("fit_dev_output_header_invalid")
    if len(records) != int(header.get("item_count", -1)) + 1:
        raise ValueError("fit_dev_output_item_count_invalid")
    if any(item.get("record_type") != "item" for item in records[1:]):
        raise ValueError("fit_dev_output_item_record_invalid")
    return records


def _load_shared_cheap_payloads(
    *,
    plan: ExecutionPlan,
    job: ExecutionJob,
    registry: FitDevRegistry,
) -> dict[str, Mapping[str, Any]]:
    if job.kind == "automix_n32_verifier" and (
        job.method != AUTOMIX or job.surface == AGENT_DIFF
    ):
        raise ExecutionContractError("automix_fit_dev_job_scope_invalid")
    upstream_id = ".".join(
        (
            SHARED_STAGE,
            "shared_endpoint_episode",
            "shared_cheap_endpoint",
            job.surface,
            job.split_role,
        )
    )
    upstream_job = plan.job(upstream_id)
    if not upstream_job.output_path.is_file():
        raise FileNotFoundError(
            f"fit_dev_upstream_output_missing:{upstream_job.job_id}"
        )
    records = read_fit_dev_job_output(upstream_job.output_path)
    header = records[0]
    if (
        header.get("plan_sha256") != plan.sha256
        or header.get("plan_protocol_semantic_sha256")
        != plan.protocol_semantic_sha256
        or header.get("job_id") != upstream_job.job_id
        or header.get("registry_sha256") != registry.registry_sha256
        or header.get("source_binding_sha256")
        != registry.source_binding_sha256
    ):
        raise ExecutionContractError("fit_dev_upstream_header_mismatch")
    by_example: dict[str, Mapping[str, Any]] = {}
    for expected, record in zip(registry.rows, records[1:], strict=True):
        if (
            record.get("example_sha256") != expected.example_sha256
            or record.get("payload_sha256") != expected.payload_sha256
        ):
            raise ExecutionContractError(
                "fit_dev_upstream_membership_mismatch"
            )
        output = record.get("output")
        if not isinstance(output, Mapping):
            raise ExecutionContractError("fit_dev_upstream_output_invalid")
        _validate_endpoint_output(
            expected,
            output,
            expected_role="cheap",
        )
        by_example[expected.example_sha256] = dict(output)
    return by_example


def _build_automix_request_payload(
    *,
    source: FitDevSourceRow,
    upstream_payload: Mapping[str, Any] | None,
    bound_tokenizer: BoundAutoMixTokenizer | None,
) -> dict[str, Any]:
    if upstream_payload is None:
        raise ExecutionContractError("automix_upstream_payload_missing")
    if bound_tokenizer is None:
        raise ExecutionContractError("automix_bound_tokenizer_required")
    original_request = upstream_payload.get("original_request")
    generated_answer = upstream_payload.get("generated_answer")
    public_episode = upstream_payload.get("public_episode")
    if (
        not isinstance(original_request, str)
        or not isinstance(generated_answer, str)
        or not isinstance(public_episode, list)
        or any(not isinstance(turn, Mapping) for turn in public_episode)
    ):
        raise ExecutionContractError("automix_upstream_trace_invalid")
    view = AutoMixRouterView(
        original_request=original_request,
        cheap_public_episode=tuple(
            dict(turn) for turn in public_episode
        ),
        cheap_generated_answer=generated_answer,
    )
    context = serialize_automix_agent_context(view)
    request = build_automix_verifier_request(
        context=context,
        question=original_request,
        generated_answer=generated_answer,
        tokenizer=bound_tokenizer.tokenizer,
    )
    artifact = {
        "source_example_sha256": source.example_sha256,
        "upstream_payload_sha256": _digest(upstream_payload),
        "context_sha256": hashlib.sha256(
            context.encode("utf-8")
        ).hexdigest(),
        "original_request_sha256": hashlib.sha256(
            original_request.encode("utf-8")
        ).hexdigest(),
        "generated_answer_sha256": hashlib.sha256(
            generated_answer.encode("utf-8")
        ).hexdigest(),
        "public_episode_sha256": _digest(public_episode),
        "prompt": request.prompt,
        "prompt_sha256": request.prompt_sha256,
        "prompt_template_sha256": VERIFIER_PROMPT_TEMPLATE_SHA256,
        "generation_config": request.generation_config(),
        "provider_endpoint": "chat.completions.create",
        "provider_batch_count": 1,
        "model": CHEAP_MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "tokenizer_id": request.tokenizer_id,
        "tokenizer_bundle_sha256": bound_tokenizer.bundle_sha256,
    }
    return {
        **artifact,
        "request_sha256": _digest(artifact),
    }


class _RecordingBroker(ProviderAttemptBroker):
    """Record the exact receipts returned by the real accounting broker."""

    def __init__(self, broker: ProviderAttemptBroker) -> None:
        self._broker = broker
        self._receipts: list[CallReceipt] = []

    @property
    def receipts(self) -> tuple[CallReceipt, ...]:
        return tuple(self._receipts)

    def perform(
        self,
        raw_attempt: Callable[[], CallReceipt],
    ) -> CallReceipt:
        receipt = self._broker.perform(raw_attempt)
        self._receipts.append(receipt)
        return receipt


def _validate_item_result(
    *,
    job: ExecutionJob,
    item: StageItemInput,
    result: StageItemResult,
    brokered_attempt_count: int,
    brokered_receipts: tuple[CallReceipt, ...],
) -> None:
    if not isinstance(result, StageItemResult):
        raise TypeError("fit_dev_native_result_invalid")
    if brokered_attempt_count != len(result.receipts):
        raise ExecutionContractError(
            "fit_dev_receipts_not_equal_brokered_attempts"
        )
    if tuple(
        receipt.public_record() for receipt in result.receipts
    ) != tuple(receipt.public_record() for receipt in brokered_receipts):
        raise ExecutionContractError(
            "fit_dev_receipts_not_equal_broker_settlements"
        )
    if not 1 <= brokered_attempt_count <= job.attempts_per_item:
        raise ExecutionContractError("fit_dev_item_attempt_count_invalid")
    if job.kind in {"route_embedding", "automix_n32_verifier"} and (
        brokered_attempt_count != 1
    ):
        raise ExecutionContractError("fit_dev_batch_attempt_count_not_one")
    if job.kind == "shared_endpoint_episode":
        if item.source is None:
            raise ExecutionContractError("fit_dev_endpoint_source_missing")
        if any(
            receipt.provider_calls != 1
            or receipt.retry_count != 0
            or receipt.input_tokens is None
            or receipt.cached_input_tokens is None
            or receipt.output_tokens is None
            or receipt.reasoning_tokens is None
            or receipt.provider_reported_cost_usd is None
            for receipt in result.receipts
        ):
            raise ExecutionContractError(
                "fit_dev_endpoint_provider_receipt_incomplete"
            )
        expected_role = dict(job.details)["model_role"]
        _validate_endpoint_output(
            item.source,
            result.output,
            expected_role=expected_role,
        )
        if (
            expected_role == "strong"
            and (
                item.upstream_payload is None
                or result.output.get("original_request")
                != item.upstream_payload.get("original_request")
            )
        ):
            raise ExecutionContractError(
                "fit_dev_paired_endpoint_request_mismatch"
            )
        forbidden = {
            "cheap": {"strong_agent", "router", "verifier"},
            "strong": {"cheap_agent", "router", "verifier"},
        }[expected_role]
        required = f"{expected_role}_agent"
        roles = {receipt.role for receipt in result.receipts}
        if required not in roles or roles & forbidden:
            raise ExecutionContractError(
                "fit_dev_endpoint_receipt_role_mismatch"
            )
    elif job.kind == "route_embedding":
        _validate_route_embedding_result(item, result)
    elif job.kind == "automix_n32_verifier":
        _validate_automix_result(item, result)


def _validate_endpoint_output(
    source: FitDevSourceRow,
    output: Mapping[str, Any],
    *,
    expected_role: str,
) -> None:
    if output.get("model_role") != expected_role:
        raise ExecutionContractError("fit_dev_endpoint_model_role_mismatch")
    original_request = output.get("original_request")
    if not isinstance(original_request, str) or not original_request.strip():
        raise ExecutionContractError(
            "fit_dev_endpoint_original_request_invalid"
        )
    # STATE-Bench and Agent-Diff expose a fixed public request in source. Tau2
    # obtains the opening message from its native user simulator, so pairing is
    # bound against the shared cheap output instead of a hidden task sketch.
    if (
        source.surface not in {TAU2_RETAIL, TAU2_AIRLINE}
        and original_request != source.original_request
    ):
        raise ExecutionContractError(
            "fit_dev_endpoint_original_request_mismatch"
        )
    generated = output.get("generated_answer")
    if not isinstance(generated, str) or not generated.strip():
        raise ExecutionContractError(
            "fit_dev_endpoint_generated_answer_invalid"
        )
    episode = output.get("public_episode")
    if (
        not isinstance(episode, list)
        or not episode
        or any(not isinstance(turn, Mapping) for turn in episode)
    ):
        raise ExecutionContractError("fit_dev_endpoint_episode_invalid")
    score = output.get("official_score")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
    ):
        raise ExecutionContractError("fit_dev_endpoint_score_invalid")


def _validate_route_embedding_result(
    item: StageItemInput,
    result: StageItemResult,
) -> None:
    if item.upstream_payload is None:
        raise ExecutionContractError("route_embedding_upstream_missing")
    if len(result.receipts) != 1:
        raise ExecutionContractError("route_embedding_receipt_count_invalid")
    receipt = result.receipts[0]
    if (
        receipt.role != "router"
        or receipt.model != "text-embedding-3-small"
    ):
        raise ExecutionContractError("route_embedding_receipt_invalid")
    embedding = result.output.get("embedding")
    if (
        not isinstance(embedding, list)
        or not embedding
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in embedding
        )
    ):
        raise ExecutionContractError("route_embedding_vector_invalid")
    upstream_request = item.upstream_payload.get("original_request")
    if (
        not isinstance(upstream_request, str)
        or result.output.get("input_sha256")
        != hashlib.sha256(upstream_request.encode("utf-8")).hexdigest()
    ):
        raise ExecutionContractError("route_embedding_input_mismatch")


def _validate_automix_result(
    item: StageItemInput,
    result: StageItemResult,
) -> None:
    if (
        item.source is None
        or item.upstream_payload is None
        or item.request_payload is None
    ):
        raise ExecutionContractError("automix_upstream_payload_missing")
    receipt = result.receipts[0]
    if receipt.role != "verifier" or receipt.model != "gpt-5.4-mini":
        raise ExecutionContractError("automix_verifier_receipt_invalid")
    outputs = result.output.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != VERIFIER_SAMPLES:
        raise ExecutionContractError("automix_verifier_outputs_invalid")
    score = parse_automix_verifier_outputs(outputs)
    expected = {
        "sample_count": score.sample_count,
        "valid_count": score.valid_count,
        "correct_count": score.correct_count,
        "confidence": score.confidence,
    }
    if any(result.output.get(key) != value for key, value in expected.items()):
        raise ExecutionContractError("automix_verifier_score_mismatch")
    if (
        result.output.get("prompt_sha256")
        != item.request_payload.get("prompt_sha256")
        or result.output.get("request_sha256")
        != item.request_payload.get("request_sha256")
    ):
        raise ExecutionContractError(
            "automix_verifier_request_binding_mismatch"
        )


def _item_record(
    *,
    job: ExecutionJob,
    item: StageItemInput,
    result: StageItemResult,
) -> dict[str, Any]:
    source = item.source
    return {
        "record_type": "item",
        "ordinal": item.ordinal,
        "job_id": job.job_id,
        "example_sha256": source.example_sha256 if source else None,
        "payload_sha256": source.payload_sha256 if source else None,
        "source": source.private_record() if source else None,
        "upstream_payload_sha256": (
            _digest(item.upstream_payload)
            if item.upstream_payload is not None
            else None
        ),
        "request_payload": (
            dict(item.request_payload)
            if item.request_payload is not None
            else None
        ),
        "output": dict(result.output),
        "receipts": [
            receipt.public_record() for receipt in result.receipts
        ],
    }


def _serialize_job_output(
    *,
    plan: ExecutionPlan,
    job: ExecutionJob,
    item_records: Sequence[Mapping[str, Any]],
) -> bytes:
    header = {
        "schema": FIT_DEV_OUTPUT_SCHEMA,
        "record_type": "job_header",
        "plan_sha256": plan.sha256,
        "plan_protocol_semantic_sha256": plan.protocol_semantic_sha256,
        "job_id": job.job_id,
        "stage": job.stage,
        "kind": job.kind,
        "method": job.method,
        "surface": job.surface,
        "split_role": job.split_role,
        "item_count": len(item_records),
        "registry_sha256": job.registry_sha256,
        "source_binding_sha256": job.source_binding_sha256,
        "split_gate_sha256": job.split_gate_sha256,
        "official_test_outcomes_read": False,
        "partial_resume_or_cross_run_merge_allowed": False,
        "same_job_bound_item_checkpoint_continuation_allowed": True,
        "cross_manifest_or_protocol_merge_allowed": False,
    }
    lines = (header, *item_records)
    return (
        "\n".join(
            json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for item in lines
        )
        + "\n"
    ).encode("utf-8")


def _publish_append_only_atomic(path: Path, payload: bytes) -> None:
    """Publish complete bytes atomically without ever replacing a target."""

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".complete",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        # link() is atomic and fails rather than replacing an existing target.
        os.link(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _load_or_initialize_item_checkpoint(
    prepared: PreparedFitDevJob,
    *,
    checkpoint_path: Path,
) -> tuple[dict[str, Any], ...]:
    path = Path(checkpoint_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "schema": FIT_DEV_CHECKPOINT_SCHEMA,
        "record_type": "checkpoint_header",
        "plan_sha256": prepared.plan.sha256,
        "plan_protocol_semantic_sha256": (
            prepared.plan.protocol_semantic_sha256
        ),
        "job_id": prepared.job.job_id,
        "item_count": len(prepared.items),
        "registry_sha256": prepared.job.registry_sha256,
        "source_binding_sha256": prepared.job.source_binding_sha256,
        "split_gate_sha256": prepared.job.split_gate_sha256,
        "same_job_bound_item_checkpoint_continuation_allowed": True,
        "cross_manifest_or_protocol_merge_allowed": False,
    }
    if not path.exists():
        payload = (
            json.dumps(
                header,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        try:
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            pass
        else:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
    records = _read_jsonl(path)
    if not records or records[0] != header:
        raise ExecutionContractError(
            "fit_dev_item_checkpoint_header_mismatch"
        )
    item_records = records[1:]
    if len(item_records) > len(prepared.items):
        raise ExecutionContractError(
            "fit_dev_item_checkpoint_count_exceeded"
        )
    validated: list[dict[str, Any]] = []
    for item, record in zip(
        prepared.items,
        item_records,
        strict=False,
    ):
        output = record.get("output")
        raw_receipts = record.get("receipts")
        if not isinstance(output, Mapping) or not isinstance(
            raw_receipts,
            list,
        ):
            raise ExecutionContractError(
                "fit_dev_item_checkpoint_record_invalid"
            )
        try:
            receipts = tuple(
                CallReceipt(**dict(raw))
                for raw in raw_receipts
                if isinstance(raw, Mapping)
            )
        except (TypeError, ValueError) as exc:
            raise ExecutionContractError(
                "fit_dev_item_checkpoint_receipt_invalid"
            ) from exc
        if len(receipts) != len(raw_receipts):
            raise ExecutionContractError(
                "fit_dev_item_checkpoint_receipt_invalid"
            )
        result = StageItemResult(
            output=dict(output),
            receipts=receipts,
        )
        _validate_item_result(
            job=prepared.job,
            item=item,
            result=result,
            brokered_attempt_count=len(receipts),
            brokered_receipts=receipts,
        )
        expected = _item_record(
            job=prepared.job,
            item=item,
            result=result,
        )
        if record != expected:
            raise ExecutionContractError(
                "fit_dev_item_checkpoint_membership_mismatch"
            )
        validated.append(expected)
    return tuple(validated)


def _append_item_checkpoint(
    checkpoint_path: Path,
    record: Mapping[str, Any],
) -> None:
    payload = (
        json.dumps(
            dict(record),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        Path(checkpoint_path).resolve(),
        os.O_WRONLY | os.O_APPEND,
    )
    with os.fdopen(descriptor, "ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _load_state_train(repo: Path) -> tuple[FitDevSourceRow, ...]:
    rows: list[FitDevSourceRow] = []
    for domain in ("customer_support", "shopping_assistant", "travel"):
        domain_dir = repo / "state_bench" / "domains" / domain
        split_path = domain_dir / "splits" / "train_test.json"
        split = _read_mapping(split_path, "state_split")
        raw_train = split.get("splits")
        if not isinstance(raw_train, Mapping) or not isinstance(
            raw_train.get("train"),
            list,
        ):
            raise TypeError("state_train_split_invalid")
        for raw_id in raw_train["train"]:
            task_id = str(raw_id)
            if Path(task_id).name != task_id:
                raise ValueError("state_train_task_id_not_atomic")
            task_path = domain_dir / "tasks" / f"{task_id}.json"
            task = _read_mapping(task_path, "state_task")
            if str(task.get("task_id")) != task_id:
                raise ValueError("state_train_task_id_mismatch")
            env_raw = task.get("task_env_path")
            if not isinstance(env_raw, str) or not env_raw:
                raise ValueError("state_train_env_path_invalid")
            env_path = (repo / env_raw).resolve()
            if not env_path.is_relative_to(repo.resolve()):
                raise ValueError("state_train_env_path_outside_repo")
            environment = _read_mapping(env_path, "state_environment")
            payload = {"task": task, "initial_environment": environment}
            opening = task.get("opening_message")
            rows.append(
                _source_row(
                    surface=STATE_BENCH,
                    stratum=domain,
                    source_identity=task_id,
                    original_request=opening,
                    payload=payload,
                    native_locator={
                        "domain": domain,
                        "task_id": task_id,
                        "task_path": str(task_path.resolve()),
                        "initial_environment_path": str(env_path),
                    },
                )
            )
    return tuple(rows)


def _load_tau2_train(
    repo: Path,
    *,
    domain: str,
) -> tuple[FitDevSourceRow, ...]:
    domain_dir = repo / "data" / "tau2" / "domains" / domain
    split_path = domain_dir / "split_tasks.json"
    tasks_path = domain_dir / "tasks.json"
    split = _read_mapping(split_path, "tau2_split")
    tasks_raw = json.loads(tasks_path.read_text(encoding="utf-8"))
    if not isinstance(tasks_raw, list):
        raise TypeError("tau2_tasks_invalid")
    tasks = {
        str(item["id"]): item
        for item in tasks_raw
        if isinstance(item, Mapping)
    }
    raw_train = split.get("train")
    if not isinstance(raw_train, list):
        raise TypeError("tau2_train_split_invalid")
    surface = TAU2_RETAIL if domain == "retail" else TAU2_AIRLINE
    rows: list[FitDevSourceRow] = []
    for raw_id in raw_train:
        task_id = str(raw_id)
        task = tasks.get(task_id)
        if task is None:
            raise ValueError("tau2_train_task_missing")
        request = _tau2_original_request(task)
        rows.append(
            _source_row(
                surface=surface,
                stratum=domain,
                source_identity=task_id,
                original_request=request,
                payload=task,
                native_locator={
                    "domain": domain,
                    "task_id": task_id,
                    "tasks_path": str(tasks_path.resolve()),
                },
            )
        )
    return tuple(rows)


def _load_agent_diff_train(
    dataset: Path,
) -> tuple[FitDevSourceRow, ...]:
    train_path = dataset / "train.jsonl"
    rows: list[FitDevSourceRow] = []
    for ordinal, raw in enumerate(
        _read_jsonl(train_path),
        start=1,
    ):
        service = raw.get("service")
        test_id = raw.get("test_id")
        question = raw.get("question")
        rows.append(
            _source_row(
                surface=AGENT_DIFF,
                stratum=service,
                source_identity=test_id,
                original_request=question,
                payload=raw,
                native_locator={
                    "service": str(service),
                    "test_id": str(test_id),
                    "train_ordinal": str(ordinal),
                    "train_jsonl": str(train_path.resolve()),
                },
            )
        )
    return tuple(rows)


def _source_row(
    *,
    surface: str,
    stratum: Any,
    source_identity: Any,
    original_request: Any,
    payload: Any,
    native_locator: Mapping[str, str],
) -> FitDevSourceRow:
    if not isinstance(stratum, str) or not stratum.strip():
        raise ValueError("fit_dev_source_stratum_invalid")
    if not isinstance(source_identity, str) or not source_identity.strip():
        raise ValueError("fit_dev_source_identity_invalid")
    if not isinstance(original_request, str) or not original_request.strip():
        raise ValueError("fit_dev_source_original_request_invalid")
    payload_sha256 = _digest(payload)
    return FitDevSourceRow(
        surface=surface,
        stratum=stratum,
        source_identity=source_identity,
        original_request=original_request,
        payload_sha256=payload_sha256,
        example_sha256=_digest(
            {
                "surface": surface,
                "stratum": stratum,
                "source_identity": source_identity,
                "payload_sha256": payload_sha256,
            }
        ),
        native_locator=native_locator,
    )


def _tau2_original_request(task: Mapping[str, Any]) -> str:
    scenario = task.get("user_scenario")
    if not isinstance(scenario, Mapping):
        raise TypeError("tau2_user_scenario_invalid")
    instructions = scenario.get("instructions")
    if not isinstance(instructions, Mapping):
        raise TypeError("tau2_user_instructions_invalid")
    parts = tuple(
        value.strip()
        for key in ("task_instructions", "reason_for_call")
        if isinstance((value := instructions.get(key)), str)
        and value.strip()
    )
    if not parts:
        raise ValueError("tau2_original_request_missing")
    return "\n\n".join(parts)


def _registry_sha256(
    *,
    role: str,
    surface: str,
    source_binding_sha256: str,
    rows: Sequence[FitDevSourceRow],
) -> str:
    return _digest(
        {
            "role": role,
            "surface": surface,
            "source_binding_sha256": source_binding_sha256,
            "example_sha256s": tuple(
                row.example_sha256 for row in rows
            ),
        }
    )


def _read_mapping(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label}_not_mapping")
    return payload


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows = tuple(
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if any(not isinstance(row, dict) for row in rows):
        raise TypeError("fit_dev_jsonl_row_invalid")
    return rows


def _require_sha256(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name}_invalid")
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "FIT_DEV_CHECKPOINT_SCHEMA",
    "FIT_DEV_DISPATCH_SCHEMA",
    "FIT_DEV_OUTPUT_SCHEMA",
    "PRODUCTION_WIRED_KINDS",
    "BoundAutoMixTokenizer",
    "FitDevRegistry",
    "FitDevSourceRow",
    "PreparedFitDevJob",
    "StageItemExecutor",
    "StageItemInput",
    "StageItemResult",
    "deterministic_fit_dev",
    "build_openai_compatible_stage_executor",
    "execute_prepared_fit_dev_job",
    "fit_dev_checkpoint_path",
    "inspect_fit_dev_item_checkpoint",
    "load_fit_dev_registry",
    "load_bound_automix_tokenizer",
    "openai_compatible_executor_factory",
    "prepare_fit_dev_job",
    "read_fit_dev_job_output",
]
