"""Cost pool descriptors and usage summaries for CertSpec runs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, computed_field


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CostPoolSpec(_StrictModel):
    """One model-family/reasoning-effort/provider cost pool."""

    pool_id: str = Field(min_length=1)
    pool_label: str | None = None
    role: str | None = None
    model_family: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    provider: str | None = None
    api_base: str | None = None
    cost: float = Field(default=0.0, ge=0.0)
    latency: float = Field(default=0.0, ge=0.0)
    input_price_per_1m: float | None = Field(default=None, ge=0.0)
    cached_input_price_per_1m: float | None = Field(default=None, ge=0.0)
    output_price_per_1m: float | None = Field(default=None, ge=0.0)
    pricing_source: str | None = None
    pricing_snapshot_date: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CostPoolUsage(_StrictModel):
    """Aggregated observed usage for a cost pool."""

    pool_id: str = Field(min_length=1)
    pool_label: str | None = None
    model_family: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    provider: str | None = None
    api_base: str | None = None
    trajectory_count: int = Field(default=0, ge=0)
    accepted_count: int = Field(default=0, ge=0)
    cost: float = Field(default=0.0, ge=0.0)
    latency: float = Field(default=0.0, ge=0.0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def average_latency(self) -> float | None:
        if not self.trajectory_count:
            return None
        return _round(self.latency / self.trajectory_count)


def build_cost_config(
    *,
    planner_model: str | None,
    cheap_model: str | None,
    evaluator_model: str | None,
    strong_model: str | None,
    cheap_medium_model: str | None = None,
    user_model: str | None = None,
    strong_reasoning_effort: str | None = None,
    cheap_reasoning_effort: str | None = None,
    cheap_medium_reasoning_effort: str | None = None,
    user_reasoning_effort: str | None = None,
    evaluator_reasoning_effort: str | None = None,
    api_base: str | None = None,
    provider: str | None = None,
) -> dict[str, Any]:
    """Build the legacy role cost config plus normalized pool descriptors."""

    cost_config: dict[str, Any] = {
        "planner": {
            "model": planner_model,
            "fixed_cost": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
        },
        "speculator": {"model": cheap_model, "fixed_cost": 1.0},
        # Legacy frontier reports expect a verifier role. In tau2 live runs the
        # CLI ``evaluator_model`` is the benchmark judge, not a PAPC runtime
        # verifier; observed runtime verifier/final-gate costs come from live
        # controller ledgers and agent_cost.
        "verifier": {"model": evaluator_model, "fixed_cost": 0.0},
        "benchmark_evaluator": {
            "model": evaluator_model,
            "fixed_cost": 0.0,
            "excluded_from_method_runtime_cost": True,
        },
        "actor": {"model": strong_model, "fixed_cost": 10.0},
        "selectgate_baseline": {"fixed_cost": 0.0},
        "reasoning_effort": {
            "strong": strong_reasoning_effort,
            "cheap": cheap_reasoning_effort,
            "cheap_medium": cheap_medium_reasoning_effort,
            "user": user_reasoning_effort,
            "evaluator": evaluator_reasoning_effort,
        },
        "cost_accounting": {
            "method_runtime_cost_excludes_benchmark_evaluator": True,
            "benchmark_evaluator_cost_reported_separately": True,
            "benchmark_evaluator_cost_observed_in_standard_tau_results": False,
            "method_runtime_cost_sources": [
                "tau simulation agent_cost",
                "controller ledger final-gate or optional residual-router costs when merged into agent outputs",
            ],
            "evaluation_cost_sources": [
                "benchmark NL assertion evaluator calls, when the runner records them separately",
            ],
        },
        "note": (
            "Relative fixed costs; replace with token-priced config for paper "
            "analysis. Benchmark evaluator cost is not method runtime cost."
        ),
    }
    if user_model:
        cost_config["user"] = {"model": user_model, "fixed_cost": 0.0}
    cost_config["cost_pools"] = [
        pool.model_dump(mode="json")
        for pool in cost_pools_from_config(
            cost_config,
            provider=provider,
            api_base=api_base,
        )
    ]
    cost_config["cost_pool_registry"] = [
        pool.model_dump(mode="json")
        for pool in build_gpt_cost_pool_registry(
            cheap_model=cheap_model,
            cheap_reasoning_effort=cheap_reasoning_effort,
            cheap_medium_model=cheap_medium_model,
            cheap_medium_reasoning_effort=cheap_medium_reasoning_effort,
            strong_model=strong_model,
            strong_reasoning_effort=strong_reasoning_effort,
            evaluator_model=evaluator_model,
            evaluator_reasoning_effort=evaluator_reasoning_effort,
            user_model=user_model,
            user_reasoning_effort=user_reasoning_effort,
            provider=provider,
            api_base=api_base,
        )
    ]
    return cost_config


OPENAI_PUBLIC_PRICING_SNAPSHOT_DATE = "2026-08-28"
OPENAI_PUBLIC_PRICING_SOURCE = "https://developers.openai.com/api/docs/models/compare"
GOOGLE_PUBLIC_PRICING_SNAPSHOT_DATE = "2026-08-20"
GOOGLE_PUBLIC_PRICING_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing"
ANTHROPIC_PUBLIC_PRICING_SNAPSHOT_DATE = "2026-08-20"
ANTHROPIC_PUBLIC_PRICING_SOURCE = (
    "https://platform.claude.com/docs/en/about-claude/pricing"
)
XAI_PUBLIC_PRICING_SNAPSHOT_DATE = "2026-08-20"
XAI_PUBLIC_PRICING_SOURCE = "https://docs.x.ai/developers/pricing"
DEEPSEEK_PUBLIC_PRICING_SNAPSHOT_DATE = "2026-08-22"
DEEPSEEK_PUBLIC_PRICING_SOURCE = (
    "https://api-docs.deepseek.com/quick_start/pricing/"
)
SUPPORTED_REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh")

PUBLIC_OPENAI_GPT_PRICES_PER_1M = {
    # GPT-5.6 slugs are listed explicitly; the bare "gpt-5.6" alias is left out
    # on purpose so an unpinned alias fails loudly instead of silently pricing
    # as another tier.
    "gpt-5.6-sol": {
        "input_price_per_1m": 4.00,
        "cached_input_price_per_1m": 0.40,
        "output_price_per_1m": 20.00,
    },
    "gpt-5.6-terra": {
        "input_price_per_1m": 2.00,
        "cached_input_price_per_1m": 0.20,
        "output_price_per_1m": 12.00,
    },
    # One dated snapshot prices the whole GPT-5.6 pool. Sol's listed rate is a
    # promotion through 2026-11-21; raw usage remains the authority for later
    # repricing. A paid artifact keeps its launch-era dollars for provenance,
    # while current paper/readout costs are rebuilt from that raw usage here.
    "gpt-5.6-luna": {
        "input_price_per_1m": 0.20,
        "cached_input_price_per_1m": 0.02,
        "output_price_per_1m": 1.20,
    },
    "gpt-5.5": {
        "input_price_per_1m": 5.00,
        "cached_input_price_per_1m": 0.50,
        "output_price_per_1m": 30.00,
    },
    "gpt-5.4": {
        "input_price_per_1m": 2.50,
        "cached_input_price_per_1m": 0.25,
        "output_price_per_1m": 15.00,
    },
    "gpt-5.4-mini": {
        "input_price_per_1m": 0.75,
        "cached_input_price_per_1m": 0.075,
        "output_price_per_1m": 4.50,
    },
    "gpt-5.4-nano": {
        "input_price_per_1m": 0.20,
        "cached_input_price_per_1m": 0.02,
        "output_price_per_1m": 1.25,
    },
}

# Standard synchronous prices. Gemini 3.7 Flash's introductory rate is
# published through 2026-12-31 and is limited to prompts at or below 200k
# tokens. Every paid PAPC run must retain per-call usage so model-specific
# pricing conditions can be audited before these rates enter a paper table.
PUBLIC_GOOGLE_GEMINI_PRICES_PER_1M = {
    # The native bridge appends ``-high`` to encode reasoning effort. Prefix
    # matching below therefore binds ``gemini-3.5-flash-high`` to Google's
    # official ``gemini-3.5-flash`` billing model.
    "gemini-3.5-flash": {
        "input_price_per_1m": 1.50,
        "cached_input_price_per_1m": 0.15,
        "output_price_per_1m": 9.00,
    },
    # Google applies the Gemini 3.7 Flash introductory rate to 3.6 Flash
    # through 2026-12-31. The ``-high`` suffix remains a local effort alias.
    "gemini-3.6-flash": {
        "input_price_per_1m": 0.75,
        "cached_input_price_per_1m": 0.075,
        "output_price_per_1m": 3.75,
    },
    "gemini-3.7-flash": {
        "input_price_per_1m": 0.75,
        "cached_input_price_per_1m": 0.075,
        "output_price_per_1m": 3.75,
    },
    "gemini-3.1-pro-preview": {
        "input_price_per_1m": 2.00,
        "cached_input_price_per_1m": 0.20,
        "output_price_per_1m": 12.00,
    },
}

# Sonnet 5's current standard global API rate. The cache read rate is retained
# for audit, but PAPC's primary cost always prices every input token at 2.00.
PUBLIC_ANTHROPIC_CLAUDE_PRICES_PER_1M = {
    "claude-sonnet-5": {
        "input_price_per_1m": 2.00,
        "cached_input_price_per_1m": 0.20,
        "output_price_per_1m": 10.00,
    },
    "claude-opus-5": {
        "input_price_per_1m": 5.00,
        "cached_input_price_per_1m": 0.50,
        "output_price_per_1m": 25.00,
    },
    # The dated id is what the provider answers to, and the lookup matches on a
    # prefix, so this one key covers every point release of Haiku 4.5. Its
    # absence was not a missing number but a missing arm: a run that used Haiku
    # as its prefix model finished all fifty episodes and then raised on the
    # cost step, losing the artifact for work that had already been paid for.
    "claude-haiku-4-5": {
        "input_price_per_1m": 1.00,
        "cached_input_price_per_1m": 0.10,
        "output_price_per_1m": 5.00,
    },
}

# Grok 4.6 short-context prices. xAI switches the entire request to the long
# rate at 200k prompt tokens, so paid artifacts must retain per-call prompt
# usage and fail paper-cost validation if any call crosses that threshold.
PUBLIC_XAI_GROK_PRICES_PER_1M = {
    "grok-4.6": {
        "input_price_per_1m": 2.00,
        "cached_input_price_per_1m": 0.50,
        "output_price_per_1m": 6.00,
    },
}

# Peak cache-miss and output rates are frozen for time-of-day-neutral paper
# accounting. DeepSeek's off-peak rates are half of these values, but both
# tiers retain the same exact 3x Flash-to-Pro price ratio. PAPC prices every
# input token at the cache-miss rate, so cache-hit variation remains excluded.
PUBLIC_DEEPSEEK_PRICES_PER_1M = {
    "deepseek-v4-flash": {
        "input_price_per_1m": 0.44,
        "cached_input_price_per_1m": 0.014,
        "output_price_per_1m": 1.32,
    },
    "deepseek-v4-pro": {
        "input_price_per_1m": 1.32,
        "cached_input_price_per_1m": 0.044,
        "output_price_per_1m": 3.96,
    },
}

PUBLIC_MODEL_PRICES_PER_1M = {
    **PUBLIC_OPENAI_GPT_PRICES_PER_1M,
    **PUBLIC_GOOGLE_GEMINI_PRICES_PER_1M,
    **PUBLIC_ANTHROPIC_CLAUDE_PRICES_PER_1M,
    **PUBLIC_XAI_GROK_PRICES_PER_1M,
    **PUBLIC_DEEPSEEK_PRICES_PER_1M,
}

# Price tables in the order a model id is matched against them, paired with the
# vendor whose reporting convention that table follows.
_VENDOR_PRICE_TABLES = (
    ("openai", PUBLIC_OPENAI_GPT_PRICES_PER_1M),
    ("google", PUBLIC_GOOGLE_GEMINI_PRICES_PER_1M),
    ("anthropic", PUBLIC_ANTHROPIC_CLAUDE_PRICES_PER_1M),
    ("xai", PUBLIC_XAI_GROK_PRICES_PER_1M),
    ("deepseek", PUBLIC_DEEPSEEK_PRICES_PER_1M),
)

# Whether a vendor's reported output/completion token count already contains the
# reasoning tokens it also reports separately.
#
# OpenAI defines ``output_tokens`` as the whole generated count and
# ``output_tokens_details.reasoning_tokens`` as a subset of it; the other
# OpenAI-compatible surfaces here follow the same containment. Google reports
# ``candidatesTokenCount`` and ``thoughtsTokenCount`` as disjoint counts and
# bills both at the output rate, so Gemini usage must have its reasoning added
# back before it is priced. Pricing Gemini output alone understated every
# Gemini arm's agent-side cost until 2026-08-29.
REASONING_TOKENS_INCLUDED_IN_OUTPUT_BY_VENDOR = {
    "openai": True,
    "google": False,
    "anthropic": True,
    "xai": True,
    "deepseek": True,
}


def build_gpt_cost_pool_registry(
    *,
    cheap_model: str | None,
    cheap_reasoning_effort: str | None,
    cheap_medium_model: str | None = None,
    cheap_medium_reasoning_effort: str | None = None,
    strong_model: str | None,
    strong_reasoning_effort: str | None,
    evaluator_model: str | None = None,
    evaluator_reasoning_effort: str | None = None,
    user_model: str | None = None,
    user_reasoning_effort: str | None = None,
    provider: str | None = None,
    api_base: str | None = None,
) -> list[CostPoolSpec]:
    """Return the method-level GPT cost pools used by CTOS controllers.

    Unit prices are a public pricing snapshot. Live benchmark artifacts remain the
    source of truth for observed ``agent_cost``, token use, provider retries,
    and latency.
    """

    resolved_provider = provider or _provider_from_api_base(api_base)
    cheap_medium_model = cheap_medium_model or cheap_model
    cheap_medium_reasoning_effort = cheap_medium_reasoning_effort or "medium"
    cheap_reasoning_effort = cheap_reasoning_effort or "low"
    strong_reasoning_effort = strong_reasoning_effort or "xhigh"
    registry = [
        _registry_pool(
            pool_id="cheap-low",
            pool_label="cheap-low",
            role="speculator",
            model=cheap_model,
            reasoning_effort=cheap_reasoning_effort,
            provider=resolved_provider,
            api_base=api_base,
            purpose="low-cost initial or branch outcome draft",
            controller_policy="not a blanket default; use when the selector predicts cheap exploration is enough",
            default_enabled=False,
        ),
        _registry_pool(
            pool_id="cheap-medium",
            pool_label="cheap-medium",
            role="speculator",
            model=cheap_medium_model,
            reasoning_effort=cheap_medium_reasoning_effort,
            provider=resolved_provider,
            api_base=api_base,
            purpose="default low-cost draft/completion pool",
            controller_policy=(
                "default proposal pool; model family and reasoning effort are "
                "calibration variables, not the answer-level mechanism"
            ),
            default_enabled=True,
        ),
        _registry_pool(
            pool_id="mid-high",
            pool_label="mid-high",
            role="actor",
            model=strong_model,
            reasoning_effort="high",
            provider=resolved_provider,
            api_base=api_base,
            purpose="conditional high-reasoning probe before xhigh repair",
            controller_policy="use only when calibration says high reasoning can close the failure cheaper than xhigh",
            default_enabled=False,
        ),
        _registry_pool(
            pool_id="xhigh-router",
            pool_label="xhigh-router",
            role="actor",
            model=strong_model,
            reasoning_effort=strong_reasoning_effort,
            provider=resolved_provider,
            api_base=api_base,
            purpose="read-heavy controller checkpoint that emits a tiny routing decision",
            controller_policy=(
                "optional calibration/debug residual router; output must be "
                "short structured JSON and cannot solve the task"
            ),
            default_enabled=False,
        ),
        _registry_pool(
            pool_id="strong-repair",
            pool_label="strong-repair",
            role="actor",
            model=strong_model,
            reasoning_effort=strong_reasoning_effort,
            provider=resolved_provider,
            api_base=api_base,
            purpose="targeted obligation-level certificate completion",
            controller_policy="trigger only for concrete failed obligations with visible evidence and dominated cheaper alternatives",
            default_enabled=False,
        ),
        _registry_pool(
            pool_id="strong-full",
            pool_label="strong-full",
            role="actor",
            model=strong_model,
            reasoning_effort=strong_reasoning_effort,
            provider=resolved_provider,
            api_base=api_base,
            purpose="direct target comparator, not a CTOS default path",
            controller_policy="formal paired baseline only",
            default_enabled=False,
        ),
    ]
    if evaluator_model:
        registry.append(
            _registry_pool(
                pool_id="evaluator",
                pool_label="evaluator",
                role="benchmark_evaluator",
                model=evaluator_model,
                reasoning_effort=evaluator_reasoning_effort,
                provider=resolved_provider,
                api_base=api_base,
                purpose="natural-language assertion evaluator",
                controller_policy=(
                    "benchmark judge only; excluded from PAPC method runtime "
                    "cost and reported as evaluation cost"
                ),
                default_enabled=True,
                excluded_from_method_runtime_cost=True,
            )
        )
    if user_model:
        registry.append(
            _registry_pool(
                pool_id="user-simulator",
                pool_label="user-simulator",
                role="user",
                model=user_model,
                reasoning_effort=user_reasoning_effort,
                provider=resolved_provider,
                api_base=api_base,
                purpose="benchmark user simulator",
                controller_policy="environment role, reported separately from agent budget",
                default_enabled=True,
            )
        )
    return registry


def public_openai_price_for_model(model: str | None) -> dict[str, float] | None:
    """Return a public unit-price snapshot for a registered model family.

    The historical name is retained for callers. New non-OpenAI pools use the
    same cache-neutral accounting function but retain their own official source.
    """

    matched = _matched_vendor_price(model)
    return dict(matched[1]) if matched else None


def _matched_vendor_price(
    model: str | None,
) -> tuple[str, dict[str, float]] | None:
    """Return the ``(vendor, prices)`` pair a model id matches, if any."""

    if not model:
        return None
    normalized = model.rsplit("/", 1)[-1].lower()
    for vendor, table in _VENDOR_PRICE_TABLES:
        for prefix, prices in sorted(
            table.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if normalized.startswith(prefix):
                return vendor, prices
    return None


def public_pricing_vendor_for_model(model: str | None) -> str | None:
    """Return the vendor whose reporting convention a model id follows."""

    matched = _matched_vendor_price(model)
    return matched[0] if matched else None


def reasoning_tokens_included_in_output(model: str | None) -> bool | None:
    """Return whether this model's reported output already covers reasoning.

    ``None`` means the model is not in a frozen price table, so its reporting
    convention has not been declared and its usage cannot be priced here.
    """

    vendor = public_pricing_vendor_for_model(model)
    if vendor is None:
        return None
    return REASONING_TOKENS_INCLUDED_IN_OUTPUT_BY_VENDOR[vendor]


def billable_output_tokens(
    *,
    model: str,
    output_tokens: int,
    reasoning_tokens: int | None,
) -> int:
    """Return the output tokens a vendor actually bills for one usage record.

    ``reasoning_tokens`` of ``None`` means the run never recorded reasoning
    usage. That is not the same as a run that recorded zero: for a vendor whose
    output excludes reasoning, an unrecorded count leaves the true billable
    output unknown, so callers that need a defensible number must check
    :func:`reasoning_usage_required_for_pricing` first rather than read the
    understated figure returned here.
    """

    included = reasoning_tokens_included_in_output(model)
    if included is None:
        raise ValueError(f"no frozen public price for model: {model}")
    if reasoning_tokens is None:
        return output_tokens
    if reasoning_tokens < 0:
        raise ValueError("token counts must be non-negative")
    if included:
        if reasoning_tokens > output_tokens:
            raise ValueError(
                f"{model}: reasoning_tokens={reasoning_tokens} exceeds "
                f"output_tokens={output_tokens}, so this vendor's output "
                "cannot be the superset its convention declares"
            )
        return output_tokens
    return output_tokens + reasoning_tokens


def reasoning_usage_required_for_pricing(model: str | None) -> bool:
    """Return whether pricing this model needs a recorded reasoning count."""

    return reasoning_tokens_included_in_output(model) is False


def public_pricing_metadata_for_model(
    model: str | None,
) -> tuple[str | None, str | None]:
    """Return the official source and snapshot date for a registered model."""

    if not model:
        return None, None
    normalized = model.rsplit("/", 1)[-1].lower()
    if any(
        normalized.startswith(prefix)
        for prefix in PUBLIC_OPENAI_GPT_PRICES_PER_1M
    ):
        return OPENAI_PUBLIC_PRICING_SOURCE, OPENAI_PUBLIC_PRICING_SNAPSHOT_DATE
    if any(
        normalized.startswith(prefix)
        for prefix in PUBLIC_GOOGLE_GEMINI_PRICES_PER_1M
    ):
        return GOOGLE_PUBLIC_PRICING_SOURCE, GOOGLE_PUBLIC_PRICING_SNAPSHOT_DATE
    if any(
        normalized.startswith(prefix)
        for prefix in PUBLIC_ANTHROPIC_CLAUDE_PRICES_PER_1M
    ):
        return (
            ANTHROPIC_PUBLIC_PRICING_SOURCE,
            ANTHROPIC_PUBLIC_PRICING_SNAPSHOT_DATE,
        )
    if any(
        normalized.startswith(prefix)
        for prefix in PUBLIC_XAI_GROK_PRICES_PER_1M
    ):
        return XAI_PUBLIC_PRICING_SOURCE, XAI_PUBLIC_PRICING_SNAPSHOT_DATE
    if any(
        normalized.startswith(prefix)
        for prefix in PUBLIC_DEEPSEEK_PRICES_PER_1M
    ):
        return DEEPSEEK_PUBLIC_PRICING_SOURCE, DEEPSEEK_PUBLIC_PRICING_SNAPSHOT_DATE
    return None, None


def cache_neutral_cost_usd(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int | None = None,
) -> float:
    """Reprice total model usage without a provider cache discount.

    ``input_tokens`` must be the provider-reported total input count, including
    any cached-input subset. Cached input is therefore priced once, at the
    ordinary input rate. This paper-facing estimand removes cache-hit variation
    across request order, concurrency, and provider infrastructure.

    ``reasoning_tokens`` is the separately reported reasoning count. Pass it
    whenever the run recorded one: for a vendor whose output count excludes
    reasoning, omitting it prices only the visible output and understates the
    arm. Callers that must not publish an understated figure should gate on
    :func:`reasoning_usage_required_for_pricing` before calling.
    """

    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts must be non-negative")
    prices = public_openai_price_for_model(model)
    if prices is None:
        raise ValueError(f"no frozen public price for model: {model}")
    billable_output = billable_output_tokens(
        model=model,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
    )
    return (
        input_tokens * prices["input_price_per_1m"]
        + billable_output * prices["output_price_per_1m"]
    ) / 1_000_000


def cache_neutral_cost_breakdown(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int | None = None,
) -> dict[str, Any]:
    """Price one usage record and say whether the result can be published.

    ``reasoning_usage_missing`` marks the case this function exists for: a
    vendor that bills reasoning on top of output, priced from a run that never
    recorded a reasoning count. The returned cost is then a lower bound, and
    the caller must not present it as the arm's agent-side cost.
    """

    included = reasoning_tokens_included_in_output(model)
    if included is None:
        raise ValueError(f"no frozen public price for model: {model}")
    missing = reasoning_tokens is None and not included
    return {
        "model": model,
        "cost_usd": cache_neutral_cost_usd(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
        ),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "billable_output_tokens": billable_output_tokens(
            model=model,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
        ),
        "reasoning_included_in_output": included,
        "reasoning_usage_missing": missing,
        "cost_is_lower_bound": missing,
    }


def _registry_pool(
    *,
    pool_id: str,
    pool_label: str,
    role: str,
    model: str | None,
    reasoning_effort: str | None,
    provider: str | None,
    api_base: str | None,
    purpose: str,
    controller_policy: str,
    default_enabled: bool,
    excluded_from_method_runtime_cost: bool = False,
) -> CostPoolSpec:
    prices = public_openai_price_for_model(model) or {}
    pricing_source, pricing_snapshot_date = public_pricing_metadata_for_model(model)
    return CostPoolSpec(
        pool_id=pool_id,
        pool_label=pool_label,
        role=role,
        model_family=infer_model_family(model),
        model=model,
        reasoning_effort=reasoning_effort,
        provider=provider,
        api_base=api_base,
        input_price_per_1m=prices.get("input_price_per_1m"),
        cached_input_price_per_1m=prices.get("cached_input_price_per_1m"),
        output_price_per_1m=prices.get("output_price_per_1m"),
        pricing_source=pricing_source if prices else None,
        pricing_snapshot_date=pricing_snapshot_date if prices else None,
        metadata={
            "purpose": purpose,
            "controller_policy": controller_policy,
            "default_enabled": default_enabled,
            "excluded_from_method_runtime_cost": excluded_from_method_runtime_cost,
            "supported_reasoning_efforts": list(SUPPORTED_REASONING_EFFORTS),
            "cost_accounting_note": (
                "Unit prices are public pricing snapshots. Observed benchmark "
                "agent_cost, token usage, provider retries, and latency remain "
                "the paper evidence."
            ),
        },
    )


def cost_pools_from_config(
    cost_config: dict[str, Any],
    *,
    provider: str | None = None,
    api_base: str | None = None,
) -> list[CostPoolSpec]:
    """Normalize legacy role-level cost config into pool specs."""

    if isinstance(cost_config.get("cost_pools"), list):
        return [
            CostPoolSpec.model_validate(pool)
            for pool in cost_config["cost_pools"]
            if isinstance(pool, dict)
        ]

    effort_by_tier = cost_config.get("reasoning_effort")
    if not isinstance(effort_by_tier, dict):
        effort_by_tier = {}
    resolved_provider = provider or _provider_from_config(cost_config, api_base)
    pools: list[CostPoolSpec] = []
    for role in ("planner", "speculator", "verifier", "actor", "user"):
        role_config = cost_config.get(role)
        if not isinstance(role_config, dict):
            continue
        model = _optional_str(role_config.get("model"))
        role_api_base = _optional_str(role_config.get("api_base")) or api_base
        role_provider = (
            _optional_str(role_config.get("provider"))
            or resolved_provider
            or _provider_from_api_base(role_api_base)
        )
        tier = _role_tier(role)
        reasoning_effort = _optional_str(
            role_config.get("reasoning_effort") or effort_by_tier.get(tier)
        )
        family = _optional_str(role_config.get("model_family")) or infer_model_family(
            model
        )
        pools.append(
            CostPoolSpec(
                pool_id=_pool_id(
                    role=role,
                    model_family=family,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    provider=role_provider,
                    api_base=role_api_base,
                ),
                pool_label=_pool_label(
                    role=role,
                    model_tier=tier,
                    reasoning_effort=reasoning_effort,
                    metadata={},
                ),
                role=role,
                model_family=family,
                model=model,
                reasoning_effort=reasoning_effort,
                provider=role_provider,
                api_base=role_api_base,
                cost=_role_cost(role_config),
                latency=_non_negative_float(role_config.get("latency"), default=0.0),
            )
        )
    return pools


def summarize_pool_usage(
    records: Iterable[Any] = (),
    cost_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize pool usage from trajectory/certificate metadata.

    If no records are supplied, the summary still exposes configured pools with
    zero usage so downstream analysis can inspect the available cost pool grid.
    """

    configured = {
        pool.pool_id: pool for pool in cost_pools_from_config(cost_config or {})
    }
    rows: dict[str, _UsageAccumulator] = defaultdict(_UsageAccumulator)
    for pool in configured.values():
        rows[pool.pool_id].apply_pool(pool)

    for record in records:
        metadata = _record_metadata(record)
        spec = _pool_from_metadata(metadata, configured)
        rows[spec.pool_id].apply_pool(spec)
        rows[spec.pool_id].trajectory_count += 1
        rows[spec.pool_id].accepted_count += int(_record_accepted(record, metadata))
        rows[spec.pool_id].cost += _metadata_cost(metadata, default=spec.cost)
        rows[spec.pool_id].latency += _metadata_latency(metadata, default=spec.latency)

    usages = [
        accumulator.to_usage(pool_id) for pool_id, accumulator in sorted(rows.items())
    ]
    return {
        "pool_count": len(usages),
        "trajectory_count": sum(row.trajectory_count for row in usages),
        "accepted_count": sum(row.accepted_count for row in usages),
        "total_cost": _round(sum(row.cost for row in usages)),
        "total_latency": _round(sum(row.latency for row in usages)),
        "pools": [row.model_dump(mode="json") for row in usages],
    }


def infer_model_family(model: str | None) -> str | None:
    if not model:
        return None
    if "/" in model:
        model = model.rsplit("/", 1)[1]
    lowered = model.lower()
    for prefix in ("gpt-5", "gpt-4.1", "gpt-4o", "o4-mini", "o3-mini", "o3"):
        if lowered.startswith(prefix):
            return prefix
    return model.split("-", 1)[0] if "-" in model else model


class _UsageAccumulator:
    def __init__(self) -> None:
        self.pool_label: str | None = None
        self.model_family: str | None = None
        self.model: str | None = None
        self.reasoning_effort: str | None = None
        self.provider: str | None = None
        self.api_base: str | None = None
        self.trajectory_count = 0
        self.accepted_count = 0
        self.cost = 0.0
        self.latency = 0.0

    def apply_pool(self, pool: CostPoolSpec) -> None:
        self.pool_label = self.pool_label or pool.pool_label
        self.model_family = self.model_family or pool.model_family
        self.model = self.model or pool.model
        self.reasoning_effort = self.reasoning_effort or pool.reasoning_effort
        self.provider = self.provider or pool.provider
        self.api_base = self.api_base or pool.api_base

    def to_usage(self, pool_id: str) -> CostPoolUsage:
        return CostPoolUsage(
            pool_id=pool_id,
            pool_label=self.pool_label,
            model_family=self.model_family,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            provider=self.provider,
            api_base=self.api_base,
            trajectory_count=self.trajectory_count,
            accepted_count=self.accepted_count,
            cost=_round(self.cost),
            latency=_round(self.latency),
        )


def _pool_from_metadata(
    metadata: dict[str, Any],
    configured: dict[str, CostPoolSpec],
) -> CostPoolSpec:
    pool_id = _optional_str(metadata.get("pool_id") or metadata.get("cost_pool_id"))
    if pool_id and pool_id in configured:
        return configured[pool_id]
    model = _optional_str(metadata.get("model"))
    role = _optional_str(metadata.get("role"))
    model_tier = _optional_str(metadata.get("model_tier"))
    role_key = _role_from_metadata(role, model_tier)
    effort = _optional_str(metadata.get("reasoning_effort"))
    if role_key:
        for pool in configured.values():
            if (
                pool.role == role_key
                and (model is None or pool.model == model)
                and (effort is None or pool.reasoning_effort == effort)
            ):
                return _pool_with_metadata_label(
                    pool,
                    role=role_key,
                    model_tier=model_tier,
                    reasoning_effort=effort or pool.reasoning_effort,
                    metadata=metadata,
                )
    family = _optional_str(metadata.get("model_family")) or infer_model_family(model)
    provider = _optional_str(metadata.get("provider"))
    api_base = _optional_str(metadata.get("api_base"))
    return CostPoolSpec(
        pool_id=pool_id
        or _pool_id(
            role=role_key,
            model_family=family,
            model=model,
            reasoning_effort=effort,
            provider=provider,
            api_base=api_base,
        ),
        pool_label=_pool_label(
            role=role_key,
            model_tier=model_tier,
            reasoning_effort=effort,
            metadata=metadata,
        ),
        role=role_key,
        model_family=family,
        model=model,
        reasoning_effort=effort,
        provider=provider,
        api_base=api_base,
        cost=_metadata_cost(metadata, default=0.0),
        latency=_metadata_latency(metadata, default=0.0),
    )


def _pool_with_metadata_label(
    pool: CostPoolSpec,
    *,
    role: str | None,
    model_tier: str | None,
    reasoning_effort: str | None,
    metadata: dict[str, Any],
) -> CostPoolSpec:
    label = _pool_label(
        role=role,
        model_tier=model_tier,
        reasoning_effort=reasoning_effort,
        metadata=metadata,
    )
    if not label or label == pool.pool_label:
        return pool
    return pool.model_copy(
        update={"pool_id": f"{pool.pool_id}|{_safe_id(label)}", "pool_label": label}
    )


def _role_from_metadata(role: str | None, model_tier: str | None) -> str | None:
    if role in {"actor", "speculator", "verifier", "planner", "user"}:
        return role
    if role in {"strong", "baseline"} or model_tier == "strong":
        return "actor"
    if role == "weak" or model_tier in {"cheap", "cheap_medium"}:
        return "speculator"
    if model_tier == "evaluator":
        return "verifier"
    return None


def _record_metadata(record: Any) -> dict[str, Any]:
    if isinstance(record, dict):
        metadata = record.get("metadata")
        return dict(metadata) if isinstance(metadata, dict) else {}
    metadata = getattr(record, "metadata", None)
    return dict(metadata) if isinstance(metadata, dict) else {}


def _record_accepted(record: Any, metadata: dict[str, Any]) -> bool:
    if isinstance(record, dict) and isinstance(record.get("accepted"), bool):
        return bool(record["accepted"])
    accepted = getattr(record, "accepted", None)
    if isinstance(accepted, bool):
        return accepted
    if isinstance(record, dict):
        reward = record.get("reward", metadata.get("reward", None))
        return isinstance(reward, int | float) and float(reward) > 0.0
    reward = getattr(record, "reward", metadata.get("reward", None))
    return isinstance(reward, int | float) and float(reward) > 0.0


def _metadata_cost(metadata: dict[str, Any], *, default: float) -> float:
    for key in ("cost", "total_cost", "dollar_cost"):
        value = _non_negative_float(metadata.get(key), default=-1.0)
        if value >= 0.0:
            return value
    return default


def _metadata_latency(metadata: dict[str, Any], *, default: float) -> float:
    for key in ("latency", "latency_seconds", "wall_time_seconds"):
        value = _non_negative_float(metadata.get(key), default=-1.0)
        if value >= 0.0:
            return value
    return default


def _provider_from_config(cost_config: dict[str, Any], api_base: str | None) -> str:
    provider = _optional_str(cost_config.get("provider"))
    return provider or _provider_from_api_base(api_base)


def _provider_from_api_base(api_base: str | None) -> str:
    return "openai_compatible" if api_base else "openai"


def _pool_id(
    *,
    role: str | None,
    model_family: str | None,
    model: str | None,
    reasoning_effort: str | None,
    provider: str | None,
    api_base: str | None,
) -> str:
    parts = [
        role or "unknown_role",
        model_family or model or "unknown_model",
        reasoning_effort or "default_effort",
        provider or _provider_from_api_base(api_base),
        api_base or "default_api_base",
    ]
    return "|".join(_safe_id(part) for part in parts)


def _pool_label(
    *,
    role: str | None,
    model_tier: str | None,
    reasoning_effort: str | None,
    metadata: dict[str, Any],
) -> str | None:
    explicit = _optional_str(metadata.get("pool_label"))
    if explicit:
        return explicit
    tier = _optional_str(model_tier) or _role_tier(role or "")
    arm = str(metadata.get("arm") or "").lower()
    if tier in {"cheap", "cheap_medium"}:
        if tier == "cheap_medium":
            return "cheap-medium"
        return f"cheap-{reasoning_effort}" if reasoning_effort else None
    if tier == "strong":
        if "repair" in arm:
            return "strong-repair"
        if arm in {"strong-only", "strong_only"} or "strong-only" in arm:
            return "strong-full"
        if "fallback" in arm:
            return "strong-fallback"
        return f"strong-{reasoning_effort}" if reasoning_effort else None
    if tier in {"evaluator", "verifier"} or role == "verifier":
        return "mid"
    if tier == "user":
        return f"user-{reasoning_effort}" if reasoning_effort else "user"
    return None


def _role_tier(role: str) -> str:
    return {
        "actor": "strong",
        "speculator": "cheap",
        "verifier": "evaluator",
        "user": "user",
        "planner": "planner",
    }.get(role, role)


def _role_cost(role_config: dict[str, Any]) -> float:
    fixed = _non_negative_float(
        role_config.get("fixed_cost", role_config.get("cost", 0.0)),
        default=0.0,
    )
    input_tokens = _non_negative_float(role_config.get("input_tokens"), default=0.0)
    output_tokens = _non_negative_float(role_config.get("output_tokens"), default=0.0)
    input_price = _non_negative_float(
        role_config.get("input_price_per_1m"),
        default=0.0,
    )
    output_price = _non_negative_float(
        role_config.get("output_price_per_1m"),
        default=0.0,
    )
    return _round(
        fixed
        + input_tokens * input_price / 1_000_000.0
        + output_tokens * output_price / 1_000_000.0
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _safe_id(value: str) -> str:
    return value.replace("/", "_").replace(":", "_").replace(" ", "_")


def _non_negative_float(value: Any, *, default: float) -> float:
    if isinstance(value, int | float):
        return max(float(value), 0.0)
    return default


def _round(value: float) -> float:
    return round(float(value), 10)
