"""Cache-neutral agent-side cost readout for PAPC public run artifacts.

One artifact goes in and one row comes out: which surface and cost pool the run
used, how many agent-side tokens the cheap and the strong role spent, what those
tokens cost at the frozen public price snapshot with no cache discount, and how
that total compares with the same-pool official cheap single-run denominator.

Three accounting rules are enforced here and nowhere else:

* Agent side means the cheap role plus the strong role. The benchmark user
  simulator and the benchmark evaluator are harness roles; they never enter the
  total, even when the artifact reports their tokens.
* Cache neutral means the full provider-reported ``input_tokens`` count is
  priced at the ordinary input rate. The cached subset is priced once, at that
  same rate, never at the discounted rate.
* Prices come from :mod:`papc.certspec.costing`, so the repository keeps exactly
  one frozen price table.

A role whose token fields are absent while the role made calls is unavailable,
not zero: reading such an artifact raises :class:`ValueError` naming the file
and the field. A role that made zero calls contributes zero dollars.

Partial token coverage inside a role, meaning a nonzero
``token_usage_unavailable_count``, is reported as observed by the runners and is
summed as given; the reader does not attempt to extrapolate the missing calls.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from papc.certspec.costing import (
    cache_neutral_cost_usd,
    reasoning_tokens_included_in_output,
)

__all__ = [
    "CHEAP_ANCHORS",
    "DENOMINATOR_ROWS",
    "format_latency_table",
    "CheapAnchor",
    "CostReadout",
    "RoleUsage",
    "format_table",
    "read_artifact",
]


@dataclass(frozen=True)
class RoleUsage:
    """Observed provider usage for one agent-side role.

    ``latency_seconds`` is the summed provider latency of that role's calls,
    not wall clock. Wall clock is a property of the concurrency a run happened
    to use, so it says nothing comparable across runs; the summed provider
    latency is what one row would cost in time if it ran alone. Artifacts that
    do not carry it report ``None`` rather than zero, because a role that was
    not timed is unmeasured, not instant.
    """

    calls: int | None
    input_tokens: int
    output_tokens: int
    latency_seconds: float | None = None
    # Separately reported reasoning usage. ``None`` means the artifact carries
    # no reasoning field for this role, which for a vendor that bills reasoning
    # on top of output makes the role's cost a lower bound rather than a number.
    reasoning_tokens: int | None = None


@dataclass(frozen=True)
class CheapAnchor:
    """Official cheap single-run denominator for one surface and cost pool."""

    surface: str
    pool: str
    quality_label: str
    agent_side_usd: float
    source_note: str


@dataclass(frozen=True)
class CostReadout:
    """Recomputed agent-side cost for one run artifact."""

    artifact: str
    surface: str
    pool: str
    cheap_model: str | None
    strong_model: str | None
    quality_label: str | None
    cheap: RoleUsage
    strong: RoleUsage
    cheap_usd: float
    strong_usd: float
    agent_side_usd: float
    anchor_usd: float | None
    multiple: float | None
    within_two_x: bool | None
    reported_agent_side_usd: float | None
    discrepancy_usd: float | None


CHEAP_ANCHORS: Mapping[tuple[str, str], CheapAnchor] = MappingProxyType(
    {
        ("state", "old"): CheapAnchor(
            surface="state",
            pool="old",
            quality_label="79/150",
            agent_side_usd=5.424551,
            source_note="gpt-5.4-mini single run via benchmark run_batch.py",
        ),
        ("state", "new"): CheapAnchor(
            surface="state",
            pool="new",
            quality_label="81/150",
            agent_side_usd=1.1841922,
            source_note=(
                "gpt-5.6-luna single run via benchmark run_batch.py; raw "
                "4616375 input / 217431 output repriced at the 2026-08-28 "
                "public snapshot"
            ),
        ),
        ("retail", "old"): CheapAnchor(
            surface="retail",
            pool="old",
            quality_label="26/40",
            agent_side_usd=2.226476,
            source_note="gpt-5.4-mini test40 single run",
        ),
        ("retail", "new"): CheapAnchor(
            surface="retail",
            pool="new",
            quality_label="27/40",
            agent_side_usd=0.5016936,
            source_note=(
                "gpt-5.6-luna test40 single run; raw 2204640 input / 50638 "
                "output repriced at the 2026-08-28 public snapshot"
            ),
        ),
        ("airline", "old"): CheapAnchor(
            surface="airline",
            pool="old",
            quality_label="32/50",
            agent_side_usd=2.682097,
            source_note=(
                "outputs/papc/..._official_cheap_base50_gpt54mini_high_mc5_"
                "20260701_codex_v1_public.json"
            ),
        ),
        ("airline", "new"): CheapAnchor(
            surface="airline",
            pool="new",
            quality_label="36/50",
            agent_side_usd=0.671334,
            source_note=(
                "gpt-5.6-luna base50 single run; raw 2842224 input / 85741 "
                "output repriced at the 2026-08-28 public snapshot"
            ),
        ),
        ("agent-diff", "old"): CheapAnchor(
            surface="agent-diff",
            pool="old",
            quality_label="39/45",
            agent_side_usd=6.421196,
            source_note=(
                "outputs/papc/agent_diff_hf_test45_cheap_baseline_public_mini_"
                "high_current_20260810_v1.json (2026-08-10 current-code run; the "
                "older $4.949594 offline rescore is a demoted historical snapshot "
                "and must NOT be used)"
            ),
        ),
        ("agent-diff", "new"): CheapAnchor(
            surface="agent-diff",
            pool="new",
            quality_label="28/45",
            agent_side_usd=0.7025854,
            source_note=(
                "gpt-5.6-luna test45 single run; raw 2213381 input / 216591 "
                "output repriced at the 2026-08-28 public snapshot"
            ),
        ),
        ("toolsandbox", "old"): CheapAnchor(
            surface="toolsandbox",
            pool="old",
            quality_label="strict 34",
            agent_side_usd=2.310910,
            source_note=(
                "outputs/papc/toolsandbox_cheap_denominator_remeasure129_"
                "20260816_v1_exec.json (2026-08-16 re-measurement of the same "
                "cheap model on the same 129 rows with current code). The "
                "2026-07-01 anchor it replaces, "
                "toolsandbox_base129_baseline_accounting_cheap_full_20260701_"
                "codex_gpt54user_v1.json at $2.0060655 and strict 31, is stale "
                "rather than wrong: it consumed 651 input tokens per call where "
                "the same model on the same rows now consumes 968. The staleness "
                "was established by matching on row ordinal rather than by "
                "argument, running the fresh cheap arm on ordinals 0 to 19 and "
                "comparing per-call input against the method arm on the same "
                "twenty rows, 800 against 806, agreeing to 0.75 percent while "
                "the frozen anchor sits 20 percent below both. Note the "
                "replacement also raises the denominator's own quality from 31 "
                "to 34, which narrows the method's reported quality gain."
            ),
        ),
        ("toolsandbox", "new"): CheapAnchor(
            surface="toolsandbox",
            pool="new",
            quality_label="strict 34 / 0.666445",
            agent_side_usd=0.3206706,
            source_note=(
                "gpt-5.6-luna base129 single run; raw 787623 input / 135955 "
                "output repriced at the 2026-08-28 public snapshot"
            ),
        ),
    }
)

# Pool membership is a property of the model pair, not of the run name. The
# cheap side decides; the strong side is only consulted when the artifact hides
# its cheap model, and the file name is only consulted when the artifact hides
# both.
_CHEAP_MODEL_POOL = {
    "gpt-5.4-mini": "old",
    "gpt-5.6-luna": "new",
    "claude-sonnet-5": "sonnet5",
    "claude-opus-5": "opus5",
    "gemini-3.7-flash": "gemini37",
    "grok-4.6": "grok46",
    "deepseek-v4-flash": "deepseekv4",
}
_STRONG_MODEL_POOL = {
    "gpt-5.5": "old",
    "gpt-5.6-sol": "new",
    "deepseek-v4-pro": "deepseekv4",
}
_POOL_MODELS = {
    "old": {"cheap": "gpt-5.4-mini", "strong": "gpt-5.5"},
    "new": {"cheap": "gpt-5.6-luna", "strong": "gpt-5.6-sol"},
    "sonnet5": {"cheap": "claude-sonnet-5", "strong": None},
    "opus5": {"cheap": "claude-opus-5", "strong": None},
    "gemini37": {"cheap": "gemini-3.7-flash", "strong": None},
    "grok46": {"cheap": "grok-4.6", "strong": None},
    "deepseekv4": {
        "cheap": "deepseek-v4-flash",
        "strong": "deepseek-v4-pro",
    },
}
_FILENAME_POOL_TOKENS = {
    "old": ("g54m", "gpt54mini", "gpt-5.4-mini", "54mini", "oldpool"),
    "new": ("luna", "gpt56luna", "gpt-5.6-luna", "newpool"),
    "sonnet5": ("sonnet5", "claude-sonnet-5"),
    "opus5": ("opus5", "claude-opus-5"),
    "gemini37": ("gemini37", "gemini-3.7-flash"),
    "grok46": ("grok46", "grok-4.6"),
    "deepseekv4": ("deepseekv4", "deepseek-v4-flash", "deepseek-v4-pro"),
}

# The only two roles that are agent side. Every other role an artifact reports,
# including user_simulator, evaluator, judge and router, is harness cost and is
# never read, so it can never reach a total by accident.
_AGENT_ROLE_KEYS = ("cheap", "strong")

# The successor is named for its job in some artifacts and for its tier in
# others. Both name the same agent-side role, and refusing to read one of them
# is what left a registered figure pointing at a markdown file instead of the
# run that produced it.
_ROLE_ALIASES: Mapping[str, tuple[str, ...]] = {
    "strong": ("residual_strong",),
}

_CHEAP_MODEL_PATHS = (
    "models.cheap",
    "models.cheap_model",
    "model_plan.cheap_model",
    "gate_bindings.config.cheap_model",
)
_STRONG_MODEL_PATHS = (
    "models.strong",
    "models.strong_model",
    "model_plan.residual_strong_model",
    "model_plan.strong_model",
    "gate_bindings.config.strong_model",
)
_SOLO_MODEL_PATHS = ("models.agent", "model_plan.agent_model", "model_plan.model")
_SOLO_ROLE_PATHS = (
    "models.agent_role",
    "model_plan.agent_role",
    "model_plan.method_role",
    "arm",
    "baseline_arm",
)

# Role-keyed ledgers, most specific container first. Every entry maps a dotted
# path to a mapping that carries a "cheap" and a "strong" block.
_LEDGER_ROOTS = (
    "summary.runtime_ledger.per_role",
    "resource_accounting.method_resources.runtime_ledger.per_role",
    "resource_accounting.runtime_ledger.per_role",
    "resource_accounting.method_resources.runtime_breakdown",
    "resource_accounting.runtime_breakdown",
    "runtime_ledger.per_role",
    "summary.runtime_breakdown",
    # Offline reducers that rebuild one arm's ledger publish it at the top
    # level, with the model named inside each role block.
    "per_role",
    # The ToolSandbox shard aggregate names its roles by job rather than tier.
    "runtime",
)

# The native single-agent tau2 runner publishes its only arm as ``arm`` rather
# than wrapping it in an ``arms`` mapping. Keep that public shape small and
# teach the readout to consume it directly.
_SINGLE_ARM_LEDGER_ROOT = "arm.runtime_ledger.per_role"

_REPORTED_COST_PATHS = (
    "agent_side_raw_cost_usd",
    "cost_accounting.agent.cache_neutral.total_cost_usd",
    "summary.runtime_breakdown.primary_resource_comparison.roles.agent_side.cost_usd",
    "resource_accounting.method_resources.runtime_breakdown.agent_cache_neutral_cost"
    ".agent_side_total_cost_usd",
    "resource_accounting.runtime_breakdown.agent_cache_neutral_cost"
    ".agent_side_total_cost_usd",
    "paper_acceptance_gate.agent_side_cache_neutral_cost_usd",
    "agent_side_cache_neutral_cost_usd",
)
_REPORTED_COST_ARM_PATHS = (
    "agent_side_raw_cost_usd",
    "agent_side_cost_accounting.cache_neutral_cost_usd",
)

_USAGE_SHAPES_HELP = (
    "cost_accounting.agent.{logical_input_tokens,output_tokens}, "
    "model_resource_summary.per_arm.<arm>.{cheap,strong}, "
    "arms.<arm>.runtime_ledger.per_role.{cheap,strong}, "
    "arm.runtime_ledger.per_role.{cheap,strong}, "
    "summary.runtime_ledger.per_role.{cheap,strong}, "
    "per_role.{cheap,strong}, "
    "resource_accounting[.method_resources].runtime_ledger.per_role.{cheap,strong}, "
    "agent_resource.{cheap_usage,strong_usage}, "
    "agent_resource.usage"
)


def read_artifact(path: str | Path) -> CostReadout:
    """Recompute the cache-neutral agent-side cost of one run artifact.

    The artifact is never modified and no provider call is made. Raises
    ``ValueError`` when the file carries no usable cheap or strong usage, or
    when a role that made calls reports no token counts.
    """

    artifact = str(path)
    doc = _load(artifact)
    surface = _detect_surface(doc, artifact)
    cheap_model, strong_model, solo_role = _resolve_models(doc)
    pool = _detect_pool(cheap_model, strong_model, artifact)
    # The tau2 public reports redact every model id, so a role that the artifact
    # does not name is priced at the pool it belongs to. Pool membership is the
    # model pair by definition, so this fills a redacted name rather than
    # guessing an unknown one, and it only happens once the pool is known.
    if pool in _POOL_MODELS:
        cheap_model = cheap_model or _POOL_MODELS[pool]["cheap"]
        strong_model = strong_model or _POOL_MODELS[pool]["strong"]
    cheap, strong, arm = _extract_usage(doc, artifact, solo_role)

    cheap_usd = _price(cheap, cheap_model, artifact, "cheap")
    strong_usd = _price(strong, strong_model, artifact, "strong")
    agent_side_usd = cheap_usd + strong_usd

    anchor = CHEAP_ANCHORS.get((surface, pool))
    anchor_usd = anchor.agent_side_usd if anchor is not None else None
    multiple = agent_side_usd / anchor_usd if anchor_usd else None
    within_two_x = multiple <= 2.0 if multiple is not None else None

    reported = _reported_cost(doc, arm)
    discrepancy = agent_side_usd - reported if reported is not None else None

    return CostReadout(
        artifact=artifact,
        surface=surface,
        pool=pool,
        cheap_model=cheap_model,
        strong_model=strong_model,
        quality_label=_quality_label(doc, surface, arm),
        cheap=cheap,
        strong=strong,
        cheap_usd=cheap_usd,
        strong_usd=strong_usd,
        agent_side_usd=agent_side_usd,
        anchor_usd=anchor_usd,
        multiple=multiple,
        within_two_x=within_two_x,
        reported_agent_side_usd=reported,
        discrepancy_usd=discrepancy,
    )


DENOMINATOR_ROWS: Mapping[str, int] = MappingProxyType(
    {
        "state": 150,
        "retail": 40,
        "airline": 50,
        "agent-diff": 45,
        "toolsandbox": 129,
    }
)


def format_latency_table(readouts: Sequence[CostReadout]) -> str:
    """Render summed provider latency per role, and per denominator row.

    Wall clock is omitted on purpose: it is a function of the concurrency the
    run happened to use. Summed provider latency divided by the denominator is
    the comparable quantity, namely what one row costs in time when nothing
    else is running.
    """

    header = (
        "artifact",
        "surface",
        "pool",
        "cheap calls",
        "cheap s",
        "strong calls",
        "strong s",
        "agent s",
        "rows",
        "s/row",
    )
    rows = [header]
    for readout in readouts:
        cheap_s = readout.cheap.latency_seconds
        strong_s = readout.strong.latency_seconds
        total = None
        if cheap_s is not None or strong_s is not None:
            total = (cheap_s or 0.0) + (strong_s or 0.0)
        denominator = DENOMINATOR_ROWS.get(readout.surface)
        rows.append(
            (
                readout.artifact,
                readout.surface,
                readout.pool,
                "-" if readout.cheap.calls is None else str(readout.cheap.calls),
                "-" if cheap_s is None else f"{cheap_s:.1f}",
                "-" if readout.strong.calls is None else str(readout.strong.calls),
                "-" if strong_s is None else f"{strong_s:.1f}",
                "-" if total is None else f"{total:.1f}",
                "-" if denominator is None else str(denominator),
                "-"
                if total is None or not denominator
                else f"{total / denominator:.1f}",
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    lines = [_format_row(rows[0], widths), "  ".join("-" * w for w in widths)]
    lines.extend(_format_row(row, widths) for row in rows[1:])
    return "\n".join(lines)


def format_table(readouts: Sequence[CostReadout]) -> str:
    """Render readouts as one aligned text table."""

    header = (
        "artifact",
        "surface",
        "pool",
        "quality",
        "cheap $",
        "strong $",
        "total $",
        "anchor $",
        "x anchor",
        "<= 2x",
    )
    rows = [header]
    for readout in readouts:
        rows.append(
            (
                readout.artifact,
                readout.surface,
                readout.pool,
                readout.quality_label or "-",
                f"{readout.cheap_usd:.6f}",
                f"{readout.strong_usd:.6f}",
                f"{readout.agent_side_usd:.6f}",
                "-" if readout.anchor_usd is None else f"{readout.anchor_usd:.6f}",
                "-" if readout.multiple is None else f"{readout.multiple:.3f}x",
                _two_x_mark(readout.within_two_x),
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    lines = [_format_row(rows[0], widths), "  ".join("-" * w for w in widths)]
    lines.extend(_format_row(row, widths) for row in rows[1:])
    return "\n".join(lines)


def _format_row(row: Sequence[str], widths: Sequence[int]) -> str:
    cells = [row[0].ljust(widths[0])]
    cells.extend(cell.ljust(widths[i]) for i, cell in enumerate(row[1:4], start=1))
    cells.extend(cell.rjust(widths[i]) for i, cell in enumerate(row[4:], start=4))
    return "  ".join(cells).rstrip()


def _two_x_mark(within_two_x: bool | None) -> str:
    if within_two_x is None:
        return "?"
    return "PASS" if within_two_x else "FAIL"


def _load(artifact: str) -> Mapping[str, Any]:
    file_path = Path(artifact)
    if not file_path.is_file():
        raise ValueError(f"{artifact}: no such artifact file")
    with file_path.open(encoding="utf-8") as handle:
        doc = json.load(handle)
    if not isinstance(doc, Mapping):
        raise ValueError(f"{artifact}: artifact root is not a JSON object")
    return doc


def _detect_surface(doc: Mapping[str, Any], artifact: str) -> str:
    schema = _text(doc.get("schema")) or ""
    surface_field = _text(doc.get("surface")) or ""
    domain = (_text(doc.get("domain")) or "").lower()
    haystack = f"{schema} {surface_field}".lower()
    if "agent_diff" in haystack or "agentdiff" in haystack:
        return "agent-diff"
    if "toolsandbox" in haystack:
        return "toolsandbox"
    if "state_bench" in haystack:
        return "state"
    if "airline" in haystack or domain == "airline":
        return "airline"
    if "retail" in haystack or domain == "retail":
        return "retail"
    name = Path(artifact).name.lower()
    for token, surface in (
        ("agent_diff", "agent-diff"),
        ("toolsandbox", "toolsandbox"),
        ("state_bench", "state"),
        ("airline", "airline"),
        ("retail", "retail"),
    ):
        if token in name:
            return surface
    raise ValueError(
        f"{artifact}: cannot tell which surface produced this artifact "
        f"(schema={schema!r}, surface={surface_field!r}, domain={domain!r})"
    )


def _resolve_models(
    doc: Mapping[str, Any],
) -> tuple[str | None, str | None, str | None]:
    cheap = _first_text(doc, _CHEAP_MODEL_PATHS)
    strong = _first_text(doc, _STRONG_MODEL_PATHS)
    if _text(_get(doc, "frontier_policy.arm")) == "cheap_anchor" and cheap:
        # STATE keeps a residual model in its generic launch plan even though
        # this arm mechanically disables every strong call. It is a measured
        # single-agent denominator, so do not invent a second model role.
        return cheap, None, "cheap"
    if cheap is not None or strong is not None:
        return cheap, strong, None
    solo_model = _first_text(doc, _SOLO_MODEL_PATHS)
    if solo_model is None:
        return None, None, None
    solo_role = _solo_role(doc, solo_model)
    if solo_role == "strong":
        return None, solo_model, "strong"
    return solo_model, None, "cheap"


def _solo_role(doc: Mapping[str, Any], model: str) -> str:
    hint = " ".join(
        text for path in _SOLO_ROLE_PATHS if (text := _text(_get(doc, path)))
    ).lower()
    if "cheap" in hint:
        return "cheap"
    if "strong" in hint:
        return "strong"
    return "cheap" if _normalize_model(model) in _CHEAP_MODEL_POOL else "strong"


def _detect_pool(
    cheap_model: str | None, strong_model: str | None, artifact: str
) -> str:
    """Name the pool only when every model present agrees on one.

    A named pool is a *pair*, and its denominator is the cheap single run of
    that pair. Deciding the pool from whichever role happened to match first
    would price an exploratory pairing such as ``gpt-5.4-mini`` with
    ``gpt-5.4`` against the old pool's denominator and report a multiple
    against an arm that was never run. A model outside both named arms, or two
    roles pointing at different arms, therefore leaves the pool unknown and the
    multiple unreported.
    """

    votes: list[str] = []
    for model, table in (
        (cheap_model, _CHEAP_MODEL_POOL),
        (strong_model, _STRONG_MODEL_POOL),
    ):
        if not model:
            continue
        pool = _pool_for_model(model, table)
        if pool is None:
            return "unknown"
        votes.append(pool)
    if votes:
        return votes[0] if len(set(votes)) == 1 else "unknown"
    name = Path(artifact).name.lower()
    matched = {
        pool
        for pool, tokens in _FILENAME_POOL_TOKENS.items()
        if any(token in name for token in tokens)
    }
    if len(matched) == 1:
        return matched.pop()
    return "unknown"


def _pool_for_model(model: str | None, table: Mapping[str, str]) -> str | None:
    if not model:
        return None
    normalized = _normalize_model(model)
    if normalized in table:
        return table[normalized]
    for known, pool in table.items():
        if normalized.startswith(known):
            return pool
    return None


def _normalize_model(model: str) -> str:
    return model.rsplit("/", 1)[-1].strip().lower()


def _extract_usage(
    doc: Mapping[str, Any], artifact: str, solo_role: str | None
) -> tuple[RoleUsage, RoleUsage, str | None]:
    claude_official = _claude_official_usage(doc, artifact, solo_role)
    if claude_official is not None:
        return (*claude_official, _text(doc.get("baseline_arm")))

    arm_name, roles = _per_arm_roles(doc, artifact)
    if roles is not None:
        return (
            _role_usage(roles, "cheap", artifact, f"model_resource_summary.per_arm.{arm_name}"),
            _role_usage(roles, "strong", artifact, f"model_resource_summary.per_arm.{arm_name}"),
            arm_name,
        )

    arm_name, roles = _arm_ledger_roles(doc, artifact)
    if roles is not None:
        where = f"arms.{arm_name}.runtime_ledger.per_role"
        return (
            _role_usage(roles, "cheap", artifact, where),
            _role_usage(roles, "strong", artifact, where),
            arm_name,
        )

    roles = _get(doc, _SINGLE_ARM_LEDGER_ROOT)
    if _is_role_mapping(roles):
        arm_name = _text(_get(doc, "arm.arm")) or _text(doc.get("arm_name"))
        return (
            _role_usage(roles, "cheap", artifact, _SINGLE_ARM_LEDGER_ROOT),
            _role_usage(roles, "strong", artifact, _SINGLE_ARM_LEDGER_ROOT),
            arm_name,
        )

    for root in _LEDGER_ROOTS:
        candidate = _get(doc, root)
        if _is_role_mapping(candidate):
            return (
                _role_usage(candidate, "cheap", artifact, root),
                _role_usage(candidate, "strong", artifact, root),
                None,
            )

    agent_resource = _get(doc, "agent_resource")
    if isinstance(agent_resource, Mapping):
        return (*_agent_resource_usage(agent_resource, artifact, solo_role), None)

    raise ValueError(
        f"{artifact}: no agent-side cheap/strong usage found; expected one of: "
        f"{_USAGE_SHAPES_HELP}"
    )


def _claude_official_usage(
    doc: Mapping[str, Any], artifact: str, solo_role: str | None
) -> tuple[RoleUsage, RoleUsage] | None:
    """Read logical input from the subscription runner's cache ledger.

    The Claude SDK reports uncached, cache-write, and cache-read input
    separately.  The generic tau2 ledger retains only the uncached fragment,
    while ``cost_accounting.agent.logical_input_tokens`` is their full logical
    sum.  Cache-neutral accounting must price that full sum at the ordinary
    input rate, not silently treat cache reads as free.
    """

    if _text(doc.get("schema")) != "papc.tau2.claude_official_baseline_public.v1":
        return None
    block = _get(doc, "cost_accounting.agent")
    if not isinstance(block, Mapping):
        return None
    calls = _as_int(block.get("calls"))
    input_tokens = _as_int(block.get("logical_input_tokens"))
    output_tokens = _as_int(block.get("output_tokens"))
    if input_tokens is None or output_tokens is None:
        missing = [
            field
            for field, value in (
                ("logical_input_tokens", input_tokens),
                ("output_tokens", output_tokens),
            )
            if value is None
        ]
        raise ValueError(
            f"{artifact}: cost_accounting.agent.{'/'.join(missing)} is unavailable "
            f"(calls={calls!r}); a missing token field is unavailable, not zero"
        )
    role = "strong" if solo_role == "strong" else "cheap"
    arm = _text(doc.get("baseline_arm"))
    latency = _as_float(
        _get(doc, f"arms.{arm}.runtime_ledger.per_role.{role}.latency_seconds")
        if arm is not None
        else None
    )
    usage = RoleUsage(
        calls=calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_seconds=latency,
    )
    empty = RoleUsage(calls=0, input_tokens=0, output_tokens=0)
    return (empty, usage) if role == "strong" else (usage, empty)


def _per_arm_roles(
    doc: Mapping[str, Any], artifact: str
) -> tuple[str | None, Mapping[str, Any] | None]:
    per_arm = _get(doc, "model_resource_summary.per_arm")
    if not isinstance(per_arm, Mapping) or not per_arm:
        return None, None
    arm = _select_arm(
        {name: block for name, block in per_arm.items() if _is_role_mapping(block)},
        artifact,
        "model_resource_summary.per_arm",
    )
    if arm is None:
        return None, None
    return arm, per_arm[arm]


def _arm_ledger_roles(
    doc: Mapping[str, Any], artifact: str
) -> tuple[str | None, Mapping[str, Any] | None]:
    arms = _get(doc, "arms")
    if not isinstance(arms, Mapping) or not arms:
        return None, None
    candidates: dict[str, Any] = {}
    for name, block in arms.items():
        if not isinstance(block, Mapping):
            continue
        per_role = _get(block, "runtime_ledger.per_role")
        if _is_role_mapping(per_role):
            candidates[name] = per_role
    arm = _select_arm(candidates, artifact, "arms.<arm>.runtime_ledger.per_role")
    if arm is None:
        return None, None
    return arm, candidates[arm]


def _select_arm(
    candidates: Mapping[str, Any], artifact: str, where: str
) -> str | None:
    if not candidates:
        return None
    if len(candidates) == 1:
        return next(iter(candidates))
    preferred = [
        name
        for name in candidates
        if any(token in name.lower() for token in ("method", "cfrc"))
    ]
    if len(preferred) == 1:
        return preferred[0]
    raise ValueError(
        f"{artifact}: {where} carries {len(candidates)} arms "
        f"({sorted(candidates)}) and none is the single method arm; "
        "split the artifact or read the arm directly"
    )


def _is_role_mapping(candidate: Any) -> bool:
    return isinstance(candidate, Mapping) and any(
        isinstance(candidate.get(role), Mapping) for role in _AGENT_ROLE_KEYS
    )


def _role_usage(
    roles: Mapping[str, Any], role: str, artifact: str, where: str
) -> RoleUsage:
    block = roles.get(role)
    name = role
    if not isinstance(block, Mapping):
        for alias in _ROLE_ALIASES.get(role, ()):
            candidate = roles.get(alias)
            if isinstance(candidate, Mapping):
                block, name = candidate, alias
                break
    if not isinstance(block, Mapping):
        raise ValueError(
            f"{artifact}: {where}.{role} is missing; an absent agent-side role is "
            "unavailable, not zero"
        )
    return _usage_from_block(block, artifact, f"{where}.{name}")


def _usage_from_block(
    block: Mapping[str, Any], artifact: str, where: str
) -> RoleUsage:
    calls = _as_int(block.get("calls"))
    input_tokens = _as_int(block.get("input_tokens"))
    output_tokens = _as_int(block.get("output_tokens"))
    latency = _as_float(
        block.get("latency_seconds")
        if block.get("latency_seconds") is not None
        else block.get("latency_seconds_total")
    )
    if input_tokens is None or output_tokens is None:
        if calls == 0:
            return RoleUsage(
                calls=0, input_tokens=0, output_tokens=0, latency_seconds=latency
            )
        missing = [
            field
            for field, value in (
                ("input_tokens", input_tokens),
                ("output_tokens", output_tokens),
            )
            if value is None
        ]
        raise ValueError(
            f"{artifact}: {where}.{'/'.join(missing)} is unavailable "
            f"(calls={calls!r}); a missing token field is unavailable, not zero"
        )
    return RoleUsage(
        calls=calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_seconds=latency,
        reasoning_tokens=_as_int(block.get("reasoning_tokens")),
    )


def _agent_resource_usage(
    agent_resource: Mapping[str, Any], artifact: str, solo_role: str | None
) -> tuple[RoleUsage, RoleUsage]:
    cheap_block = agent_resource.get("cheap_usage")
    strong_block = agent_resource.get("strong_usage")
    if isinstance(cheap_block, Mapping) or isinstance(strong_block, Mapping):
        return (
            _paired_agent_usage(
                agent_resource, "cheap_usage", "cheap_provider_calls", artifact
            ),
            _paired_agent_usage(
                agent_resource, "strong_usage", "strong_provider_calls", artifact
            ),
        )
    solo_block = agent_resource.get("usage")
    if isinstance(solo_block, Mapping):
        # A single-agent baseline artifact structurally has one role. The other
        # role is absent because it never existed in that run, so it is zero.
        usage = _usage_from_block(solo_block, artifact, "agent_resource.usage")
        calls = _as_int(agent_resource.get("provider_calls"))
        usage = RoleUsage(
            calls=usage.calls if usage.calls is not None else calls,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            latency_seconds=(
                usage.latency_seconds
                if usage.latency_seconds is not None
                else _as_float(agent_resource.get("model_latency_seconds"))
            ),
        )
        empty = RoleUsage(calls=0, input_tokens=0, output_tokens=0)
        if solo_role == "strong":
            return empty, usage
        return usage, empty
    raise ValueError(
        f"{artifact}: agent_resource carries neither "
        "{cheap_usage,strong_usage} nor usage"
    )


def _paired_agent_usage(
    agent_resource: Mapping[str, Any], usage_key: str, calls_key: str, artifact: str
) -> RoleUsage:
    block = agent_resource.get(usage_key)
    calls = _as_int(agent_resource.get(calls_key))
    if not isinstance(block, Mapping):
        if calls == 0:
            return RoleUsage(calls=0, input_tokens=0, output_tokens=0)
        raise ValueError(
            f"{artifact}: agent_resource.{usage_key} is unavailable "
            f"({calls_key}={calls!r}); a missing role is unavailable, not zero"
        )
    usage = _usage_from_block(block, artifact, f"agent_resource.{usage_key}")
    # The runner keeps this role's latency beside the usage block rather than
    # inside it, under the role's own name.
    sidecar = _as_float(
        agent_resource.get(usage_key.replace("_usage", "_model_latency_seconds"))
    )
    return RoleUsage(
        calls=usage.calls if usage.calls is not None else calls,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        latency_seconds=(
            usage.latency_seconds if usage.latency_seconds is not None else sidecar
        ),
        reasoning_tokens=usage.reasoning_tokens,
    )


def _price(
    usage: RoleUsage, model: str | None, artifact: str, role: str
) -> float:
    if usage.input_tokens == 0 and usage.output_tokens == 0:
        return 0.0
    if not model:
        raise ValueError(
            f"{artifact}: the {role} role spent tokens but the artifact carries "
            "no model id for it, so it cannot be priced"
        )
    return cache_neutral_cost_usd(
        model=model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        reasoning_tokens=usage.reasoning_tokens,
    )


def reasoning_pricing_status(
    usage: RoleUsage, model: str | None
) -> dict[str, Any]:
    """Report whether this role's priced cost covers its reasoning usage."""

    included = reasoning_tokens_included_in_output(model) if model else None
    spent = usage.input_tokens > 0 or usage.output_tokens > 0
    return {
        "role_model": model,
        "reasoning_tokens": usage.reasoning_tokens,
        "reasoning_included_in_output": included,
        "reasoning_usage_missing": (
            spent and included is False and usage.reasoning_tokens is None
        ),
    }


def _reported_cost(doc: Mapping[str, Any], arm: str | None) -> float | None:
    for path in _REPORTED_COST_PATHS:
        value = _as_float(_get(doc, path))
        if value is not None:
            return value
    if arm:
        arm_block = _selected_arm_block(doc, arm)
        if isinstance(arm_block, Mapping):
            for path in _REPORTED_COST_ARM_PATHS:
                value = _as_float(_get(arm_block, path))
                if value is not None:
                    return value
    return None


def _quality_label(
    doc: Mapping[str, Any], surface: str, arm: str | None
) -> str | None:
    if surface == "agent-diff":
        passed = _as_int(_get(doc, "aggregate.passed"))
        rows = _as_int(_get(doc, "aggregate.evaluated")) or _as_int(
            _get(doc, "aggregate.rows")
        )
        if passed is not None and rows:
            return f"{passed}/{rows}"
        return None
    if surface == "toolsandbox":
        strict = _as_int(_get(doc, "summary.strict_complete_count"))
        if strict is None:
            strict = _as_int(_get(doc, "summary.milestone_complete_count"))
        similarity = _as_float(_get(doc, "summary.official_avg_similarity_score"))
        if strict is None:
            return None
        if similarity is None:
            return f"strict {strict}"
        # Both numbers are counts of different things, so the second one says
        # which it is rather than leaving the reader to infer it from scale.
        return f"strict {strict} / similarity {similarity:.6f}"
    if surface == "state":
        # STATE execution artifacts carry no official outcome: task completion
        # is produced by a separate scoring pass over the private trajectories.
        return None
    reward = _as_float(doc.get("reward_sum"))
    denominator = _as_int(doc.get("reward_evaluable_count"))
    if (reward is None or denominator is None) and arm:
        arm_block = _selected_arm_block(doc, arm)
        if isinstance(arm_block, Mapping):
            reward = _as_float(arm_block.get("reward_sum")) if reward is None else reward
            if denominator is None:
                denominator = _as_int(arm_block.get("reward_evaluable_count"))
    if reward is None or not denominator:
        return None
    rendered = f"{int(reward)}" if float(reward).is_integer() else f"{reward:.2f}"
    return f"{rendered}/{denominator}"


def _selected_arm_block(
    doc: Mapping[str, Any], arm_name: str
) -> Mapping[str, Any] | None:
    block = _get(doc, f"arms.{arm_name}")
    if isinstance(block, Mapping):
        return block
    single = doc.get("arm")
    if isinstance(single, Mapping):
        name = _text(single.get("arm")) or _text(doc.get("arm_name"))
        if name == arm_name:
            return single
    return None


def _get(doc: Mapping[str, Any], dotted: str) -> Any:
    current: Any = doc
    for key in dotted.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _first_text(doc: Mapping[str, Any], paths: Sequence[str]) -> str | None:
    for path in paths:
        text = _text(_get(doc, path))
        if text:
            return text
    return None


def _text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    return None
