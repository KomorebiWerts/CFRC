#!/usr/bin/env python3
"""The incumbent operating point of each surface, as a runnable declaration.

Every paid run in this project is launched by hand-assembling fifteen or more
runner flags. Two kinds of damage follow from that, and both are already in the
record: a flag that silently differs between two shots makes their numbers
incomparable (the Agent-Diff ``adaptive`` versus ``high`` pair, the STATE user
simulator model), and a flag that is simply forgotten costs a launch (the STATE
private output root). This module states each incumbent point once, in code, so
a rerun is a name rather than a transcription.

A point carries four things: the flags that *define* the arm, the same-pool
official cheap single-run denominator its multiple is read against, whatever
result is currently registered for it, and the acceptance floor a rerun has to
clear. Output paths are derived from a run id, never declared, so two runs of
the same point cannot collide or overwrite each other.

Surfaces whose launch layout is not reproduced here declare
``command_supported=False`` rather than guessing: an approximate command is
worse than none, because it would be spent before it was found wrong.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from papc.paths import DATA_ROOT

PUBLIC_OUTPUT_DIR = "outputs/papc"
PRIVATE_OUTPUT_ROOT = f"{DATA_ROOT}/papc"

STATE_APPROVAL_SCOPE = "append_only_private_state_bench_cfrc_method_arm_execution"

# Successor calls per denominator row, below which a point is not measuring the
# method. Set from the observed spread: the eight engaged points sit at 0.96 to
# 2.48, and the one disengaged point sat at 0.10.
MINIMUM_ENGAGEMENT = 0.5

# Cross-pool differences are split by what they can support. Only a reward-blind
# numeric budget remains eligible for a strict one-method comparison. A semantic
# boundary is retained in the operating-point registry so the measured run is
# reproducible, but both cells are excluded from unified paper aggregates. Any
# undeclared difference is still rejected by the test below this registry.
# Points that exist to state a DIFFERENT mechanism rather than to be a pool's
# method arm. They are excluded from the cross-pool difference check by name, not
# by a pattern, so adding one is a deliberate act.
DECLARED_TRADEOFF_POINTS: frozenset[str] = frozenset()


PER_POOL_NUMERIC_BUDGET_PARAMETERS: Mapping[str, str] = {
    "--exact-contract-provider-call-cap": (
        "Proposal samples per exact contract: one on the old pool and two on "
        "the new pool. At the launch-era price snapshot, cap one on the new "
        "pool measured 40/45 at $9.543471 against 43.5/45 at $9.406813, worse "
        "on both axes. The registered parent is repriced below from raw usage; "
        "the integer budget does not alter contract, admission, tools, or "
        "checked execution."
    ),
}


PER_POOL_SEMANTIC_BOUNDARIES: Mapping[str, str] = {
    "--refuse-unclosable-last-slot": (
        "At the same last budget slot and empty candidate, off continues while "
        "on returns a safety block. It therefore changes whether completion or "
        "bottom remains reachable, not only a loop bound. New on is 91/150 at "
        "$4.751875 under the 2026-08-28 price snapshot (the frozen artifact "
        "reports $10.560896); new off is 90/150 and old on is 81/150 against "
        "95/150 off. STATE is excluded from strict unified aggregates."
    ),
    "--residual-recertifies-after-user-turn": (
        "On clears the graph and successor tenure, returns control to cheap, "
        "and requires a new reached-state certification; off keeps the waiting "
        "residual. This changes authority and state transitions. It cuts old "
        "cost by about 14.4 percent, while enabling it on new changed quality "
        "from 44/50 to 39/50. Airline is excluded from unified aggregates."
    ),
}


PER_POOL_PARAMETERS: Mapping[str, str] = {
    **PER_POOL_NUMERIC_BUDGET_PARAMETERS,
    **PER_POOL_SEMANTIC_BOUNDARIES,
}


# Per-surface online concurrency ceilings. Agent-Diff returns HTTP 503 from the
# relay at fourteen and holds at five; the other four surfaces have been run at
# fourteen without incident. A run is still one surface at a time.
ONLINE_CONCURRENCY_CEILING: Mapping[str, int] = {
    "state": 8,
    "airline": 8,
    "retail": 8,
    "agent-diff": 5,
    "toolsandbox": 8,
}


@dataclass(frozen=True)
class RegisteredResult:
    """What this point last measured, and on which artifact.

    ``strong_calls_per_row`` is the guard against reading a cheap point as an
    efficient one. A point whose successor almost never runs is cheap because
    the method is not executing, not because it is executing efficiently, and
    the Retail old pool sat in the table at 1.227x for exactly that reason:
    four successor calls across forty tasks. Any point below
    :data:`MINIMUM_ENGAGEMENT` is reported as a symptom rather than a result.
    """

    quality_label: str
    agent_side_usd: float
    multiple: float
    public_artifact: str
    source_bound_to_head: bool
    strong_calls: int | None = None
    denominator_rows: int | None = None
    note: str = ""
    # Every artifact this point's figure is the mean of. Agent-Diff's old pool
    # spreads $2.60 across same-configuration repeats, so a single shot there
    # reports where the dice landed rather than what the arm costs. When this
    # is set, ``agent_side_usd`` is the mean over these and ``public_artifact``
    # names whichever one sits closest to it.
    sample_artifacts: tuple[str, ...] = ()

    # How many tool effect units the accepted graphs actually executed that the
    # cheap prefix had not already proposed. The call rate alone cannot see the
    # case this exists for: the STATE budget-1 point cleared the rate guard at
    # 0.82 calls per row while admitting zero actions on all 150 rows, because
    # a residual round that emits tool calls exhausts the budget before it can
    # write a closing message and the compiler then discards the whole graph.
    # A successor that is called and refused is not a successor that ran.
    admitted_residual_actions: int | None = None

    @property
    def strong_calls_per_row(self) -> float | None:
        if self.strong_calls is None or not self.denominator_rows:
            return None
        return self.strong_calls / self.denominator_rows

    @property
    def method_engaged(self) -> bool | None:
        # An admitted action settles the question by itself, and the call rate
        # must not be allowed to overrule it. The Retail closure arms route most
        # rows through deterministic transcription and call the successor only
        # where transcription cannot serve, so they sit at 0.13 to 0.43 calls
        # per row by design while executing residual actions the cheap candidate
        # never proposed. Reading that rate as idleness would reject the arm for
        # being efficient. The rate is evidence of nothing when the action count
        # is known; it is the fallback for points measured before anyone counted
        # actions.
        if self.admitted_residual_actions is not None:
            return self.admitted_residual_actions > 0
        rate = self.strong_calls_per_row
        return None if rate is None else rate >= MINIMUM_ENGAGEMENT


@dataclass(frozen=True)
class Acceptance:
    """The floor a rerun of this point has to clear to stay registered.

    The current policy is Pareto-safe: a candidate may not score below or cost
    more than the registered point in its own surface and pool. A measured
    incumbent therefore uses ``hold_incumbent`` even when it is already inside
    the historical two-times budget.
    """

    reward_floor: float | None
    cost_ceiling_usd: float | None
    secondary_quality_floors: tuple[tuple[str, float], ...] = ()
    ceiling_intent: str = "two_x_budget"
    infrastructure_errors_allowed: int = 0

    def __post_init__(self) -> None:
        if self.ceiling_intent not in ("two_x_budget", "hold_incumbent"):
            raise ValueError(f"unknown ceiling intent: {self.ceiling_intent}")


@dataclass(frozen=True)
class OperatingPoint:
    name: str
    surface: str
    pool: str
    model_pool_arm: str
    arm_flags: tuple[str, ...]
    denominator_label: str
    denominator_usd: float
    acceptance: Acceptance
    registered: RegisteredResult | None = None
    command_supported: bool = True
    papc_run_surface: str = ""
    papc_run_mode: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def two_x_budget_usd(self) -> float:
        return 2.0 * self.denominator_usd

    @property
    def concurrency_ceiling(self) -> int:
        return ONLINE_CONCURRENCY_CEILING[self.surface]


def _state_point(
    *,
    name: str,
    pool: str,
    arm: str,
    arm_flags: tuple[str, ...],
    denominator_label: str,
    denominator_usd: float,
    acceptance: Acceptance,
    registered: RegisteredResult | None,
    notes: tuple[str, ...] = (),
) -> OperatingPoint:
    return OperatingPoint(
        name=name,
        surface="state",
        pool=pool,
        model_pool_arm=arm,
        arm_flags=arm_flags,
        denominator_label=denominator_label,
        denominator_usd=denominator_usd,
        acceptance=acceptance,
        registered=registered,
        papc_run_surface="state",
        papc_run_mode="method",
        notes=notes,
    )


STATE_SHARED_FLAGS: tuple[str, ...] = (
    "--split",
    "test",
    "--max-total-rows",
    "150",
    "--max-residual-strong-calls",
    "2",
    "--domain-prompt-suffix",
    "on",
    "--user-simulator-model",
    "gpt-5.5",
    "--user-simulator-reasoning-effort",
    "high",
)

AIRLINE_SHARED_FLAGS: tuple[str, ...] = (
    "--task-split",
    "base",
    "--tasks-per-domain",
    "50",
    "--num-trials",
    "1",
    # The runner requires all four; the pool arm only names the agent pair, and
    # a shot that omits these dies at argparse after the gate is already spent.
    "--user-model",
    "gpt-5.5",
    "--user-reasoning-effort",
    "high",
    "--evaluator-model",
    "gpt-5.5",
    "--evaluator-reasoning-effort",
    "high",
)

RETAIL_SHARED_FLAGS: tuple[str, ...] = (
    "--domain",
    "retail",
    "--domain-split",
    "test",
    "--tasks-per-domain",
    "40",
    "--num-trials",
    "1",
    "--user-model",
    "gpt-5.5",
    "--user-reasoning-effort",
    "high",
    "--evaluator-model",
    "gpt-5.5",
    "--evaluator-reasoning-effort",
    "high",
    "--method-only",
    "--request-contract",
    "--method-mode",
    "frontier",
    "--frontier-timing",
    "pre_write_candidate",
    "--pre-write-residual-policy",
    "first_write_candidate",
    "--compiler-setting",
    "strict_core",
    # The gate binds all four of these and the runner defaults disagree with it
    # on the last two, so a launch that leaves them out is refused after the
    # gate has already been written.
    "--request-contract-private-replica",
    "--strong-call-cap",
    "2",
    "--suffix-tool-call-budget",
    "6",
)


OPERATING_POINTS: Mapping[str, OperatingPoint] = {
    "state-old": _state_point(
        name="state-old",
        pool="old",
        arm="gpt54mini-gpt55-high",
        # Both pools now run the same flags. The one-call variant this point
        # used to carry made STATE the third surface whose two pools differ
        # inside the same surface, and it bought its lower bill by disabling
        # the component the table credits: at one call the round that emits
        # tool calls has nothing left to pay for its closing message, so the
        # compiler discards the residual whole and the arm admitted zero
        # actions across all 150 rows while still clearing the engagement
        # guard at 0.82 calls per row.
        arm_flags=STATE_SHARED_FLAGS,
        denominator_label="79/150",
        denominator_usd=5.424551,
        acceptance=Acceptance(
            reward_floor=95.0,
            cost_ceiling_usd=12.411678,
            ceiling_intent="hold_incumbent",
        ),
        registered=RegisteredResult(
            quality_label="95/150",
            agent_side_usd=12.411678,
            multiple=2.2881,
            public_artifact=(
                "outputs/papc/state_bench_v11_budget2_gpt54mini_gpt55_high"
                "_full150_c10_20260810_v11_exec.json"
            ),
            source_bound_to_head=False,
            strong_calls=242,
            denominator_rows=150,
            admitted_residual_actions=141,
            note=(
                "Replaces the one-call point at 92/150 and $10.797128 = 1.990x. "
                "That point was cheaper and scored what the strong agent alone "
                "scores, but it admitted zero residual actions on all 150 rows: "
                "the executed graphs summed to no tool nodes at all, 100 rows "
                "blocked on strong_budget_exhausted_before_final and the other "
                "23 accepted a message-only graph. Two calls per row costs "
                "$1.614550 more and 0.298x, and buys 141 executed residual "
                "actions on 89 of 150 rows, three task points and a unified "
                "flag set with the other pool. Scored by the official protocol "
                "judge, gpt-5.4 at high effort."
            ),
        ),
        notes=(
            "Lowering the closing call's reasoning effort is measured and "
            "negative here: 85/150 at $11.806457 = 2.176x against 95/150 at "
            "$12.411678. The knob works, and the mechanism is the finding: the "
            "closing message came back the same length, 154 prose tokens "
            "against 188, with the reasoning share cut from 72.1 to 13.5 "
            "percent, and it scored ten points worse. The deliberation behind "
            "that message is doing disclosure work rather than padding, which "
            "is the same thing the cheap-authored repair said at sixteen "
            "points. Both ledger sides moved again: strong fell $1.049 and "
            "cheap rose $0.444.",
            "The last-slot refusal is measured and negative here: 81/150 at "
            "$12.658603 against 95/150 at $12.4117 without it, a fourteen point "
            "drop well outside the 85 to 95 band, so this pool runs without it "
            "while the other pool runs with it. The switch decides what happens "
            "to a residual that cannot be closed inside the remaining budget, "
            "and both settings run all three components; which one is better is "
            "a property of how often the pool's successor reaches that state, "
            "not of the method.",
        ),
    ),
    "state-new": _state_point(
        name="state-new",
        pool="new",
        arm="gpt56-luna-sol-high",
        arm_flags=STATE_SHARED_FLAGS
        + ("--refuse-unclosable-last-slot", "on"),
        denominator_label="81/150",
        denominator_usd=1.1841922,
        acceptance=Acceptance(
            reward_floor=91.0,
            cost_ceiling_usd=4.7518752,
            ceiling_intent="hold_incumbent",
        ),
        registered=RegisteredResult(
            quality_label="91/150",
            agent_side_usd=4.7518752,
            multiple=4.0128,
            public_artifact=(
                "outputs/papc/state_bench_v15_noemptyslot_luna_sol"
                "_full150_c10_20260812_v1_exec.json"
            ),
            source_bound_to_head=False,
            strong_calls=212,
            denominator_rows=150,
            admitted_residual_actions=145,
            note=(
                "Raw usage repriced at the 2026-08-28 public snapshot gives "
                "$4.751875 = 4.0128x; the frozen artifact reports $10.560896 "
                "under its launch-era schedule. The refusal was measured twice "
                "on this pool and retained as a semantic diagnostic, not a "
                "unified aggregate. All three components still run: 115 "
                "accepted graphs carry 145 executed residual actions. The other "
                "pool runs without this switch because there it costs fourteen "
                "task points."
            ),
        ),
        notes=(
            "task_completion swings 85-95 on this benchmark's judge; the stable "
            "axis is state_pass, so read a one-point move as noise.",
            "The launch-era on/off comparison reported an 11.9 percent cost "
            "difference; it is preserved as historical semantic evidence, not "
            "mixed with the current-price absolute registration.",
        ),
    ),
    "airline-old": OperatingPoint(
        name="airline-old",
        surface="airline",
        pool="old",
        model_pool_arm="gpt54mini-gpt55-high",
        arm_flags=AIRLINE_SHARED_FLAGS
        + (
            "--successor-batches-actions",
            "--residual-recertifies-after-user-turn",
            "--frontier-activation-policy",
            "all_write",
            "--lean-residual-packet",
        ),
        denominator_label="32/50",
        denominator_usd=2.682097,
        acceptance=Acceptance(
            reward_floor=39.0,
            cost_ceiling_usd=6.111525,
            ceiling_intent="hold_incumbent",
        ),
        registered=RegisteredResult(
            quality_label="39.0/50 (mean of 3: 38, 40, 39)",
            agent_side_usd=6.111525,
            multiple=2.2786,
            public_artifact=(
                "outputs/papc/tau2_airline_allwrite_lean_g54m_g55_base50"
                "_20260817_v2_exec_public.json"
            ),
            sample_artifacts=(
                "outputs/papc/tau2_airline_allwrite_lean_g54m_g55_base50"
                "_20260817_v1_exec_public.json",
                "outputs/papc/tau2_airline_allwrite_lean_g54m_g55_base50"
                "_20260817_v2_exec_public.json",
                "outputs/papc/tau2_airline_allwrite_lean_g54m_g55_base50"
                "_20260817_v3_exec_public.json",
            ),
            source_bound_to_head=False,
            strong_calls=79,
            denominator_rows=50,
            note=(
                "A hand-off to a human no longer opens a commitment "
                "frontier. It changes no state and leaves no residual behind "
                "it, so on the method's own definition of a commitment it was "
                "never one, and this is the definition applied rather than a "
                "lever pulled. Against the two shots of the predicate it "
                "replaces, 41.0 at $8.296629, this is 39.0 at $6.111525: two "
                "reward points for 26.3 percent of the bill. Report it at the "
                "mean; the three shots are 38, 40 and 39 at $5.310297, "
                "$6.280591 and $6.743685, so the lowest, which reads 1.980x, "
                "is the luckiest sample and not the arm. Quality is the stable "
                "axis, standard deviation 0.82, while cost spreads 9.8 percent "
                "against 0.4 percent for the replaced predicate, because a "
                "prefix that is no longer interrupted takes a freer route. The "
                "same definition is used on the new pool, where it costs three "
                "points and saves 1.3 percent: certifying an escalation is "
                "nearly free there because that successor emits 250.7 output "
                "tokens per call against 508.8 here."
            ),
        ),
        papc_run_surface="tau2",
        papc_run_mode="airline",
        notes=(
            "Superseded 2026-08-17: the previous point was 42/50 at $8.281077 "
            "= 3.088x with the same flags minus the escalation-frontier round "
            "cap and the lean packet. The recertification finding that "
            "produced it still holds and is why that flag is still set: "
            "recertifying after a pause returns the floor to the prefix "
            "model at the one moment the successor has stopped acting, and "
            "buys the same 42/50 for 12.5 percent less. It costs the new pool "
            "five points, so it is a declared per-pool parameter over one "
            "implementation, not a code fork.",
            "The first all-write sample reached 1.980x at 38/50, but it is the "
            "lowest-cost draw rather than the arm. The registered point is the "
            "three-shot mean, 39/50 at 2.2786x.",
            "Quality is 42/50 with or without the recertification switch; only "
            "the cost moves, so unifying on off holds the effect.",
            "This arm rewrites the published batching prohibition and must be "
            "labelled wherever it is cited.",
        ),
    ),
    "airline-new": OperatingPoint(
        name="airline-new",
        surface="airline",
        pool="new",
        model_pool_arm="gpt56-luna-sol-high",
        arm_flags=AIRLINE_SHARED_FLAGS
        + (
            "--successor-batches-actions",
            "--frontier-activation-policy",
            "all_write",
            "--lean-residual-packet",
        ),
        denominator_label="36/50",
        denominator_usd=0.671334,
        acceptance=Acceptance(
            reward_floor=41.0,
            cost_ceiling_usd=3.3573628,
            ceiling_intent="hold_incumbent",
        ),
        registered=RegisteredResult(
            quality_label="41.0/50 (mean of 2: 41, 41)",
            agent_side_usd=3.3573628,
            multiple=5.0010,
            public_artifact=(
                "outputs/papc/tau2_airline_allwrite_lean_luna_sol_base50"
                "_20260817_v1_exec_public.json"
            ),
            sample_artifacts=(
                "outputs/papc/tau2_airline_allwrite_lean_luna_sol_base50"
                "_20260817_v1_exec_public.json",
                "outputs/papc/tau2_airline_allwrite_lean_luna_sol_base50"
                "_20260817_v2_exec_public.json",
            ),
            source_bound_to_head=False,
            strong_calls=79,
            denominator_rows=50,
            note=(
                "Uses the same all-write plus lean definition as the old pool. "
                "The two shots both score 41/50 and their raw usage averages "
                "$3.357363 = 5.0010x at the 2026-08-28 price snapshot; their "
                "frozen launch-era mean is $6.547922. Against the superseded "
                "escalation-certifying point it costs three task points."
            ),
        ),
        papc_run_surface="tau2",
        papc_run_mode="airline",
        notes=(
            "Three pre-rearm shots read 44/43/44 of 50 at $7.0046/$6.9915/"
            "$6.4815, mean $6.826; the spread is the eight percent per-row band.",
            "The head-bound shot needed two infrastructure patches to close its "
            "fifty-row denominator; the relay's auth pool ran dry at ten and "
            "again at five.",
            "This arm rewrites the published batching prohibition and must be "
            "labelled wherever it is cited.",
        ),
    ),
    "retail-old": OperatingPoint(
        name="retail-old",
        surface="retail",
        pool="old",
        model_pool_arm="gpt54mini-gpt55-high",
        arm_flags=RETAIL_SHARED_FLAGS
        + (
            "--contract-builder-variant",
            "deterministic_candidate_grounded",
            "--deterministic-contract-closure",
            "--zero-obligation-request-fallback",
            "--contract-builder-executor",
            "cheap_prefix_model",
        ),
        denominator_label="26/40",
        denominator_usd=2.226476,
        acceptance=Acceptance(
            reward_floor=29.6,
            cost_ceiling_usd=3.750596,
            ceiling_intent="hold_incumbent",
        ),
        registered=RegisteredResult(
            quality_label="29.6/40 (mean of 5, SD 3.78)",
            agent_side_usd=3.750596,
            multiple=1.6845,
            public_artifact=(
                "outputs/papc/retail_test40_closurecheap_zeroobl_g54m_g55"
                "_20260816_r2_public.json"
            ),
            source_bound_to_head=False,
            strong_calls=9,
            denominator_rows=40,
            admitted_residual_actions=7,
            sample_artifacts=(
                "outputs/papc/retail_test40_closurecheap_zeroobl_g54m_g55"
                "_20260816_v1_public.json",
                "outputs/papc/retail_test40_closurecheap_zeroobl_g54m_g55"
                "_20260816_r2_public.json",
                "outputs/papc/retail_test40_closurecheap_zeroobl_g54m_g55"
                "_20260816_r3_public.json",
                "outputs/papc/retail_test40_closurecheap_zeroobl_g54m_g55"
                "_20260816_r4_public.json",
                "outputs/papc/retail_test40_closurecheap_zeroobl_g54m_g55"
                "_20260816_r5_public.json",
            ),
            note=(
                "Replaces the 30/40 at 2.292x. Shots read 30, 36, 27, 28, 27; "
                "the mean is 0.24 standard errors below the strong model alone "
                "at 30/40, and the bill is 26.8 percent of that model's "
                "$14.02. strong_calls and admitted_residual_actions are this "
                "artifact's own, not the five-shot sum; the action count is the "
                "rows whose executed graph "
                "carried an action the withheld cheap candidate did not, which "
                "rose from 6 to 6.6 per shot when the zero-obligation fallback "
                "was added, so the lower bill is not the method standing idle."
            ),
        ),
        papc_run_surface="tau2",
        papc_run_mode="retail",
        notes=(
            "Contract construction runs on the prefix model: grounding is "
            "deterministic, residual closure is one cheap call that returns "
            "only the difference against the whole visible request, and the "
            "zero-obligation fallback re-grounds from the request on the rows "
            "where the cheap agent handed off instead of committing. "
            "Per-shot reward SD is 3.78 over five shots, and the variance is "
            "the user simulator rewriting its own opening message every run "
            "rather than anything in this arm: the first user message is "
            "byte-identical on one of forty tasks across two runs of the same "
            "configuration, and the information it discloses differs.",
        ),
    ),
    "retail-new": OperatingPoint(
        name="retail-new",
        surface="retail",
        pool="new",
        model_pool_arm="gpt56-luna-sol-high",
        arm_flags=RETAIL_SHARED_FLAGS
        + (
            "--contract-builder-variant",
            "deterministic_candidate_grounded",
            "--deterministic-contract-closure",
            "--zero-obligation-request-fallback",
            "--contract-builder-executor",
            "cheap_prefix_model",
        ),
        denominator_label="27/40",
        denominator_usd=0.5016936,
        acceptance=Acceptance(
            reward_floor=30.5,
            cost_ceiling_usd=0.9791697,
            ceiling_intent="hold_incumbent",
        ),
        registered=RegisteredResult(
            quality_label="30.5/40 (mean of 2, SD 2.12)",
            agent_side_usd=0.9791697,
            multiple=1.9517,
            public_artifact=(
                "outputs/papc/retail_test40_closurecheap_zeroobl_luna_sol"
                "_20260816_r3_public.json"
            ),
            source_bound_to_head=False,
            strong_calls=6,
            denominator_rows=40,
            admitted_residual_actions=6,
            sample_artifacts=(
                "outputs/papc/retail_test40_closurecheap_zeroobl_luna_sol"
                "_20260816_r3_public.json",
                "outputs/papc/retail_test40_closurecheap_zeroobl_luna_sol"
                "_20260816_r4_public.json",
            ),
            note=(
                "Only r3 and r4 completed all 40 rows: they scored 32 and 29 "
                "at current-price $0.963389 and $0.994950, respectively; the "
                "frozen artifacts report $3.351460 and $3.308197. v1, r2 and "
                "r5 each missed one row and are excluded from both means. "
                "strong_calls and admitted_residual_actions are r3's own; the "
                "action count is rows whose executed graph carried an action "
                "the withheld cheap candidate did not."
            ),
        ),
        papc_run_surface="tau2",
        papc_run_mode="retail",
    ),
    "agent-diff-old": OperatingPoint(
        name="agent-diff-old",
        surface="agent-diff",
        pool="old",
        model_pool_arm="gpt54mini-gpt55-high",
        arm_flags=("--exact-contract-provider-call-cap", "1"),
        denominator_label="39/45",
        denominator_usd=6.421196,
        acceptance=Acceptance(
            reward_floor=43.0,
            cost_ceiling_usd=12.721529,
            ceiling_intent="hold_incumbent",
        ),
        registered=RegisteredResult(
            quality_label="43/45",
            agent_side_usd=12.721529,
            multiple=1.9812,
            public_artifact=(
                "outputs/papc/agent_diff_hf_test45_cfrc_public_agentdiff"
                "_cap1_rep2_g54m_g55_high_full45_c5_20260814_v1.json"
            ),
            sample_artifacts=(
                "outputs/papc/agent_diff_hf_test45_cfrc_public_agentdiff"
                "_cap1_rep2_g54m_g55_high_full45_c5_20260814_v1.json",
            ),
            source_bound_to_head=False,
            strong_calls=54,
            denominator_rows=45,
            note=(
                "At its recorded runner commit, service-scoped cap-1 n=1: "
                "43/45 at "
                "$12.721529 = 1.9812x. The former three-artifact mean was not "
                "same-configuration evidence: it mixed a legacy merged chain, "
                "a current runner with service scoping off, and this artifact "
                "with service scoping on. It is therefore retired as a mean, "
                "not silently retained as n=3. This artifact is the provisional "
                "historical parent for an adaptive-budget comparison; after any "
                "runner change it must be requalified before a current-source "
                "claim. Any n=3 claim requires three byte-compatible repeats."
            ),
        ),
        command_supported=False,
        papc_run_surface="agent-diff",
        papc_run_mode="method",
        notes=(
            "The cap binds both MAX_EXACT_CONTRACT_PROVIDER_CALLS and "
            "MAX_EXACT_CONTRACT_PROVIDER_CALLS_AFTER_WITNESS_FAILURE. It "
            "removes five of fifty-four successor calls but thirty-one percent "
            "of successor output tokens, because a repair round averages 13,376 "
            "output tokens against an ordinary round's 3,015.",
            "The same cap is measured negative on the new pool: 40/45 at "
            "$9.543471 against a registered 43.5/45 at $9.4068. gpt-5.6-sol "
            "does not spend the wasteful repair rounds that gpt-5.5 does.",
            "The old-pool registered cost is the service-scoped cap1 artifact "
            "at its recorded runner commit, $12.721529, not the retired "
            "mixed-source mean of $13.612; requalify it after runner changes.",
            "Launch layout not reproduced here. Assemble it from the manifest.",
        ),
    ),
    "agent-diff-new": OperatingPoint(
        name="agent-diff-new",
        surface="agent-diff",
        pool="new",
        model_pool_arm="gpt56-luna-sol-high",
        arm_flags=("--exact-contract-provider-call-cap", "2"),
        denominator_label="28/45",
        denominator_usd=0.7025854,
        acceptance=Acceptance(
            reward_floor=43.5,
            cost_ceiling_usd=5.1243536,
            ceiling_intent="hold_incumbent",
        ),
        registered=RegisteredResult(
            quality_label="43.5/45",
            agent_side_usd=5.1243536,
            multiple=7.2936,
            public_artifact=(
                "outputs/papc/agent_diff_hf_test45_cfrc_public_agentdiff_"
                "annotation_luna_sol_high_full45_c4_execute_20260806_v1.json"
            ),
            sample_artifacts=(
                "outputs/papc/agent_diff_hf_test45_cfrc_public_agentdiff_"
                "annotation_luna_sol_high_full45_c4_execute_20260806_v1.json",
                "outputs/papc/agent_diff_hf_test45_cfrc_public_agentdiff_"
                "annotation_luna_sol_high_full45_c4_execute_20260806_r2.json",
            ),
            source_bound_to_head=False,
            strong_calls=60,
            denominator_rows=45,
            note=(
                "The two raw token ledgers reprice to $4.804653 and $5.444054 "
                "under the 2026-08-28 public snapshot; their launch-era mean "
                "was $9.4068."
            ),
        ),
        command_supported=False,
        papc_run_surface="agent-diff",
        papc_run_mode="method",
        notes=(
            "Blocked by reasoning, which is 51 percent of the whole bill once "
            "the cheap role is counted (its output is 77.3 percent reasoning). "
            "Adaptive effort is not a way out: it scores 44 and 42 of 45 "
            "against high's 45, so it is a tradeoff, and no baseline can "
            "instantiate a rule that reads the contract, so it cannot be "
            "compared like for like either.",
            "Launch layout not reproduced here. Assemble it from the manifest.",
        ),
    ),
    "toolsandbox-old": OperatingPoint(
        name="toolsandbox-old",
        surface="toolsandbox",
        pool="old",
        model_pool_arm="gpt54mini-gpt55-high",
        arm_flags=(),
        denominator_label="strict 34",
        denominator_usd=2.310910,
        acceptance=Acceptance(
            reward_floor=0.713458,
            cost_ceiling_usd=3.840395,
            secondary_quality_floors=(("strict_complete_count", 40.0),),
            ceiling_intent="hold_incumbent",
        ),
        registered=RegisteredResult(
            quality_label="strict 40 / similarity 0.713458",
            agent_side_usd=3.840395,
            multiple=1.6619,
            strong_calls=124,
            denominator_rows=129,
            public_artifact=(
                "outputs/papc/toolsandbox_cfrc_oldpool_c7_20260810"
                "_v1_exec_shard{0..6}.json"
            ),
            source_bound_to_head=False,
            admitted_residual_actions=26,
            note=(
                "The arm did not change; the denominator did. It was 1.9144x "
                "against the 2026-07-01 cheap anchor, whose per-call input is 20 "
                "percent below what the same model consumes on the same rows "
                "today. Against a same-code re-measurement it is 1.6619x. The "
                "correction also costs the method three quality points of "
                "reported headroom, 40 against 34 rather than 40 against 31."
            ),
        ),
        command_supported=False,
        papc_run_surface="toolsandbox",
        papc_run_mode="method",
        notes=("Seven-way sharded; cost is the sum over shards.",),
    ),
    "toolsandbox-new": OperatingPoint(
        name="toolsandbox-new",
        surface="toolsandbox",
        pool="new",
        model_pool_arm="gpt56-luna-sol-high",
        arm_flags=(),
        denominator_label="strict 34 / 0.666445",
        denominator_usd=0.3206706,
        acceptance=Acceptance(
            reward_floor=0.705414,
            cost_ceiling_usd=1.2878284,
            secondary_quality_floors=(("strict_complete_count", 43.0),),
            ceiling_intent="hold_incumbent",
        ),
        registered=RegisteredResult(
            quality_label="strict 43 / similarity 0.705414",
            agent_side_usd=1.2878284,
            multiple=4.0160,
            public_artifact=(
                "outputs/papc/toolsandbox_cfrc_luna_sol_high_base129_c7"
                "_20260801_v1_aggregate.json"
            ),
            source_bound_to_head=False,
            strong_calls=125,
            denominator_rows=129,
            admitted_residual_actions=26,
            note=(
                "The pointer this replaced named a markdown file, which left "
                "the figure undatable and let a repeat be read as same-source "
                "when it was not. Its raw ledger reprices to $1.287828 = "
                "4.0160x under the 2026-08-28 snapshot; the artifact's frozen "
                "launch-era total is $2.697147. Two later current-source "
                "requalifications remain source-drift diagnostics and are not "
                "spliced into this registration."
            ),
        ),
        command_supported=False,
        papc_run_surface="toolsandbox",
        papc_run_mode="method",
    ),
}


class UnsupportedLaunch(ValueError):
    """Raised when a point declares no reproduced launch layout."""


def _require_command(point: OperatingPoint) -> None:
    if not point.command_supported:
        raise UnsupportedLaunch(
            f"{point.name} declares no reproduced launch layout; "
            "assemble it from docs/papc/current_evidence_manifest.md"
        )


def _state_argv(
    point: OperatingPoint, *, run_id: str, concurrency: int, execute: bool
) -> list[str]:
    public = f"{PUBLIC_OUTPUT_DIR}/{run_id}_{'exec' if execute else 'noapi'}.json"
    argv = [
        "papc-run",
        "state",
        "method",
        "--model-pool-arm",
        point.model_pool_arm,
        *point.arm_flags,
        "--max-concurrency",
        str(concurrency),
        "--mode",
        "true_live_cfrc" if execute else "noapi",
        "--public-output-json",
        public,
    ]
    if execute:
        argv += [
            "--expected-noapi-report-json",
            f"{PUBLIC_OUTPUT_DIR}/{run_id}_noapi.json",
            "--private-output-root",
            f"{PRIVATE_OUTPUT_ROOT}/{run_id}",
            "--approve-paid-method-arm-run",
            "--approval-scope",
            STATE_APPROVAL_SCOPE,
        ]
    argv += ["--api-base", "$OPENAI_BASE_URL", "--api-key-env", "OPENAI_API_KEY"]
    if execute:
        argv.append("--execute")
    return argv


def _airline_argv(
    point: OperatingPoint, *, run_id: str, concurrency: int, execute: bool
) -> list[str]:
    if not execute:
        return [
            ".venv/bin/python",
            "scripts/tau2/papc_tau2_airline_residual_gate_noapi.py",
            "--output",
            f"{PUBLIC_OUTPUT_DIR}/{run_id}_gate.json",
            "--projected-pass-count",
            str(int(point.acceptance.reward_floor or 0)),
            "--projection-basis",
            f"matched airline base50 {point.name} arm on the same pool",
        ]
    return [
        "papc-run",
        "tau2",
        "airline",
        "--model-pool-arm",
        point.model_pool_arm,
        *point.arm_flags,
        "--run-name",
        run_id,
        "--max-concurrency",
        str(concurrency),
        "--reward-floor",
        str(point.acceptance.reward_floor),
        "--agent-side-cost-ceiling-usd",
        str(point.acceptance.cost_ceiling_usd),
        "--compiler-gate-json",
        f"{PUBLIC_OUTPUT_DIR}/{run_id}_gate.json",
        "--public-output-json",
        f"{PUBLIC_OUTPUT_DIR}/{run_id}_exec_public.json",
        "--output-root",
        PRIVATE_OUTPUT_ROOT,
        "--api-base",
        "$OPENAI_BASE_URL",
        "--api-key-env",
        "OPENAI_API_KEY",
        "--execute",
    ]


def _retail_argv(
    point: OperatingPoint, *, run_id: str, concurrency: int, execute: bool
) -> list[str]:
    if not execute:
        flags = list(point.arm_flags)

        def flag_value(flag: str, default: str) -> str:
            return flags[flags.index(flag) + 1] if flag in flags else default

        argv = [
            ".venv/bin/python",
            "scripts/tau2/papc_tau2_request_contract_gate_noapi.py",
            "--surface",
            "retail",
            "--full-run",
            "--split",
            flag_value("--domain-split", "test"),
            "--user-authorized",
            "--model-pool-arm",
            point.model_pool_arm,
            "--contract-builder-variant",
            flag_value(
                "--contract-builder-variant",
                "deterministic_candidate_grounded",
            ),
            "--contract-builder-executor",
            flag_value("--contract-builder-executor", "successor"),
            "--max-concurrency",
            str(concurrency),
            "--reward-floor",
            str(point.acceptance.reward_floor),
            "--cost-ceiling-usd",
            str(point.acceptance.cost_ceiling_usd),
            "--output",
            f"{PUBLIC_OUTPUT_DIR}/{run_id}_gate.json",
        ]
        for boolean_flag in (
            "--deterministic-contract-closure",
            "--zero-obligation-request-fallback",
            "--grounded-terminal-literals",
            "--transcription-is-fallback",
        ):
            if boolean_flag in flags:
                argv.append(boolean_flag)
        return argv
    return [
        "papc-run",
        "tau2",
        "retail",
        "--model-pool-arm",
        point.model_pool_arm,
        *point.arm_flags,
        "--run-name",
        run_id,
        "--max-concurrency",
        str(concurrency),
        "--reward-floor",
        str(point.acceptance.reward_floor),
        "--cost-ceiling-usd",
        str(point.acceptance.cost_ceiling_usd),
        "--request-contract-gate-json",
        f"{PUBLIC_OUTPUT_DIR}/{run_id}_gate.json",
        "--public-output-json",
        f"{PUBLIC_OUTPUT_DIR}/{run_id}.json",
        "--output-root",
        PRIVATE_OUTPUT_ROOT,
        "--api-base",
        "$OPENAI_BASE_URL",
        "--api-key-env",
        "OPENAI_API_KEY",
        "--execute",
    ]


_ARGV_BUILDERS = {
    "state": _state_argv,
    "airline": _airline_argv,
    "retail": _retail_argv,
}


def launch_argv(
    point: OperatingPoint,
    *,
    run_id: str,
    execute: bool,
    concurrency: int | None = None,
) -> list[str]:
    """Return the gate command (execute=False) or the paid command for a point.

    The concurrency defaults to the surface ceiling and is refused above it,
    because the ceiling is a measured property of the relay under load rather
    than a preference.
    """

    _require_command(point)
    resolved = point.concurrency_ceiling if concurrency is None else int(concurrency)
    if resolved > point.concurrency_ceiling:
        raise ValueError(
            f"{point.surface} runs at most {point.concurrency_ceiling} in parallel; "
            f"asked for {resolved}"
        )
    if resolved < 1:
        raise ValueError("concurrency must be at least one")
    builder = _ARGV_BUILDERS[point.surface]
    return builder(point, run_id=run_id, concurrency=resolved, execute=execute)


def format_points(points: Sequence[OperatingPoint] | None = None) -> str:
    """Render the incumbent points as a table."""

    rows = list(points if points is not None else OPERATING_POINTS.values())
    header = (
        f"{'point':17} {'quality':22} {'agent-side':>11} {'anchor':>9} "
        f"{'x':>7} {'2x':>4} {'head':>5} {'s/row':>6}"
    )
    lines = [header, "-" * len(header)]
    for point in rows:
        reg = point.registered
        quality = (reg.quality_label if reg else "unmeasured")[:22]
        cost = f"${reg.agent_side_usd:.4f}" if reg else "--"
        mult = f"{reg.multiple:.3f}x" if reg else "--"
        # A point where the successor barely runs is cheap for the wrong
        # reason, so its budget mark is withheld rather than granted.
        engaged = reg.method_engaged if reg else None
        inside = (
            "--"
            if reg is None
            else "IDLE"
            if engaged is False
            else "yes"
            if reg.multiple <= 2.0
            else "no"
        )
        head = "yes" if reg and reg.source_bound_to_head else "no"
        rate = reg.strong_calls_per_row if reg else None
        rate_text = f"{rate:.2f}" if rate is not None else "-"
        lines.append(
            f"{point.name:17} {quality:22} {cost:>11} "
            f"${point.denominator_usd:8.4f} {mult:>7} {inside:>4} {head:>5} {rate_text:>6}"
        )
    counted = [
        p
        for p in rows
        if p.registered
        and p.registered.multiple <= 2.0
        and p.registered.method_engaged is not False
    ]
    head_count = sum(
        1 for p in rows if p.registered and p.registered.source_bound_to_head
    )
    idle = [p.name for p in rows if p.registered and p.registered.method_engaged is False]
    lines.append("")
    lines.append(
        f"{len(counted)}/{len(rows)} inside the two-times budget with the method "
        f"engaged, {head_count}/{len(rows)} bound to the current source."
    )
    if idle:
        lines.append(
            "IDLE means the successor runs on fewer than "
            f"{MINIMUM_ENGAGEMENT:.1f} calls per row, so the number measures the "
            "method not running: " + ", ".join(idle)
        )
    return "\n".join(lines)
