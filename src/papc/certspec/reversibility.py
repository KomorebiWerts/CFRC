"""Schema-derived reversibility classification and restoration cost (Lambda grounding).

This module is benchmark-independent. It replaces hand-set per-field
restoration-cost tables with a cost **derived from observable tool-schema
structure**, so the restoration-cost measure ``Lambda`` is auditable against the
environment's own tool contract rather than tuned to manufacture a separation.

A write-unit's restoration cost answers a single question: *if a committed value
of this unit were overwritten, how costly is it to restore?* We read the answer
from the tool schema, not from a gold outcome, expected value, task id, or fixed
argument:

* **REVERSIBLE** -- an *in-place update* tool can re-set the unit to an arbitrary
  value with no money / points / fee / lifecycle-confirm semantics. A second
  update call restores the prior value at ~one tool call. Cost
  :data:`REVERSIBLE_RESTORATION_COST` (``< 1``).
* **IRREVERSIBLE** -- the unit is writable only through create/book/cancel tools
  (no in-place update path), or its own field semantics are money / points /
  fee / refund / cancellation / lifecycle-confirm (no inverse). Restoring it
  needs cancel+rebook+penalty or is non-recoverable. Cost
  :data:`IRREVERSIBLE_RESTORATION_COST` (``>> threshold``).
* **UNKNOWN** -- the schema does not expose enough to classify; falls back to the
  uniform unit weight :data:`DEFAULT_RESTORATION_COST` (``== 1``), i.e. ``Lambda``
  reduces to the counting measure ``mu`` for that unit.

Why this is *not* a tuned magnitude
------------------------------------
The two roles that were previously conflated are now separated:

1. the **classification** of each unit (reversible vs irreversible) is read from
   the schema (in-place update tool present? money/fee/lifecycle semantics?), so
   it cannot be hand-picked to be convenient for a particular bundle;
2. the **budget** ``rho_alpha`` is calibrated separately (see
   :mod:`papc.certspec.calibration`), so it is not fit to a target bundle cost.

The remaining magnitudes are the method's *stated modeling scale*: a reversible
write costs a fraction of a unit (one in-place re-call), an irreversible write
costs much more than one unit (no inverse), and one irreversible unit alone
exceeds :data:`IRREVERSIBLE_DAMAGE_THRESHOLD`. The genuine mechanism a
cost-weighted ``Lambda`` then has over the uniform counting measure ``mu`` is:
when the collision budget is denominated in *restoration cost*, a cost-aware
projection can pack a bundle of cheap (schema-reversible) colliding units whose
*count* exceeds the budget but whose *cost* fits, while a count-aware projection
cannot. The separation is therefore earned by the schema's reversibility
structure, not by the magnitudes.

No API, no dataset names, no task ids, no gold answer sketches.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

# --- method modeling scale (see module docstring) ---------------------------
# Centralized here so dataset adapters import the scale rather than re-declaring
# magic magnitudes. Only the CLASSIFICATION is dataset-derived; this scale is the
# benchmark-independent method assumption.
REVERSIBLE_RESTORATION_COST = 0.5  # < 1: one in-place update call restores it
DEFAULT_RESTORATION_COST = 1.0  # unknown reversibility -> the uniform mu weight
IRREVERSIBLE_RESTORATION_COST = 50.0  # >> threshold: no inverse; cancel+rebook+penalty
IRREVERSIBLE_DAMAGE_THRESHOLD = 2.0  # one irreversible unit alone exceeds this

ReversibilityClass = Literal["reversible", "irreversible", "unknown"]

# --- observable text signals (read from schema descriptions, never gold) -----
# These are conservative substring cues over the tool/field schema text. They
# are heuristics over *observable schema documentation*, not over outcomes.
_MONEY_POINTS_TOKENS: tuple[str, ...] = (
    "price",
    "cash",
    "points",
    "payment",
    "mileage",
    "loyalty",
    "balance",
    "charge",
)
_FEE_PENALTY_TOKENS: tuple[str, ...] = (
    "fee",
    "penalt",
    "fare difference",
    "fare_difference",
    "refund",
    "compensation",
    "cancellation",
)
_CONFIRM_LIFECYCLE_TOKENS: tuple[str, ...] = (
    "confirm",
    "preview the cancellation",
    "cancel an existing",
    "cancel a ",
)


def looks_like_money_or_points(text: str) -> bool:
    """True if schema text indicates a money/points movement (no inverse)."""

    low = text.lower()
    return any(tok in low for tok in _MONEY_POINTS_TOKENS)


def looks_like_fee_or_penalty(text: str) -> bool:
    """True if schema text indicates a fee / fare-difference / refund / penalty."""

    low = text.lower()
    return any(tok in low for tok in _FEE_PENALTY_TOKENS)


def looks_like_confirm_lifecycle(text: str) -> bool:
    """True if schema text indicates a preview->confirm / cancellation lifecycle."""

    low = text.lower()
    return any(tok in low for tok in _CONFIRM_LIFECYCLE_TOKENS)


@dataclass(frozen=True)
class WriteUnitSignal:
    """Observable, schema-derived features of one write-unit (``entity.field`` family).

    Every field is read from the tool schema's structure or documentation text.
    No gold outcome, expected value, task id, or fixed tool argument is consumed.
    """

    unit_family: str
    in_place_update_tool: bool = False  # an update-style tool can re-set this field
    money_or_points: bool = False  # field moves money or loyalty points (no inverse)
    fee_or_penalty: bool = (
        False  # field is/charges a fee, fare diff, refund, compensation
    )
    lifecycle_confirm: bool = False  # status set via a preview->confirm cancel/commit
    create_or_cancel_only: bool = (
        False  # only writable via create/book/cancel (no update)
    )
    reason: str = ""  # human-readable schema citation for the classification


def classify_reversibility(signal: WriteUnitSignal) -> ReversibilityClass:
    """Classify a write-unit from its observable schema signal.

    Order matters: any non-invertible *semantics* (money/points, fee/penalty,
    lifecycle confirm) makes the unit irreversible even if an update tool can
    nominally re-pass it, because restoring the prior value is not free. A unit
    that is only reachable through create/book/cancel (no in-place update) is
    also irreversible (restore = cancel+rebook). Only a unit an in-place update
    tool can freely re-set, with no such semantics, is reversible.
    """

    if signal.money_or_points or signal.fee_or_penalty or signal.lifecycle_confirm:
        return "irreversible"
    if signal.create_or_cancel_only and not signal.in_place_update_tool:
        return "irreversible"
    if signal.in_place_update_tool:
        return "reversible"
    return "unknown"


def restoration_cost_from_signal(signal: WriteUnitSignal) -> float:
    """Map a schema signal to a per-unit restoration cost on the method scale."""

    cls = classify_reversibility(signal)
    if cls == "reversible":
        return REVERSIBLE_RESTORATION_COST
    if cls == "irreversible":
        return IRREVERSIBLE_RESTORATION_COST
    return DEFAULT_RESTORATION_COST


def restoration_cost_for_family(
    signals: Mapping[str, WriteUnitSignal],
    unit_family: str,
) -> float:
    """Restoration cost for a unit family, defaulting to the uniform weight."""

    signal = signals.get(unit_family)
    if signal is None:
        return DEFAULT_RESTORATION_COST
    return restoration_cost_from_signal(signal)


def restoration_cost_vector(
    units: Sequence[str],
    signals: Mapping[str, WriteUnitSignal],
    *,
    family_of: "Callable[[str], str] | None" = None,
) -> dict[str, float]:
    """Build a per-unit restoration-cost vector from schema-derived signals.

    ``units`` are concrete unit tokens (e.g. ``entity.field#k``); ``family_of``
    strips them to the schema family the ``signals`` map is keyed by. With no
    ``family_of`` the token is used verbatim.
    """

    fam = family_of or (lambda u: u)
    return {unit: restoration_cost_for_family(signals, fam(unit)) for unit in units}


def is_non_uniform(cost_vector: Mapping[str, float]) -> bool:
    """True iff the cost vector has at least two distinct values.

    A uniform vector cannot let ``Lambda`` differ from the counting measure
    ``mu``; a non-uniform one can. (The customer_support failure mode is uniform.)
    """

    return len({round(float(v), 6) for v in cost_vector.values()}) > 1


__all__ = [
    "REVERSIBLE_RESTORATION_COST",
    "DEFAULT_RESTORATION_COST",
    "IRREVERSIBLE_RESTORATION_COST",
    "IRREVERSIBLE_DAMAGE_THRESHOLD",
    "ReversibilityClass",
    "WriteUnitSignal",
    "looks_like_money_or_points",
    "looks_like_fee_or_penalty",
    "looks_like_confirm_lifecycle",
    "classify_reversibility",
    "restoration_cost_from_signal",
    "restoration_cost_for_family",
    "restoration_cost_vector",
    "is_non_uniform",
]
