"""STATE-Bench *travel* projection into the shared RCBE state-write contract.

This is the dataset adapter the repo boundary requires to live in
``src/papc/certspec``: it turns a runtime-visible travel row/receipt into the
benchmark-independent method types
(:class:`~papc.certspec.state_write_contract.StateWriteContract`,
:class:`~papc.certspec.state_write_contract.ProposedTargetAction`) AND a **real
per-unit restoration-cost vector** over committed travel units. Experiment
scripts under ``scripts/`` must call this module; they must not re-implement the
projection, cost mapping, or arm logic inline.

Restoration cost is SCHEMA-DERIVED, not hand-listed
---------------------------------------------------
Each committed unit's restoration cost is computed by
:func:`travel_write_unit_signal` ->
:func:`papc.certspec.reversibility.classify_reversibility` from the *observable*
travel tool contract, not from a hand-picked field list:

* a unit is **reversible** only if an in-place update tool can freely re-set it
  with no money / fee / lifecycle semantics (a second ``update_booking`` call
  restores the prior value) -- cost ``CHEAP_RESTORATION_COST``;
* otherwise it is **irreversible** -- it moves money/points, charges a
  fee/fare-difference/refund, drives a cancel/confirm lifecycle, or is reachable
  only through create/book/cancel (no in-place update path), so restoring it
  needs cancel+rebook+penalty -- cost ``IRREVERSIBLE_RESTORATION_COST``.

Grounding the cost in the schema deliberately REMOVES the hand-tuned
convenience the old list encoded. Reading ``tools.py`` reclassifies several
fields the previous hand-set table called "cheap" as irreversible:
``bookings.cabin_class`` (``update_booking`` documents "upgrade fees apply"),
``hotels.room_type``, ``car_rentals.car_class`` and
``car_rentals.insurance_included`` (no in-place update tool exists -- only
book + cancel). This dissolves the cheap-carrier bundle that earlier travel
replays relied on; that collapse is intended, not a regression.

Because committed state still spans two genuine cost regimes, ``Lambda``
(restoration-cost-weighted collision) *can* separate from the uniform counting
measure ``mu`` when a row actually mixes schema-reversible and irreversible
colliding units. Whether any natural row does is an empirical question answered
by the prevalence audit, not assumed here.

Source of the magnitudes
------------------------
The reversible/default/irreversible magnitudes are the benchmark-independent
method scale defined in :mod:`papc.certspec.reversibility`
(:data:`CHEAP_RESTORATION_COST` < :data:`IRREVERSIBLE_DAMAGE_THRESHOLD` <=
:data:`IRREVERSIBLE_RESTORATION_COST`). Only the per-unit *classification* is
dataset-derived. No task ids, entity ids, expected values, or fixed tool
arguments are consumed or emitted.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from papc.certspec.reversibility import (
    DEFAULT_RESTORATION_COST,
    IRREVERSIBLE_DAMAGE_THRESHOLD,
    IRREVERSIBLE_RESTORATION_COST,
    REVERSIBLE_RESTORATION_COST,
    WriteUnitSignal,
    classify_reversibility,
    looks_like_fee_or_penalty,
    looks_like_money_or_points,
    restoration_cost_from_signal,
)
from papc.certspec.state_write_contract import (
    Obligation,
    ProposedTargetAction,
    StateWriteContract,
)

SURFACE = "state_bench_travel"

# --- restoration-cost: SCHEMA-DERIVED, not hand-set --------------------------
# Magnitudes are the benchmark-independent method scale, imported from
# papc.certspec.reversibility. The CLASSIFICATION below is derived from the
# travel tool schema, so it is auditable against tools.py rather than tuned.
CHEAP_RESTORATION_COST = REVERSIBLE_RESTORATION_COST

# Schema-structural facts read from state_bench/domains/travel/tools.py. The ONLY
# in-place update tool is ``update_booking`` (bookings); hotels and car rentals
# expose no update tool (book + cancel only, per WRITE_TOOL_NAMES), so their
# committed fields cannot be re-set without cancel+rebook (a penalty).
_ENTITIES_WITH_INPLACE_UPDATE: frozenset[str] = frozenset({"bookings"})

# ``update_booking`` parameters that re-set a stored value with no fee / money /
# lifecycle semantics. ``cabin_class`` is EXCLUDED: its schema says "Upgrade
# cabin class on the same flight (upgrade fees apply)"; ``flight_id`` is a
# rebooking whose ``change_reason`` "affects fee calculation".
_INPLACE_UPDATE_SETTABLE_FIELDS: frozenset[str] = frozenset(
    {
        "seat_type",
        "meal_preference",
        "add_wifi",
        "add_extra_legroom",
        "add_insurance",
        "paid_checked_bags",
    }
)

# Fields an update tool can re-pass but whose schema doc attaches a fee, so the
# token scan over the bare field name would otherwise miss them.
_FEE_BEARING_UPDATE_FIELDS: frozenset[str] = frozenset({"cabin_class", "flight_id"})


def travel_write_unit_signal(family: str) -> WriteUnitSignal:
    """Build the observable reversibility signal for a travel ``entity.field``.

    Reads only the tool-schema structure (which entity exposes an in-place update
    tool, which update params re-set a value) and the field-name/doc semantics
    (money/points, fee/penalty, lifecycle). No gold outcome or expected value.
    """

    entity, _, field_name = family.partition(".")
    has_update_tool = entity in _ENTITIES_WITH_INPLACE_UPDATE
    in_place = has_update_tool and field_name in _INPLACE_UPDATE_SETTABLE_FIELDS
    money = looks_like_money_or_points(field_name)
    fee = (
        looks_like_fee_or_penalty(field_name)
        or field_name in _FEE_BEARING_UPDATE_FIELDS
    )
    lifecycle = field_name == "status"
    return WriteUnitSignal(
        unit_family=family,
        in_place_update_tool=in_place,
        money_or_points=money,
        fee_or_penalty=fee,
        lifecycle_confirm=lifecycle,
        create_or_cancel_only=not has_update_tool,
        reason=(
            f"entity={entity} in_place_update={in_place} money={money} "
            f"fee={fee} lifecycle={lifecycle}"
        ),
    )


# The travel committed-state field universe (schema-level entity.field families).
# The two public frozensets below are DERIVED from the schema signal, not
# hand-authored, so they are auditable against tools.py (see audit test).
_TRAVEL_COMMITTED_FIELD_UNIVERSE: tuple[str, ...] = (
    "bookings.cabin_class",
    "bookings.seat_type",
    "bookings.meal_preference",
    "bookings.add_wifi",
    "bookings.add_extra_legroom",
    "bookings.add_insurance",
    "bookings.paid_checked_bags",
    "bookings.status",
    "bookings.flight_id",
    "bookings.price_paid",
    "bookings.cash_amount",
    "bookings.points_used",
    "bookings.payment_method",
    "bookings.cancellation_fee",
    "bookings.refund_amount",
    "bookings.change_fee",
    "bookings.fare_difference",
    "bookings.delay_compensation",
    "hotels.room_type",
    "hotels.status",
    "hotels.cancellation_fee",
    "hotels.refund_amount",
    "car_rentals.car_class",
    "car_rentals.insurance_included",
    "car_rentals.status",
    "car_rentals.cancellation_fee",
    "car_rentals.refund_amount",
    "users.loyalty_points",
    "users.mileage_balance",
    "users.points_balance",
)

_TRAVEL_WRITE_SIGNALS: dict[str, WriteUnitSignal] = {
    family: travel_write_unit_signal(family)
    for family in _TRAVEL_COMMITTED_FIELD_UNIVERSE
}

_DERIVED_REVERSIBILITY: dict[str, str] = {
    family: classify_reversibility(signal)
    for family, signal in _TRAVEL_WRITE_SIGNALS.items()
}

# Cheaply restorable committed units = DERIVED reversible class. Note the schema
# reclassifies cabin_class (upgrade fee), hotels.room_type and
# car_rentals.car_class/insurance_included (no in-place update tool) as NOT cheap
# -- exactly the hand-tuned convenience the grounding removes.
CHEAP_RESTORABLE_FIELDS: frozenset[str] = frozenset(
    f for f, c in _DERIVED_REVERSIBILITY.items() if c == "reversible"
)

# Runtime-enforced carrier fields for ``update_booking(flight_id=...)``. The
# STATE-Bench travel tool schema and environment both require these fields to be
# passed explicitly when changing flights. They are **argument_carriers**:
# value-unchanged required tool arguments that make the call executable. They are
# NOT committed writes and are kept out of the forbidden set ``C(z)`` (see the
# typed split in ``TravelProjectionRow`` / ``projection_row_from_state_requirements``).
BOOKING_FLIGHT_CHANGE_CARRIER_FIELDS: tuple[str, ...] = (
    "bookings.cabin_class",
    "bookings.seat_type",
    "bookings.meal_preference",
    "bookings.add_wifi",
    "bookings.add_extra_legroom",
    "bookings.add_insurance",
)

# Irreversible / penalty-bearing committed units = DERIVED irreversible class.
# This includes lifecycle/money fields plus fee-bearing update fields
# (bookings.cabin_class, bookings.flight_id) and create/cancel-only fields such
# as hotels.room_type and car_rentals.car_class.
IRREVERSIBLE_FIELDS: frozenset[str] = frozenset(
    f for f, c in _DERIVED_REVERSIBILITY.items() if c == "irreversible"
)


def unit_family(unit: str) -> str:
    """Strip the per-row occurrence suffix (``entity.field#k``) -> ``entity.field``."""

    return unit.split("#", 1)[0]


def restoration_cost_for_unit(unit: str) -> float:
    """Per-unit restoration cost from the travel schema field family.

    Uses only the abstract ``entity.field`` family, never gold/expected values.
    """

    family = unit_family(unit)
    signal = _TRAVEL_WRITE_SIGNALS.get(family)
    if signal is None:
        return DEFAULT_RESTORATION_COST
    return restoration_cost_from_signal(signal)


def restoration_cost_vector(units: Sequence[str]) -> dict[str, float]:
    """Build the real per-unit restoration-cost vector for committed travel units.

    This is the key object customer_support lacked: a *non-uniform* map so that
    restoration-cost ``Lambda`` separates cheaply restorable preference writes
    from irreversible money/lifecycle writes.
    """

    return {unit: restoration_cost_for_unit(unit) for unit in units}


def is_non_uniform(cost_vector: Mapping[str, float]) -> bool:
    """True iff the restoration-cost vector has at least two distinct values.

    A uniform vector (the customer_support failure mode) cannot let ``Lambda``
    differ from the counting measure ``mu``; a non-uniform one can.
    """

    return len(set(round(float(v), 6) for v in cost_vector.values())) > 1


def abstract_units(state_requirements: Sequence[Any]) -> list[str]:
    """Project private travel state-requirements into abstract unit tokens.

    Each token is ``entity.field#occurrence`` -- the schema field family plus a
    per-row de-duplication index. It deliberately excludes record keys, entity
    ids, expected values, and fixed tool arguments.
    """

    units: list[str] = []
    seen: Counter[str] = Counter()
    for req_value in state_requirements:
        req = _mapping(req_value)
        if not req:
            continue
        entity = str(req.get("entity_type") or "unknown")
        fields: list[str] = []
        field_name = req.get("field")
        if field_name not in (None, ""):
            fields.append(str(field_name))
        for key in ("match_fields", "expected_fields"):
            for item in _mapping(req.get(key)).keys():
                fields.append(str(item))
        if not fields:
            fields.append("record")
        for name in fields:
            family = f"{entity}.{name}"
            seen[family] += 1
            units.append(f"{family}#{seen[family]}")
    return units


@dataclass(frozen=True)
class TravelProjectionRow:
    """Public-safe abstract projection candidate for one travel task/receipt.

    All fields are schema-level abstract unit tokens plus a restoration-cost
    vector. There are no task ids, entity ids, expected values, or fixed tool
    arguments.

    Typed three-role split (the unit's role IS its set membership):

    * ``committed_units`` -- **changed_effect / committed-collider**: real
      committed writes. These become ``forbidden_writes`` (``C(z)``) in the
      contract, so an over-broad proposal that touches them collides and
      ``pi_C`` projects them out (``Lambda`` may re-admit a cheap one).
    * ``action_closure_units`` -- **argument_carrier**: value-unchanged required
      tool arguments (the six ``update_booking`` booking-preference fields). They
      are required for the call to *execute* (they stay in ``proposal_units``,
      ``touched_units`` and the required-residual footprint), but they are NOT
      committed writes and are deliberately kept OUT of ``committed_units`` /
      ``forbidden_writes`` -- a carrier is not a collision and must never be
      scored as ``Lambda`` re-admission work.
    * ``residual_units`` -- the withheld required residual write (the open
      bounded obligation the proposer may legitimately close).
    """

    row_key: str
    task_type: str
    committed_units: frozenset[str]
    residual_units: frozenset[str]
    proposal_units: frozenset[str]
    restoration_cost: Mapping[str, float] = field(default_factory=dict)
    action_closure_units: frozenset[str] = field(default_factory=frozenset)
    action_closure_reason: str = ""
    touched_units: frozenset[str] = field(default_factory=frozenset)
    closure_metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def collision_units(self) -> frozenset[str]:
        return self.committed_units & self.proposal_units

    @property
    def has_partial_collision(self) -> bool:
        return bool(self.collision_units) and bool(
            self.residual_units & self.proposal_units
        )

    @property
    def has_cheap_collision(self) -> bool:
        return any(
            self.restoration_cost.get(unit, IRREVERSIBLE_RESTORATION_COST)
            < IRREVERSIBLE_DAMAGE_THRESHOLD
            for unit in self.collision_units
        )

    @property
    def has_irreversible_collision(self) -> bool:
        return any(
            self.restoration_cost.get(unit, 0.0) >= IRREVERSIBLE_DAMAGE_THRESHOLD
            for unit in self.collision_units
        )

    @property
    def has_mixed_cost_collision(self) -> bool:
        return self.has_cheap_collision and self.has_irreversible_collision

    def to_state_write_contract(self) -> StateWriteContract:
        """Build the shared :class:`StateWriteContract` for this row.

        Committed units (changed_effect) become forbidden writes (the closed
        footprint ``C(z)``); the withheld residual unit is the open bounded
        obligation. The residual obligation's required footprint also carries the
        ``action_closure_units`` (argument_carrier) so the call is executable --
        but those carriers are NOT in ``forbidden_writes`` (they are not committed
        writes), so ``pi_C`` keeps them freely and ``Lambda`` is never charged to
        re-admit them. This is the typed-split fix for the carrier miscredit.
        """

        committed = frozenset(self.committed_units)
        closed = frozenset(
            Obligation(
                obligation_id=f"{self.row_key}.committed.{unit}",
                write_footprint=frozenset({unit}),
                postcondition_witness="committed_travel_state_unit",
                surface=SURFACE,
                status="closed",
            )
            for unit in sorted(committed)
        )
        required_residual_footprint = self.residual_units | self.action_closure_units
        residual = frozenset(
            {
                Obligation(
                    obligation_id=f"{self.row_key}.residual",
                    write_footprint=required_residual_footprint,
                    postcondition_witness=(
                        "withheld_residual_travel_unit"
                        if not self.action_closure_units
                        else "withheld_residual_travel_unit_with_schema_required_carriers"
                    ),
                    surface=SURFACE,
                    status="open_bounded",
                )
            }
        )
        return StateWriteContract(
            surface=SURFACE,
            derivation_timing="terminal",
            closed=closed,
            residual=residual,
            forbidden_writes=committed,
            legal_actions=frozenset({"bounded_repair", "abstain"}),  # type: ignore[arg-type]
            state_change_required=True,
            commitment_call_count=len(committed),
            commitment_footprint_size=len(committed),
            commitment_app_count=len(
                {unit_family(u).split(".", 1)[0] for u in committed}
            ),
            soundness_assumptions=(
                "travel_state_unit_families_are_schema_level_not_entity_specific",
                "restoration_cost_comes_from_field_family_not_gold_outcome",
                "residual_unit_is_a_withheld_writable_scope_not_a_handle",
            ),
            touched_units=self.touched_units,
        )

    def to_proposed_target_action(self) -> ProposedTargetAction:
        """Build the strong-proposer write footprint for this row."""

        return ProposedTargetAction(
            surface=SURFACE,
            action_id=self.row_key,
            write_footprint=self.proposal_units,
            read_footprint=self.touched_units,
            writable_scope=self.residual_units | self.action_closure_units,
        )

    def to_closure_aware_target_action(self) -> ProposedTargetAction:
        """Build the proposal with its dependency-closure structure ``F(a,z)`` declared.

        For an existing-booking flight change, ``update_booking(flight_id=...)``
        is one ATOMIC transaction: the residual ``flight_id`` write and its six
        schema-required carrier fields must execute together. We declare that as a
        single :attr:`ProposedTargetAction.effect_groups` bundle, so closure-aware
        ``pi_C`` cannot keep half the transaction (e.g. change the flight but drop
        a required carrier whose committed value is irreversible). Rows with no
        action closure fall back to the flat proposal -- so this never changes the
        flat comparator path, only the closure-aware audit can read the group.
        """

        if not self.action_closure_units:
            return self.to_proposed_target_action()
        transaction = frozenset(self.residual_units | self.action_closure_units)
        return ProposedTargetAction(
            surface=SURFACE,
            action_id=self.row_key,
            write_footprint=self.proposal_units,
            read_footprint=self.touched_units,
            writable_scope=self.residual_units | self.action_closure_units,
            effect_groups=(transaction,),
        )


def choose_residual_unit(units: Sequence[str], *, task_type: str = "") -> str:
    """Pick an outcome-relevant residual unit, preferring an irreversible field.

    The residual is the writable scope withheld from the committed set so the
    proposer has something to legitimately close; preferring an irreversible
    family makes the residual a real commitment, not a handle.
    """

    # For existing-booking flight-change rows, ``flight_id`` is the primary
    # tool action. Fee/price fields are effects of that action, not the residual
    # write authority itself. Prefer it when present so replay cannot "solve" a
    # flight change by writing only a fee field.
    if task_type != "new_booking":
        for unit in units:
            if unit_family(unit) == "bookings.flight_id":
                return unit
    for unit in units:
        if unit_family(unit) in {
            "bookings.status",
            "hotels.status",
            "car_rentals.status",
        }:
            return unit
    for unit in units:
        if unit_family(unit) in IRREVERSIBLE_FIELDS:
            return unit
    return units[0]


def booking_flight_change_closure_units(
    residual_units: Sequence[str],
) -> frozenset[str]:
    """Schema-derived carrier units required by ``update_booking``.

    These units are not gold outcome labels. They come from the public travel
    tool contract: changing ``bookings.flight_id`` requires explicitly carrying
    the six booking preference fields. The occurrence suffix is synthetic and
    row-local because the public RCBE unit space deliberately strips concrete
    booking ids/values.
    """

    if not any(unit_family(unit) == "bookings.flight_id" for unit in residual_units):
        return frozenset()
    return frozenset(
        f"{family}#closure1" for family in BOOKING_FLIGHT_CHANGE_CARRIER_FIELDS
    )


def executable_proposal_units(
    row: TravelProjectionRow,
    proposal_units: Sequence[str] | frozenset[str],
) -> frozenset[str]:
    """Normalize a raw proposal into the executable travel tool footprint.

    The strong-proposer probe emits an abstract footprint of slots it thinks are
    relevant. For existing-booking flight changes, the real STATE-Bench travel
    tool contract is stricter: any executable ``update_booking(flight_id=...)``
    call must also carry the six booking preference fields explicitly. Those
    carrier units are schema-required tool-call arguments, not gold answers.

    Keep this normalization in the travel adapter so replay scripts do not
    silently reinterpret raw provider slots as legal tool calls.
    """

    units = frozenset(str(unit) for unit in proposal_units)
    if row.action_closure_units and any(
        unit_family(unit) == "bookings.flight_id" for unit in units
    ):
        return frozenset(units | row.action_closure_units)
    return units


def choose_committed_colliders(committed: Sequence[str]) -> list[str]:
    """Pick committed units the over-broad proposal will also touch.

    We take one cheap and one irreversible colliding unit when available so the
    row carries a mixed-cost collision (the precondition for ``Lambda`` to
    bite). Falls back to the first committed unit if neither family is present.
    """

    cheap = [u for u in committed if unit_family(u) in CHEAP_RESTORABLE_FIELDS]
    irreversible = [u for u in committed if unit_family(u) in IRREVERSIBLE_FIELDS]
    colliders: list[str] = []
    if cheap:
        colliders.append(cheap[0])
    if irreversible:
        colliders.append(irreversible[0])
    if not colliders and committed:
        colliders.append(committed[0])
    return colliders


def projection_row_from_state_requirements(
    state_requirements: Sequence[Any],
    *,
    row_key: str,
    task_type: str = "unknown",
) -> TravelProjectionRow | None:
    """Project one travel row's state-requirements into a projection candidate.

    Returns ``None`` when the row is degenerate for ``pi_C`` (fewer than two
    abstract units, or no committed unit the proposal would collide with).
    """

    units = abstract_units(state_requirements)
    if len(units) < 2:
        return None
    residual = choose_residual_unit(units, task_type=task_type)
    closure_units = (
        frozenset()
        if task_type == "new_booking"
        else booking_flight_change_closure_units([residual])
    )
    # TYPED SPLIT (argument_carrier vs changed_effect). Closure carriers are
    # value-unchanged required ``update_booking(flight_id=...)`` arguments: they
    # make the call executable but do NOT change committed state. They must NOT
    # enter ``committed_units`` -> ``forbidden_writes`` (``C(z)``); otherwise
    # ``pi_C`` scores carrier-completion as a forbidden collision and ``Lambda``
    # is mis-credited for "re-admitting" them (the conflation that produced the
    # spurious decision-change fingerprint). Carriers live ONLY in
    # ``action_closure_units`` (the argument_carrier role), ``proposal_units``,
    # ``touched_units`` and the required-residual footprint (executability) --
    # never in the committed/forbidden set. Only true committed writes
    # (changed_effect / committed colliders) are committed here.
    committed = sorted(frozenset(unit for unit in units if unit != residual))
    if closure_units:
        colliders = [
            unit
            for unit in choose_committed_colliders(committed)
            if unit_family(unit) not in CHEAP_RESTORABLE_FIELDS
        ]
    else:
        colliders = choose_committed_colliders(committed)
    if not colliders and not closure_units:
        return None
    proposal = frozenset([residual, *colliders, *closure_units])
    return TravelProjectionRow(
        row_key=row_key,
        task_type=task_type,
        committed_units=frozenset(committed),
        residual_units=frozenset({residual}),
        proposal_units=proposal,
        # Cost vector is over COMMITTED writes only. Carriers (argument_carrier)
        # are value-unchanged required args, not committed overwrites: they must
        # NOT enter committed_restoration_cost, or writing a required carrier of
        # an otherwise fee-bearing field (e.g. cabin_class re-passed unchanged)
        # would be mis-counted as irreversible damage by the ledger.
        restoration_cost=restoration_cost_vector(committed),
        action_closure_units=closure_units,
        action_closure_reason=(
            "update_booking_flight_id_requires_explicit_booking_preferences"
            if closure_units
            else ""
        ),
        touched_units=closure_units,
        closure_metadata=(
            {
                "schema_required_carrier_units": tuple(sorted(closure_units)),
                "schema_required_carrier_count": len(closure_units),
                "schema_required_carrier_source": "update_booking_flight_id_tool_schema",
                "closure_units_are_not_forbidden_writes": True,
            }
            if closure_units
            else {}
        ),
    )


def build_projection_rows(
    tasks: Sequence[Mapping[str, Any]],
    *,
    task_type_normalizer: Any | None = None,
) -> list[TravelProjectionRow]:
    """Project a sequence of travel tasks into non-degenerate projection rows."""

    rows: list[TravelProjectionRow] = []
    for task in tasks:
        task_map = _mapping(task)
        raw_type = str(task_map.get("task_type") or "unknown")
        task_type = task_type_normalizer(raw_type) if task_type_normalizer else raw_type
        row = projection_row_from_state_requirements(
            _sequence(task_map.get("state_requirements")),
            row_key=f"{SURFACE}#{len(rows):03d}",
            task_type=task_type,
        )
        if row is not None:
            rows.append(row)
    return rows


def gold_required_field_reversibility_census(
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Structural ceiling for a travel distinctive-trio positive (NO-API, gold-only).

    Counts every DIRECT gold field-requirement (``entity.field`` on an existing
    record) across ``tasks`` by schema-derived reversibility class. A
    distinctive-trio positive needs a CHEAP-REVERSIBLE field to be a REQUIRED write
    (so ``Lambda`` can re-admit it against the budget while uniform ``pi_C`` drops
    it); it needs >=2 DISTINCT such field families to clear the ">=2 distinct
    structures" bar. This census reports how many cheap-reversible required field
    families exist at all. If fewer than two occur, no defensible adapter
    re-modeling can manufacture >=2 Lambda-load-bearing structures -- the surface
    is structurally capped, independent of typing/calibration choices.

    Created-record requirements (``match_fields``/``expected_fields`` with no
    ``field``) are clean ADDs on brand-new bookings, never in-place colliders, so
    they are excluded. Uses only gold + the schema reversibility classifier; no
    task ids, entity ids, expected values, or task_env pre-state are consumed.
    """

    task_list = list(tasks)
    direct_total = 0
    by_class: Counter[str] = Counter()
    cheap_field_families: Counter[str] = Counter()
    for task in task_list:
        for req_value in _sequence(_mapping(task).get("state_requirements")):
            req = _mapping(req_value)
            field_name = req.get("field")
            if field_name in (None, ""):
                continue
            direct_total += 1
            family = f"{req.get('entity_type') or 'unknown'}.{field_name}"
            by_class[classify_reversibility(travel_write_unit_signal(family))] += 1
            if family in CHEAP_RESTORABLE_FIELDS:
                cheap_field_families[family] += 1
    distinct_cheap = len(cheap_field_families)
    return {
        "task_count": len(task_list),
        "direct_field_requirement_count": direct_total,
        "by_reversibility_class": dict(sorted(by_class.items())),
        "cheap_reversible_required_field_count": int(
            sum(cheap_field_families.values())
        ),
        "cheap_reversible_required_field_families": dict(
            sorted(cheap_field_families.items())
        ),
        "distinct_cheap_reversible_required_field_families": distinct_cheap,
        "distinctive_positive_structurally_possible": distinct_cheap >= 2,
        "ceiling_reason": (
            "A distinctive-trio positive needs >=2 distinct cheap-reversible REQUIRED "
            "field families so Lambda re-admits a required cheap residual that uniform "
            f"pi_C drops. The corpus has {distinct_cheap}; the cheap preference fields "
            "are argument_carriers (value-unchanged required tool args), not required "
            "state writes, so they cannot be the load-bearing residual."
        ),
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return []


__all__ = [
    "SURFACE",
    "CHEAP_RESTORATION_COST",
    "IRREVERSIBLE_RESTORATION_COST",
    "IRREVERSIBLE_DAMAGE_THRESHOLD",
    "DEFAULT_RESTORATION_COST",
    "CHEAP_RESTORABLE_FIELDS",
    "BOOKING_FLIGHT_CHANGE_CARRIER_FIELDS",
    "IRREVERSIBLE_FIELDS",
    "TravelProjectionRow",
    "unit_family",
    "restoration_cost_for_unit",
    "restoration_cost_vector",
    "is_non_uniform",
    "abstract_units",
    "choose_residual_unit",
    "booking_flight_change_closure_units",
    "executable_proposal_units",
    "choose_committed_colliders",
    "projection_row_from_state_requirements",
    "build_projection_rows",
    "gold_required_field_reversibility_census",
]
