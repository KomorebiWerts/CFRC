"""Typed executable closure operator gamma_z (benchmark-independent core).

``gamma_z`` is the post-commit operator that turns an INCOMPLETE semantic residual
proposal into an EXECUTABLE official tool call. Given a reached-state certificate
``z``, it recovers the schema-required *argument carriers* -- values an official
tool call must retransmit unchanged (e.g. existing record fields, original values
read back from committed state) -- and adds them to the proposal's executable
footprint.

Position in the method chain::

    semantic residual proposal
      -> gamma_z (typed executable closure)            <-- THIS module
      -> pi_C over SEMANTIC effects only
      -> Lambda(E(a') cap C(z)) <= rho_alpha^(d) ONLY if semantic committed effects collide

The defining contract (why ``gamma_z`` is a separate operator, not part of
``pi_C`` / ``Lambda``): argument carriers are EXECUTABILITY requirements, not state
changes. They are value-unchanged retransmissions recovered from ``z``. They MUST
stay out of the committed/forbidden set ``C(z)``, out of ``pi_C``'s collision
constraint, and out of ``Lambda``'s restoration-cost budget -- carrying a required
argument is not damage and is never restoration-cost work. ``gamma_z`` therefore
may ADD ``argument_carrier`` units but must NOT add any semantic write.

This module is dataset-independent: it carries only abstract unit tokens and
provenance TAGS, never recovered concrete values, so it is public-safe. A dataset
adapter (e.g. STATE-Bench travel) instantiates ``gamma_z`` by recovering carriers
from its own reached-state representation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# Provenance is a TAG only -- never a concrete recovered value, so an
# ExecutableClosure is safe to surface in a public/no-API artifact.
PROVENANCE_UNRECOVERABLE = "unrecoverable"
PROVENANCE_SELECTED_SUFFIX_NODE = "produced_by_selected_suffix_node"
CARRIER_PROVENANCE_TAGS: frozenset[str] = frozenset(
    {
        "observed_committed_record",
        "observed_account_preference",
        PROVENANCE_SELECTED_SUFFIX_NODE,
        PROVENANCE_UNRECOVERABLE,
    }
)


@dataclass(frozen=True)
class ExecutableClosure:
    """Result of applying ``gamma_z`` to one residual proposal against reached-state ``z``.

    All units are abstract tokens (``entity.field#occ``). ``carrier_provenance``
    maps each carrier unit to a provenance TAG, never to a recovered value, so the
    object is public-safe.
    """

    semantic_effects: frozenset[str]
    argument_carriers: frozenset[str]
    carrier_provenance: Mapping[str, str] = field(default_factory=dict)
    executable_footprint: frozenset[str] = field(default_factory=frozenset)

    @property
    def adds_no_semantic_write(self) -> bool:
        """gamma_z invariant: a carrier must never be a semantic write."""

        return self.argument_carriers.isdisjoint(self.semantic_effects)

    @property
    def footprint_is_consistent(self) -> bool:
        return self.executable_footprint == (
            self.semantic_effects | self.argument_carriers
        )

    @property
    def is_valid(self) -> bool:
        return self.adds_no_semantic_write and self.footprint_is_consistent

    @property
    def recovered_carriers(self) -> frozenset[str]:
        """Carriers ``gamma_z`` resolved uniquely from ``z`` (tag != unrecoverable)."""

        return frozenset(
            unit
            for unit in self.argument_carriers
            if self.carrier_provenance.get(unit, PROVENANCE_UNRECOVERABLE)
            != PROVENANCE_UNRECOVERABLE
        )

    @property
    def fully_executable(self) -> bool:
        """True iff every required carrier was recovered from ``z`` (call is issuable)."""

        return self.recovered_carriers == self.argument_carriers

    def assert_valid(self) -> None:
        if not self.adds_no_semantic_write:
            raise ValueError(
                "gamma_z invariant violated: argument_carriers overlap semantic_effects "
                "(a carrier was scored as a semantic write)"
            )
        if not self.footprint_is_consistent:
            raise ValueError(
                "gamma_z invariant violated: executable_footprint != "
                "semantic_effects | argument_carriers"
            )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "papc.executable_closure.v1",
            "semantic_effects": sorted(self.semantic_effects),
            "argument_carriers": sorted(self.argument_carriers),
            "carrier_provenance": {
                unit: self.carrier_provenance[unit]
                for unit in sorted(self.carrier_provenance)
            },
            "executable_footprint": sorted(self.executable_footprint),
            "adds_no_semantic_write": self.adds_no_semantic_write,
            "fully_executable": self.fully_executable,
            "is_valid": self.is_valid,
        }


def build_executable_closure(
    *,
    semantic_effects: frozenset[str],
    argument_carriers: frozenset[str],
    carrier_provenance: Mapping[str, str] | None = None,
) -> ExecutableClosure:
    """Apply ``gamma_z``: combine semantic effects with recovered argument carriers.

    Derives ``executable_footprint = semantic_effects | argument_carriers`` and
    enforces the ``gamma_z`` invariant (raises if a carrier would add a semantic
    write). The caller supplies carriers + provenance tags already recovered from
    its reached-state ``z``; this function does not read any dataset state itself.
    """

    semantic = frozenset(semantic_effects)
    carriers = frozenset(argument_carriers)
    closure = ExecutableClosure(
        semantic_effects=semantic,
        argument_carriers=carriers,
        carrier_provenance=dict(carrier_provenance or {}),
        executable_footprint=semantic | carriers,
    )
    closure.assert_valid()
    return closure


def output_only_recovered_carriers(closure: ExecutableClosure) -> frozenset[str]:
    """Carriers an OUTPUT-ONLY escalation policy could retransmit unchanged: none.

    An output-only policy is a deterministic function of the prompt ``x`` and the cheap
    output ``y_c`` only; by its information set it never reads the reached-state
    certificate ``z``. Every argument carrier is, by construction, a value recovered FROM
    ``z`` (its provenance tag is one of :data:`CARRIER_PROVENANCE_TAGS` -- an observed
    committed record / account preference, or ``unrecoverable`` -- never "derivable from
    prompt or cheap output"). Hence the set of carriers an output-only policy can supply
    is empty whenever the official call requires any carrier. This is the output-only side
    of :func:`executable_closure_separation` (Theorem 8 in the proof appendix).
    """

    return frozenset()


def build_executable_closure_output_only(
    *,
    semantic_effects: frozenset[str],
    argument_carriers: frozenset[str],
) -> ExecutableClosure:
    """Build the OUTPUT-ONLY twin of a ``gamma_z`` closure.

    The twin shares the same semantic effects and the same schema-required argument
    carriers, but recovers NONE of them (every carrier provenance is forced to
    :data:`PROVENANCE_UNRECOVERABLE`), because an output-only policy cannot read ``z``.
    Its :attr:`ExecutableClosure.fully_executable` is therefore ``False`` whenever any
    carrier is required, while a ``gamma_z`` closure that recovered the carriers from
    ``z`` is issuable. Carries only abstract unit tokens, so it is public-safe.
    """

    semantic = frozenset(semantic_effects)
    carriers = frozenset(argument_carriers)
    closure = ExecutableClosure(
        semantic_effects=semantic,
        argument_carriers=carriers,
        carrier_provenance={unit: PROVENANCE_UNRECOVERABLE for unit in carriers},
        executable_footprint=semantic | carriers,
    )
    closure.assert_valid()
    return closure


def executable_closure_separation(closure: ExecutableClosure) -> dict[str, Any]:
    """Contrast a ``gamma_z`` closure against its output-only twin on EXECUTABILITY.

    Returns a no-API executable-closure separation witness: an output-only policy, being a function of
    ``(x, y_c)`` only, recovers zero schema-required carriers, so its official call is NOT
    issuable whenever a carrier is required; ``gamma_z`` reads ``z`` and -- when the
    carriers are recoverable -- issues the call. ``separates`` is ``True`` iff at least one
    carrier is required AND ``gamma_z`` is fully executable while the output-only twin is
    not. All fields are computed from the abstract closure, so the result is public-safe.
    """

    twin = build_executable_closure_output_only(
        semantic_effects=closure.semantic_effects,
        argument_carriers=closure.argument_carriers,
    )
    requires_carrier = bool(closure.argument_carriers)
    return {
        "schema": "papc.executable_closure_separation.v1",
        "requires_carrier": requires_carrier,
        "gamma_z_fully_executable": closure.fully_executable,
        "output_only_fully_executable": twin.fully_executable,
        "output_only_recovered_carrier_count": len(
            output_only_recovered_carriers(closure)
        ),
        "separates": (
            requires_carrier and closure.fully_executable and not twin.fully_executable
        ),
    }


__all__ = [
    "ExecutableClosure",
    "build_executable_closure",
    "build_executable_closure_output_only",
    "executable_closure_separation",
    "output_only_recovered_carriers",
    "CARRIER_PROVENANCE_TAGS",
    "PROVENANCE_SELECTED_SUFFIX_NODE",
    "PROVENANCE_UNRECOVERABLE",
]
