"""Slot-aware candidate generator for typed branch proposal graphs.

This module is benchmark-independent PAPC core. It bounds the candidate space the
minimum-authority selector must project: given a typed branch scaffold expressed
as ``one_of`` slot groups plus ``requires`` / ``excludes`` constraints, it
enumerates only the *legal slot assignments* (dependency- and requires-closed,
exclude-respecting, deterministic, capped) instead of the ``2**n``
dependency-closed subset enumeration the strict selector uses.

Why this exists. The strict selector enumerates every dependency-closed subset of
the proposal graph. That is fine for a single-slot scaffold, but the
authority-lifted online profile issues *branch* proposals (one residual slot with
several authority alternatives ``ell0``/``ell1``/``ell2``/``ell3``, plus
preview/confirm dependency nodes). Enumerating all subsets of such a graph is
exponential, and each subset triggers an inner projection enumeration. Slot-aware
enumeration keeps the number of candidate subgraphs bounded by the product of the
slot-group sizes, so adding authority alternatives does not blow up runtime.

Scorer-blind / public-safe. The generator operates only on opaque node ids and a
dependency map. It carries no dataset payloads, scorer fields, task ids, gold
labels, fixed tool arguments, or answer sketches; node-id public-safety is owned
by the :mod:`papc.certspec.cfrc_ir` proposal-graph validation that produced them.

Semantics.

* ``one_of`` group: an at-most-one slot. Each group contributes either one of its
  members or nothing. The empty (all-none) assignment is dropped, which is the
  correct fail-closed behavior (the selector then abstains).
* ``requires`` edge ``(a, b)``: if ``a`` is selected, ``b`` must be selected. It is
  merged with the proposal graph's own ``dependency_map`` into a single closure
  relation. A chosen node whose closure pulls in an id outside ``node_ids`` makes
  that assignment infeasible and it is dropped (only that alternative is lost).
* ``excludes`` pair ``(a, b)``: an assignment containing both is dropped.
* Ungrouped nodes are never independently toggled; they enter an assignment only
  by being pulled in through the closure relation. This is what keeps the
  candidate count bounded regardless of how many ungrouped nodes exist.

When no ``one_of`` group is declared the generator falls back to guarded
dependency-closed subset enumeration (with the same ``requires`` / ``excludes``
filters), so a constraints object with only ``excludes`` still behaves correctly;
callers that need the strict ``2**n`` behavior keep using the selector's default
path instead of constructing a constraints object.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass


class SlotEnumerationError(RuntimeError):
    """Raised when a slot scaffold's base assignment space is unsafely large.

    This is surfaced (rather than silently truncated) so a malformed scaffold is
    caught by the no-API dry-run gate instead of degrading an online run.
    """


@dataclass(frozen=True)
class SlotConstraints:
    """Typed branch-scaffold constraints over proposal-graph node ids.

    ``candidate_cap`` deterministically bounds the number of emitted assignments.
    ``max_base_combinations`` bounds the raw assignment space before enumeration
    and raises :class:`SlotEnumerationError` when exceeded.
    """

    one_of: tuple[tuple[str, ...], ...] = ()
    requires: tuple[tuple[str, str], ...] = ()
    excludes: tuple[tuple[str, str], ...] = ()
    candidate_cap: int = 128
    max_base_combinations: int = 4096

    def __post_init__(self) -> None:
        normalized_groups = tuple(
            tuple(_dedupe(group)) for group in self.one_of if tuple(group)
        )
        for group in normalized_groups:
            for node_id in group:
                if not str(node_id).strip():
                    raise ValueError("one_of node ids must be non-empty")
        object.__setattr__(self, "one_of", normalized_groups)
        object.__setattr__(
            self, "requires", tuple((str(a), str(b)) for a, b in self.requires)
        )
        object.__setattr__(
            self, "excludes", tuple((str(a), str(b)) for a, b in self.excludes)
        )
        if self.candidate_cap < 1:
            raise ValueError("candidate_cap must be >= 1")
        if self.max_base_combinations < 1:
            raise ValueError("max_base_combinations must be >= 1")

    @property
    def has_groups(self) -> bool:
        return bool(self.one_of)


def enumerate_slot_assignments(
    *,
    node_ids: Sequence[str],
    dependency_map: Mapping[str, Iterable[str]],
    constraints: SlotConstraints,
) -> list[tuple[str, ...]]:
    """Enumerate legal slot assignments as sorted, deduplicated id tuples.

    Each assignment is dependency- and requires-closed, exclude-respecting, and
    non-empty. The result is ordered by ``(len, ids)``. If the legal assignment
    count exceeds ``constraints.candidate_cap``, the scaffold is rejected rather
    than truncated; silently dropping assignments can remove the true
    minimum-authority candidate.
    """

    available = {str(node_id) for node_id in node_ids}
    closure_edges = _closure_edges(available, dependency_map, constraints.requires)
    excludes = tuple(
        (a, b) for a, b in constraints.excludes if a in available and b in available
    )

    if constraints.has_groups:
        assignments = _grouped_assignments(
            available, closure_edges, excludes, constraints
        )
    else:
        assignments = _ungrouped_assignments(
            available, closure_edges, excludes, constraints
        )

    ordered = sorted(assignments, key=lambda assignment: (len(assignment), assignment))
    if len(ordered) > constraints.candidate_cap:
        raise SlotEnumerationError(
            "slot scaffold legal assignment count exceeds "
            f"candidate_cap={constraints.candidate_cap}"
        )
    return ordered


def assignment_respects_slot_constraints(
    *,
    assignment: Iterable[str],
    node_ids: Sequence[str],
    dependency_map: Mapping[str, Iterable[str]],
    constraints: SlotConstraints,
) -> bool:
    """Validate one final assignment against every scaffold constraint.

    This is also used after the selector attaches a required dependent such as a
    terminal communication node.  Revalidation is necessary because that node was
    not part of the seed assignment enumerated by the slot generator.
    """

    available = {str(node_id) for node_id in node_ids}
    selected = frozenset(str(node_id) for node_id in assignment)
    if not selected or not selected <= available:
        return False
    closure_edges = _closure_edges(available, dependency_map, constraints.requires)
    if not _is_closed(selected, closure_edges):
        return False
    if _violates_excludes(selected, constraints.excludes):
        return False
    return not _violates_one_of(selected, constraints.one_of)


def _grouped_assignments(
    available: set[str],
    closure_edges: Mapping[str, set[str]],
    excludes: tuple[tuple[str, str], ...],
    constraints: SlotConstraints,
) -> set[tuple[str, ...]]:
    groups = [
        tuple(sorted(member for member in group if member in available))
        for group in constraints.one_of
    ]
    groups = [group for group in groups if group]
    if not groups:
        return _ungrouped_assignments(available, closure_edges, excludes, constraints)

    base_count = 1
    for group in groups:
        base_count *= len(group) + 1  # "+1" for the choose-none option
        if base_count > constraints.max_base_combinations:
            raise SlotEnumerationError(
                "slot scaffold base assignment space exceeds "
                f"max_base_combinations={constraints.max_base_combinations}"
            )

    choice_lists = [[None, *group] for group in groups]
    results: set[tuple[str, ...]] = set()
    for combo in itertools.product(*choice_lists):
        seed = {choice for choice in combo if choice is not None}
        if not seed:
            continue
        closed = _close(seed, available, closure_edges)
        if closed is None:
            continue
        if _violates_excludes(closed, excludes) or _violates_one_of(
            closed, constraints.one_of
        ):
            continue
        results.add(tuple(sorted(closed)))
    return results


def _ungrouped_assignments(
    available: set[str],
    closure_edges: Mapping[str, set[str]],
    excludes: tuple[tuple[str, str], ...],
    constraints: SlotConstraints,
) -> set[tuple[str, ...]]:
    nodes = sorted(available)
    count = len(nodes)
    if (1 << count) > constraints.max_base_combinations:
        raise SlotEnumerationError(
            f"ungrouped slot enumeration space 2**{count} exceeds "
            f"max_base_combinations={constraints.max_base_combinations}"
        )
    results: set[tuple[str, ...]] = set()
    for mask in range(1, 1 << count):
        subset = frozenset(
            nodes[index] for index in range(count) if mask & (1 << index)
        )
        if not _is_closed(subset, closure_edges):
            continue
        if _violates_excludes(subset, excludes) or _violates_one_of(
            subset, constraints.one_of
        ):
            continue
        results.add(tuple(sorted(subset)))
    return results


def _closure_edges(
    available: set[str],
    dependency_map: Mapping[str, Iterable[str]],
    requires: tuple[tuple[str, str], ...],
) -> dict[str, set[str]]:
    edges: dict[str, set[str]] = {}
    for node_id, deps in dependency_map.items():
        node = str(node_id)
        if node not in available:
            continue
        edges.setdefault(node, set()).update(str(dep) for dep in deps)
    for source, target in requires:
        edges.setdefault(str(source), set()).add(str(target))
    return edges


def _close(
    seed: Iterable[str],
    available: set[str],
    closure_edges: Mapping[str, set[str]],
) -> frozenset[str] | None:
    """Close ``seed`` under the closure relation, or ``None`` if infeasible.

    Infeasible means a pulled-in prerequisite id is not in ``available`` (e.g. it
    was filtered out as a non-executable node), so the alternative cannot be
    realized within the current graph.
    """

    work = set(seed)
    stack = list(seed)
    while stack:
        node = stack.pop()
        for required in closure_edges.get(node, ()):  # noqa: SIM118 - set membership
            if required not in available:
                return None
            if required not in work:
                work.add(required)
                stack.append(required)
    return frozenset(work)


def _is_closed(subset: frozenset[str], closure_edges: Mapping[str, set[str]]) -> bool:
    for node in subset:
        for required in closure_edges.get(node, ()):  # noqa: SIM118 - set membership
            if required not in subset:
                return False
    return True


def _violates_excludes(
    assignment: Iterable[str],
    excludes: tuple[tuple[str, str], ...],
) -> bool:
    members = set(assignment)
    return any(a in members and b in members for a, b in excludes)


def _violates_one_of(
    assignment: Iterable[str],
    groups: tuple[tuple[str, ...], ...],
) -> bool:
    members = set(assignment)
    return any(len(members & set(group)) > 1 for group in groups)


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


__all__ = [
    "SlotConstraints",
    "SlotEnumerationError",
    "assignment_respects_slot_constraints",
    "enumerate_slot_assignments",
]
