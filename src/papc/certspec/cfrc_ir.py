"""Shared benchmark-independent CFRC IR objects.

These objects are public-safe runtime/core types. They carry abstract units,
authority labels, dependency structure, and provenance references only. They
must not carry dataset task ids, scorer text, reward signals, fixed tool
arguments, fixed KB queries, private paths, private receipts, or answer sketches.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from papc.certspec.executable_closure import ExecutableClosure

ProposalNodeKind = Literal["tool_call", "message", "repair", "abstain"]
ProposalNodeRole = Literal[
    "obligation_effect",
    "evidence_acquisition",
    "terminal_communication",
    "discharge_declaration",
]

_FORBIDDEN_PUBLIC_TOKENS: tuple[str, ...] = (
    "task_id",
    "gold",
    "direct_outcome",
    "scorer",
    "answer_sketch",
    "private_root",
    "private_receipt",
    "fixed_args",
    "fixed_query",
    "fixed_kb_query",
    "kb_query",
    "reward",
)


def _validate_public_safe_text(value: str, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    lowered = text.lower()
    token_text = lowered.replace("-", "_").replace(" ", "_")
    if text.startswith("/"):
        raise ValueError(f"{field_name} must not contain absolute paths")
    if "\\" in text:
        raise ValueError(f"{field_name} must not contain filesystem separators")
    if ".json" in lowered:
        raise ValueError(f"{field_name} must not contain artifact filenames")
    for token in _FORBIDDEN_PUBLIC_TOKENS:
        if token in token_text:
            raise ValueError(
                f"{field_name} contains forbidden runtime-visible token: {token}"
            )
    return text


def _ordered_unique(values: Iterable[str], *, field_name: str) -> tuple[str, ...]:
    items: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = _validate_public_safe_text(str(raw), field_name=field_name)
        if value in seen:
            continue
        seen.add(value)
        items.append(value)
    return tuple(items)


def _sorted_unique(values: Iterable[str], *, field_name: str) -> tuple[str, ...]:
    return tuple(sorted(_ordered_unique(values, field_name=field_name)))


def _validate_public_safe_mapping(
    values: Mapping[str, str],
    *,
    key_field_name: str,
    value_field_name: str,
) -> dict[str, str]:
    sanitized: dict[str, str] = {}
    for key, value in values.items():
        safe_key = _validate_public_safe_text(str(key), field_name=key_field_name)
        safe_value = _validate_public_safe_text(str(value), field_name=value_field_name)
        sanitized[safe_key] = safe_value
    return sanitized


def _is_opaque_sha256_ref(value: str) -> bool:
    prefix = "sha256:"
    text = str(value)
    if not text.startswith(prefix):
        return False
    digest = text[len(prefix) :]
    return len(digest) == 64 and all(
        character in "0123456789abcdef" for character in digest
    )


@dataclass(frozen=True)
class Commitment:
    """Abstract committed unit bundle in the fixed certificate boundary."""

    commitment_id: str
    unit_tokens: tuple[str, ...]
    effect_families: tuple[str, ...] = ()
    provenance_refs: tuple[str, ...] = ()
    authority_label: str = ""

    def __post_init__(self) -> None:
        _validate_public_safe_text(self.commitment_id, field_name="commitment_id")
        _ordered_unique(self.unit_tokens, field_name="unit_tokens")
        _ordered_unique(self.effect_families, field_name="effect_families")
        _ordered_unique(self.provenance_refs, field_name="provenance_refs")
        if self.authority_label:
            _validate_public_safe_text(
                self.authority_label, field_name="authority_label"
            )

    def to_json(self) -> dict[str, Any]:
        return {
            "commitment_id": self.commitment_id,
            "unit_tokens": list(self.unit_tokens),
            "effect_families": list(self.effect_families),
            "provenance_refs": list(self.provenance_refs),
            "authority_label": self.authority_label,
        }


@dataclass(frozen=True)
class ResidualObligation:
    """Abstract residual obligation over units, not dataset-specific answers."""

    obligation_id: str
    required_unit_tokens: tuple[str, ...]
    effect_families: tuple[str, ...] = ()
    provenance_refs: tuple[str, ...] = ()
    full_discharge_required: bool = True

    def __post_init__(self) -> None:
        _validate_public_safe_text(self.obligation_id, field_name="obligation_id")
        _ordered_unique(self.required_unit_tokens, field_name="required_unit_tokens")
        _ordered_unique(self.effect_families, field_name="effect_families")
        _ordered_unique(self.provenance_refs, field_name="provenance_refs")

    def to_json(self) -> dict[str, Any]:
        return {
            "obligation_id": self.obligation_id,
            "required_unit_tokens": list(self.required_unit_tokens),
            "effect_families": list(self.effect_families),
            "provenance_refs": list(self.provenance_refs),
            "full_discharge_required": self.full_discharge_required,
        }


@dataclass(frozen=True)
class EvidenceFact:
    """Visible evidence reference carried into the fixed certificate boundary."""

    fact_id: str
    fact_type: str
    unit_tokens: tuple[str, ...] = ()
    provenance_refs: tuple[str, ...] = ()
    authority_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_public_safe_text(self.fact_id, field_name="fact_id")
        _validate_public_safe_text(self.fact_type, field_name="fact_type")
        _ordered_unique(self.unit_tokens, field_name="unit_tokens")
        _ordered_unique(self.provenance_refs, field_name="provenance_refs")
        _ordered_unique(self.authority_labels, field_name="authority_labels")

    def to_json(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "fact_type": self.fact_type,
            "unit_tokens": list(self.unit_tokens),
            "provenance_refs": list(self.provenance_refs),
            "authority_labels": list(self.authority_labels),
        }


@dataclass(frozen=True)
class ProposalCoverageClaim:
    """Proposal-time obligation coverage asserted by one native action."""

    obligation_id: str
    unit_tokens: tuple[str, ...]
    replica_receipt_ref: str

    def __post_init__(self) -> None:
        _validate_public_safe_text(self.obligation_id, field_name="obligation_id")
        units = tuple(
            _validate_public_safe_text(str(unit), field_name="coverage_unit_tokens")
            for unit in self.unit_tokens
        )
        if len(units) != len(set(units)):
            raise ValueError("coverage unit tokens must be unique")
        object.__setattr__(self, "unit_tokens", units)
        _validate_public_safe_text(
            self.replica_receipt_ref,
            field_name="replica_receipt_ref",
        )
        if not _is_opaque_sha256_ref(self.replica_receipt_ref):
            raise ValueError("replica_receipt_ref must be an opaque sha256 digest")

    def to_json(self) -> dict[str, Any]:
        return {
            "obligation_id": self.obligation_id,
            "unit_tokens": list(self.unit_tokens),
            "replica_receipt_ref": self.replica_receipt_ref,
        }


@dataclass(frozen=True)
class ProposalNode:
    """One typed node in a benchmark-independent proposal graph."""

    node_id: str
    kind: ProposalNodeKind
    authority_label: str
    node_role: ProposalNodeRole = "obligation_effect"
    semantic_effects: tuple[str, ...] = ()
    effect_families: tuple[str, ...] = ()
    evidence_output_refs: tuple[str, ...] = ()
    dependency_ids: tuple[str, ...] = ()
    provenance_refs: tuple[str, ...] = ()
    evidence_fact_ids: tuple[str, ...] = ()
    discharged_obligation_ids: tuple[str, ...] = ()
    coverage_claims: tuple[ProposalCoverageClaim, ...] = ()
    executable_closure: ExecutableClosure | None = None
    tool_call_count: int = 0

    def __post_init__(self) -> None:
        _validate_public_safe_text(self.node_id, field_name="node_id")
        _validate_public_safe_text(self.authority_label, field_name="authority_label")
        if self.node_role not in {
            "obligation_effect",
            "evidence_acquisition",
            "terminal_communication",
            "discharge_declaration",
        }:
            raise ValueError("proposal node role is invalid")
        _ordered_unique(self.semantic_effects, field_name="semantic_effects")
        _ordered_unique(self.effect_families, field_name="effect_families")
        _ordered_unique(self.evidence_output_refs, field_name="evidence_output_refs")
        _ordered_unique(self.dependency_ids, field_name="dependency_ids")
        _ordered_unique(self.provenance_refs, field_name="provenance_refs")
        _ordered_unique(self.evidence_fact_ids, field_name="evidence_fact_ids")
        _ordered_unique(
            self.discharged_obligation_ids,
            field_name="discharged_obligation_ids",
        )
        if self.discharged_obligation_ids and self.kind != "message":
            raise ValueError(
                "explicit obligation discharge is supported only by message nodes"
            )
        if self.coverage_claims and self.kind != "tool_call":
            raise ValueError("coverage claims are supported only by tool-call nodes")
        if self.tool_call_count < 0:
            raise ValueError("tool_call_count must be non-negative")
        if self.executable_closure is not None:
            self.executable_closure.assert_valid()
            if (
                frozenset(self.semantic_effects)
                != self.executable_closure.semantic_effects
            ):
                raise ValueError(
                    "semantic_effects must match executable_closure.semantic_effects"
                )
            _validate_public_safe_mapping(
                self.executable_closure.carrier_provenance,
                key_field_name="carrier_provenance_key",
                value_field_name="carrier_provenance_value",
            )
        if self.kind == "abstain" and self.semantic_effects:
            raise ValueError("abstain nodes must not declare semantic effects")
        if self.node_role == "discharge_declaration":
            # A declaration names the obligation for which the model requests a
            # failure return. It is not successful coverage, and its zero
            # authority has to be a property of the type rather than a
            # convention the constructing bridge happens to honor.
            if self.kind != "message":
                raise ValueError("discharge declaration nodes must be messages")
            if not self.discharged_obligation_ids:
                raise ValueError(
                    "discharge declaration nodes must name an obligation"
                )
            if self.semantic_effects or self.effect_families:
                raise ValueError(
                    "discharge declaration nodes must not declare semantic effects"
                )
            if self.evidence_output_refs:
                raise ValueError("discharge declaration nodes must not produce evidence")
            if (
                self.executable_closure is not None
                and self.executable_closure.executable_footprint
            ):
                raise ValueError(
                    "discharge declaration nodes must carry an empty executable footprint"
                )
        elif self.node_role == "evidence_acquisition":
            if self.kind != "tool_call":
                raise ValueError("evidence acquisition nodes must be tool calls")
            if self.semantic_effects or self.effect_families:
                raise ValueError(
                    "evidence acquisition nodes must not declare semantic effects"
                )
            if not self.evidence_output_refs:
                raise ValueError(
                    "evidence acquisition nodes require abstract evidence outputs"
                )
        elif self.node_role == "terminal_communication":
            if self.kind != "message":
                raise ValueError("terminal communication nodes must be messages")
            if self.semantic_effects or self.effect_families:
                raise ValueError(
                    "terminal communication nodes must not declare semantic effects"
                )
            if self.evidence_output_refs:
                raise ValueError(
                    "terminal communication nodes must not produce evidence"
                )
            if not self.dependency_ids:
                raise ValueError(
                    "terminal communication nodes require a selected dependency"
                )
        elif self.evidence_output_refs and self.kind != "tool_call":
            raise ValueError(
                "only tool-call nodes may produce receipt-supported evidence outputs"
            )

    @property
    def semantic_effect_set(self) -> frozenset[str]:
        return frozenset(self.semantic_effects)

    @property
    def executable(self) -> bool:
        return (
            self.executable_closure is None or self.executable_closure.fully_executable
        )

    @property
    def effective_tool_call_count(self) -> int:
        if self.tool_call_count > 0:
            return self.tool_call_count
        return 1 if self.kind == "tool_call" else 0

    def to_json(self) -> dict[str, Any]:
        payload = {
            "node_id": self.node_id,
            "kind": self.kind,
            "node_role": self.node_role,
            "authority_label": self.authority_label,
            "semantic_effects": list(self.semantic_effects),
            "effect_families": list(self.effect_families),
            "evidence_output_refs": list(self.evidence_output_refs),
            "dependency_ids": list(self.dependency_ids),
            "provenance_refs": list(self.provenance_refs),
            "evidence_fact_ids": list(self.evidence_fact_ids),
            "tool_call_count": self.effective_tool_call_count,
            "has_executable_closure": self.executable_closure is not None,
            "executable_closure": (
                self.executable_closure.to_json()
                if self.executable_closure is not None
                else None
            ),
        }
        if self.discharged_obligation_ids:
            payload["discharged_obligation_ids"] = list(
                self.discharged_obligation_ids
            )
        if self.coverage_claims:
            payload["coverage_claims"] = [
                claim.to_json() for claim in self.coverage_claims
            ]
        return payload


@dataclass(frozen=True)
class ProposalGraph:
    """Typed proposal graph over public-safe nodes and stable dependencies."""

    nodes: tuple[ProposalNode, ...]
    graph_id: str = "proposal_graph"
    commitment_ids: tuple[str, ...] = ()
    residual_obligation_ids: tuple[str, ...] = ()
    evidence_fact_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_public_safe_text(self.graph_id, field_name="graph_id")
        _ordered_unique(self.commitment_ids, field_name="commitment_ids")
        _ordered_unique(
            self.residual_obligation_ids, field_name="residual_obligation_ids"
        )
        _ordered_unique(self.evidence_fact_ids, field_name="evidence_fact_ids")
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("proposal graph node ids must be unique")
        evidence_outputs = [
            output_ref
            for node in self.nodes
            for output_ref in node.evidence_output_refs
        ]
        if len(evidence_outputs) != len(set(evidence_outputs)):
            raise ValueError("proposal graph evidence output refs must be unique")
        known_ids = set(node_ids)
        terminal_nodes = tuple(
            node for node in self.nodes if node.node_role == "terminal_communication"
        )
        if len(terminal_nodes) > 1:
            raise ValueError(
                "proposal graph supports at most one terminal communication node"
            )
        for node in self.nodes:
            missing = set(node.dependency_ids) - known_ids
            if missing:
                raise ValueError(
                    f"proposal node {node.node_id} depends on unknown ids: {sorted(missing)}"
                )
        # Candidate graphs may be incomplete.  In particular, a proposer can
        # acquire evidence and then terminate without producing an obligation
        # effect.  CFRC owns global discharge and must abstain on that graph;
        # reject-only intentionally exposes the result of omitting that check.

    def node_by_id(self) -> dict[str, ProposalNode]:
        return {node.node_id: node for node in self.nodes}

    def stable_node_ids(self) -> tuple[str, ...]:
        return tuple(
            node.node_id for node in sorted(self.nodes, key=lambda item: item.node_id)
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "papc.cfrc_ir.proposal_graph.v1",
            "graph_id": self.graph_id,
            "commitment_ids": list(self.commitment_ids),
            "residual_obligation_ids": list(self.residual_obligation_ids),
            "evidence_fact_ids": list(self.evidence_fact_ids),
            "stable_node_ids": list(self.stable_node_ids()),
            "nodes": [
                node.to_json()
                for node in sorted(self.nodes, key=lambda item: item.node_id)
            ],
        }


@dataclass(frozen=True)
class AuthorityLevel:
    """Configurable authority label with adapter-supplied rank."""

    label: str
    rank: int
    effect_families: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        _validate_public_safe_text(self.label, field_name="label")
        if self.rank < 0:
            raise ValueError("authority rank must be non-negative")
        _ordered_unique(self.effect_families, field_name="effect_families")
        if self.description:
            _validate_public_safe_text(self.description, field_name="description")

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "rank": self.rank,
            "effect_families": list(self.effect_families),
            "description": self.description,
        }


@dataclass(frozen=True)
class AuthorityLattice:
    """Adapter-configured authority ordering with no baked-in benchmark semantics."""

    levels: tuple[AuthorityLevel, ...]

    def __post_init__(self) -> None:
        labels = [level.label for level in self.levels]
        if len(labels) != len(set(labels)):
            raise ValueError("authority labels must be unique")

    @classmethod
    def from_ordered_labels(cls, labels: Sequence[str]) -> AuthorityLattice:
        return cls(
            tuple(
                AuthorityLevel(label=str(label).strip(), rank=index)
                for index, label in enumerate(labels)
            )
        )

    def rank_for(self, label: str) -> int:
        for level in self.levels:
            if level.label == label:
                return level.rank
        raise KeyError(f"unknown authority label: {label}")

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "papc.cfrc_ir.authority_lattice.v1",
            "levels": [
                level.to_json()
                for level in sorted(
                    self.levels, key=lambda item: (item.rank, item.label)
                )
            ],
        }


__all__ = [
    "AuthorityLattice",
    "AuthorityLevel",
    "Commitment",
    "EvidenceFact",
    "ProposalCoverageClaim",
    "ProposalGraph",
    "ProposalNode",
    "ProposalNodeKind",
    "ProposalNodeRole",
    "ResidualObligation",
]
