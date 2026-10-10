"""Derive which earlier receipt produced each argument value of a later call.

The shared compiler already enforces that a value created during completion may
be consumed only through a receipt from a selected predecessor node: a carrier
tagged :data:`PROVENANCE_SELECTED_SUFFIX_NODE` must name an evidence output of
an earlier selected node, and the consumer must declare that node as a
dependency.  Surface bridges never supplied those carriers, so the enforcement
path stayed inert and the proposal graph carried order edges only.

This module supplies the missing derivation.  It is benchmark-independent and
reads only the arguments a call actually used and the results earlier calls
actually returned.  It records opaque node identifiers, result paths, and an
encoding tag, never a recovered value, so its output is public-safe.

Attribution is deliberately conservative, because a false edge would make the
compiler reject a suffix it admits today.  A value is attributed to a receipt
only when it was never visible as an input before that receipt returned it,
which keeps request-supplied identifiers out of the derived dependencies.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, unquote

from papc.certspec.executable_closure import PROVENANCE_SELECTED_SUFFIX_NODE

SCHEMA = "papc.receipt_provenance.v1"
CARRIER_PREFIX = "grounded_receipt_"
ENCODING_IDENTITY = "identity"
ENCODING_PERCENT = "percent"
MATCH_VALUE = "value"
MATCH_SUBSTRING = "substring"
DEFAULT_MINIMUM_VALUE_LENGTH = 4
MINIMUM_SUBSTRING_VALUE_LENGTH = 6
# A surface whose call arguments are a command string carries the produced value
# inside a larger scalar, so a token-delimited containment is the only available
# match.  The delimiter class keeps a short identifier from matching the middle
# of an unrelated token.
_TOKEN_BOUNDARY = r"[A-Za-z0-9_.@%-]"
_RFC3339_LIKE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}"
    r"(?:[Tt ]\d{2}:\d{2}"
    r"(?::\d{2}(?:\.\d+)?)?"
    r"(?:[Zz]|[+-]\d{2}:\d{2})?)?$"
)


@dataclass(frozen=True)
class ValueProvenance:
    """One argument value traced back to the receipt that produced it."""

    argument_path: str
    source_node_id: str
    source_result_path: str
    encoding: str = ENCODING_IDENTITY
    match_kind: str = MATCH_VALUE

    def __post_init__(self) -> None:
        if not str(self.argument_path).strip():
            raise ValueError("value provenance requires an argument path")
        if not str(self.source_node_id).strip():
            raise ValueError("value provenance requires a source node id")
        if not str(self.source_result_path).strip():
            raise ValueError("value provenance requires a source result path")
        if self.encoding not in {ENCODING_IDENTITY, ENCODING_PERCENT}:
            raise ValueError("value provenance encoding is not supported")
        if self.match_kind not in {MATCH_VALUE, MATCH_SUBSTRING}:
            raise ValueError("value provenance match kind is not supported")

    @property
    def carrier_unit(self) -> str:
        return carrier_unit_for(self.source_node_id)

    def to_json(self) -> dict[str, Any]:
        return {
            "argument_path": self.argument_path,
            "source_node_id": self.source_node_id,
            "source_result_path": self.source_result_path,
            "encoding": self.encoding,
            "match_kind": self.match_kind,
        }


def carrier_unit_for(node_id: str) -> str:
    """Return the carrier unit naming the receipt of one grounded node."""

    return f"{CARRIER_PREFIX}{node_id}"


def derive_value_provenance(
    *,
    arguments: Mapping[str, Any],
    prior_calls: Sequence[Mapping[str, Any]],
    frontier_values: frozenset[str] = frozenset(),
    minimum_value_length: int = DEFAULT_MINIMUM_VALUE_LENGTH,
    include_numbers: bool = False,
) -> tuple[ValueProvenance, ...]:
    """Trace each argument scalar of one call to the receipt that produced it.

    Every entry of ``prior_calls`` is a mapping with ``node_id``, ``arguments``,
    and ``result``, given in the order the calls ran.  A value counts as
    receipt-produced only when no earlier argument and no frontier value already
    carried it, so an identifier taken from the request is never mistaken for a
    value that execution must obtain.
    """

    known: set[str] = {str(value) for value in frontier_values}
    produced: dict[str, tuple[str, str]] = {}
    for row in prior_calls:
        node_id = str(row.get("node_id") or "")
        if not node_id:
            raise ValueError("prior grounded call requires a node id")
        raw_arguments = row.get("arguments")
        for _, value in _scalar_paths(
            raw_arguments if isinstance(raw_arguments, (Mapping, list, tuple)) else {},
            minimum_value_length=minimum_value_length,
            include_numbers=include_numbers,
        ):
            known.add(value)
        for path, value in _scalar_paths(
            row.get("result"),
            minimum_value_length=minimum_value_length,
            include_numbers=include_numbers,
        ):
            if value in known or value in produced:
                continue
            produced[value] = (node_id, path)

    records: list[ValueProvenance] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for argument_path, value in _scalar_paths(
        arguments,
        minimum_value_length=minimum_value_length,
        include_numbers=include_numbers,
    ):
        for node_id, result_path, encoding, match_kind in _match_produced_values(
            value, produced
        ):
            key = (argument_path, node_id, result_path, encoding, match_kind)
            if key in seen:
                continue
            seen.add(key)
            records.append(
                ValueProvenance(
                    argument_path=argument_path,
                    source_node_id=node_id,
                    source_result_path=result_path,
                    encoding=encoding,
                    match_kind=match_kind,
                )
            )
    return tuple(records)


def provenance_carrier_units(
    provenance: Sequence[ValueProvenance],
) -> frozenset[str]:
    """Return the carrier units required by one call, one per source node."""

    return frozenset(record.carrier_unit for record in provenance)


def provenance_carrier_tags(
    provenance: Sequence[ValueProvenance],
) -> dict[str, str]:
    """Tag every derived carrier as produced by a node of the selected suffix."""

    return {
        record.carrier_unit: PROVENANCE_SELECTED_SUFFIX_NODE for record in provenance
    }


def provenance_dependency_ids(
    provenance: Sequence[ValueProvenance],
) -> tuple[str, ...]:
    """Return the source nodes a consumer depends on, in stable order."""

    return tuple(dict.fromkeys(record.source_node_id for record in provenance))


def provenance_public_summary(
    provenance: Sequence[ValueProvenance],
) -> dict[str, Any]:
    """Summarize derived provenance without disclosing any recovered value."""

    encodings: dict[str, int] = {}
    match_kinds: dict[str, int] = {}
    for record in provenance:
        encodings[record.encoding] = encodings.get(record.encoding, 0) + 1
        match_kinds[record.match_kind] = match_kinds.get(record.match_kind, 0) + 1
    return {
        "schema": SCHEMA,
        "value_provenance_edge_count": len(provenance),
        "value_provenance_source_node_count": len(
            provenance_dependency_ids(provenance)
        ),
        "value_provenance_encodings": {
            key: encodings[key] for key in sorted(encodings)
        },
        "value_provenance_match_kinds": {
            key: match_kinds[key] for key in sorted(match_kinds)
        },
        "recovered_values_public": False,
    }


def _match_produced_values(
    value: str,
    produced: Mapping[str, tuple[str, str]],
) -> list[tuple[str, str, str, str]]:
    """Return every produced value this argument scalar carries.

    A structured argument carries one value and matches by equality.  A command
    string carries the value inside a larger scalar and matches by
    token-delimited containment, so one such argument can depend on several
    predecessors at once.
    """

    direct = produced.get(value)
    if direct is not None:
        return [(direct[0], direct[1], ENCODING_IDENTITY, MATCH_VALUE)]
    decoded = unquote(value)
    if decoded != value:
        encoded_source = produced.get(decoded)
        if encoded_source is not None:
            return [
                (encoded_source[0], encoded_source[1], ENCODING_PERCENT, MATCH_VALUE)
            ]

    matches: list[tuple[str, str, str, str]] = []
    seen_sources: set[tuple[str, str]] = set()
    for candidate in sorted(produced, key=len, reverse=True):
        if len(candidate) >= len(value) or not _is_identifier_like(candidate):
            continue
        node_id, path = produced[candidate]
        if (node_id, path) in seen_sources:
            continue
        if _contains_token(value, candidate):
            matches.append((node_id, path, ENCODING_IDENTITY, MATCH_SUBSTRING))
            seen_sources.add((node_id, path))
            continue
        encoded = quote(candidate, safe="")
        if encoded != candidate and _contains_token(value, encoded):
            matches.append((node_id, path, ENCODING_PERCENT, MATCH_SUBSTRING))
            seen_sources.add((node_id, path))
    return matches


def _is_identifier_like(value: str) -> bool:
    """Report whether a produced value may be matched inside a larger scalar.

    Equality already proves that a structured argument carries exactly one
    produced value.  Containment does not, since a later call may happen to
    write the same ordinary word, so containment is restricted to values that
    the environment plausibly minted.  Refusing a real edge only leaves the
    call as unbound as it is today, while accepting a false one would reject a
    suffix the compiler admits.
    """

    if len(value) < MINIMUM_SUBSTRING_VALUE_LENGTH:
        return False
    if any(character.isspace() for character in value):
        return False
    if _RFC3339_LIKE_RE.fullmatch(value):
        return False
    return any(character.isdigit() for character in value) or any(
        character in "-_@:./" for character in value
    )


def _embedded_json(text: str) -> Any | None:
    stripped = text.lstrip()
    if not stripped or stripped[0] not in "{[":
        return None
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, (Mapping, list)) else None


def _contains_token(haystack: str, needle: str) -> bool:
    pattern = (
        rf"(?<!{_TOKEN_BOUNDARY}){re.escape(needle)}(?!{_TOKEN_BOUNDARY})"
    )
    return re.search(pattern, haystack) is not None


def _scalar_paths(
    value: Any,
    *,
    minimum_value_length: int,
    include_numbers: bool,
    prefix: str = "",
) -> Iterator[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key in value:
            child_prefix = (
                f"{prefix}{key}"
                if not prefix or prefix.endswith("::")
                else f"{prefix}.{key}"
            )
            yield from _scalar_paths(
                value[key],
                minimum_value_length=minimum_value_length,
                include_numbers=include_numbers,
                prefix=child_prefix,
            )
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _scalar_paths(
                item,
                minimum_value_length=minimum_value_length,
                include_numbers=include_numbers,
                prefix=f"{prefix}[{index}]",
            )
        return
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, str):
        text = value.strip()
        # A tool result often arrives as a JSON document inside one string
        # field. Treat that document as a container rather than also publishing
        # the serialized container as one overlapping scalar.
        embedded = _embedded_json(text)
        if embedded is not None:
            yield from _scalar_paths(
                embedded,
                minimum_value_length=minimum_value_length,
                include_numbers=include_numbers,
                prefix=f"{prefix}::" if prefix else "self::",
            )
            return
        if len(text) >= max(1, int(minimum_value_length)):
            yield prefix or "self", text
        return
    if isinstance(value, (int, float)) and include_numbers:
        yield prefix or "self", repr(value)


__all__ = [
    "CARRIER_PREFIX",
    "DEFAULT_MINIMUM_VALUE_LENGTH",
    "MINIMUM_SUBSTRING_VALUE_LENGTH",
    "ENCODING_IDENTITY",
    "ENCODING_PERCENT",
    "MATCH_SUBSTRING",
    "MATCH_VALUE",
    "SCHEMA",
    "ValueProvenance",
    "carrier_unit_for",
    "derive_value_provenance",
    "provenance_carrier_tags",
    "provenance_carrier_units",
    "provenance_dependency_ids",
    "provenance_public_summary",
]
