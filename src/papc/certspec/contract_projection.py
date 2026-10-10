"""Contract projection: what the successor is allowed to see.

A reached-state contract is a bounded typed projection of the accepted prefix.
Everything the successor reads -- the request semantics, the accepted receipts,
the callable surface -- reaches it through that projection, never as the raw
transcript the cheap prefix model produced.

This module holds the surface-independent part: the callable surface. tau2 and
ToolSandbox hand a model whole OpenAI schemas, argument descriptions included.
STATE-Bench already drops that prose from its successor's copy and keeps the
argument contract -- names, types, enums, defaults, required flags -- because
the argument contract is what the successor spends its output on, while the
prose is restating a policy the successor already has. The catalog rides on
every strong call, so the difference is paid once per call.

Measured catalog sizes, whole against projected:

    tau2 airline    14 tools    2379 tok -> 1744 -> 1413
    tau2 retail     16 tools    2940 tok -> 1552 -> 1158

The agent's own copy of the catalog is never touched; only the copy the
successor and the contract builder read passes through here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

__all__ = [
    "CompactTool",
    "compact_tool_schema",
    "compact_successor_tools",
    "project_tool_schemas",
]


@dataclass(frozen=True)
class CompactTool:
    """One tool as the successor sees it: same callable surface, less prose."""

    name: str
    openai_schema: Mapping[str, Any]


def compact_tool_schema(
    schema: Mapping[str, Any], *, keep_tool_description: bool = True
) -> dict[str, Any]:
    """Drop argument prose, and optionally the tool's own sentence.

    Names, types, enums, defaults and required flags survive untouched: the
    projection removes restatement, not the callable contract.
    """

    compact = deepcopy(dict(schema))
    function = (
        compact.get("function")
        if isinstance(compact.get("function"), dict)
        else compact
    )
    if not isinstance(function, dict):
        return compact
    if not keep_tool_description:
        function.pop("description", None)
    parameters = function.get("parameters")
    if not isinstance(parameters, dict):
        return compact
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        return compact
    for value in properties.values():
        if not isinstance(value, dict):
            continue
        value.pop("description", None)
        for nested_key in ("items", "additionalProperties"):
            nested = value.get(nested_key)
            if isinstance(nested, dict):
                nested.pop("description", None)
    return compact


def compact_successor_tools(
    tools: Sequence[Any], *, keep_tool_description: bool = True
) -> list[Any]:
    """The successor's copy of the catalog. The agent's own copy is untouched."""

    compacted: list[Any] = []
    for tool in tools:
        schema = getattr(tool, "openai_schema", None)
        if not isinstance(schema, Mapping):
            compacted.append(tool)
            continue
        compacted.append(
            CompactTool(
                name=str(getattr(tool, "name", "") or ""),
                openai_schema=compact_tool_schema(
                    schema, keep_tool_description=keep_tool_description
                ),
            )
        )
    return compacted


def project_tool_schemas(
    schemas: Sequence[Mapping[str, Any]], *, keep_tool_description: bool = True
) -> list[dict[str, Any]]:
    """Project an already-extracted list of schemas, in place of the raw list."""

    return [
        compact_tool_schema(
            schema, keep_tool_description=keep_tool_description
        )
        if isinstance(schema, Mapping)
        else dict(schema)
        for schema in schemas
    ]
