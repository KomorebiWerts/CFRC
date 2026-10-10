"""Request-independent reached-state evidence shared by PAPC surfaces.

This module may project environment facts into ``E`` and public tool schemas
into the native interface record. It deliberately does not derive ``O``:
residual obligations require the frozen request scope as well as trusted
reached-state evidence. A receipt listing an entity is evidence that the
entity exists, not evidence that the retained request requires inspecting it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from papc.certspec.cfrc_ir import EvidenceFact


@dataclass(frozen=True)
class EnvironmentContext:
    """Reached-state facts that are true regardless of what was requested.

    These are properties of the environment the successor inherits, not
    answers to the task: a clock reading resolves relative dates, and an
    actor identity resolves "my" in a request without disclosing what to do
    about it.
    """

    observed_at: str | None = None
    actor: str | None = None
    locale: str | None = None
    timezone: str | None = None

    def as_evidence(self) -> dict[str, Any]:
        fields = {
            "observed_at": self.observed_at,
            "actor": self.actor,
            "locale": self.locale,
            "timezone": self.timezone,
        }
        return {key: value for key, value in fields.items() if value is not None}


def environment_context_evidence(context: EnvironmentContext | None) -> dict[str, Any]:
    """Project reached-state environment facts into contract evidence."""

    if context is None:
        return {}
    return context.as_evidence()


def operation_grammar(
    tool_schemas: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Project native tool schemas into the admissible action space ``S``.

    Reports, per operation, the arguments it accepts, which are required,
    and any default the schema itself declares. A default that the schema
    states is part of what the tools permit, not a hint about what to do:
    withholding it would only make the successor rediscover the same fact by
    calling the tool.
    """

    grammar: dict[str, dict[str, Any]] = {}
    for name in sorted(tool_schemas):
        schema = tool_schemas[name]
        parameters = schema.get("parameters")
        if not isinstance(parameters, Mapping):
            parameters = schema
        properties = parameters.get("properties")
        properties = properties if isinstance(properties, Mapping) else {}
        required = parameters.get("required")
        required_names = (
            tuple(sorted(str(item) for item in required))
            if isinstance(required, Sequence) and not isinstance(required, str)
            else ()
        )
        arguments: dict[str, dict[str, Any]] = {}
        for argument in sorted(properties):
            spec = properties[argument]
            spec = spec if isinstance(spec, Mapping) else {}
            entry: dict[str, Any] = {"required": argument in required_names}
            argument_type = spec.get("type")
            if argument_type is not None:
                entry["type"] = argument_type
            if "default" in spec:
                entry["default"] = spec["default"]
            enum = spec.get("enum")
            if isinstance(enum, Sequence) and not isinstance(enum, str):
                entry["enum"] = list(enum)
            arguments[argument] = entry
        operation: dict[str, Any] = {"arguments": arguments}
        for passthrough in ("methods", "markers", "effect_unit"):
            if passthrough in schema:
                operation[passthrough] = schema[passthrough]
        grammar[name] = operation
    return grammar


ENVIRONMENT_FACT_ID = "reached_state_environment"
ENVIRONMENT_FACT_TYPE = "environment_context"


def environment_facts(context: EnvironmentContext | None) -> tuple[EvidenceFact, ...]:
    """Carry reached-state environment facts in that same shape."""

    evidence = environment_context_evidence(context)
    if not evidence:
        return ()
    return (
        EvidenceFact(
            fact_id=ENVIRONMENT_FACT_ID,
            fact_type=ENVIRONMENT_FACT_TYPE,
            unit_tokens=tuple(f"{key}={evidence[key]}" for key in sorted(evidence)),
        ),
    )
