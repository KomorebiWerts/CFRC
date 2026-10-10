"""Private ToolSandbox strong-response bridge for a future base129 pilot.

The shared CFRC IR stays public-safe, so concrete ToolSandbox tool arguments
must not enter ``ProposalGraph``.  This module parses a private strong response
into a public proposal graph plus private suffix bindings that an approved
runtime adapter can execute later.
"""

from __future__ import annotations

import ast
import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from papc.certspec.cfrc_ir import ProposalGraph, ProposalNode
from papc.certspec.cfrc_online_runner import assert_public_summary_safe
from papc.certspec.executable_closure import build_executable_closure
from papc.certspec.runtime_ledger import RoleCall, role_call_from_usage

SCHEMA = "papc.toolsandbox.base129_proposer_response_bridge.v1"
EXPECTED_RESPONSE_SCHEMA = "papc.toolsandbox.base129_proposer_response.v1"
READY_DECISION = "toolsandbox_base129_proposer_response_ready_private"
BLOCKED_DECISION = "blocked_toolsandbox_base129_proposer_response_private"
NODE_ID = "toolsandbox_typed_suffix_node"


@dataclass(frozen=True)
class ToolSandboxProposerResponseBridge:
    decision: str
    proposal_graph: ProposalGraph | None
    private_suffix_payload_by_node: Mapping[str, Any] = field(default_factory=dict)
    public_summary: Mapping[str, Any] = field(default_factory=dict)
    strong_call: RoleCall | None = None
    blockers: tuple[str, ...] = ()


def parse_toolsandbox_proposer_response(
    *,
    response: Mapping[str, Any],
    prompt_payload: Mapping[str, Any],
) -> ToolSandboxProposerResponseBridge:
    """Parse one private strong response into public graph + private binding."""

    response_schema = _mapping(prompt_payload.get("response_schema"))
    prompt_contract = _mapping(prompt_payload.get("prompt_contract"))
    tool_calls = _tool_calls(response.get("tool_calls"))
    code = _code_payload(response=response, response_schema=response_schema)
    assistant_response = _assistant_response(
        response=response,
        response_schema=response_schema,
    )
    assistant_template = _assistant_response_template(
        response=response,
        response_schema=response_schema,
    )
    assistant_bindings = _assistant_response_bindings(
        response=response,
        response_schema=response_schema,
    )
    expected_sequence = tuple(str(name) for name in response_schema.get("allowed_tool_sequence") or ())
    support_call_names = _support_call_names(response_schema)
    code_sequence, code_blockers = _code_call_sequence(
        code,
        allowed_names=frozenset(expected_sequence),
        support_names=support_call_names,
    )
    declared_code_sequence = _declared_code_sequence(response=response, code=code)
    actual_sequence = (
        declared_code_sequence
        or code_sequence
        or tuple(str(call.get("tool_name") or "") for call in tool_calls)
    )
    mutating_tool_name = str(response_schema.get("mutating_tool_name") or "")
    blockers = _blockers(
        response=response,
        response_schema=response_schema,
        prompt_contract=prompt_contract,
        tool_calls=tool_calls,
        code=code,
        assistant_response=assistant_response,
        assistant_template=assistant_template,
        assistant_bindings=assistant_bindings,
        code_blockers=code_blockers,
        expected_sequence=expected_sequence,
        actual_sequence=actual_sequence,
        mutating_tool_name=mutating_tool_name,
    )
    public_summary = _public_summary(
        response=response,
        decision=READY_DECISION if not blockers else BLOCKED_DECISION,
        prompt_payload=prompt_payload,
        tool_calls=tool_calls,
        code=code,
        assistant_response=assistant_response,
        assistant_template=assistant_template,
        assistant_bindings=assistant_bindings,
        expected_sequence=expected_sequence,
        actual_sequence=actual_sequence,
        mutating_tool_name=mutating_tool_name,
        blockers=blockers,
    )
    assert_public_summary_safe(public_summary)
    if blockers:
        return ToolSandboxProposerResponseBridge(
            decision=BLOCKED_DECISION,
            proposal_graph=None,
            public_summary=public_summary,
            blockers=tuple(blockers),
        )

    effect_units = tuple(str(unit) for unit in prompt_contract.get("allowed_effect_units") or ())
    effect_families = _effect_families(
        response_schema=response_schema,
        effect_units=effect_units,
    )
    template_id = _template_id(response_schema)
    authority_label = _authority_label(response, prompt_contract)
    node = ProposalNode(
        node_id=NODE_ID,
        kind="tool_call",
        authority_label=authority_label,
        semantic_effects=effect_units,
        effect_families=effect_families,
        executable_closure=build_executable_closure(
            semantic_effects=frozenset(effect_units),
            argument_carriers=frozenset(),
        ),
        tool_call_count=len(actual_sequence),
    )
    private_payload = {
        "template_id": template_id,
        "tool_sequence": list(actual_sequence),
        "mutating_tool_name": mutating_tool_name,
        "raw_payload_values_private": True,
    }
    if code:
        private_payload["code"] = code
    else:
        private_payload["tool_calls"] = copy.deepcopy(tool_calls)
    if assistant_response:
        private_payload["assistant_response"] = assistant_response
    if assistant_template:
        private_payload["assistant_response_template"] = assistant_template
        private_payload["assistant_response_bindings"] = copy.deepcopy(assistant_bindings)
    return ToolSandboxProposerResponseBridge(
        decision=READY_DECISION,
        proposal_graph=ProposalGraph(nodes=(node,), graph_id="toolsandbox_proposer_response_graph"),
        private_suffix_payload_by_node={NODE_ID: private_payload},
        public_summary=public_summary,
        strong_call=_strong_call(response),
        blockers=(),
    )


def _blockers(
    *,
    response: Mapping[str, Any],
    response_schema: Mapping[str, Any],
    prompt_contract: Mapping[str, Any],
    tool_calls: Sequence[Mapping[str, Any]],
    code: str,
    assistant_response: str,
    assistant_template: str,
    assistant_bindings: Mapping[str, Any],
    code_blockers: Sequence[str],
    expected_sequence: tuple[str, ...],
    actual_sequence: tuple[str, ...],
    mutating_tool_name: str,
) -> list[str]:
    blockers: list[str] = []
    response_schema_value = response.get("schema")
    if response_schema_value and response_schema_value != EXPECTED_RESPONSE_SCHEMA:
        blockers.append("response_schema_mismatch")
    if response.get("decision") != "propose":
        blockers.append("proposer_did_not_propose")
    if not prompt_contract.get("allowed_effect_units"):
        blockers.append("prompt_contract_missing_effect_units")
    if not expected_sequence:
        blockers.append("expected_tool_sequence_missing")
    if not actual_sequence or not _is_ordered_subsequence(actual_sequence, expected_sequence):
        blockers.append("tool_sequence_mismatch")
    if code:
        blockers.extend(code_blockers)
        blockers.extend(
            _tool_code_argument_blockers(
                code=code,
                response_schema=response_schema,
            )
        )
        blockers.extend(
            _mutating_code_argument_blockers(
                code=code,
                response_schema=response_schema,
                mutating_tool_name=mutating_tool_name,
            )
        )
    if not code and not tool_calls:
        blockers.append("tool_calls_missing")
    if (
        response_schema.get("assistant_response_required") is True
        and not assistant_response
        and not assistant_template
    ):
        blockers.append("assistant_response_missing")
    if (
        response_schema.get("assistant_response_template_required") is True
        and not assistant_template
    ):
        blockers.append("assistant_response_template_missing")
    if assistant_template and not assistant_bindings:
        blockers.append("assistant_response_bindings_missing")
    if assistant_bindings and not _assistant_response_bindings_well_formed(
        assistant_bindings
    ):
        blockers.append("assistant_response_binding_schema_mismatch")
    if (
        response_schema.get("mutating_tool_must_be_last", True) is not False
        and actual_sequence
        and actual_sequence[-1] != mutating_tool_name
    ):
        blockers.append("mutating_tool_not_last")
    if not code:
        blockers.extend(
            _mutating_argument_blockers(
                tool_calls=tool_calls,
                response_schema=response_schema,
                mutating_tool_name=mutating_tool_name,
            )
        )
    return sorted(set(blockers))


def _mutating_argument_blockers(
    *,
    tool_calls: Sequence[Mapping[str, Any]],
    response_schema: Mapping[str, Any],
    mutating_tool_name: str,
) -> list[str]:
    if not tool_calls:
        return []
    mutating_call = tool_calls[-1]
    if str(mutating_call.get("tool_name") or "") != mutating_tool_name:
        return []
    args = _mapping(mutating_call.get("arguments"))
    required = {
        str(item.get("key")): str(item.get("type"))
        for item in response_schema.get("mutating_arg_key_types") or ()
        if isinstance(item, Mapping) and item.get("key")
    }
    optional = {
        str(item.get("key")): str(item.get("type"))
        for item in response_schema.get("optional_mutating_arg_key_types") or ()
        if isinstance(item, Mapping) and item.get("key")
    }
    expected = {**required, **optional}
    blockers: list[str] = []
    if not optional and set(args) != set(required):
        blockers.append("mutating_argument_key_mismatch")
    if optional and (
        not set(required).issubset(set(args))
        or not set(args).issubset(set(expected))
    ):
        blockers.append("mutating_argument_key_mismatch")
    for key, expected_type in expected.items():
        if key not in args:
            continue
        if _public_type(args[key]) != expected_type:
            blockers.append("mutating_argument_type_mismatch")
            break
    return blockers


def _public_summary(
    *,
    response: Mapping[str, Any],
    decision: str,
    prompt_payload: Mapping[str, Any],
    tool_calls: Sequence[Mapping[str, Any]],
    code: str,
    assistant_response: str,
    assistant_template: str,
    assistant_bindings: Mapping[str, Any],
    expected_sequence: tuple[str, ...],
    actual_sequence: tuple[str, ...],
    mutating_tool_name: str,
    blockers: Sequence[str],
) -> dict[str, Any]:
    public_ordinal = _int(prompt_payload.get("public_ordinal"))
    response_schema = _mapping(prompt_payload.get("response_schema"))
    prompt_contract = _mapping(prompt_payload.get("prompt_contract"))
    effect_units = tuple(str(unit) for unit in prompt_contract.get("allowed_effect_units") or ())
    effect_families = _effect_families(
        response_schema=response_schema,
        effect_units=effect_units,
    )
    return {
        "schema": SCHEMA,
        "surface": "toolsandbox",
        "decision": decision,
        "public_ordinal": public_ordinal,
        "candidate_id": str(prompt_payload.get("candidate_id") or ""),
        "template_id": _template_id(response_schema),
        "effect_family_count": len(effect_families),
        "tool_call_count": len(actual_sequence),
        "expected_tool_sequence": list(expected_sequence),
        "actual_tool_sequence": list(actual_sequence),
        "mutating_tool_name": mutating_tool_name,
        "response_schema_present": bool(response.get("schema")),
        "response_schema_matched": response.get("schema") == EXPECTED_RESPONSE_SCHEMA,
        "mutating_argument_key_count": (
            _mutating_code_argument_key_count(code, mutating_tool_name)
            if code
            else len(_mapping(tool_calls[-1].get("arguments"))) if tool_calls else 0
        ),
        "code_payload_present": bool(code),
        "assistant_response_present": bool(assistant_response),
        "assistant_response_template_present": bool(assistant_template),
        "assistant_response_binding_count": len(assistant_bindings),
        "raw_payload_values_excluded": True,
        "private_suffix_binding_required": decision == READY_DECISION,
        "blockers": list(blockers),
    }


def _template_id(response_schema: Mapping[str, Any]) -> str:
    template_id = response_schema.get("template_id")
    return str(template_id) if template_id else "toolsandbox.add_reminder.v1"


def _effect_families(
    *,
    response_schema: Mapping[str, Any],
    effect_units: Sequence[str],
) -> tuple[str, ...]:
    explicit = tuple(
        str(item)
        for item in response_schema.get("effect_families") or ()
        if str(item)
    )
    if explicit:
        return explicit
    inferred = tuple(
        sorted(
            {
                ".".join(unit.split(".")[:2])
                for unit in effect_units
                if unit.startswith("toolsandbox.") and len(unit.split(".")) >= 2
            }
        )
    )
    return inferred or ("toolsandbox.reminder",)


def _strong_call(response: Mapping[str, Any]) -> RoleCall:
    usage = _mapping(response.get("usage"))
    return role_call_from_usage(
        "strong",
        usage,
        latency_seconds=_optional_float(response.get("latency_seconds")),
    )


def _authority_label(
    response: Mapping[str, Any],
    prompt_contract: Mapping[str, Any],
) -> str:
    authority = str(response.get("authority_label") or "")
    allowed = [str(item) for item in prompt_contract.get("allowed_authorities") or ()]
    if authority:
        return authority
    if allowed:
        return allowed[-1]
    return "ell1"


def _tool_calls(raw: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(raw, list):
        return ()
    calls: list[Mapping[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        call = {
            "tool_name": str(item.get("tool_name") or item.get("name") or ""),
            "arguments": dict(_mapping(item.get("arguments"))),
        }
        if item.get("allow_exception") is True:
            call["allow_exception"] = True
            expected = item.get("expected_exception_substring")
            if isinstance(expected, str) and expected:
                call["expected_exception_substring"] = expected
        calls.append(
            call
        )
    return tuple(calls)


def _code_payload(
    *,
    response: Mapping[str, Any],
    response_schema: Mapping[str, Any],
) -> str:
    if response_schema.get("code_payload_allowed") is not True:
        return ""
    code = response.get("code")
    return code.strip() if isinstance(code, str) else ""


def _assistant_response(
    *,
    response: Mapping[str, Any],
    response_schema: Mapping[str, Any],
) -> str:
    if response_schema.get("assistant_response_allowed") is not True:
        return ""
    value = response.get("assistant_response")
    return value.strip() if isinstance(value, str) else ""


def _assistant_response_template(
    *,
    response: Mapping[str, Any],
    response_schema: Mapping[str, Any],
) -> str:
    if response_schema.get("assistant_response_template_allowed") is not True:
        return ""
    value = response.get("assistant_response_template")
    return value.strip() if isinstance(value, str) else ""


def _assistant_response_bindings(
    *,
    response: Mapping[str, Any],
    response_schema: Mapping[str, Any],
) -> Mapping[str, Any]:
    if response_schema.get("assistant_response_template_allowed") is not True:
        return {}
    value = response.get("assistant_response_bindings")
    if not isinstance(value, Mapping):
        return {}
    return _normalize_assistant_response_bindings(value)


def _is_code_output_source(value: Any) -> bool:
    return str(value) in {"code_output", "code_stdout"}


def _code_output_index(binding: Mapping[str, Any]) -> Any:
    if "output_index" in binding:
        return binding["output_index"]
    if "index" in binding:
        return binding["index"]
    return None


def _normalize_assistant_response_bindings(
    bindings: Mapping[str, Any],
) -> Mapping[str, Any]:
    normalized: dict[str, Any] = {}
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            normalized[str(name)] = copy.deepcopy(binding)
            continue
        if "tool_output_index" in binding:
            normalized[str(name)] = copy.deepcopy(dict(binding))
            continue
        if "code_output_index" in binding and "path" in binding:
            normalized[str(name)] = {
                "tool_output_index": binding["code_output_index"],
                "path": copy.deepcopy(binding["path"]),
            }
            _copy_binding_options(binding, normalized[str(name)])
            continue
        if "code_output_index" in binding and "field" in binding:
            normalized[str(name)] = {
                "tool_output_index": binding["code_output_index"],
                "path": [str(binding["field"])],
            }
            _copy_binding_options(binding, normalized[str(name)])
            continue
        if (
            _is_code_output_source(binding.get("source"))
            and _code_output_index(binding) is not None
            and "path" in binding
        ):
            normalized[str(name)] = {
                "tool_output_index": _code_output_index(binding),
                "path": copy.deepcopy(binding["path"]),
            }
            _copy_binding_options(binding, normalized[str(name)])
            continue
        if (
            binding.get("source") == "tool_output"
            and "call_index" in binding
        ):
            normalized[str(name)] = {
                "tool_output_index": binding["call_index"],
                "path": copy.deepcopy(binding.get("path", [])),
            }
            _copy_binding_options(binding, normalized[str(name)])
            continue
        if (
            _is_code_output_source(binding.get("source"))
            and _code_output_index(binding) is not None
            and "key" in binding
        ):
            normalized[str(name)] = {
                "tool_output_index": _code_output_index(binding),
                "path": [str(binding["key"])],
            }
            _copy_binding_options(binding, normalized[str(name)])
            continue
        if (
            _is_code_output_source(binding.get("source"))
            and _code_output_index(binding) is not None
            and "field" in binding
        ):
            normalized[str(name)] = {
                "tool_output_index": _code_output_index(binding),
                "path": [str(binding["field"])],
            }
            _copy_binding_options(binding, normalized[str(name)])
            continue
        normalized[str(name)] = copy.deepcopy(dict(binding))
    return normalized


def _copy_binding_options(
    source: Mapping[str, Any],
    target: dict[str, Any],
) -> None:
    for key in ("round", "fallback"):
        if key in source:
            target[key] = copy.deepcopy(source[key])


def _assistant_response_bindings_well_formed(bindings: Mapping[str, Any]) -> bool:
    for name, binding in bindings.items():
        if not str(name):
            return False
        if not isinstance(binding, Mapping):
            return False
        if "tool_output_index" not in binding:
            return False
        try:
            int(binding["tool_output_index"])
        except (TypeError, ValueError):
            return False
        path = binding.get("path", [])
        if not isinstance(path, list):
            return False
        if "round" in binding:
            try:
                int(binding["round"])
            except (TypeError, ValueError):
                return False
        if "fallback" in binding and not isinstance(
            binding["fallback"],
            (str, int, float, bool),
        ):
            return False
    return True


def _code_call_sequence(
    code: str,
    *,
    allowed_names: frozenset[str],
    support_names: frozenset[str] = frozenset(),
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not code:
        return (), ()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return (), ("code_syntax_error",)
    names: list[str] = []
    blockers: list[str] = []
    for node in ast.walk(tree):
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.Lambda,
                ast.Global,
                ast.Nonlocal,
            ),
        ):
            blockers.append("code_contains_disallowed_statement")
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    ]
    calls.sort(key=lambda node: (getattr(node, "lineno", 0), getattr(node, "col_offset", 0)))
    for node in calls:
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in support_names:
                    names.append(node.func.id)
            else:
                blockers.append("code_contains_non_name_call")
    all_call_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    if any(name not in allowed_names and name not in support_names for name in all_call_names):
        blockers.append("code_contains_unapproved_call")
    return tuple(names), tuple(sorted(set(blockers)))


def _declared_code_sequence(
    *,
    response: Mapping[str, Any],
    code: str,
) -> tuple[str, ...]:
    if not code:
        return ()
    raw = response.get("tool_sequence")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return ()
    return tuple(str(name) for name in raw if str(name))


def _support_call_names(response_schema: Mapping[str, Any]) -> frozenset[str]:
    if response_schema.get("assistant_response_template_allowed") is True:
        return frozenset({"print"})
    return frozenset()


def _is_ordered_subsequence(
    actual_sequence: Sequence[str],
    expected_sequence: Sequence[str],
) -> bool:
    if not actual_sequence:
        return False
    expected_iter = iter(expected_sequence)
    return all(any(actual == expected for expected in expected_iter) for actual in actual_sequence)


def _mutating_code_argument_blockers(
    *,
    code: str,
    response_schema: Mapping[str, Any],
    mutating_tool_name: str,
) -> list[str]:
    call = _last_call_by_name(code, mutating_tool_name)
    if call is None:
        return ["mutating_tool_missing"]
    if call.args:
        return ["mutating_argument_positional_not_allowed"]
    keywords = {keyword.arg for keyword in call.keywords if keyword.arg is not None}
    if len(keywords) != len(call.keywords):
        return ["mutating_argument_star_kwargs_not_allowed"]
    required = {
        str(item.get("key"))
        for item in response_schema.get("mutating_arg_key_types") or ()
        if isinstance(item, Mapping) and item.get("key")
    }
    optional = {
        str(item.get("key"))
        for item in response_schema.get("optional_mutating_arg_key_types") or ()
        if isinstance(item, Mapping) and item.get("key")
    }
    if not required.issubset(keywords) or not keywords.issubset(required | optional):
        return ["mutating_argument_key_mismatch"]
    return []


def _tool_code_argument_blockers(
    *,
    code: str,
    response_schema: Mapping[str, Any],
) -> list[str]:
    tool_arg_schema = _mapping(response_schema.get("tool_arg_key_types"))
    if not tool_arg_schema:
        return []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    blockers: list[str] = []
    schema_by_name = {
        str(name): tuple(spec for spec in specs if isinstance(spec, Mapping))
        for name, specs in tool_arg_schema.items()
        if isinstance(specs, Sequence) and not isinstance(specs, str | bytes)
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        specs = schema_by_name.get(node.func.id)
        if specs is None:
            continue
        if node.args:
            blockers.append("code_tool_argument_positional_not_allowed")
            continue
        keyword_names = [keyword.arg for keyword in node.keywords]
        if any(name is None for name in keyword_names):
            blockers.append("code_tool_argument_star_kwargs_not_allowed")
            continue
        allowed = {
            str(spec.get("key"))
            for spec in specs
            if spec.get("key")
        }
        required = {
            str(spec.get("key"))
            for spec in specs
            if spec.get("key") and spec.get("required") is True
        }
        actual = {str(name) for name in keyword_names if name}
        if not required.issubset(actual) or not actual.issubset(allowed):
            blockers.append("code_tool_argument_key_mismatch")
    return blockers


def _mutating_code_argument_key_count(code: str, mutating_tool_name: str) -> int:
    call = _last_call_by_name(code, mutating_tool_name)
    if call is None:
        return 0
    return sum(1 for keyword in call.keywords if keyword.arg is not None)


def _last_call_by_name(code: str, name: str) -> ast.Call | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]
    return calls[-1] if calls else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _public_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    return "unknown"


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "BLOCKED_DECISION",
    "NODE_ID",
    "READY_DECISION",
    "SCHEMA",
    "ToolSandboxProposerResponseBridge",
    "parse_toolsandbox_proposer_response",
]
