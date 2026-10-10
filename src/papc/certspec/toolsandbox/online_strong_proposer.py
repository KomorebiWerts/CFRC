"""OpenAI-compatible strong proposer for ToolSandbox true-online CFRC."""

from __future__ import annotations

import copy
import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from papc.certspec.cfrc_ir import ProposalGraph
from papc.certspec.cfrc_online_runner import (
    CFRCOnlineProposalContext,
    CFRCOnlineProposerResult,
)
from papc.certspec.provider import reasoning_tokens_from_usage
from papc.certspec.runtime_ledger import RoleCall, role_call_from_usage
from papc.certspec.toolsandbox.base129_proposer_response_bridge import (
    parse_toolsandbox_proposer_response,
)
from papc.certspec.toolsandbox.online_frontier_monitor import (
    REMINDER_TOOL_SEQUENCE,
)
from papc.certspec.toolsandbox.role_bridge import (
    apply_reasoning_effort,
    build_openai_client,
)

EMPTY_GRAPH_ID = "toolsandbox_strong_proposer_blocked_empty_graph"


@dataclass(frozen=True)
class ToolSandboxOpenAIStrongProposerConfig:
    model: str
    reasoning_effort: str
    api_base: str
    api_key_env: str = "OPENAI_API_KEY"
    provider_timeout_seconds: float | None = 300.0


@dataclass
class ToolSandboxOpenAIStrongProposer:
    """Call the strong model once and keep raw suffix args private."""

    config: ToolSandboxOpenAIStrongProposerConfig
    client: Any | None = None
    _client_ready: bool = field(default=False, init=False, repr=False)
    latest_private_suffix_payload_by_node: Mapping[str, Any] = field(default_factory=dict)
    latest_private_diagnostic: Mapping[str, Any] = field(default_factory=dict)

    def __call__(self, context: CFRCOnlineProposalContext) -> CFRCOnlineProposerResult:
        prompt_payload = _prompt_payload(context)
        started = time.monotonic()
        response = self._call_provider(prompt_payload)
        response["latency_seconds"] = round(time.monotonic() - started, 6)
        bridge = parse_toolsandbox_proposer_response(
            response=response,
            prompt_payload=prompt_payload,
        )
        self.latest_private_suffix_payload_by_node = dict(
            bridge.private_suffix_payload_by_node
        )
        self.latest_private_diagnostic = {
            "bridge_decision": bridge.decision,
            "bridge_blockers": list(bridge.blockers),
            "bridge_public_summary": dict(bridge.public_summary),
            "raw_response": dict(response),
            "prompt_payload": copy.deepcopy(dict(prompt_payload)),
        }
        return CFRCOnlineProposerResult(
            proposal_graph=bridge.proposal_graph
            if bridge.proposal_graph is not None
            else ProposalGraph(nodes=(), graph_id=EMPTY_GRAPH_ID),
            strong_call=bridge.strong_call or _strong_call_from_response(response),
        )

    def _call_provider(self, prompt_payload: Mapping[str, Any]) -> dict[str, Any]:
        if not self._client_ready:
            self.client = _prepare_client(self.config, self.client)
            self._client_ready = True
        completion = self.client.chat.completions.create(
            model=self.config.model,
            messages=_messages(prompt_payload),
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content or "{}"
        payload = _json_object_from_text(content)
        payload.pop("usage", None)
        usage = getattr(completion, "usage", None)
        if usage is not None:
            if hasattr(usage, "model_dump"):
                payload["usage"] = usage.model_dump()
            elif isinstance(usage, Mapping):
                payload["usage"] = dict(usage)
        return payload


@dataclass
class ToolSandboxGroundedStepProposer:
    """Propose one native call at a time and observe each receipt.

    Receipt-grounded authoring needs the successor to see what the previous call
    actually returned before it commits to the next argument, so this proposer is
    called once per step instead of once per episode.  It accumulates provider
    usage across the steps of one authoring pass; the controller drains that
    total into the runtime ledger as a single strong role entry.
    """

    config: ToolSandboxOpenAIStrongProposerConfig
    client: Any | None = None
    _client_ready: bool = field(default=False, init=False, repr=False)
    latest_private_diagnostic: Mapping[str, Any] = field(default_factory=dict)
    _pending_calls: list[Mapping[str, Any]] = field(default_factory=list, repr=False)
    _step_records: list[Mapping[str, Any]] = field(default_factory=list, repr=False)

    def __call__(self, step_context: Mapping[str, Any]) -> Mapping[str, Any]:
        started = time.monotonic()
        response = self._call_provider(step_context)
        response["latency_seconds"] = round(time.monotonic() - started, 6)
        self._pending_calls.append(response)
        self._step_records.append(
            {
                "round_index": step_context.get("round_index"),
                "action": str(response.get("action") or ""),
                "tool_name": str(response.get("tool_name") or ""),
                "raw_response": dict(response),
            }
        )
        return response

    def drain_role_call(self) -> RoleCall:
        """Return the accumulated usage of one authoring pass and reset it."""

        calls = list(self._pending_calls)
        self._pending_calls = []
        self.latest_private_diagnostic = {
            "authoring_step_count": len(calls),
            "authoring_steps": list(self._step_records),
        }
        self._step_records = []
        if not calls:
            return RoleCall(role="strong", call_count=0, token_usage_available=False)
        usages = [
            usage
            for usage in (
                response.get("usage")
                for response in calls
            )
            if isinstance(usage, Mapping)
        ]
        return RoleCall(
            role="strong",
            call_count=len(calls),
            input_tokens=_summed(usages, ("input_tokens", "prompt_tokens")),
            output_tokens=_summed(usages, ("output_tokens", "completion_tokens")),
            reasoning_tokens=_summed_reasoning(usages),
            latency_seconds=sum(
                float(response.get("latency_seconds") or 0.0) for response in calls
            ),
            token_usage_available=all(
                _strong_call_from_response(response).token_usage_available
                for response in calls
            ),
        )

    def _call_provider(self, step_context: Mapping[str, Any]) -> dict[str, Any]:
        if not self._client_ready:
            self.client = _prepare_client(self.config, self.client)
            self._client_ready = True
        completion = self.client.chat.completions.create(
            model=self.config.model,
            messages=_step_messages(step_context),
            response_format={"type": "json_object"},
            tools=[_proposal_tool(step_context)],
            tool_choice={
                "type": "function",
                "function": {"name": PROPOSAL_TOOL_NAME},
            },
        )
        message = completion.choices[0].message
        # The step is read from the tool first and from the text second. A
        # provider that ignores the offer answers in content exactly as before,
        # and the prompt still carries the same contract, so an arm measured
        # before this existed is not disturbed by it.
        arguments = _submitted_proposal_arguments(message)
        content = arguments or (message.content or "{}")
        payload = _json_object_from_text(content)
        payload.pop("usage", None)
        usage = getattr(completion, "usage", None)
        if usage is not None:
            if hasattr(usage, "model_dump"):
                payload["usage"] = usage.model_dump()
            elif isinstance(usage, Mapping):
                payload["usage"] = dict(usage)
        return payload


def _prepare_client(config: ToolSandboxOpenAIStrongProposerConfig, client: Any) -> Any:
    """Create and configure once, lazily so a dry run needs no credentials."""

    if client is None:
        client = build_openai_client(
            api_base=config.api_base,
            api_key_env=config.api_key_env,
            timeout_seconds=config.provider_timeout_seconds,
        )
    return apply_reasoning_effort(client, config.reasoning_effort)


def _step_messages(step_context: Mapping[str, Any]) -> list[dict[str, str]]:
    allowed_tools = ", ".join(
        str(name) for name in step_context.get("allowed_tool_sequence") or ()
    )
    domain_instructions = " ".join(
        str(item) for item in step_context.get("domain_instructions") or ()
    )
    # Announcing an action the contract cannot accept costs a provider call and
    # ends in a refusal, so the prompt offers declining only where it can work.
    declaration_clause = (
        'Use {"action": "decline", "obligation_id": ..., "mode": '
        '"decline" or "clarify", "basis": "..."} only when an open '
        "obligation cannot be discharged from what is visible. "
        "Name an id from open_obligation_ids, declare it at most once, "
        "and end the pass with action=terminal on the very next step; "
        "declared_obligation_ids lists what you already declined. "
        if step_context.get("declaration_accepted") is not False
        else "Every open obligation must be discharged by a call that succeeds; "
        "this contract has no way to set one aside. If the visible prefix "
        "cannot support one, complete what it does support and say plainly in "
        "the terminal text what you could not do. "
    )
    terminal_with_last_call_clause = (
        "When the next call is the final call needed to discharge every open "
        "obligation, you may include its terminal answer in that same "
        "tool_call object. Use exactly one of: "
        '(1) "terminal_response": "..." for static text, or '
        '(2) "terminal_response_template": "...{name}..." plus a nonempty '
        '"terminal_response_bindings" mapping. Each binding must use '
        '{"tool_output_index": N, "path": [...]}, where N is the zero-based '
        "index in the complete receipt list including this new call; optional "
        '"round" is an integer. Use each binding name once or more as a simple '
        "{name} placeholder; do not use attribute/index traversal, conversions, "
        "or format specs. Do not use fallback, call_index, key, or field. "
        "The checker ignores this piggyback unless the new replica call "
        "succeeds, all obligations are covered, every binding names a selected "
        "successful receipt, and template rendering succeeds to nonempty text. "
        "If you do not piggyback, use a separate action=terminal step as usual. "
        if step_context.get("terminal_with_last_call") is True
        else ""
    )
    return [
        {
            "role": "system",
            "content": (
                "You are the strong residual completer for ToolSandbox CFRC. "
                "You continue an episode a cheaper agent already started; you "
                "never restart it and never repeat work the visible prefix "
                "already committed. "
                "Act one native call at a time. Return a single JSON object with "
                'an "action" field, and use no other response shape. '
                'Use {"action": "tool_call", "tool_name": ..., "arguments": {...}} '
                "to run the next call; you will then see its real result before "
                "choosing the following one. "
                'Use {"action": "terminal", "terminal_response": "..."} once the '
                "residual is complete, giving the final user-facing text. That "
                "text is submitted to the user as the answer for this episode, so "
                "state the outcome of what you just did. "
                f"{terminal_with_last_call_clause}"
                f"{declaration_clause}"
                f"Call only these tools, in this order: {allowed_tools}. "
                "Every argument of a state-changing call must come from the "
                "visible request text or from a result an earlier call in this "
                "pass returned. Never invent a value. "
                "Read observed_receipts for what your earlier calls returned, and "
                "verifier_feedback for why a previous attempt was refused. "
                f"{domain_instructions}"
            ),
        },
        {
            "role": "user",
            # The surface rules are already stated above. Repeating them in the
            # payload costs input tokens on every step of every episode.
            "content": json.dumps(
                {
                    key: value
                    for key, value in step_context.items()
                    if key != "domain_instructions"
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    ]


def _summed_reasoning(usages: Sequence[Mapping[str, Any]]) -> int | None:
    """Sum reasoning usage across calls, reading every shape a provider uses."""

    total = 0
    seen = False
    for usage in usages:
        value = reasoning_tokens_from_usage(usage)
        if value is not None:
            total += value
            seen = True
    return total if seen else None


def _summed(usages: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> int | None:
    total = 0
    seen = False
    for usage in usages:
        for key in keys:
            value = _optional_int(usage.get(key))
            if value is not None:
                total += value
                seen = True
                break
    return total if seen else None


def _prompt_payload(context: CFRCOnlineProposalContext) -> dict[str, Any]:
    response_schema = _response_schema(context.prompt_contract)
    return {
        "surface": "toolsandbox",
        "candidate_id": context.candidate.candidate_id,
        "public_ordinal": _public_ordinal_from_candidate(context.candidate.candidate_id),
        "prefix_messages": list(context.prefix_payload.get("conversation") or ()),
        "prompt_contract": dict(context.prompt_contract),
        "response_schema": response_schema,
        "instructions": [
            "Return one JSON object only.",
            "Use only information visible in prefix_messages.",
            "Do not restart the episode.",
            "The deterministic frontier monitor already certified this prefix as residual-closable for the selected typed suffix; prefer decision=propose and abstain only on an explicit visible contradiction or a still-missing required slot.",
            "Prefer a private code payload when the residual needs live ToolSandbox read tools.",
            "If the visible prefix still lacks the user intent or target time, set decision to abstain.",
            *list(response_schema.get("domain_instructions") or ()),
        ],
    }


def _response_schema(prompt_contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    contract = prompt_contract if isinstance(prompt_contract, Mapping) else {}
    typed_suffix = contract.get("typed_suffix")
    if isinstance(typed_suffix, Mapping):
        schema = _generic_response_schema(
            typed_suffix=typed_suffix,
            prompt_contract=contract,
        )
    else:
        schema = _reminder_response_schema()
    return schema


def _generic_response_schema(
    *,
    typed_suffix: Mapping[str, Any],
    prompt_contract: Mapping[str, Any],
) -> dict[str, Any]:
    allowed_sequence = [
        str(name)
        for name in typed_suffix.get("suffix_tool_sequence") or ()
        if str(name)
    ]
    code_payload_allowed = typed_suffix.get("code_payload_allowed", True) is not False
    mutating_tool_name = str(typed_suffix.get("mutating_tool_name") or "")
    mutating_tool_must_be_last = typed_suffix.get(
        "mutating_tool_must_be_last",
        True,
    ) is not False
    sequence_rule = (
        f"Code must end its tool effects with {mutating_tool_name}."
        if mutating_tool_must_be_last
        else (
            f"Code may call {mutating_tool_name} before later allowed read tools; "
            "preserve allowed_tool_sequence order."
        )
    )
    domain_sequence_rule = (
        f"Allowed suffix must be an ordered subsequence ending in {mutating_tool_name}."
        if mutating_tool_must_be_last
        else (
            f"Allowed suffix is an ordered subsequence and may continue with later allowed read tools "
            f"after {mutating_tool_name}."
        )
    )
    schema: dict[str, Any] = {
        "schema": "papc.toolsandbox.base129_proposer_response.v1",
        "decision_values": ["propose", "abstain"],
        "template_id": str(typed_suffix.get("template_id") or ""),
        "effect_families": list(
            typed_suffix.get("effect_families")
            or _effect_families_from_units(prompt_contract.get("allowed_effect_units") or ())
        ),
        "allowed_tool_sequence": allowed_sequence,
        "code_payload_allowed": code_payload_allowed,
        "assistant_response_allowed": (
            typed_suffix.get("assistant_response_allowed", True) is not False
        ),
        "assistant_response_template_allowed": (
            typed_suffix.get("assistant_response_template_allowed", True) is not False
        ),
        "assistant_response_template_required": bool(
            typed_suffix.get("assistant_response_template_required") is True
        ),
        "tool_call_argument_binding_allowed": True,
        "expected_tool_exception_continue_allowed": True,
        "assistant_response_required": bool(
            typed_suffix.get("assistant_response_required") is True
        ),
        "code_rules": [
            *(
                [
                    "Code may call only the allowed_tool_sequence functions.",
                    "Code must call an ordered subsequence of allowed_tool_sequence.",
                    sequence_rule,
                    "No imports, class/function definitions, lambdas, globals, or non-name calls.",
                    f"Use keyword arguments for {mutating_tool_name}.",
                    "If assistant_response_template needs runtime values, end code with print({...}) containing only bound fields.",
                ]
                if code_payload_allowed
                else [
                    "Do not include a code payload for this suffix; use ordered tool_calls.",
                    f"Tool_calls must follow allowed_tool_sequence and may call {mutating_tool_name} before later allowed read tools when needed.",
                ]
            ),
        ],
        "code_example_shape": _code_example_shape(
            typed_suffix=typed_suffix,
            allowed_sequence=allowed_sequence,
            mutating_tool_name=mutating_tool_name,
        ),
        "mutating_tool_name": mutating_tool_name,
        "mutating_tool_must_be_last": mutating_tool_must_be_last,
        "mutating_arg_key_types": [
            dict(item)
            for item in typed_suffix.get("mutating_arg_key_types") or ()
            if isinstance(item, Mapping)
        ],
        "optional_mutating_arg_key_types": [
            dict(item)
            for item in typed_suffix.get("optional_mutating_arg_key_types") or ()
            if isinstance(item, Mapping)
        ],
        "tool_arg_key_types": _tool_arg_key_types(typed_suffix),
        "domain_instructions": [
            domain_sequence_rule,
            "If the residual requires a final user-facing answer after tools run, include assistant_response with that exact final text.",
            "If the final answer depends on runtime tool outputs, include assistant_response_template and assistant_response_bindings instead of guessing the output value.",
            "For tool_calls, an argument may bind to an earlier tool result with source=tool_output, for example {'source': 'tool_output', 'call_index': 0, 'path': ['field']}; use this for values such as unit_conversion.amount.",
            "A template binding may include fallback only when an allowed read tool can legitimately return no value; use a neutral value such as 'not available', never a guessed answer.",
            "If an allowed read tool is expected to fail before a repair action, mark that tool call with allow_exception=true and expected_exception_substring; continue only when the substring matches.",
            "For template bindings from code payloads, print a final private dict such as print({'field': value}) and bind to that output index.",
            "For assistant_response_template, use Python format placeholders such as {field}, not double braces.",
        ],
    }
    if schema["assistant_response_required"]:
        schema["domain_instructions"].append(
            "assistant_response is required for this typed suffix; do not omit the final user-facing confirmation."
        )
    schema["domain_instructions"].extend(
        str(item)
        for item in typed_suffix.get("domain_instructions") or ()
        if str(item)
    )
    if mutating_tool_name == "add_reminder":
        reminder_schema = _reminder_response_schema()
        for key in (
            "datetime_info_fields",
            "weekday_code_example_shape",
            "multi_turn_slot_merge_rule",
        ):
            schema.setdefault(key, copy.deepcopy(reminder_schema[key]))
        if "code_example_shape" not in typed_suffix:
            schema["code_example_shape"] = copy.deepcopy(
                reminder_schema["code_example_shape"]
            )
        if "assistant_response_template_allowed" not in typed_suffix:
            schema["assistant_response_template_allowed"] = reminder_schema[
                "assistant_response_template_allowed"
            ]
        if (
            "optional_mutating_arg_key_types" not in typed_suffix
            and str(typed_suffix.get("template_id") or "")
            != "toolsandbox.add_reminder.with_location.v1"
        ):
            schema["optional_mutating_arg_key_types"] = copy.deepcopy(
                reminder_schema["optional_mutating_arg_key_types"]
            )
        schema["code_rules"].extend(
            rule
            for rule in reminder_schema["code_rules"]
            if rule not in schema["code_rules"]
        )
        schema["domain_instructions"].extend(
            instruction
            for instruction in reminder_schema["domain_instructions"]
            if instruction not in schema["domain_instructions"]
        )
    return schema


def _code_example_shape(
    *,
    typed_suffix: Mapping[str, Any],
    allowed_sequence: Sequence[str],
    mutating_tool_name: str,
) -> Mapping[str, Any]:
    raw = typed_suffix.get("code_example_shape")
    if isinstance(raw, Mapping):
        return dict(raw)
    return {
        "decision": "propose",
        "authority_label": "ell1",
        "tool_sequence": list(allowed_sequence),
        "code": f"{mutating_tool_name}(...)",
    }


def _tool_arg_key_types(typed_suffix: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    raw = typed_suffix.get("tool_arg_key_types")
    if not isinstance(raw, Mapping):
        return {}
    result: dict[str, list[dict[str, Any]]] = {}
    for name, specs in raw.items():
        if not isinstance(specs, list):
            continue
        result[str(name)] = [dict(item) for item in specs if isinstance(item, Mapping)]
    return result


def _reminder_response_schema() -> dict[str, Any]:
    return {
        "schema": "papc.toolsandbox.base129_proposer_response.v1",
        "decision_values": ["propose", "abstain"],
        "template_id": "toolsandbox.add_reminder.v1",
        "effect_families": ["toolsandbox.reminder"],
        "allowed_tool_sequence": list(REMINDER_TOOL_SEQUENCE),
        "code_payload_allowed": True,
        "assistant_response_allowed": True,
        "assistant_response_template_allowed": False,
        "code_rules": [
            "Code may call only the allowed_tool_sequence functions.",
            "Code must call an ordered subsequence of allowed_tool_sequence.",
            "Code must end its tool effects with add_reminder.",
            "No imports, class/function definitions, lambdas, globals, or non-name calls.",
            "Use keyword arguments for add_reminder.",
            "Do not require a timezone argument; the ToolSandbox time utilities use local timestamp semantics.",
            "Reminder content must come from the visible prefix; abstain when content is missing.",
            "Use timestamp_to_datetime_info()['isoweekday']; never use a key named weekday.",
            "Content and target time may come from different visible user turns in the same prefix.",
        ],
        "datetime_info_fields": [
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "isoweekday",
        ],
        "code_example_shape": {
            "decision": "propose",
            "authority_label": "ell1",
            "tool_sequence": list(REMINDER_TOOL_SEQUENCE),
            "code": "\n".join(
                [
                    "now = get_current_timestamp()",
                    "base = timestamp_to_datetime_info(timestamp=now)",
                    "shifted = now + 86400",
                    "target_info = timestamp_to_datetime_info(timestamp=shifted)",
                    "target = datetime_info_to_timestamp(year=target_info['year'], month=target_info['month'], day=target_info['day'], hour=17, minute=0, second=0)",
                    "add_reminder(content='the requested reminder text', reminder_timestamp=target)",
                ]
            ),
        },
        "weekday_code_example_shape": {
            "decision": "propose",
            "authority_label": "ell1",
            "tool_sequence": list(REMINDER_TOOL_SEQUENCE),
            "code": "\n".join(
                [
                    "now = get_current_timestamp()",
                    "base = timestamp_to_datetime_info(timestamp=now)",
                    "days_until = (5 - base['isoweekday']) % 7",
                    "if days_until == 0:",
                    "    days_until = 7",
                    "shifted = now + days_until * 86400",
                    "target_info = timestamp_to_datetime_info(timestamp=shifted)",
                    "target = datetime_info_to_timestamp(year=target_info['year'], month=target_info['month'], day=target_info['day'], hour=17, minute=0, second=0)",
                    "add_reminder(content='the requested reminder text', reminder_timestamp=target)",
                ]
            ),
        },
        "multi_turn_slot_merge_rule": (
            "If the prefix contains a reminder request with target time and a later "
            "user answer that supplies the reminder content, propose using the "
            "later answer as content."
        ),
        "mutating_tool_name": "add_reminder",
        "mutating_arg_key_types": [
            {"key": "content", "type": "string"},
            {"key": "reminder_timestamp", "type": "number"},
        ],
        "optional_mutating_arg_key_types": [
            {"key": "latitude", "type": "number"},
            {"key": "longitude", "type": "number"},
        ],
        "domain_instructions": [
            "Do not abstain solely because current time or converted timestamp is unknown; compute it inside code with the allowed read tools.",
            "Do not treat an omitted timezone as missing information; the allowed ToolSandbox time utilities use the episode environment's local timestamp semantics and expose no timezone argument.",
            "If the cheap assistant asks for timezone, device timezone, city, UTC offset, or location after the user has supplied reminder content and time, treat that as cheap uncertainty rather than a missing required slot; do not ask for timezone and propose the residual suffix using local timestamp semantics.",
            "For relative dates such as tomorrow, tonight, next week, or next Friday, call get_current_timestamp and timestamp_to_datetime_info, add the required whole-day delta in seconds inside code, then construct the target datetime with datetime_info_to_timestamp.",
            "timestamp_to_datetime_info returns isoweekday, not weekday; Monday is 1 and Friday is 5.",
            "For explicit absolute date/time requests, you may omit unneeded read tools and use an ordered subsequence ending in add_reminder.",
            "If the visible prefix lacks reminder content, abstain so cheap/user interaction can continue.",
            "For multi-turn clarification, combine visible user turns: an earlier turn may provide the time and a later answer may provide the reminder content.",
            "A short user answer such as 'Buy chocolate milk.' after a reminder clarification is valid reminder content.",
            "A later user turn such as 'set a reminder to buy chocolate milk next Friday at 5 PM' contains reminder content; extract the substantive action phrase after 'to' as content instead of abstaining.",
            "For a frontier-certified reminder-add suffix, do not abstain merely because the request was split across turns or because the final answer is a short fragment; propose with code that computes the timestamp and uses the visible content.",
            "Abstain only if a required reminder slot is still absent or the visible prefix explicitly contradicts the typed suffix.",
            "If confirmation text is needed after adding the reminder, include assistant_response with that final text.",
        ],
    }


PROPOSAL_TOOL_NAME = "record_authoring_step"


def _submitted_proposal_arguments(message: Any) -> str | None:
    """Arguments of the proposal call, or None when the reply carried none.

    The name is matched rather than assumed. A reply may open with a call to
    one of the tools the session already carries, and reading whichever call
    came first would hand a shell command to a parser expecting a proposal.
    """

    calls = getattr(message, "tool_calls", None)
    if calls is None and isinstance(message, Mapping):
        calls = message.get("tool_calls")
    if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes, bytearray)):
        return None
    for call in calls:
        function = getattr(call, "function", None)
        if function is None and isinstance(call, Mapping):
            function = call.get("function")
        name = getattr(function, "name", None)
        arguments = getattr(function, "arguments", None)
        if isinstance(function, Mapping):
            name = function.get("name")
            arguments = function.get("arguments")
        if str(name or "") != PROPOSAL_TOOL_NAME:
            continue
        if isinstance(arguments, str) and arguments.strip():
            return arguments
        if isinstance(arguments, Mapping):
            return json.dumps(dict(arguments), ensure_ascii=False)
    return None


def _proposal_tool(schema: Mapping[str, Any]) -> dict[str, Any]:
    """One authoring step, offered as the one action the proposer can take.

    The fields are those AuthoringStep.from_response reads, not the shape of
    the proposer's own response envelope: an authoring round is a single step,
    either a call to one allowed tool, a terminal message that closes the
    suffix, or a decline. Offering the envelope instead left every round
    missing the tool name and the terminal message at once.

    A proposer reached through an agent session arrives with a coding agent's
    tools already in its action space, and asked in prose for a step with no
    tool of its own it abstained four rounds running while writing several
    hundred tokens of deliberation. This tool executes nothing. It carries the
    step, so the proposer still only proposes.
    """

    allowed = [str(name) for name in schema.get("allowed_tool_sequence") or ()]
    properties: dict[str, Any] = {
        "action": {
            "type": "string",
            "enum": ["tool_call", "terminal", "decline"],
        },
        "tool_name": (
            {"type": "string", "enum": allowed} if allowed else {"type": "string"}
        ),
        "arguments": {"type": "object"},
        "terminal_response": {
            "type": "string",
            "description": (
                "The message the user receives once this step closes the "
                "suffix. Required on a terminal action, and allowed beside a "
                "tool call when that call is the last one."
            ),
        },
        "obligation_id": {"type": "string"},
        "mode": {"type": "string", "enum": ["decline", "clarify"]},
        "basis": {"type": "string"},
    }
    if schema.get("assistant_response_template_allowed"):
        properties["terminal_response_template"] = {"type": "string"}
        properties["terminal_response_bindings"] = {"type": "object"}
    return {
        "type": "function",
        "function": {
            "name": PROPOSAL_TOOL_NAME,
            "description": (
                "Record the next authoring step. This is the only action "
                "available here: it records the step and runs nothing. Use "
                "action=tool_call with tool_name and arguments to call one "
                "allowed tool, action=terminal with terminal_response to close "
                "the suffix, or action=decline."
            ),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": ["action"],
            },
        },
    }



def _messages(prompt_payload: Mapping[str, Any]) -> list[dict[str, str]]:
    response_schema = prompt_payload.get("response_schema")
    schema = response_schema if isinstance(response_schema, Mapping) else {}
    allowed_tools = ", ".join(str(name) for name in schema.get("allowed_tool_sequence") or ())
    mutating_tool = str(schema.get("mutating_tool_name") or "the mutating tool")
    domain_instructions = " ".join(str(item) for item in schema.get("domain_instructions") or ())
    code_instruction = (
        "When live timestamp or calendar conversion is needed, return a "
        "private Python code string instead of fixed tool_calls. The code "
        f"may call only these tools as an ordered subsequence: {allowed_tools}. "
        if schema.get("code_payload_allowed") is not False
        else (
            "Use ordered tool_calls for this suffix and do not include a code field. "
            f"Tool calls may use only these tools as an ordered subsequence: {allowed_tools}. "
        )
    )
    return [
        {
            "role": "system",
            "content": (
                "You are the strong residual proposer for ToolSandbox CFRC. "
                "Return valid JSON matching response_schema. Proposed residuals "
                f"must be executable in order and must end with {mutating_tool}. "
                f"{code_instruction}"
                "The deterministic frontier monitor already certified the "
                "selected typed suffix as residual-closable from the visible "
                "prefix; prefer decision=propose unless a required slot is "
                "still absent or the prefix explicitly contradicts the suffix. "
                "If the residual needs a final user-facing answer, include "
                "assistant_response in the same JSON object. "
                "If the response_schema allows assistant_response_template, use "
                "template bindings for values that only exist after tool execution. "
                f"{domain_instructions}"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                prompt_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    ]


def _json_object_from_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return {
            "schema": "papc.toolsandbox.base129_proposer_response.v1",
            "decision": "abstain",
            "tool_calls": [],
        }
    return payload if isinstance(payload, dict) else {}


def _strong_call_from_response(response: Mapping[str, Any]) -> RoleCall:
    usage = response.get("usage") if isinstance(response.get("usage"), Mapping) else {}
    return role_call_from_usage(
        "strong",
        usage,
        latency_seconds=_optional_float(response.get("latency_seconds")),
    )


def _public_ordinal_from_candidate(candidate_id: str) -> int:
    match = re.search(r"_(\d+)$", candidate_id)
    if not match:
        return 0
    return int(match.group(1))


def _effect_families_from_units(units: Any) -> list[str]:
    return sorted(
        {
            ".".join(str(unit).split(".")[:2])
            for unit in units
            if str(unit).startswith("toolsandbox.") and len(str(unit).split(".")) >= 2
        }
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "EMPTY_GRAPH_ID",
    "ToolSandboxGroundedStepProposer",
    "ToolSandboxOpenAIStrongProposer",
    "ToolSandboxOpenAIStrongProposerConfig",
]
