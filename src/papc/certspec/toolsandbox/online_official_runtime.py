"""Official ToolSandbox runtime adapter for true-online CFRC episodes."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
import signal
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from papc.certspec.toolsandbox.method_runtime import (
    _patch_numpy_compat,
    _prepare_toolsandbox_imports,
    named_scenarios_for_checkout,
    official_evaluation_summary,
)
from papc.certspec.toolsandbox.role_bridge import (
    ToolSandboxRoleBridgeConfig,
    openai_compatible_role_factories,
)
from papc.certspec.toolsandbox.runtime_accounting import (
    ToolSandboxRuntimeAccountingRecorder,
)

EXPECTED_OFFICIAL_ENGINE = "tool_sandbox Scenario.play_and_evaluate"
_UUID_TOKEN_RE = re.compile(
    r"(?i)(?<![0-9a-f])"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}"
    r"(?![0-9a-f])"
)


@dataclass(frozen=True)
class ToolSandboxOfficialOnlineRuntimeConfig:
    checkout: Path
    api_base: str
    api_key_env: str
    cheap_model: str
    cheap_reasoning_effort: str
    user_model: str
    user_reasoning_effort: str
    provider_timeout_seconds: float | None = 300.0
    role_factories: Mapping[str, Callable[[], Any]] | None = None


class ToolSandboxOfficialOnlineRuntime:
    """Step a live ToolSandbox episode and execute residual suffixes officially."""

    surface = "toolsandbox"

    def __init__(self, config: ToolSandboxOfficialOnlineRuntimeConfig) -> None:
        self.config = config
        self._scenario: Any | None = None
        self._scenario_name = ""
        self._roles: dict[Any, Any] = {}
        self._context: Any | None = None
        self._runtime_accounting = ToolSandboxRuntimeAccountingRecorder()

    def start_episode(self, scenario_name: str) -> None:
        _prepare_toolsandbox_imports(self.config.checkout)
        _patch_numpy_compat()

        from tool_sandbox.common.execution_context import (  # type: ignore
            DatabaseNamespace,
            RoleType,
            get_current_context,
            set_current_context,
        )
        from tool_sandbox.common.tool_discovery import ToolBackend  # type: ignore
        from tool_sandbox.roles.execution_environment import (
            ExecutionEnvironment,  # type: ignore
        )

        scenarios = named_scenarios_for_checkout(
            self.config.checkout,
            preferred_tool_backend=ToolBackend.DEFAULT,
        )
        if scenario_name not in scenarios:
            raise KeyError(f"unknown ToolSandbox scenario: {scenario_name}")
        scenario = scenarios[scenario_name]
        self._runtime_accounting = ToolSandboxRuntimeAccountingRecorder()
        context = copy.deepcopy(scenario.starting_context)
        set_current_context(context)
        factories = self._role_factories()
        self._roles = {
            RoleType.USER: factories["user"](),
            RoleType.AGENT: factories["agent"](),
            RoleType.EXECUTION_ENVIRONMENT: ExecutionEnvironment(),
        }
        self._wrap_accounted_roles(RoleType)
        sandbox_db = context.get_database(
            DatabaseNamespace.SANDBOX,
            drop_sandbox_message_index=False,
            get_all_history_snapshots=True,
        )
        max_index = context.max_sandbox_message_index
        for message_index in range(max_index + 1):
            if (
                sandbox_db["recipient"][message_index] == RoleType.EXECUTION_ENVIRONMENT
                and sandbox_db["sender"][message_index] == RoleType.SYSTEM
            ):
                recipient = sandbox_db["recipient"][message_index]
                _call_with_deadline(
                    lambda recipient=recipient, message_index=message_index: self._roles[
                        recipient
                    ].respond(ending_index=message_index),
                    timeout_seconds=self.config.provider_timeout_seconds,
                    label=f"ToolSandbox setup respond({recipient})",
                )
        if get_current_context().max_sandbox_message_index != max_index:
            raise RuntimeError("ToolSandbox system setup unexpectedly changed message count")
        self._scenario = scenario
        self._scenario_name = scenario_name
        self._context = get_current_context()

    def is_active(self) -> bool:
        scenario = self._require_scenario()
        context = self._require_context()
        sandbox_db = _sandbox_db(context)
        return bool(sandbox_db["conversation_active"][-1]) and int(
            sandbox_db["sandbox_message_index"][-1]
        ) < int(scenario.max_messages + context.max_sandbox_message_index)

    def step(self) -> Mapping[str, Any]:
        if not self.is_active():
            return {"advanced": False}
        from tool_sandbox.common.execution_context import (  # type: ignore
            DatabaseNamespace,
            get_current_context,
        )

        context = self._require_context()
        sandbox_db = context.get_database(
            DatabaseNamespace.SANDBOX,
            drop_sandbox_message_index=False,
        )
        recipient = sandbox_db["recipient"][-1]
        _call_with_deadline(
            lambda: self._roles[recipient].respond(),
            timeout_seconds=self.config.provider_timeout_seconds,
            label=f"ToolSandbox role respond({recipient})",
        )
        self._context = get_current_context()
        return {"advanced": True, "recipient": str(recipient)}

    def visible_conversation(self) -> Sequence[Mapping[str, Any]]:
        return _visible_openai_messages(self._require_context())

    def available_tool_names(self) -> Sequence[str]:
        """Return the same public tool allow-list exposed to the agent role."""

        tool_allow_list = self._require_context().tool_allow_list
        return tuple(sorted(str(name) for name in tool_allow_list or ()))

    def snapshot_context(self) -> Any:
        """Return an independent native checkpoint of the reached state."""

        return copy.deepcopy(self._require_context())

    def restore_context(self, checkpoint: Any) -> None:
        """Install one independent checkpoint as the active native state."""

        _prepare_toolsandbox_imports(self.config.checkout)
        from tool_sandbox.common.execution_context import (  # type: ignore
            set_current_context,
        )

        context = copy.deepcopy(checkpoint)
        set_current_context(context)
        self._context = context

    def context_sha256(self, checkpoint: Any | None = None) -> str:
        """Hash state databases/config while excluding the live console object."""

        context = checkpoint if checkpoint is not None else self._require_context()
        serialized = context.to_dict(serialize_console=False)
        return hashlib.sha256(
            json.dumps(
                serialized,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()

    def context_replay_sha256(self, reached_checkpoint: Any) -> str:
        """Hash post-state modulo IDs freshly allocated after the reached state."""

        reached_serialized = _without_tool_traces(
            reached_checkpoint.to_dict(serialize_console=False)
        )
        stable_uuids = _uuid_tokens(reached_serialized)
        stable_creation_timestamps = _creation_timestamps(reached_serialized)
        serialized = _without_tool_traces(
            self._require_context().to_dict(serialize_console=False)
        )
        normalized = _normalize_fresh_uuids(
            serialized,
            stable_uuids=stable_uuids,
            stable_creation_timestamps=stable_creation_timestamps,
        )
        return hashlib.sha256(
            json.dumps(
                normalized,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()

    def execute_suffix(self, suffix_payloads: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        errors: list[str] = []
        executed = 0
        assistant_responses = 0
        for payload in suffix_payloads:
            payload_failed = False
            tool_outputs: list[str] = []
            code = payload.get("code")
            if isinstance(code, str) and code.strip():
                result = _execute_code(code.strip())
                tool_outputs.extend(_string_outputs(result.get("tool_outputs")))
                executed += _sequence_count(payload.get("tool_sequence")) or 1
                if result.get("tool_call_exception"):
                    errors.append(str(result["tool_call_exception"]))
                    payload_failed = True
                assistant_response, render_error = _assistant_response(
                    payload,
                    tool_outputs=tool_outputs,
                )
                if render_error:
                    errors.append(render_error)
                    payload_failed = True
                if assistant_response and not payload_failed:
                    _append_assistant_response(assistant_response)
                    assistant_responses += 1
                continue
            for call in _tool_calls(payload):
                try:
                    arguments = _resolve_tool_arguments(
                        _mapping(call.get("arguments")),
                        tool_outputs=tool_outputs,
                    )
                except (KeyError, TypeError, ValueError, IndexError) as exc:
                    errors.append(f"tool_argument_binding_failed:{type(exc).__name__}")
                    payload_failed = True
                    break
                result = _execute_tool_call(
                    tool_name=str(call["tool_name"]),
                    arguments=arguments,
                )
                executed += 1
                content = result.get("content")
                if isinstance(content, str):
                    tool_outputs.append(content)
                tool_call_exception = result.get("tool_call_exception")
                if tool_call_exception and not _allowed_tool_exception(
                    call,
                    str(tool_call_exception),
                ):
                    errors.append(str(result["tool_call_exception"]))
                    payload_failed = True
                    break
            assistant_response, render_error = _assistant_response(
                payload,
                tool_outputs=tool_outputs,
            )
            if render_error:
                errors.append(render_error)
                payload_failed = True
            if assistant_response and not payload_failed:
                _append_assistant_response(assistant_response)
                assistant_responses += 1
        self._context = _current_execution_context(self._context)
        evaluation = self.evaluate()
        return {
            "official_evaluator_used": evaluation["official_evaluator_used"],
            "official_engine": EXPECTED_OFFICIAL_ENGINE,
            "runtime_error_count": len(errors),
            "runtime_errors": errors,
            "executed_tool_call_count": executed,
            "assistant_response_appended_count": assistant_responses,
            "tool_trace_recorded": _tool_trace_recorded(self._require_context()),
            "evaluation": dict(evaluation["evaluation"]),
        }

    def execute_grounded_call(
        self,
        *,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Execute one native call on the active context and return its receipt.

        Receipt-grounded authoring needs the concrete result of each call before
        the successor proposes the next one.  ``execute_suffix`` cannot supply
        that: it reports aggregate counts for a whole payload list.  Snapshot and
        restore stay with the caller, so this method never decides which state a
        call runs against.
        """

        result = _execute_tool_call(tool_name=str(tool_name), arguments=dict(arguments))
        self._context = _current_execution_context(self._context)
        exception = result.get("tool_call_exception")
        return {
            "tool_name": str(tool_name),
            "arguments": dict(arguments),
            "succeeded": not exception,
            "result": str(result.get("content") or ""),
            "tool_call_exception": str(exception) if exception else "",
        }

    def evaluate(self) -> Mapping[str, Any]:
        scenario = self._require_scenario()
        context = self._require_context()
        evaluation_result = scenario.evaluation.evaluate(
            execution_context=context,
            max_turn_count=scenario.max_messages,
        )
        return {
            "official_evaluator_used": True,
            "official_engine": EXPECTED_OFFICIAL_ENGINE,
            "evaluation": official_evaluation_summary(evaluation_result),
        }

    def runtime_ledger_summary(self) -> Mapping[str, Any]:
        return self._runtime_accounting.public_summary()

    def _role_factories(self) -> Mapping[str, Callable[[], Any]]:
        if self.config.role_factories is not None:
            return self.config.role_factories
        return openai_compatible_role_factories(
            ToolSandboxRoleBridgeConfig(
                agent_model=self.config.cheap_model,
                user_model=self.config.user_model,
                api_base=self.config.api_base,
                api_key_env=self.config.api_key_env,
                agent_reasoning_effort=self.config.cheap_reasoning_effort,
                user_reasoning_effort=self.config.user_reasoning_effort,
                provider_timeout_seconds=self.config.provider_timeout_seconds,
            )
        )

    def _require_scenario(self) -> Any:
        if self._scenario is None:
            raise RuntimeError("ToolSandbox episode has not started")
        return self._scenario

    def _require_context(self) -> Any:
        if self._context is None:
            raise RuntimeError("ToolSandbox episode has not started")
        return self._context

    def _wrap_accounted_roles(self, role_type: Any) -> None:
        agent = self._roles.get(role_type.AGENT)
        user = self._roles.get(role_type.USER)
        if agent is not None and hasattr(agent, "model_inference"):
            self._runtime_accounting.wrap_role_model_inference(
                agent,
                role_name="agent",
                runtime_role="cheap",
            )
        if user is not None and hasattr(user, "model_inference"):
            self._runtime_accounting.wrap_role_model_inference(
                user,
                role_name="user",
                runtime_role="user_simulator",
            )


def _call_with_deadline(
    call: Callable[[], Any],
    *,
    timeout_seconds: float | None,
    label: str,
) -> Any:
    if timeout_seconds is None or timeout_seconds <= 0:
        return call()
    if threading.current_thread() is not threading.main_thread():
        return call()

    def _timeout_handler(signum: int, frame: Any) -> None:
        raise TimeoutError(f"{label} timed out after {timeout_seconds:g}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
    try:
        return call()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def _current_execution_context(fallback: Any) -> Any:
    try:
        from tool_sandbox.common.execution_context import (  # type: ignore
            get_current_context,
        )
    except ModuleNotFoundError:
        return fallback

    return get_current_context()


def _visible_openai_messages(context: Any) -> list[Mapping[str, Any]]:
    import polars as pl  # type: ignore
    from tool_sandbox.common.execution_context import (  # type: ignore
        DatabaseNamespace,
        RoleType,
    )
    from tool_sandbox.common.message_conversion import (  # type: ignore
        Message,
        to_openai_messages,
    )

    message_subset_database = context.get_database(
        DatabaseNamespace.SANDBOX,
        drop_sandbox_message_index=False,
        get_all_history_snapshots=True,
    ).filter(
        ((pl.col("sender") == RoleType.AGENT) | (pl.col("recipient") == RoleType.AGENT))
        & ((pl.col("visible_to") != [RoleType.USER]) | (pl.col("visible_to").is_null()))
    )
    if message_subset_database.is_empty():
        return []
    messages = [
        Message(**row)
        for row in message_subset_database.drop("sandbox_message_index").to_dicts()
    ]
    openai_messages, _ = to_openai_messages(messages=messages)
    return [dict(message) for message in openai_messages]


def _execute_tool_call(*, tool_name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    from tool_sandbox.common.execution_context import (  # type: ignore
        DatabaseNamespace,
        RoleType,
        get_current_context,
        set_current_context,
    )
    from tool_sandbox.common.message_conversion import Message  # type: ignore
    from tool_sandbox.roles.execution_environment import (
        ExecutionEnvironment,  # type: ignore
    )

    if not tool_name.isidentifier():
        return {
            "tool_call_exception": f"unsafe tool name: {tool_name}",
            "response_present": False,
        }
    context = get_current_context()
    set_current_context(context)
    code = _tool_call_code(tool_name=tool_name, arguments=arguments)
    ExecutionEnvironment.add_messages(
        [
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content=code,
                visible_to=[RoleType.AGENT, RoleType.EXECUTION_ENVIRONMENT],
            )
        ]
    )
    ExecutionEnvironment().respond()
    sandbox = get_current_context().get_database(
        DatabaseNamespace.SANDBOX,
        get_all_history_snapshots=True,
        drop_sandbox_message_index=False,
    )
    last = sandbox.to_dicts()[-1] if not sandbox.is_empty() else {}
    return {
        "tool_call_exception": last.get("tool_call_exception"),
        "response_present": bool(last.get("content")),
        "content": str(last.get("content") or ""),
    }


def _execute_code(code: str) -> dict[str, Any]:
    chunks = _code_statement_chunks(code)
    if not chunks:
        return {"tool_call_exception": "empty code payload", "response_present": False}
    last: dict[str, Any] = {
        "tool_call_exception": None,
        "response_present": False,
    }
    outputs: list[str] = []
    for chunk in chunks:
        last = _execute_code_chunk(chunk)
        if not last.get("tool_call_exception"):
            binding = _assignment_tool_result_binding(
                chunk,
                tool_traces=last.get("tool_traces"),
            )
            if binding is not None:
                _install_console_binding(*binding)
        content = last.get("content")
        if isinstance(content, str) and content.strip():
            outputs.append(content)
        if last.get("tool_call_exception"):
            last["tool_outputs"] = outputs
            return last
    last["tool_outputs"] = outputs
    return last


def _execute_code_chunk(code: str) -> dict[str, Any]:
    from tool_sandbox.common.execution_context import (  # type: ignore
        DatabaseNamespace,
        RoleType,
        get_current_context,
        set_current_context,
    )
    from tool_sandbox.common.message_conversion import Message  # type: ignore
    from tool_sandbox.roles.execution_environment import (
        ExecutionEnvironment,  # type: ignore
    )

    context = get_current_context()
    set_current_context(context)
    ExecutionEnvironment.add_messages(
        [
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content=code,
                visible_to=[RoleType.AGENT, RoleType.EXECUTION_ENVIRONMENT],
            )
        ]
    )
    ExecutionEnvironment().respond()
    sandbox = get_current_context().get_database(
        DatabaseNamespace.SANDBOX,
        get_all_history_snapshots=True,
        drop_sandbox_message_index=False,
    )
    last = sandbox.to_dicts()[-1] if not sandbox.is_empty() else {}
    return {
        "tool_call_exception": last.get("tool_call_exception"),
        "response_present": bool(last.get("content")),
        "content": str(last.get("content") or ""),
        "tool_traces": _parsed_tool_traces(last.get("tool_trace")),
    }


def _append_assistant_response(content: str) -> None:
    from tool_sandbox.common.execution_context import (  # type: ignore
        RoleType,
        get_current_context,
        set_current_context,
    )
    from tool_sandbox.common.message_conversion import Message  # type: ignore
    from tool_sandbox.roles.execution_environment import (  # type: ignore
        ExecutionEnvironment,
    )

    context = get_current_context()
    set_current_context(context)
    ExecutionEnvironment.add_messages(
        [
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.USER,
                content=content,
                visible_to=[RoleType.AGENT, RoleType.USER],
            )
        ]
    )


def _code_statement_chunks(code: str) -> tuple[str, ...]:
    tree = ast.parse(code)
    lines = code.splitlines()
    chunks: list[str] = []
    for statement in tree.body:
        start = getattr(statement, "lineno", None)
        end = getattr(statement, "end_lineno", None)
        if start is None or end is None:
            return (code,)
        segment = "\n".join(lines[start - 1 : end]).strip()
        if segment:
            chunks.append(segment)
    return tuple(chunks)


def _parsed_tool_traces(raw: Any) -> tuple[Mapping[str, Any], ...]:
    traces: list[Mapping[str, Any]] = []

    def collect(value: Any) -> None:
        if isinstance(value, str):
            try:
                collect(json.loads(value))
            except json.JSONDecodeError:
                return
        elif isinstance(value, Mapping):
            traces.append(dict(value))
        elif isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            for item in value:
                collect(item)

    collect(raw)
    return tuple(traces)


def _assignment_tool_result_binding(
    code: str,
    *,
    tool_traces: Any,
) -> tuple[str, Any] | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        return None
    assignment = tree.body[0]
    if len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Name):
        return None
    if not isinstance(assignment.value, ast.Call) or not isinstance(
        assignment.value.func,
        ast.Name,
    ):
        return None
    tool_name = assignment.value.func.id
    for trace in reversed(_parsed_tool_traces(tool_traces)):
        if trace.get("tool_name") == tool_name and "result" in trace:
            return assignment.targets[0].id, copy.deepcopy(trace["result"])
    return None


def _install_console_binding(name: str, value: Any) -> None:
    from tool_sandbox.common.execution_context import (  # type: ignore
        get_current_context,
        set_current_context,
    )

    context = get_current_context()
    context.interactive_console.locals[name] = copy.deepcopy(value)
    set_current_context(context)


def _tool_call_code(*, tool_name: str, arguments: Mapping[str, Any]) -> str:
    return (
        "_papc_parameters = "
        + repr(json.loads(json.dumps(dict(arguments), ensure_ascii=False)))
        + "\n"
        + f"_papc_response = {tool_name}(**_papc_parameters)\n"
        + "print(repr(_papc_response))"
    )


def _tool_calls(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw = payload.get("tool_calls")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return ()
    return tuple(item for item in raw if isinstance(item, Mapping))


def _resolve_tool_arguments(
    arguments: Mapping[str, Any],
    *,
    tool_outputs: Sequence[str],
) -> dict[str, Any]:
    return {
        str(name): _resolve_tool_argument_value(value, tool_outputs=tool_outputs)
        for name, value in arguments.items()
    }


def _resolve_tool_argument_value(value: Any, *, tool_outputs: Sequence[str]) -> Any:
    if isinstance(value, Mapping) and value.get("source") == "tool_output":
        return _raw_binding_value(value, tool_outputs=tool_outputs)
    if isinstance(value, list):
        return [
            _resolve_tool_argument_value(item, tool_outputs=tool_outputs)
            for item in value
        ]
    if isinstance(value, Mapping):
        return {
            str(name): _resolve_tool_argument_value(item, tool_outputs=tool_outputs)
            for name, item in value.items()
        }
    return value


def _allowed_tool_exception(call: Mapping[str, Any], exception_text: str) -> bool:
    if call.get("allow_exception") is not True:
        return False
    expected = call.get("expected_exception_substring")
    if isinstance(expected, str) and expected:
        if expected in exception_text:
            return True
        normalized_expected = expected.lower()
        normalized_exception = exception_text.lower()
        compact_expected = re.sub(r"[^a-z0-9]+", "", normalized_expected)
        compact_exception = re.sub(r"[^a-z0-9]+", "", normalized_exception)
        if compact_expected and compact_expected in compact_exception:
            return True
        if "disabled" in normalized_expected and "not enabled" in normalized_exception:
            return True
        return False
    return True


def _assistant_response(
    payload: Mapping[str, Any],
    *,
    tool_outputs: Sequence[str] = (),
) -> tuple[str, str | None]:
    template = payload.get("assistant_response_template")
    if isinstance(template, str) and template.strip():
        try:
            rendered = render_assistant_response_template(
                template=template,
                bindings=payload.get("assistant_response_bindings"),
                tool_outputs=tool_outputs,
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            return "", f"assistant_response_render_failed:{type(exc).__name__}"
        return rendered, None
    value = payload.get("assistant_response")
    if isinstance(value, str) and value.strip():
        return value.strip(), None
    return "", None


def render_assistant_response_template(
    *,
    template: str,
    bindings: Any,
    tool_outputs: Sequence[str],
) -> str:
    """Render a receipt-bound assistant response or raise on an invalid binding."""

    return _render_assistant_response_template(
        template=template,
        bindings=bindings,
        tool_outputs=tool_outputs,
    )


def _render_assistant_response_template(
    *,
    template: str,
    bindings: Any,
    tool_outputs: Sequence[str],
) -> str:
    if not isinstance(bindings, Mapping):
        raise TypeError("assistant_response_bindings must be a mapping")
    values = {
        str(name): _binding_value(binding, tool_outputs=tool_outputs)
        for name, binding in bindings.items()
        if isinstance(binding, Mapping)
    }
    return template.format_map(values).strip()


def _binding_value(binding: Mapping[str, Any], *, tool_outputs: Sequence[str]) -> str:
    value = _raw_binding_value(binding, tool_outputs=tool_outputs)
    if "round" in binding:
        digits = int(binding["round"])
        number = round(float(value), digits)
        if digits == 0:
            return str(int(number))
        return f"{number:.{digits}f}"
    return str(value)


def _raw_binding_value(binding: Mapping[str, Any], *, tool_outputs: Sequence[str]) -> Any:
    index_value = binding.get("tool_output_index")
    if index_value is None:
        index_value = binding.get("call_index")
    index = int(index_value)
    try:
        raw = tool_outputs[index]
        value: Any = _parse_tool_output(raw)
        path = binding.get("path")
        if path is None:
            key = binding.get("key", binding.get("field"))
            path = [key] if key is not None else []
        for part in path or ():
            if isinstance(value, Mapping):
                key = str(part)
                if key in value:
                    value = value[key]
                elif key == "temperature" and "current_temperature" in value:
                    value = value["current_temperature"]
                elif key == "temperature" and "average_temperature" in value:
                    value = value["average_temperature"]
                else:
                    raise KeyError(key)
            elif isinstance(value, Sequence) and not isinstance(
                value, (str, bytes, bytearray)
            ):
                if isinstance(part, str) and value and isinstance(value[0], Mapping):
                    value = value[0][part]
                else:
                    value = value[int(part)]
            elif str(part) in {"amount", "converted_amount", "value", "result"}:
                continue
            else:
                raise TypeError("cannot traverse assistant response binding")
    except (IndexError, KeyError, TypeError):
        if "fallback" not in binding:
            raise
        return copy.deepcopy(binding["fallback"])
    if "fallback" in binding and (value is None or value == ""):
        return copy.deepcopy(binding["fallback"])
    return value


def _parse_tool_output(raw: str) -> Any:
    text = raw.strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text


def _string_outputs(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence_count(value: Any) -> int:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return 0
    return len(value)


def _sandbox_db(context: Any, *, get_all_history_snapshots: bool = False) -> Any:
    from tool_sandbox.common.execution_context import DatabaseNamespace  # type: ignore

    return context.get_database(
        DatabaseNamespace.SANDBOX,
        drop_sandbox_message_index=False,
        get_all_history_snapshots=get_all_history_snapshots,
    )


def _tool_trace_recorded(context: Any) -> bool:
    sandbox = _sandbox_db(context, get_all_history_snapshots=True)
    try:
        values = sandbox["tool_trace"].to_list()
    except Exception:  # noqa: BLE001 - public receipt only needs a coarse flag.
        return False
    return any(value is not None for value in values)


def _uuid_tokens(value: Any) -> frozenset[str]:
    tokens: set[str] = set()

    def collect(item: Any) -> None:
        if isinstance(item, str):
            tokens.update(match.group(0).lower() for match in _UUID_TOKEN_RE.finditer(item))
        elif isinstance(item, Mapping):
            for key in sorted(item, key=lambda candidate: str(candidate)):
                collect(key)
                collect(item[key])
        elif isinstance(item, Sequence) and not isinstance(
            item,
            (str, bytes, bytearray),
        ):
            for child in item:
                collect(child)

    collect(value)
    return frozenset(tokens)


def _normalize_fresh_uuids(
    value: Any,
    *,
    stable_uuids: frozenset[str],
    stable_creation_timestamps: frozenset[float] = frozenset(),
) -> Any:
    replacements: dict[str, str] = {}
    creation_timestamp_replacements: dict[float, str] = {}

    def normalize(item: Any) -> Any:
        if isinstance(item, str):
            def replace(match: re.Match[str]) -> str:
                token = match.group(0).lower()
                if token in stable_uuids:
                    return match.group(0)
                if token not in replacements:
                    replacements[token] = f"<papc-generated-uuid-{len(replacements)}>"
                return replacements[token]

            return _UUID_TOKEN_RE.sub(replace, item)
        if isinstance(item, Mapping):
            normalized: dict[str, Any] = {}
            for key in sorted(item, key=lambda candidate: str(candidate)):
                key_text = str(key)
                child = item[key]
                if (
                    key_text == "creation_timestamp"
                    and isinstance(child, (int, float))
                    and not isinstance(child, bool)
                    and float(child) not in stable_creation_timestamps
                ):
                    timestamp = float(child)
                    if timestamp not in creation_timestamp_replacements:
                        creation_timestamp_replacements[timestamp] = (
                            "<papc-generated-creation-timestamp-"
                            f"{len(creation_timestamp_replacements)}>"
                        )
                    normalized[key_text] = creation_timestamp_replacements[timestamp]
                else:
                    normalized[key_text] = normalize(child)
            return normalized
        if isinstance(item, Sequence) and not isinstance(
            item,
            (str, bytes, bytearray),
        ):
            return [normalize(child) for child in item]
        return item

    return normalize(value)


def _without_tool_traces(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _without_tool_traces(child)
            for key, child in value.items()
            if str(key) != "tool_trace"
        }
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [_without_tool_traces(child) for child in value]
    return value


def _creation_timestamps(value: Any) -> frozenset[float]:
    timestamps: set[float] = set()

    def collect(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if (
                    str(key) == "creation_timestamp"
                    and isinstance(child, (int, float))
                    and not isinstance(child, bool)
                ):
                    timestamps.add(float(child))
                else:
                    collect(child)
        elif isinstance(item, Sequence) and not isinstance(
            item,
            (str, bytes, bytearray),
        ):
            for child in item:
                collect(child)

    collect(value)
    return frozenset(timestamps)


__all__ = [
    "EXPECTED_OFFICIAL_ENGINE",
    "ToolSandboxOfficialOnlineRuntime",
    "ToolSandboxOfficialOnlineRuntimeConfig",
    "render_assistant_response_template",
]
