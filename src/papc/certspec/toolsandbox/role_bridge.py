"""Repo-local OpenAI-compatible ToolSandbox role bridge.

The upstream ToolSandbox OpenAI roles intentionally hardcode the public OpenAI
base URL.  PAPC online runs use an explicit OpenAI-compatible endpoint instead,
so bounded ToolSandbox probes must construct roles through this bridge rather
than patching the upstream checkout.
"""

from __future__ import annotations

import hashlib
import inspect
import keyword
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from types import SimpleNamespace
from typing import Any

DEFAULT_PROVIDER_TIMEOUT_SECONDS = 300.0


@dataclass(frozen=True)
class ToolSandboxRoleBridgeConfig:
    agent_model: str
    user_model: str
    api_base: str
    api_key_env: str = "OPENAI_API_KEY"
    agent_reasoning_effort: str | None = None
    user_reasoning_effort: str | None = None
    provider_timeout_seconds: float | None = DEFAULT_PROVIDER_TIMEOUT_SECONDS


def openai_client_kwargs(
    *,
    api_base: str,
    api_key_env: str,
    timeout_seconds: float | None = DEFAULT_PROVIDER_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Build OpenAI SDK kwargs from an env-var name and explicit base URL."""

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"missing API key env var: {api_key_env}")
    kwargs = {"api_key": api_key}
    if api_base:
        kwargs["base_url"] = api_base
    if timeout_seconds is not None:
        kwargs["timeout"] = float(timeout_seconds)
    return kwargs


def build_openai_client(
    *,
    api_base: str,
    api_key_env: str,
    timeout_seconds: float | None = DEFAULT_PROVIDER_TIMEOUT_SECONDS,
) -> Any:
    """Construct an OpenAI client without relying on global OPENAI_BASE_URL."""

    from openai import OpenAI  # type: ignore

    try:
        return OpenAI(
            **openai_client_kwargs(
                api_base=api_base,
                api_key_env=api_key_env,
                timeout_seconds=timeout_seconds,
            )
        )
    except TypeError as exc:
        if "proxies" not in str(exc):
            raise
        from httpx import Client  # type: ignore

        return OpenAI(
            **openai_client_kwargs(
                api_base=api_base,
                api_key_env=api_key_env,
                timeout_seconds=timeout_seconds,
            ),
            http_client=Client(),
        )


def callable_accepts_reasoning_effort(create: Callable[..., Any]) -> bool:
    """Return whether a chat completions create callable accepts reasoning_effort."""

    return callable_accepts_parameter(create, "reasoning_effort")


def callable_accepts_extra_body(create: Callable[..., Any]) -> bool:
    """Return whether a chat completions create callable accepts extra_body."""

    return callable_accepts_parameter(create, "extra_body")


def callable_accepts_parameter(create: Callable[..., Any], parameter_name: str) -> bool:
    """Return whether a callable accepts a named kwarg or arbitrary kwargs."""

    try:
        signature = inspect.signature(create)
    except (TypeError, ValueError):
        return False
    if parameter_name in signature.parameters:
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def openai_sdk_supports_reasoning_effort() -> bool:
    """Check the installed OpenAI SDK before starting a paid ToolSandbox run."""

    try:
        from openai.resources.chat.completions import Completions  # type: ignore
    except Exception:  # noqa: BLE001 - source checks report capability as false.
        return False
    return callable_accepts_reasoning_effort(Completions.create)


def apply_reasoning_effort(client: Any, reasoning_effort: str | None) -> Any:
    """Attach a default reasoning effort to a ToolSandbox OpenAI client."""

    if not reasoning_effort or reasoning_effort == "none":
        return client
    original_create = client.chat.completions.create
    supports_top_level = callable_accepts_reasoning_effort(original_create)
    supports_extra_body = callable_accepts_extra_body(original_create)
    if not supports_top_level and not supports_extra_body:
        raise RuntimeError(
            "installed OpenAI SDK chat.completions.create does not support "
            "reasoning_effort or extra_body; use a PAPC runtime with a compatible OpenAI SDK"
        )

    def create_with_reasoning(*args: Any, **kwargs: Any) -> Any:
        if supports_top_level:
            kwargs.setdefault("reasoning_effort", reasoning_effort)
        else:
            extra_body = dict(kwargs.get("extra_body") or {})
            extra_body.setdefault("reasoning_effort", reasoning_effort)
            kwargs["extra_body"] = extra_body
        return original_create(*args, **kwargs)

    client.chat.completions.create = create_with_reasoning
    return client


def _python_tool_call_binding(raw_id: str) -> str:
    if raw_id.isidentifier() and not keyword.iskeyword(raw_id):
        return raw_id
    digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()
    return f"_papc_tool_call_{digest}"


def _install_toolsandbox_tool_call_id_compatibility() -> None:
    from tool_sandbox.common import message_conversion  # type: ignore

    converter = message_conversion.openai_tool_call_to_python_code
    if not getattr(converter, "_papc_tool_call_id_compatibility", False):

        @wraps(converter)
        def compatible_tool_call_to_python_code(
            tool_call: Any,
            available_tool_names: set[str],
            execution_facing_tool_name: str | None,
        ) -> str:
            raw_id = tool_call.id
            if not isinstance(raw_id, str):
                raise TypeError("ToolSandbox tool-call ID must be a string")
            binding = _python_tool_call_binding(raw_id)
            if binding == raw_id:
                return converter(
                    tool_call,
                    available_tool_names,
                    execution_facing_tool_name,
                )
            return converter(
                SimpleNamespace(id=binding, function=tool_call.function),
                available_tool_names,
                execution_facing_tool_name,
            )

        compatible_tool_call_to_python_code._papc_tool_call_id_compatibility = True  # type: ignore[attr-defined]
        message_conversion.openai_tool_call_to_python_code = (
            compatible_tool_call_to_python_code
        )

    serializer = message_conversion.to_openai_messages
    if not getattr(serializer, "_papc_tool_call_id_compatibility", False):

        @wraps(serializer)
        def compatible_to_openai_messages(
            messages: list[Any],
        ) -> tuple[list[dict[str, Any]], list[list[int]]]:
            openai_messages, indices_mapping = serializer(messages)
            for openai_message, source_indices in zip(
                openai_messages,
                indices_mapping,
                strict=True,
            ):
                tool_calls = openai_message.get("tool_calls")
                if not tool_calls:
                    continue
                source_messages = [
                    messages[index]
                    for index in source_indices
                    if messages[index].sender == message_conversion.RoleType.AGENT
                    and messages[index].recipient
                    == message_conversion.RoleType.EXECUTION_ENVIRONMENT
                ]
                if len(source_messages) != len(tool_calls):
                    raise RuntimeError(
                        "ToolSandbox tool-call serialization lost its source-message mapping"
                    )
                for tool_call, source_message in zip(
                    tool_calls,
                    source_messages,
                    strict=True,
                ):
                    raw_id = source_message.openai_tool_call_id
                    if not raw_id:
                        raise RuntimeError(
                            "ToolSandbox assistant tool call is missing openai_tool_call_id"
                        )
                    tool_call["id"] = raw_id
            return openai_messages, indices_mapping

        compatible_to_openai_messages._papc_tool_call_id_compatibility = True  # type: ignore[attr-defined]
        message_conversion.to_openai_messages = compatible_to_openai_messages

    for module_name in (
        "tool_sandbox.roles.openai_api_agent",
        "tool_sandbox.roles.openai_api_user",
    ):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        if hasattr(module, "openai_tool_call_to_python_code"):
            module.openai_tool_call_to_python_code = (
                message_conversion.openai_tool_call_to_python_code
            )
        if hasattr(module, "to_openai_messages"):
            module.to_openai_messages = message_conversion.to_openai_messages


def openai_compatible_agent_factory(
    *,
    model_name: str,
    api_base: str,
    api_key_env: str = "OPENAI_API_KEY",
    reasoning_effort: str | None = None,
    provider_timeout_seconds: float | None = DEFAULT_PROVIDER_TIMEOUT_SECONDS,
) -> Callable[[], Any]:
    """Return a no-argument ToolSandbox agent role factory."""

    _install_toolsandbox_tool_call_id_compatibility()
    from tool_sandbox.roles.openai_api_agent import OpenAIAPIAgent  # type: ignore

    class PAPCOpenAICompatibleAgent(OpenAIAPIAgent):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            self.model_name = model_name
            self.openai_client = apply_reasoning_effort(
                build_openai_client(
                    api_base=api_base,
                    api_key_env=api_key_env,
                    timeout_seconds=provider_timeout_seconds,
                ),
                reasoning_effort,
            )

    return PAPCOpenAICompatibleAgent


def openai_compatible_user_factory(
    *,
    model_name: str,
    api_base: str,
    api_key_env: str = "OPENAI_API_KEY",
    reasoning_effort: str | None = None,
    provider_timeout_seconds: float | None = DEFAULT_PROVIDER_TIMEOUT_SECONDS,
) -> Callable[[], Any]:
    """Return a no-argument ToolSandbox user-simulator role factory."""

    _install_toolsandbox_tool_call_id_compatibility()
    from tool_sandbox.roles.openai_api_user import OpenAIAPIUser  # type: ignore

    class PAPCOpenAICompatibleUser(OpenAIAPIUser):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            self.model_name = model_name
            self.openai_client = apply_reasoning_effort(
                build_openai_client(
                    api_base=api_base,
                    api_key_env=api_key_env,
                    timeout_seconds=provider_timeout_seconds,
                ),
                reasoning_effort,
            )

    return PAPCOpenAICompatibleUser


def openai_compatible_role_factories(
    config: ToolSandboxRoleBridgeConfig,
) -> dict[str, Callable[[], Any]]:
    """Return factories keyed by ToolSandbox role name for a patchless runner."""

    return {
        "agent": openai_compatible_agent_factory(
            model_name=config.agent_model,
            api_base=config.api_base,
            api_key_env=config.api_key_env,
            reasoning_effort=config.agent_reasoning_effort,
            provider_timeout_seconds=config.provider_timeout_seconds,
        ),
        "user": openai_compatible_user_factory(
            model_name=config.user_model,
            api_base=config.api_base,
            api_key_env=config.api_key_env,
            reasoning_effort=config.user_reasoning_effort,
            provider_timeout_seconds=config.provider_timeout_seconds,
        ),
    }
