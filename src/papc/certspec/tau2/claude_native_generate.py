"""Lossless Haiku Messages transport for the active tau2 method.

OpenAI-prefixed Claude loses its thinking setting in LiteLLM, and Chat
Completions cannot carry the signed native blocks through tau2's serializer.
Keep those blocks opaque in raw_data and replay them only for their origin
model. The gateway credential stays local; it is not a Claude OAuth token.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import httpx

from papc.certspec.costing import cache_neutral_cost_usd

HAIKU_MODEL = "claude-haiku-4-5-20251001"
# Match the pinned gateway's existing effort -> manual thinking budget mapping.
THINKING_BUDGETS = {"low": 1024, "medium": 8192, "high": 24576}


def _messages(messages: Sequence[Any], model: str) -> tuple[list, list]:
    system, history = [], []
    for message in messages:
        role = message.role
        if role == "system":
            if message.content:
                system.append({"type": "text", "text": message.content})
            continue
        if role == "tool":
            blocks = [
                {
                    "type": "tool_result",
                    "tool_use_id": message.id,
                    "content": message.content,
                    "is_error": bool(message.error),
                }
            ]
            role = "user"
        elif role in {"user", "assistant"}:
            blocks = (
                [{"type": "text", "text": message.content}] if message.content else []
            )
            if role == "assistant":
                blocks.extend(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": deepcopy(call.arguments),
                    }
                    for call in message.tool_calls or []
                )
                raw = message.raw_data or {}
                if raw.get("papc_native_claude_model") == model:
                    native = raw["content"]
                    native_text = "".join(
                        b["text"] for b in native if b["type"] == "text"
                    )
                    native_calls = [
                        {key: b[key] for key in ("type", "id", "name", "input")}
                        for b in native
                        if b["type"] == "tool_use"
                    ]
                    # Never resurrect an intercepted/replaced action from raw_data.
                    if native_text != (message.content or "") or native_calls != [
                        b for b in blocks if b["type"] == "tool_use"
                    ]:
                        raise ValueError(
                            "Claude native history differs from accepted message"
                        )
                    blocks = deepcopy(native)
        else:
            raise ValueError(f"Unsupported Claude message role: {role}")
        if not blocks:
            raise ValueError("Empty Claude message")
        # Anthropic represents adjacent tool results as one user content array.
        if history and history[-1]["role"] == role:
            history[-1]["content"].extend(blocks)
        else:
            history.append({"role": role, "content": blocks})
    return system, history


def generate_claude_native(
    *,
    model: str,
    messages: Sequence[Any],
    tools: Sequence[Any] | None = None,
    call_name: str | None = None,
    api_base: str,
    reasoning_effort: str,
    api_key: str | None = None,
    temperature: float = 0.0,
    drop_params: bool = True,
    tool_choice: str | None = None,
    max_tokens: int = 32000,
    num_retries: int = 3,
    timeout: float = 300.0,
    seed: int | None = None,
) -> Any:
    """Call native /v1/messages without rewriting policy, tools or thinking.

    tau2 sets a shared simulation seed on every LLM agent. Anthropic Messages
    has no sampling seed parameter, so keep it as provenance, off the wire.
    """
    from tau2.data_model.message import AssistantMessage, ToolCall
    from tau2.utils.llm_utils import (
        _write_llm_log,
        llm_log_mode,
        validate_message_history,
    )

    native_model = model.removeprefix("openai/")
    if native_model != HAIKU_MODEL:
        raise ValueError("Native tau2 Claude transport is pinned to Haiku 4.5")
    if reasoning_effort not in THINKING_BUDGETS:
        raise ValueError("Haiku manual thinking requires low, medium or high")
    budget = THINKING_BUDGETS[reasoning_effort]
    if not budget < max_tokens <= 64000:
        raise ValueError("Haiku max_tokens must exceed thinking budget and be <= 64000")
    if tool_choice not in {None, "auto", "none"}:
        raise ValueError("Manual thinking cannot force a tool choice")
    # Manual thinking requires the provider's sampling default. Do not silently
    # turn it off to accommodate tau2's generic temperature=0/drop_params flags.
    if temperature not in {0.0, 1.0}:
        raise ValueError("Manual thinking does not support custom temperature")
    if num_retries < 0:
        raise ValueError("num_retries must be non-negative")
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_base or not key:
        raise ValueError(
            "Claude native transport requires explicit gateway and credential"
        )
    validate_message_history(list(messages))
    system, history = _messages(messages, native_model)
    payload: dict[str, Any] = {
        "model": native_model,
        "system": system,
        "messages": history,
        "max_tokens": max_tokens,
        "thinking": {"type": "enabled", "budget_tokens": budget},
        "stream": False,
    }
    if tools:
        payload["tools"] = [
            {
                "name": schema["name"],
                "description": schema.get("description", ""),
                "input_schema": deepcopy(schema["parameters"]),
            }
            for tool in tools
            for schema in [tool.openai_schema["function"]]
        ]
        payload["tool_choice"] = {"type": tool_choice or "auto"}
    base = api_base.rstrip("/")
    url = base + ("/messages" if base.endswith("/v1") else "/v1/messages")
    started = time.perf_counter()
    # Retry only infrastructure failures, never an incompatible request or a
    # malformed successful response. The same payload is used on each attempt.
    with httpx.Client(timeout=timeout, trust_env=False) as client:
        for attempt in range(num_retries + 1):
            response = client.post(
                url,
                json=payload,
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                },
            )
            if response.status_code not in {429, 500, 502, 503, 504, 529}:
                break
            if attempt < num_retries:
                time.sleep(min(2**attempt, 8))
        response.raise_for_status()
        raw = response.json()
    elapsed = time.perf_counter() - started
    # Native raw response is private, including signed/redacted thinking blocks.
    # Never log headers or API keys. Keep even a malformed paid response auditable.
    # tau2 defaults to keeping only the last call of each kind. A transport
    # audit also needs intercepted candidates and responses rejected below.
    log_mode_token = llm_log_mode.set("all")
    try:
        _write_llm_log(payload, raw, call_name=call_name)
    finally:
        llm_log_mode.reset(log_mode_token)
    if raw.get("model") != native_model or raw.get("role") != "assistant":
        raise ValueError("Claude native response model/role mismatch")
    if raw.get("stop_reason") not in {"end_turn", "tool_use", "stop_sequence"}:
        raise ValueError(f"Incomplete Claude response: {raw.get('stop_reason')}")
    content, calls = [], []
    for block in raw["content"]:
        kind = block["type"]
        if kind == "text":
            content.append(block["text"])
        elif kind == "tool_use":
            calls.append(
                ToolCall(id=block["id"], name=block["name"], arguments=block["input"])
            )
        elif kind == "thinking":
            if not block.get("signature"):
                raise ValueError("Claude thinking block has no replayable signature")
        elif kind == "redacted_thinking":
            if not block.get("data"):
                raise ValueError("Claude redacted thinking block has no data")
        else:
            raise ValueError(f"Unsupported Claude content block: {kind}")
    usage = raw.get("usage")
    if not isinstance(usage, Mapping) or any(
        type(usage.get(k)) is not int or usage[k] < 0
        for k in ("input_tokens", "output_tokens")
    ):
        raise ValueError("Claude native response is missing token usage")
    cached = usage.get("cache_read_input_tokens", 0)
    created = usage.get("cache_creation_input_tokens", 0)
    if any(type(value) is not int or value < 0 for value in (cached, created)):
        raise ValueError("Claude native response has invalid cache usage")
    total_input = usage["input_tokens"] + cached + created
    normalized_usage = {
        "prompt_tokens": total_input,
        "completion_tokens": usage["output_tokens"],
        "prompt_tokens_details": {
            "cached_tokens": cached,
            "cache_creation_tokens": created,
        },
    }
    # Native output_tokens already includes thinking. Never estimate reasoning
    # tokens from displayed text or add them to the output count a second time.
    thinking_tokens = (usage.get("output_tokens_details") or {}).get("thinking_tokens")
    if thinking_tokens is not None:
        if (
            type(thinking_tokens) is not int
            or not 0 <= thinking_tokens <= usage["output_tokens"]
        ):
            raise ValueError("Claude native response has invalid thinking usage")
        normalized_usage["completion_tokens_details"] = {
            "reasoning_tokens": thinking_tokens
        }
    raw["papc_native_claude_model"] = native_model
    raw["papc_thinking_budget_tokens"] = budget
    raw["papc_orchestration_seed"] = seed
    return AssistantMessage(
        role="assistant",
        content="".join(content) or None,
        tool_calls=calls or None,
        usage=normalized_usage,
        raw_data=raw,
        generation_time_seconds=elapsed,
        cost=cache_neutral_cost_usd(
            model=native_model,
            input_tokens=total_input,
            output_tokens=usage["output_tokens"],
        ),
    )
