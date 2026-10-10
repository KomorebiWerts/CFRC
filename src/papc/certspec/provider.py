"""Benchmark-independent OpenAI provider plumbing for RCBE runtimes.

This module factors out the duplicated provider boilerplate that previously
lived inline in several G8.x experiment scripts (g8184 / g8205 / g8207): Chat
Completions / Responses client construction, message-content text extraction,
response diagnostics, retry/error classification, and usage-token accounting.

It is pure plumbing. It MUST NOT mention dataset names, task ids, benchmark
surfaces, authority posets, slot semantics, or any other experiment logic. It
reads the API key from an env-var *name* supplied by the caller and never
inlines a key. The experiment scripts remain responsible for choosing model /
limit / output paths and for all benchmark-specific reduction.

The two response shapes served here mirror the two OpenAI endpoints used by the
RCBE runtimes:

* Chat Completions (``response.choices[].message.content``) - used by g8207.
* Responses (``response.output[].content[].text`` / ``output_text``) - used by
  g8184 / g8205.

Both callers return the same provider-response envelope dict so the downstream
``call_and_parse`` plumbing in the scripts is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

__all__ = [
    "DEFAULT_RETRYABLE_ERROR_TYPES",
    "DEFAULT_RETRYABLE_ERROR_MARKERS",
    "DEFAULT_FATAL_ERROR_TYPES",
    "DEFAULT_FATAL_ERROR_MARKERS",
    "DEFAULT_FATAL_ERROR_CODE_MARKERS",
    "ProviderCaller",
    "ProviderErrorCategory",
    "as_mapping",
    "message_content_text",
    "responses_output_text",
    "chat_response_diagnostics",
    "chat_request_payload_audit",
    "chat_completion_json_object_receipt",
    "classify_provider_response",
    "is_retryable_provider_response",
    "is_fatal_provider_response",
    "UsageAccumulator",
    "build_chat_completions_caller",
    "chat_completion_raw_http_with_retries",
    "chat_completion_response_has_message",
]

ProviderCaller = Callable[[Mapping[str, Any]], Mapping[str, Any]]
ProviderErrorCategory = Literal["ok", "retryable", "fatal", "schema_invalid"]

# Transient provider conditions that justify another attempt. Mirrors the union
# of the retryable sets used by the migrated scripts (timeouts, connection
# resets, 429/502/503/524, origin response timeouts).
DEFAULT_RETRYABLE_ERROR_TYPES: frozenset[str] = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
        "TimeoutError",
    }
)
DEFAULT_RETRYABLE_ERROR_MARKERS: tuple[str, ...] = (
    "concurrency limit exceeded",
    "error code: 429",
    "error code: 502",
    "error code: 503",
    "error code: 524",
    "origin_response_timeout",
    "retryable",
    "timeout",
)

# Permanent provider conditions (auth / billing / quota) that must abort rather
# than retry.
DEFAULT_FATAL_ERROR_TYPES: frozenset[str] = frozenset(
    {"AuthenticationError", "PermissionDeniedError"}
)
DEFAULT_FATAL_ERROR_MARKERS: tuple[str, ...] = (
    "authentication_error",
    "auth_unavailable",
    "billing",
    "forbidden",
    "insufficient_quota",
    "invalidated",
    "quota",
    "unauthorized",
)
DEFAULT_FATAL_ERROR_CODE_MARKERS: tuple[str, ...] = (
    "error code: 401",
    "error code: 403",
)

RAW_HTTP_CHAT_ALLOWED_PAYLOAD_KEYS: frozenset[str] = frozenset(
    {
        "frequency_penalty",
        "max_completion_tokens",
        "max_tokens",
        "messages",
        "model",
        "presence_penalty",
        "reasoning_effort",
        "response_format",
        "seed",
        "stop",
        "stream",
        "temperature",
        "tool_choice",
        "tools",
        "top_p",
    }
)
RAW_HTTP_CHAT_MAX_ATTEMPT_TIMEOUT_SECONDS = 20.0


def as_mapping(value: Any) -> dict[str, Any]:
    """Return ``value`` as a plain dict if it is a mapping, else ``{}``.

    Equivalent to the ``base.mapping`` helper the scripts use to defensively
    coerce optional SDK / JSON fields.
    """

    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def first_tool_call_arguments(response: Any, *, name: str) -> str | None:
    """Arguments of the first call to ``name`` in a Chat Completions response.

    The name is required rather than optional. A model reached through an agent
    session arrives with its own tools already in context, so the first call in
    a reply is often to one of those: reading whichever call came first would
    hand a shell command to a parser expecting a graph. Returns None when the
    reply carried no call to this tool, so a caller keeps the text path it had.
    """

    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, Mapping):
        choices = response.get("choices")
    if not isinstance(choices, Sequence) or isinstance(
        choices, (str, bytes, bytearray)
    ):
        return None
    for choice in choices:
        message = _chat_object_mapping(choice).get("message")
        message_map = (
            dict(message)
            if isinstance(message, Mapping)
            else {"tool_calls": getattr(message, "tool_calls", None)}
        )
        calls = message_map.get("tool_calls")
        if not isinstance(calls, Sequence) or isinstance(
            calls, (str, bytes, bytearray)
        ):
            continue
        for call in calls:
            call_map = dict(call) if isinstance(call, Mapping) else {
                "function": getattr(call, "function", None)
            }
            function = call_map.get("function")
            function_map = (
                dict(function)
                if isinstance(function, Mapping)
                else {
                    "name": getattr(function, "name", None),
                    "arguments": getattr(function, "arguments", None),
                }
            )
            if str(function_map.get("name") or "") != name:
                continue
            arguments = function_map.get("arguments")
            if isinstance(arguments, str) and arguments.strip():
                return arguments
            if isinstance(arguments, Mapping):
                return json.dumps(dict(arguments), ensure_ascii=False)
    return None


def _chat_object_mapping(value: Any) -> dict[str, Any]:
    """Coerce a Chat Completions choice / message / part into a dict.

    Matches the inline ``object_mapping`` previously in the Chat-Completions
    script: when the value is not already a mapping, pull the attributes the
    Chat shape exposes (``message`` / ``content`` / ``text`` / ``finish_reason``).
    """

    if isinstance(value, Mapping):
        return dict(value)
    return {
        "message": getattr(value, "message", None),
        "content": getattr(value, "content", None),
        "text": getattr(value, "text", None),
        "tool_calls": getattr(value, "tool_calls", None),
        "finish_reason": getattr(value, "finish_reason", None),
    }


def _responses_object_mapping(value: Any) -> dict[str, Any]:
    """Coerce a Responses output item / part into a dict.

    Matches the inline ``object_mapping`` previously in the Responses scripts:
    pull the Responses-shape attributes (``type`` / ``text`` / ``content``).
    """

    if isinstance(value, Mapping):
        return dict(value)
    return {
        "type": getattr(value, "type", None),
        "text": getattr(value, "text", None),
        "content": getattr(value, "content", None),
    }


def _usage_map(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if hasattr(usage, "model_dump"):
        dumped = usage.model_dump()
        return dict(dumped) if isinstance(dumped, Mapping) else None
    if isinstance(usage, Mapping):
        return dict(usage)
    return None


def message_content_text(response: Any) -> str:
    """Extract concatenated assistant text from a Chat Completions response.

    Handles both the SDK object shape and a plain-dict shape. ``content`` may be
    a string or a list of content parts each carrying ``text``.
    """

    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, Mapping):
        choices = response.get("choices")
    if not isinstance(choices, Sequence) or isinstance(
        choices, (str, bytes, bytearray)
    ):
        return ""
    chunks: list[str] = []
    for choice in choices:
        choice_map = _chat_object_mapping(choice)
        message = choice_map.get("message")
        message_map = _chat_object_mapping(message)
        content = message_map.get("content")
        if isinstance(content, str):
            chunks.append(content)
            continue
        if isinstance(content, Sequence) and not isinstance(
            content, (str, bytes, bytearray)
        ):
            for part in content:
                part_map = _chat_object_mapping(part)
                text = part_map.get("text")
                if text:
                    chunks.append(str(text))
    return "".join(chunks)


def responses_output_text(response: Any) -> str:
    """Extract concatenated text from a Responses-API response.

    Prefers the SDK ``output_text`` convenience attribute, then walks the
    ``output`` items for ``output_text`` / ``text`` parts.
    """

    try:
        text = getattr(response, "output_text", None)
    except TypeError:
        text = None
    if text:
        return str(text)
    return _responses_output_text_from_output(getattr(response, "output", None))


def _responses_output_text_from_output(output: Any) -> str:
    if not isinstance(output, Sequence) or isinstance(output, (str, bytes, bytearray)):
        return ""
    chunks: list[str] = []
    for item in output:
        item_map = _responses_object_mapping(item)
        if item_map.get("type") in {"output_text", "text"} and item_map.get("text"):
            chunks.append(str(item_map["text"]))
            continue
        if item_map.get("type") != "message":
            continue
        content = item_map.get("content")
        if not isinstance(content, Sequence) or isinstance(
            content, (str, bytes, bytearray)
        ):
            continue
        for part in content:
            part_map = _responses_object_mapping(part)
            if part_map.get("type") in {"output_text", "text"} and part_map.get("text"):
                chunks.append(str(part_map["text"]))
    return "".join(chunks)


def chat_response_diagnostics(response: Any, *, raw_text: str) -> dict[str, Any]:
    """Per-call diagnostics for a Chat Completions response.

    Reports choice_count, finish_reasons, whether any text was present, and the
    normalized usage map. Shape matches the inline diagnostics the script wrote
    into its envelope.
    """

    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, Mapping):
        choices = response.get("choices")
    choice_count = (
        len(choices)
        if isinstance(choices, Sequence)
        and not isinstance(choices, (str, bytes, bytearray))
        else 0
    )
    finish_reasons: list[str] = []
    for choice in choices or []:
        reason = _chat_object_mapping(choice).get("finish_reason")
        if reason:
            finish_reasons.append(str(reason))
    return {
        "endpoint": "chat.completions",
        "choice_count": choice_count,
        "text_present": bool(raw_text.strip()),
        "finish_reasons": finish_reasons,
        "usage": _usage_map(response),
    }


def chat_completion_response_has_message(response: Any) -> bool:
    """True when a Chat Completions response has text or tool calls.

    OpenAI-compatible providers can return a 200 response containing an empty
    server-sent-event chunk (``choices: []``). That is transport-successful but
    unusable for benchmark rows, so callers should treat it as provider invalid
    and retry.
    """

    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, Mapping):
        choices = response.get("choices")
    if not isinstance(choices, Sequence) or isinstance(
        choices, (str, bytes, bytearray)
    ):
        return False
    for choice in choices:
        choice_map = _chat_object_mapping(choice)
        message = choice_map.get("message")
        message_map = _chat_object_mapping(message)
        content = message_map.get("content")
        if isinstance(content, str) and content.strip():
            return True
        tool_calls = message_map.get("tool_calls")
        if isinstance(tool_calls, Sequence) and not isinstance(
            tool_calls, (str, bytes, bytearray)
        ):
            if len(tool_calls) > 0:
                return True
    return False


def chat_request_payload_audit(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Non-secret audit of an outgoing Chat Completions payload."""

    messages = payload.get("messages")
    max_tokens = payload.get("max_tokens")
    return {
        "endpoint": "chat.completions",
        "message_count": len(messages)
        if isinstance(messages, Sequence)
        and not isinstance(messages, (str, bytes, bytearray))
        else 0,
        "max_tokens": int(max_tokens)
        if isinstance(max_tokens, int) and not isinstance(max_tokens, bool)
        else 0,
        "reasoning_effort": payload.get("reasoning_effort"),
        "stream": payload.get("stream") is True,
        "structured_output": "response_format" in payload,
    }


def chat_completion_json_object_receipt(
    *,
    api_base: str,
    api_key_env: str,
    model: str,
    reasoning_effort: str,
    messages: Sequence[Mapping[str, Any]],
    timeout_seconds: float = 60.0,
    max_tokens: int = 1200,
    provider_caller: Callable[[Mapping[str, Any]], tuple[Any, list[dict[str, Any]]]]
    | None = None,
) -> dict[str, Any]:
    """Call Chat Completions once and parse a JSON-object assistant message.

    This is provider plumbing, not a benchmark case abstraction. Callers decide
    whether the parsed JSON is a strong proposal, evaluator response, or another
    private artifact.
    """

    if not api_base:
        raise RuntimeError("json_chat_api_base_missing")
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise RuntimeError("json_chat_api_key_missing")
    payload = {
        "model": model,
        "messages": [dict(message) for message in messages],
        "reasoning_effort": reasoning_effort,
        "max_tokens": int(max_tokens),
        "response_format": {"type": "json_object"},
    }
    caller = provider_caller or (
        lambda request: chat_completion_raw_http_with_retries(
            api_base=api_base,
            api_key=api_key,
            payload=request,
            timeout_seconds=timeout_seconds,
            max_attempts=1,
        )
    )
    response, attempts = caller(payload)
    raw_text = message_content_text(response)
    parsed = _parse_json_object(raw_text)
    return {
        "schema": "papc.provider_chat_json_object_private_receipt.v1",
        "json_object": parsed,
        "provider_call_count": 1,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "request_payload_audit": chat_request_payload_audit(payload),
        "response_diagnostics": chat_response_diagnostics(response, raw_text=raw_text),
        "attempts": attempts,
        "raw_text_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
    }


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("json_chat_response_not_json") from exc
    if not isinstance(parsed, Mapping):
        raise RuntimeError("json_chat_response_not_object")
    return dict(parsed)


def chat_completion_raw_http_with_retries(
    *,
    api_base: str,
    api_key: str,
    payload: Mapping[str, Any],
    timeout_seconds: float,
    max_attempts: int = 3,
    retry_delay_seconds: float = 0.5,
    allow_reasoning_only_message: bool = False,
    http_post: Callable[[str, bytes, Mapping[str, str], float], tuple[int, str]]
    | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Call OpenAI-compatible Chat Completions via raw HTTP with shape retries.

    This is a narrow compatibility path for providers whose OpenAI-compatible
    endpoint accepts curl-like HTTP requests but rejects SDK/LiteLLM default
    request headers or occasionally returns ``200`` with an empty SSE chunk.
    It returns a LiteLLM ``ModelResponse`` so existing tau2 post-processing can
    keep using the same response object shape.
    """

    from litellm.main import ModelResponse  # type: ignore

    if not api_base:
        raise RuntimeError("raw_http_chat_api_base_missing")
    if not api_key:
        raise RuntimeError("raw_http_chat_api_key_missing")
    attempts: list[dict[str, Any]] = []
    post = http_post or _http_post_chat_completion
    raw_payload = _raw_http_chat_payload(payload)
    url = api_base.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "curl/8.5.0",
    }
    last_error = "raw_http_chat_no_attempts"
    for attempt_index in range(max(int(max_attempts), 1)):
        started = time.monotonic()
        attempt: dict[str, Any] = {
            "attempt": attempt_index + 1,
            "transport": "raw_http_chat_completions",
        }
        try:
            status, body = post(
                url,
                json.dumps(raw_payload).encode("utf-8"),
                headers,
                timeout_seconds,
            )
            attempt["status"] = status
            attempt["latency_seconds"] = round(time.monotonic() - started, 3)
            attempt["body_shape"] = _chat_body_shape(body)
            if status != 200:
                last_error = f"raw_http_chat_status:{status}"
                attempt["ok"] = False
                attempts.append(attempt)
                if not _raw_http_chat_status_retryable(status):
                    break
            else:
                data = _parse_chat_completion_body(body)
                response = ModelResponse(**data)
                if chat_completion_response_has_message(response) or (
                    allow_reasoning_only_message
                    and _chat_completion_has_reasoning_only_message(data)
                ):
                    attempt["ok"] = True
                    attempts.append(attempt)
                    return response, attempts
                last_error = "raw_http_chat_empty_message"
                attempt["response_shape"] = _chat_response_shape(data)
                attempt["ok"] = False
                attempts.append(attempt)
        except Exception as exc:
            last_error = f"{type(exc).__name__}:{str(exc)[:160]}"
            attempt.update(
                {
                    "ok": False,
                    "latency_seconds": round(time.monotonic() - started, 3),
                    "error_type": type(exc).__name__,
                    "error_message_prefix": str(exc)[:160],
                }
            )
            attempts.append(attempt)
        if attempt_index + 1 < max(int(max_attempts), 1):
            time.sleep(max(float(retry_delay_seconds), 0.0))
    error = RuntimeError(last_error)
    setattr(error, "raw_http_attempts", attempts)
    raise error


def _chat_completion_has_reasoning_only_message(
    data: Mapping[str, Any],
) -> bool:
    """Accept a structurally valid DeepSeek turn whose final content is empty.

    The caller still receives an empty public message. Agent runtimes can then
    apply their normal format-repair policy without executing private
    ``reasoning_content``.
    """

    choices = data.get("choices")
    if not isinstance(choices, Sequence) or not choices:
        return False
    first = choices[0]
    if not isinstance(first, Mapping):
        return False
    message = first.get("message")
    if not isinstance(message, Mapping):
        return False
    return bool(str(message.get("reasoning_content") or "").strip())


def _http_post_chat_completion(
    url: str,
    body: bytes,
    headers: Mapping[str, str],
    timeout_seconds: float,
) -> tuple[int, str]:
    request = Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            return response.status, response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")
    except URLError as exc:
        raise RuntimeError(str(exc.reason)) from exc


def _raw_http_chat_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    clean = {
        key: value
        for key, value in payload.items()
        if key in RAW_HTTP_CHAT_ALLOWED_PAYLOAD_KEYS and value is not None
    }
    clean["model"] = _raw_http_model_name(str(clean.get("model") or ""))
    if clean["model"].startswith("gpt-5"):
        # The local GPT-5 direct shim rejects Chat Completions temperature.
        clean.pop("temperature", None)
    clean["stream"] = False
    return clean


def _raw_http_model_name(model: str) -> str:
    return model[len("openai/") :] if model.startswith("openai/") else model


def _raw_http_chat_status_retryable(status: int) -> bool:
    return status in {408, 409, 429, 500, 502, 503, 504, 524}


def _chat_body_shape(body: str) -> str:
    text = body.lstrip()
    if text.startswith("data:"):
        return "sse"
    if text.startswith("{"):
        return "json"
    return "text"


def _parse_chat_completion_body(body: str) -> dict[str, Any]:
    text = body.strip()
    if not text.startswith("data:"):
        parsed = json.loads(text)
        if not isinstance(parsed, Mapping):
            raise RuntimeError("raw_http_chat_body_not_object")
        return dict(parsed)
    chunks = _parse_sse_json_chunks(text)
    for chunk in chunks:
        choices = chunk.get("choices")
        if choices:
            return chunk
    if chunks:
        return chunks[-1]
    raise RuntimeError("raw_http_chat_empty_sse")


def _parse_sse_json_chunks(text: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("data:"):
            continue
        data = stripped[len("data:") :].strip()
        if not data or data == "[DONE]":
            continue
        parsed = json.loads(data)
        if isinstance(parsed, Mapping):
            chunks.append(dict(parsed))
    return chunks


def _chat_response_shape(data: Mapping[str, Any]) -> dict[str, Any]:
    choices = data.get("choices")
    first_choice = choices[0] if isinstance(choices, list) and choices else {}
    message = first_choice.get("message") if isinstance(first_choice, Mapping) else {}
    content = message.get("content") if isinstance(message, Mapping) else None
    tool_calls = message.get("tool_calls") if isinstance(message, Mapping) else None
    return {
        "choice_count": len(choices) if isinstance(choices, list) else 0,
        "finish_reason": first_choice.get("finish_reason")
        if isinstance(first_choice, Mapping)
        else None,
        "message_keys": sorted(message.keys()) if isinstance(message, Mapping) else [],
        "content_length": len(content) if isinstance(content, str) else None,
        "tool_call_count": len(tool_calls) if isinstance(tool_calls, list) else None,
    }


def classify_provider_response(
    response: Mapping[str, Any],
    *,
    schema_invalid: bool = False,
    retryable_error_types: frozenset[str] = DEFAULT_RETRYABLE_ERROR_TYPES,
    retryable_error_markers: Sequence[str] = DEFAULT_RETRYABLE_ERROR_MARKERS,
    fatal_error_types: frozenset[str] = DEFAULT_FATAL_ERROR_TYPES,
    fatal_error_markers: Sequence[str] = DEFAULT_FATAL_ERROR_MARKERS,
    fatal_error_code_markers: Sequence[str] = DEFAULT_FATAL_ERROR_CODE_MARKERS,
) -> ProviderErrorCategory:
    """Classify a provider-response envelope into a typed category.

    Returns one of ``"fatal"``, ``"retryable"``, ``"schema_invalid"``, ``"ok"``.

    A ``provider_error`` entry (raised exception) is classified as ``fatal`` if
    its type / message matches the fatal vocabulary, else ``retryable`` if it
    matches the transient vocabulary; an otherwise-unrecognized provider error
    is conservatively treated as ``retryable`` only when its markers match,
    falling back to ``ok`` classification of the error envelope is left to the
    caller's retry policy. When there is no provider error, a caller-supplied
    ``schema_invalid`` flag maps to ``"schema_invalid"`` and everything else is
    ``"ok"``.

    Fatal is checked before retryable so an auth/billing 401/403 never loops.
    """

    error = as_mapping(response.get("provider_error"))
    if error:
        error_type = str(error.get("type") or "")
        message = str(error.get("message") or "").lower()
        if error_type in fatal_error_types:
            return "fatal"
        if any(marker in message for marker in fatal_error_code_markers):
            return "fatal"
        if any(marker in message for marker in fatal_error_markers):
            return "fatal"
        if error_type in retryable_error_types:
            return "retryable"
        if any(marker in message for marker in retryable_error_markers):
            return "retryable"
        # An unrecognized hard error is not transient; treat as fatal so the
        # runtime stops rather than burning the retry budget on a permanent
        # failure.
        return "fatal"
    if schema_invalid:
        return "schema_invalid"
    return "ok"


def is_retryable_provider_response(
    response: Mapping[str, Any],
    *,
    retryable_error_types: frozenset[str] = DEFAULT_RETRYABLE_ERROR_TYPES,
    retryable_error_markers: Sequence[str] = DEFAULT_RETRYABLE_ERROR_MARKERS,
) -> bool:
    """True if the response carries a transient provider error worth retrying.

    Preserves the exact predicate the Chat-Completions script used: only a
    ``provider_error`` whose type or (lowercased) message matches the transient
    vocabulary is retryable; a fatal-looking message that also contains a
    transient marker is *not* special-cased here (callers that need fatal
    precedence should use :func:`classify_provider_response`).
    """

    error = as_mapping(response.get("provider_error"))
    if not error:
        return False
    error_type = str(error.get("type") or "")
    if error_type in retryable_error_types:
        return True
    message = str(error.get("message") or "").lower()
    return any(marker in message for marker in retryable_error_markers)


def is_fatal_provider_response(
    response: Mapping[str, Any],
    *,
    fatal_error_types: frozenset[str] = DEFAULT_FATAL_ERROR_TYPES,
    fatal_error_markers: Sequence[str] = DEFAULT_FATAL_ERROR_MARKERS,
    fatal_error_code_markers: Sequence[str] = DEFAULT_FATAL_ERROR_CODE_MARKERS,
) -> bool:
    """True if the response carries a permanent (auth/billing/quota) error."""

    error = as_mapping(response.get("provider_error"))
    if not error:
        return False
    error_type = str(error.get("type") or "")
    if error_type in fatal_error_types:
        return True
    message = str(error.get("message") or "").lower()
    if any(marker in message for marker in fatal_error_code_markers):
        return True
    return any(marker in message for marker in fatal_error_markers)


def reasoning_tokens_from_usage(usage: Mapping[str, Any]) -> int | None:
    """Return the reasoning count a usage record carries, or ``None``.

    Endpoints put this in three places: nested under the Responses shape, nested
    under the Chat shape, or flat once the mapping has already been normalized.
    Reading only one of them silently records zero reasoning for a provider that
    reports it elsewhere, which then prices as if the model had not thought.
    ``None`` distinguishes a record that carries no reasoning field at all from
    one that genuinely reports zero.
    """

    usage = as_mapping(usage)
    output_details = as_mapping(usage.get("output_tokens_details"))
    completion_details = as_mapping(usage.get("completion_tokens_details"))
    for candidate in (
        output_details.get("reasoning_tokens"),
        completion_details.get("reasoning_tokens"),
        usage.get("reasoning_tokens"),
        usage.get("reasoning_output_tokens"),
    ):
        if candidate is not None:
            return max(0, int(candidate))
    return None


class UsageAccumulator:
    """Accumulate normalized token usage across Responses + Chat usage shapes.

    Normalizes the divergent field names the two endpoints use:

    * input  : ``input_tokens``  (Responses) / ``prompt_tokens``     (Chat)
    * output : ``output_tokens`` (Responses) / ``completion_tokens`` (Chat)
    * reason : nested endpoint details, or flat ``reasoning_tokens`` when
               re-accumulating an already normalized usage mapping
    """

    __slots__ = (
        "input_tokens",
        "cached_input_tokens",
        "cached_input_usage_call_count",
        "cached_input_usage_unavailable_count",
        "output_tokens",
        "reasoning_tokens",
    )

    def __init__(self) -> None:
        self.input_tokens = 0
        self.cached_input_tokens = 0
        self.cached_input_usage_call_count = 0
        self.cached_input_usage_unavailable_count = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0

    def add_usage(self, usage: Mapping[str, Any]) -> "UsageAccumulator":
        """Add a single normalized usage mapping into the running totals."""

        usage = as_mapping(usage)
        input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        self.input_tokens += input_tokens
        self.output_tokens += int(
            usage.get("output_tokens") or usage.get("completion_tokens") or 0
        )
        output_details = as_mapping(usage.get("output_tokens_details"))
        completion_details = as_mapping(usage.get("completion_tokens_details"))
        self.reasoning_tokens += int(
            output_details.get("reasoning_tokens")
            or completion_details.get("reasoning_tokens")
            or usage.get("reasoning_tokens")
            or 0
        )
        if "cached_input_usage_call_count" in usage:
            self.cached_input_usage_call_count += int(
                usage.get("cached_input_usage_call_count") or 0
            )
            self.cached_input_usage_unavailable_count += int(
                usage.get("cached_input_usage_unavailable_count") or 0
            )
            if usage.get("cached_input_tokens") is not None:
                self.cached_input_tokens += min(
                    input_tokens,
                    int(usage.get("cached_input_tokens") or 0),
                )
        else:
            cached_input_tokens = _cached_input_tokens_from_usage(usage)
            if cached_input_tokens is None:
                self.cached_input_usage_unavailable_count += 1
            else:
                self.cached_input_usage_call_count += 1
                self.cached_input_tokens += min(input_tokens, cached_input_tokens)
        return self

    def add_response(self, response: Mapping[str, Any]) -> "UsageAccumulator":
        """Add a provider-response envelope, summing across retry attempts.

        If the envelope records per-attempt ``attempts``, every attempt's usage
        is summed; otherwise the top-level ``usage`` is added once. Mirrors the
        inline per-branch usage roll-up in the migrated script.
        """

        response = as_mapping(response)
        attempts = _sequence(response.get("attempts"))
        if attempts:
            for attempt in attempts:
                self.add_usage(as_mapping(as_mapping(attempt).get("usage")))
        else:
            self.add_usage(as_mapping(response.get("usage")))
        return self

    def as_dict(self) -> dict[str, int | bool | None]:
        """Return the running totals keyed by normalized token name."""

        cache_available = (
            self.cached_input_usage_call_count > 0
            and self.cached_input_usage_unavailable_count == 0
        )
        return {
            "input_tokens": self.input_tokens,
            "cached_input_tokens": (
                self.cached_input_tokens if cache_available else None
            ),
            "billable_input_tokens": (
                max(0, self.input_tokens - self.cached_input_tokens)
                if cache_available
                else None
            ),
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_input_usage_available": cache_available,
            "cached_input_usage_call_count": self.cached_input_usage_call_count,
            "cached_input_usage_unavailable_count": (
                self.cached_input_usage_unavailable_count
            ),
        }


def _cached_input_tokens_from_usage(usage: Mapping[str, Any]) -> int | None:
    direct = usage.get("cached_input_tokens")
    if direct is not None:
        return max(0, int(direct))
    input_details = as_mapping(usage.get("input_tokens_details"))
    prompt_details = as_mapping(usage.get("prompt_tokens_details"))
    if input_details.get("cached_tokens") is not None:
        return max(0, int(input_details["cached_tokens"]))
    if prompt_details.get("cached_tokens") is not None:
        return max(0, int(prompt_details["cached_tokens"]))
    return None


def usage_totals_from_response(
    response: Mapping[str, Any],
) -> dict[str, int | bool | None]:
    """Convenience: normalized per-response usage totals as a dict."""

    return UsageAccumulator().add_response(response).as_dict()


def build_chat_completions_caller(
    *,
    api_key_env: str,
    api_base: str,
    timeout_seconds: float,
    max_output_tokens: int,
    extra_payload: Mapping[str, Any] | None = None,
) -> ProviderCaller:  # pragma: no cover - provider/network path.
    """Construct a Chat Completions ``ProviderCaller``.

    Reads the API key from ``os.environ[api_key_env]`` (the env-var *name*, never
    an inlined key). ``base_url`` is set only when ``api_base`` is non-empty. The
    SDK's own retries are disabled (``max_retries=0``); retry policy is owned by
    the calling runtime.

    The returned callable accepts a request mapping with ``model`` / ``system``
    / ``user`` / ``reasoning_effort`` and returns the standard provider-response
    envelope: ``raw_text``, ``latency_seconds``, ``usage`` (normalized dict),
    ``provider_response_diagnostics``, ``request_payload_audit``.
    """

    from openai import OpenAI  # type: ignore

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"missing API key env: {api_key_env}")
    client_kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": timeout_seconds,
        "max_retries": 0,
    }
    if api_base:
        client_kwargs["base_url"] = api_base
    client = OpenAI(**client_kwargs)

    def call(request: Mapping[str, Any]) -> Mapping[str, Any]:
        payload: dict[str, Any] = {
            "model": str(request.get("model") or ""),
            "messages": [
                {"role": "system", "content": str(request.get("system") or "")},
                {"role": "user", "content": str(request.get("user") or "")},
            ],
            "max_tokens": max_output_tokens,
            "stream": False,
        }
        if extra_payload:
            payload.update(dict(extra_payload))
        effort = str(request.get("reasoning_effort") or "")
        if effort:
            payload["reasoning_effort"] = effort
        t0 = time.monotonic()
        raw_http_attempts: list[dict[str, Any]] = []
        transport = "openai_sdk"
        try:
            response = client.chat.completions.create(**payload)
            if not chat_completion_response_has_message(response):
                raise RuntimeError("sdk_chat_completion_empty_message")
        except Exception as exc:
            if not api_base or not _sdk_exception_allows_raw_http_fallback(exc):
                raise
            response, raw_http_attempts = chat_completion_raw_http_with_retries(
                api_base=api_base,
                api_key=api_key,
                payload=payload,
                timeout_seconds=min(
                    float(timeout_seconds), RAW_HTTP_CHAT_MAX_ATTEMPT_TIMEOUT_SECONDS
                ),
                max_attempts=10,
            )
            transport = f"raw_http_fallback_after:{type(exc).__name__}"
        latency = round(time.monotonic() - t0, 3)
        raw_text = message_content_text(response)
        usage = getattr(response, "usage", None)
        return {
            "raw_text": raw_text,
            "latency_seconds": latency,
            "usage": usage.model_dump() if hasattr(usage, "model_dump") else usage,
            "transport": transport,
            "raw_http_attempts": raw_http_attempts,
            "provider_response_diagnostics": chat_response_diagnostics(
                response, raw_text=raw_text
            ),
            "request_payload_audit": chat_request_payload_audit(payload),
        }

    return call


def _sdk_exception_allows_raw_http_fallback(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "request was blocked" in message
        or "sdk_chat_completion_empty_message" in message
        or "choices" in message
        and "empty" in message
    )
