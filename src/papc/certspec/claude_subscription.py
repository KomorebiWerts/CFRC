"""Single-user Claude subscription calls through the official Agent SDK.

This module deliberately does not expose the Anthropic Messages or OpenAI API
protocols.  It keeps the subscription credential inside Claude Code, pins one
model in server configuration, disables tools, and returns a small PAPC-owned
response envelope.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

__all__ = [
    "ClaudeSubscriptionCallError",
    "ClaudeSubscriptionClient",
    "ClaudeSubscriptionConfig",
    "ClaudeSubscriptionResponse",
    "SubscriptionConfigurationError",
    "probe_subscription_models",
    "subscription_preflight",
]

_CONFLICTING_CREDENTIALS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
_ALLOWED_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
_MODELS_WITHOUT_EFFORT = frozenset({"claude-haiku-4-5-20251001"})
_SUBSCRIPTION_TYPES = frozenset({"pro", "max", "team", "enterprise"})
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


class SubscriptionConfigurationError(RuntimeError):
    """The process cannot safely select a Claude subscription credential."""


class ClaudeSubscriptionCallError(RuntimeError):
    """The Agent SDK returned an unsuccessful inference result."""

    def __init__(self, message: str, *, api_error_status: int | None = None) -> None:
        super().__init__(message)
        self.api_error_status = api_error_status


@dataclass(frozen=True)
class ClaudeSubscriptionConfig:
    """Fixed server-side configuration for one Claude subscription model."""

    model: str
    cli_path: Path
    config_dir: Path
    base_url: str | None = None
    effort: str | None = None
    timeout_seconds: float = 300.0
    max_prompt_bytes: int = 1_048_576

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None
    ) -> "ClaudeSubscriptionConfig":
        env = dict(os.environ if environ is None else environ)
        _reject_conflicting_credentials(env)

        model = env.get("PAPC_CLAUDE_MODEL", "").strip()
        if not model.startswith("claude-"):
            raise SubscriptionConfigurationError(
                "PAPC_CLAUDE_MODEL must be a pinned Claude model ID, not an alias"
            )

        cli_value = env.get("PAPC_CLAUDE_CLI_PATH", "").strip()
        if not cli_value:
            raise SubscriptionConfigurationError("PAPC_CLAUDE_CLI_PATH is required")
        cli_path = Path(cli_value).expanduser().resolve()

        config_value = env.get("PAPC_CLAUDE_CONFIG_DIR", "~/.claude").strip()
        config_dir = Path(config_value).expanduser().resolve()

        base_url = (
            env.get("PAPC_CLAUDE_BASE_URL") or env.get("ANTHROPIC_BASE_URL") or ""
        ).strip() or None
        if base_url is not None:
            _validate_base_url(base_url)

        raw_effort = env.get("PAPC_CLAUDE_EFFORT", "").strip().lower()
        effort = None if raw_effort in {"", "none", "default"} else raw_effort
        if effort is not None and effort not in _ALLOWED_EFFORTS:
            raise SubscriptionConfigurationError(
                "PAPC_CLAUDE_EFFORT must be none/default or one of "
                f"{sorted(_ALLOWED_EFFORTS)}"
            )
        if model in _MODELS_WITHOUT_EFFORT and effort is not None:
            raise SubscriptionConfigurationError(
                f"{model} does not support effort; set PAPC_CLAUDE_EFFORT=none"
            )
        if model not in _MODELS_WITHOUT_EFFORT and effort is None:
            effort = "high"

        timeout_seconds = _positive_float(
            env.get("PAPC_CLAUDE_TIMEOUT_SECONDS", "300"),
            name="PAPC_CLAUDE_TIMEOUT_SECONDS",
        )
        max_prompt_bytes = _positive_int(
            env.get("PAPC_CLAUDE_MAX_PROMPT_BYTES", "1048576"),
            name="PAPC_CLAUDE_MAX_PROMPT_BYTES",
        )
        return cls(
            model=model,
            cli_path=cli_path,
            config_dir=config_dir,
            base_url=base_url,
            effort=effort,
            timeout_seconds=timeout_seconds,
            max_prompt_bytes=max_prompt_bytes,
        )

    def for_model(self, model: str) -> "ClaudeSubscriptionConfig":
        """Bind model-specific reasoning controls without changing auth."""

        normalized = str(model).strip()
        if not normalized.startswith("claude-"):
            raise SubscriptionConfigurationError("model must be a Claude model ID")
        effort = None if normalized in _MODELS_WITHOUT_EFFORT else "high"
        return replace(self, model=normalized, effort=effort)

    def sdk_env(self) -> dict[str, str]:
        env = {
            "CLAUDE_CONFIG_DIR": str(self.config_dir),
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }
        if self.base_url is not None:
            env["ANTHROPIC_BASE_URL"] = self.base_url
        return env


@dataclass(frozen=True)
class ClaudeSubscriptionResponse:
    """Redacted inference result suitable for PAPC receipts and local HTTP."""

    text: str
    model: str
    usage: dict[str, Any] | None
    model_usage: dict[str, Any] | None
    provider: str
    provider_calls: int
    duration_ms: int
    estimated_cost_usd: float | None = None
    structured_output: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


RunCommand = Callable[..., subprocess.CompletedProcess[str]]
QueryFunction = Callable[..., AsyncIterator[Any]]


def subscription_preflight(
    config: ClaudeSubscriptionConfig,
    *,
    environ: Mapping[str, str] | None = None,
    run_command: RunCommand = subprocess.run,
) -> dict[str, Any]:
    """Verify that Claude Code selected first-party subscription OAuth."""

    parent_env = dict(os.environ if environ is None else environ)
    _reject_conflicting_credentials(parent_env)
    if not config.cli_path.is_file():
        raise SubscriptionConfigurationError(
            f"Claude Code binary not found: {config.cli_path}"
        )
    if not config.config_dir.is_dir():
        raise SubscriptionConfigurationError(
            f"Claude config directory not found: {config.config_dir}"
        )

    child_env = {**parent_env, **config.sdk_env()}
    completed = run_command(
        [str(config.cli_path), "auth", "status", "--json"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
        env=child_env,
    )
    if completed.returncode != 0:
        raise SubscriptionConfigurationError(
            "Claude auth status failed; run /login with the configured account"
        )
    try:
        status = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise SubscriptionConfigurationError(
            "Claude auth status did not return JSON"
        ) from exc

    auth_method = str(status.get("authMethod") or "")
    api_provider = str(status.get("apiProvider") or "")
    subscription_type = str(status.get("subscriptionType") or "").lower()
    if (
        status.get("loggedIn") is not True
        or auth_method != "claude.ai"
        or api_provider != "firstParty"
        or subscription_type not in _SUBSCRIPTION_TYPES
    ):
        raise SubscriptionConfigurationError(
            "Claude must be logged in with a first-party Pro, Max, Team, or "
            "Enterprise subscription"
        )

    return {
        "schema": "papc.claude_subscription_preflight.v1",
        "ready": True,
        "auth_method": auth_method,
        "api_provider": api_provider,
        "subscription_type": subscription_type,
        "model": config.model,
        "reasoning_effort": config.effort or "none",
        "config_dir": str(config.config_dir),
        "gateway_configured": config.base_url is not None,
    }


def _carries_decision(response: ClaudeSubscriptionResponse) -> bool:
    """Whether a response holds anything the caller can act on.

    A structured reply that validated against its schema is not automatically a
    reply that says something. The envelope this server asks for is satisfied by
    a null message beside an empty list of calls, and that shape passed the
    older test for a non-empty response because the object itself existed. It
    reached the benchmark as a turn in which the model neither spoke nor acted,
    which stalls an episode exactly as a dropped stream would. Judge on the
    values when the output is structured, and fall back to the text otherwise.
    """

    output = response.structured_output
    if isinstance(output, Mapping):
        return any(value not in (None, "", [], {}) for value in output.values())
    if output is not None:
        return True
    return bool(str(response.text or "").strip())


class ClaudeSubscriptionClient:
    """Programmatic single-turn inference through the official Agent SDK."""

    def __init__(
        self,
        config: ClaudeSubscriptionConfig,
        *,
        query_fn: QueryFunction | None = None,
        max_empty_attempts: int = 3,
        empty_retry_delay_seconds: float = 2.0,
    ) -> None:
        self.config = config
        self.max_empty_attempts = max(1, int(max_empty_attempts))
        self.empty_retry_delay_seconds = max(0.0, float(empty_retry_delay_seconds))
        if query_fn is None:
            try:
                from claude_agent_sdk import query
            except ImportError as exc:  # pragma: no cover - environment failure
                raise SubscriptionConfigurationError(
                    "claude-agent-sdk is not installed; sync the "
                    "claude-subscription extra"
                ) from exc
            query_fn = query
        self._query = query_fn

    @property
    def model(self) -> str:
        return self.config.model

    async def complete(
        self,
        *,
        prompt: str,
        system_prompt: str | None = None,
        output_format: Mapping[str, Any] | None = None,
    ) -> ClaudeSubscriptionResponse:
        """One completion, retried only when the stream returned nothing.

        Concurrent sessions occasionally end with a result message that carries
        no text and no structured output. That response holds no model decision,
        so returning it would report a benchmark episode as failed on the
        strength of a transport hiccup. Retrying it is the same treatment the
        OpenAI path gives a transport failure, and it cannot select an outcome
        because there is no outcome to select from.
        """

        last: ClaudeSubscriptionResponse | None = None
        for attempt in range(1, self.max_empty_attempts + 1):
            last = await self._complete_once(
                prompt=prompt,
                system_prompt=system_prompt,
                output_format=output_format,
            )
            if _carries_decision(last):
                return last
            if attempt < self.max_empty_attempts:
                await asyncio.sleep(self.empty_retry_delay_seconds * attempt)
        assert last is not None
        return last

    async def _complete_once(
        self,
        *,
        prompt: str,
        system_prompt: str | None = None,
        output_format: Mapping[str, Any] | None = None,
    ) -> ClaudeSubscriptionResponse:
        _reject_conflicting_credentials(os.environ)
        prompt = _validated_prompt(prompt, max_bytes=self.config.max_prompt_bytes)

        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            TextBlock,
        )

        text_parts: list[str] = []
        observed_models: set[str] = set()
        assistant_usage: dict[str, Any] | None = None
        result: ResultMessage | None = None

        try:
            # A clean cwd prevents repo/user Claude memory from leaking into a
            # benchmark model call.  The directory lives for the full SDK
            # stream and contains no CLAUDE.md, settings, skills, or project data.
            with tempfile.TemporaryDirectory(prefix="papc-claude-sdk-") as clean_cwd:
                options = ClaudeAgentOptions(
                    tools=[],
                    allowed_tools=[],
                    system_prompt=system_prompt,
                    permission_mode="dontAsk",
                    # Claude Code implements JSON-schema output through one internal
                    # structured-output turn.  Allow that bookkeeping turn while
                    # keeping ordinary text probes single-turn.
                    max_turns=8 if output_format is not None else 1,
                    model=self.config.model,
                    fallback_model=None,
                    cwd=clean_cwd,
                    cli_path=self.config.cli_path,
                    env=self.config.sdk_env(),
                    setting_sources=[],
                    skills=[],
                    effort=self.config.effort,  # type: ignore[arg-type]
                    output_format=(
                        dict(output_format) if output_format is not None else None
                    ),
                )
                async with asyncio.timeout(self.config.timeout_seconds):
                    async for message in self._query(prompt=prompt, options=options):
                        if isinstance(message, AssistantMessage):
                            observed_models.add(message.model)
                            if isinstance(message.usage, Mapping):
                                assistant_usage = dict(message.usage)
                            for block in message.content:
                                if isinstance(block, TextBlock):
                                    text_parts.append(block.text)
                        elif isinstance(message, ResultMessage):
                            result = message
        except TimeoutError as exc:
            raise ClaudeSubscriptionCallError(
                f"Claude subscription call exceeded {self.config.timeout_seconds}s"
            ) from exc
        except (ClaudeSubscriptionCallError, SubscriptionConfigurationError):
            raise
        except Exception as exc:  # noqa: BLE001 - the SDK boundary lives here
            # The Agent SDK signals a failed stream by raising a plain Exception
            # rather than a typed error, "Failed to provide valid structured
            # output after 5 attempts" among them. Nothing above this line
            # catches that, so it travelled out through the request handler,
            # which then returned no response at all and dropped the connection.
            # A caller saw a disconnect instead of a status it could classify,
            # and a disconnect mid-episode is not distinguishable from the
            # benchmark's own transport faults. Naming it here gives it the
            # gateway status the caller already knows how to retry.
            raise ClaudeSubscriptionCallError(
                f"Claude Agent SDK call failed: {exc}", api_error_status=502
            ) from exc

        if result is None:
            raise ClaudeSubscriptionCallError(
                "Claude Agent SDK ended without a result message"
            )
        if result.is_error:
            detail = "; ".join(result.errors or []) or result.subtype
            raise ClaudeSubscriptionCallError(
                f"Claude Agent SDK call failed: {detail}",
                api_error_status=result.api_error_status,
            )
        if observed_models and observed_models != {self.config.model}:
            raise ClaudeSubscriptionCallError(
                "Claude model identity mismatch: expected "
                f"{self.config.model}, observed {sorted(observed_models)}"
            )

        # Some Agent SDK streams expose the final text on AssistantMessage but
        # leave ResultMessage.result as an empty string.  Prefer a non-empty
        # result field, then fall back to the collected text blocks so callers
        # do not see a false empty response.
        result_text = result.result if isinstance(result.result, str) else ""
        text = result_text.strip() or "".join(text_parts)
        usage = dict(result.usage) if isinstance(result.usage, Mapping) else assistant_usage
        model_usage = (
            dict(result.model_usage)
            if isinstance(result.model_usage, Mapping)
            else None
        )
        return ClaudeSubscriptionResponse(
            text=text,
            model=next(iter(observed_models), self.config.model),
            usage=usage,
            model_usage=model_usage,
            provider="anthropic_subscription_agent_sdk",
            provider_calls=max(0, int(result.num_turns)),
            duration_ms=max(0, int(result.duration_ms)),
            estimated_cost_usd=result.total_cost_usd,
            structured_output=result.structured_output,
        )


async def probe_subscription_models(
    config: ClaudeSubscriptionConfig,
    *,
    models: Sequence[str],
    execute: bool,
    max_provider_calls: int,
    client_factory: Callable[[ClaudeSubscriptionConfig], Any] = (
        ClaudeSubscriptionClient
    ),
) -> dict[str, Any]:
    """Probe each candidate once and report only observed callability.

    There is no subscription model-list endpoint.  A model is therefore marked
    callable only after a bounded inference succeeds with that exact model.
    """

    candidates = [str(model).strip() for model in models]
    if not candidates or any(not model.startswith("claude-") for model in candidates):
        raise SubscriptionConfigurationError(
            "model probes require one or more Claude model IDs"
        )
    if len(set(candidates)) != len(candidates):
        raise SubscriptionConfigurationError("model probe candidates must be unique")
    if max_provider_calls != len(candidates):
        raise SubscriptionConfigurationError(
            "model probe requires max_provider_calls exactly "
            f"{len(candidates)}"
        )
    if not execute:
        return {
            "schema": "papc.claude_subscription_model_probe_plan.v1",
            "decision": "ready",
            "provider": "anthropic_subscription_agent_sdk",
            "provider_attempts": 0,
            "provider_calls": 0,
            "candidate_models": candidates,
            "callable_models": [],
            "results": [],
            "public_safe": True,
        }

    results: list[dict[str, Any]] = []
    callable_models: list[str] = []
    for model in candidates:
        candidate_config = config.for_model(model)
        client = client_factory(candidate_config)
        try:
            response = await client.complete(prompt="Reply with exactly: OK")
        except ClaudeSubscriptionCallError as exc:
            provider_rejected = (
                exc.api_error_status is not None
                and 400 <= exc.api_error_status < 500
            )
            results.append(
                {
                    "model": model,
                    "callable": False if provider_rejected else None,
                    "outcome": (
                        "provider_rejected"
                        if provider_rejected
                        else "infrastructure_failure"
                    ),
                    "error_type": type(exc).__name__,
                    "api_error_status": exc.api_error_status,
                }
            )
            continue
        except Exception as exc:  # unknown failures are infrastructure, not evidence
            results.append(
                {
                    "model": model,
                    "callable": None,
                    "outcome": "infrastructure_failure",
                    "error_type": type(exc).__name__,
                }
            )
            continue
        callable_models.append(model)
        results.append(
            {
                "model": model,
                "callable": True,
                "observed_model": response.model,
            }
        )

    return {
        "schema": "papc.claude_subscription_model_probe.v1",
        "provider": "anthropic_subscription_agent_sdk",
        "provider_attempts": len(candidates),
        "candidate_models": candidates,
        "callable_models": callable_models,
        "results": results,
        "public_safe": True,
    }


def _reject_conflicting_credentials(environ: Mapping[str, str]) -> None:
    conflicts = [name for name in _CONFLICTING_CREDENTIALS if environ.get(name)]
    if conflicts:
        joined = ", ".join(conflicts)
        raise SubscriptionConfigurationError(
            f"unset {joined}; these credentials override Claude subscription OAuth"
        )


def _validate_base_url(value: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme == "https" and parsed.hostname:
        return
    if parsed.scheme == "http" and parsed.hostname in _LOOPBACK_HOSTS:
        return
    raise SubscriptionConfigurationError(
        "PAPC_CLAUDE_BASE_URL must use HTTPS or loopback HTTP"
    )


def _validated_prompt(value: str, *, max_bytes: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SubscriptionConfigurationError("prompt must be a non-empty string")
    if len(value.encode("utf-8")) > max_bytes:
        raise SubscriptionConfigurationError(
            f"prompt exceeds the configured {max_bytes}-byte limit"
        )
    return value


def _positive_float(value: str, *, name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise SubscriptionConfigurationError(f"{name} must be numeric") from exc
    if parsed <= 0:
        raise SubscriptionConfigurationError(f"{name} must be positive")
    return parsed


def _positive_int(value: str, *, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise SubscriptionConfigurationError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise SubscriptionConfigurationError(f"{name} must be positive")
    return parsed
