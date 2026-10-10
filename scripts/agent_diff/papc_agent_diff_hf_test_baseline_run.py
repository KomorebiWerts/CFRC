#!/usr/bin/env python3
"""Run the official Agent-Diff HF test split as a PAPC baseline.

This is intentionally a baseline harness, not a CFRC method runner. It follows
the upstream ReAct notebook shape: the model receives the public task question
and a service base URL, then issues one bash command at a time through the
Agent-Diff BashExecutorProxy. The hidden state-diff answer is passed only to
the Agent-Diff evaluator after the run.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import dataclasses
import hashlib
import json
import os
import re
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from papc.certspec.provider import (
    UsageAccumulator,
    chat_completion_raw_http_with_retries,
    chat_response_diagnostics,
    first_tool_call_arguments,
    message_content_text,
)
from papc.paths import DATA_ROOT
from scripts.agent_diff import papc_agent_diff_scoring as scoring

AGENT_DIFF_SDK_PATH = Path(
    f"{DATA_ROOT}/benchmarks/agent_diff/repo/sdk/agent-diff-python"
)
DEFAULT_DATASET_PATH = Path(
    f"{DATA_ROOT}/benchmarks/agent_diff/repo/datasets/agent-diff-bench/test.jsonl"
)
DEFAULT_PUBLIC_OUTPUT = Path("outputs/papc")
DEFAULT_PRIVATE_ROOT = Path(f"{DATA_ROOT}/papc")

EXPECTED_TEST_ROWS = 45
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_MAX_CONCURRENCY = 10
DEFAULT_AGENT_ROLE = "cheap_baseline_full_agent"
DEFAULT_PROMPT_PROFILE = "baseline"
PROMPT_PROFILE_CHOICES = ("baseline", "verify_before_done")

SERVICE_CONFIG = {
    "slack": {
        "name": "Slack",
        "base_url": "https://slack.com/api",
        "extra": "",
    },
    "box": {
        "name": "Box",
        "base_url": "https://api.box.com/2.0",
        "extra": "",
    },
    "calendar": {
        "name": "Google Calendar",
        "base_url": "https://www.googleapis.com/calendar/v3",
        "extra": "- **Current Date/Time**: Sunday, June 17, 2018 at 00:01 (midnight), timezone America/Los_Angeles.\n",
    },
    "linear": {
        "name": "Linear",
        "base_url": "https://api.linear.app/graphql",
        "extra": "",
    },
}

REACT_SYSTEM_PROMPT = """You are an AI assistant that completes tasks by interacting with APIs via bash commands.

## Current Session
- **Service**: {service_name}
- **Base URL**: {base_url}
{extra_context}
## Environment
- Authentication is handled automatically via proxy. Use placeholder tokens where credentials would go.
- You execute bash commands, primarily curl, to interact with the API.
- If you are not sure how to use the API, explore the endpoint, parameters, and learn how it works.

## Response Format
Respond using XML tags:

<thinking>Your reasoning</thinking>
<action>Your bash command</action>

When done:
<thinking>Your reasoning</thinking>
<done>Brief summary</done>

## Rules
1. Execute ONE command at a time, then wait for the result.
2. Parse API responses carefully. Extract IDs and data needed for subsequent calls.
3. If a command fails, analyze the error and try a different approach.
4. Only use <done> when the task is fully completed.
"""

VERIFY_BEFORE_DONE_APPENDIX = """
## Completion Verification
Before using <done>, perform a final verification pass for the changes you made:
1. Re-read the affected resource or collection after every create, update, delete, permission, membership, invite, reaction, or move operation.
2. For updates, preserve all fields not explicitly requested by the user. Prefer the smallest PATCH/PUT body that changes only the requested field(s).
3. For delete/remove tasks, verify the target object, membership, reaction, ACL, or relation is absent after deletion.
4. For create-then-update or create-then-move tasks, verify the final parent/container, relation, and visible attributes after the last write.
5. If verification shows the requested final state is not true, repair it before using <done>.
"""

ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
DONE_RE = re.compile(r"<done>(.*?)</done>", re.DOTALL | re.IGNORECASE)
# A second spelling of the same step. A model reached through an account that
# carries a coding assistant's own tool block answers a request for one shell
# command in that block's syntax rather than in this loop's tag. What it puts
# in the block is a shell command either way, so the two are read as the same
# step; a model that never saw that block is unaffected, because it never emits
# one. The tool name is checked so a call to some other tool is not mistaken
# for a shell command.
TOOL_BLOCK_RE = re.compile(
    r"<function_calls>(.*?)</function_calls>", re.DOTALL | re.IGNORECASE
)
TOOL_NAME_RE = re.compile(
    r"<invoke\s+name=\"([^\"]+)\""
    r"|<parameter\s+name=\"tool\">\s*([^<]+?)\s*</parameter>",
    re.IGNORECASE,
)
TOOL_COMMAND_RE = re.compile(
    r"<parameter\s+name=\"command\">(.*?)</parameter>", re.DOTALL | re.IGNORECASE
)
SHELL_TOOL_NAMES = frozenset({"bash", "sh", "shell", "terminal", "run_command"})
# Said only to a model that has already answered without acting. It states what
# is true of this loop and nothing about the task: there is no one to ask, the
# environment is attached, and this tag is how it is reached. Recorded runs show
# every registered GPT arm reaching neither branch, so nothing said here changes
# a measured arm.
MAX_PREMATURE_DONE_REFUSALS = 2
NO_OBSERVATION_YET_CORRECTION = (
    "You have not run any command yet, so nothing about the environment has "
    "been observed. There is no person to answer a question and no other tool "
    "channel: a shell is already attached to the task environment, and this "
    "loop is the only way to reach it. Reply with exactly one "
    "<action>bash command</action> to look at the environment first."
)
FORMAT_CORRECTION = (
    "Please respond using either <action>one bash command</action> "
    "or <done>brief summary</done>. A shell is attached to the task "
    "environment and this loop is the only way to reach it; there is no "
    "person to ask and no other tool channel."
)
PRIVATE_MARKERS = (
    str(Path.home()),
    "Authorization: Bearer",
    "Bearer ",
    "sk-",
)


@dataclasses.dataclass(frozen=True)
class AgentDiffRow:
    ordinal: int
    question: str
    answer: dict[str, Any]
    test_id: str
    test_name: str
    service: str
    info: dict[str, Any]
    metadata: dict[str, Any]

    @property
    def public_key(self) -> str:
        payload = f"{self.ordinal}:{self.service}:{self.test_name}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_official_test_rows(dataset_path: Path) -> list[AgentDiffRow]:
    rows: list[AgentDiffRow] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for ordinal, line in enumerate(handle, start=1):
            raw = json.loads(line)
            info = _json_object(raw["info"])
            answer = _json_object(raw["answer"])
            rows.append(
                AgentDiffRow(
                    ordinal=ordinal,
                    question=str(raw["question"]),
                    answer=answer,
                    test_id=str(raw["test_id"]),
                    test_name=str(raw["test_name"]),
                    service=str(raw["service"]),
                    info=info,
                    metadata={
                        "task_horizon": raw.get("task_horizon"),
                        "operation_type": raw.get("operation_type"),
                        "entity_scope": raw.get("entity_scope"),
                        "information_availability": raw.get("information_availability"),
                        "prompt_ambiguity": raw.get("prompt_ambiguity"),
                    },
                )
            )
    return rows


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    parsed = json.loads(str(value))
    if not isinstance(parsed, Mapping):
        raise RuntimeError("expected_json_object")
    return dict(parsed)


def build_system_prompt(
    service: str, prompt_profile: str = DEFAULT_PROMPT_PROFILE
) -> str:
    cfg = SERVICE_CONFIG[service]
    prompt = REACT_SYSTEM_PROMPT.format(
        service_name=cfg["name"],
        base_url=cfg["base_url"],
        extra_context=cfg["extra"],
    )
    if prompt_profile == "baseline":
        return prompt
    if prompt_profile == "verify_before_done":
        return prompt + VERIFY_BEFORE_DONE_APPENDIX
    raise ValueError(f"unknown_prompt_profile:{prompt_profile}")


def parse_react_message(text: str) -> dict[str, str | None]:
    action = ACTION_RE.search(text)
    done = DONE_RE.search(text)
    command = action.group(1).strip() if action else None
    encoding = "action_tag" if command else None
    if command is None:
        command = _shell_command_from_tool_block(text)
        encoding = "tool_block" if command else None
    return {
        "action": command,
        "done": done.group(1).strip() if done else None,
        "action_encoding": encoding,
    }


def _shell_command_from_tool_block(text: str) -> str | None:
    """Read one shell command out of a coding assistant's tool block."""

    for block in TOOL_BLOCK_RE.findall(text):
        name_match = TOOL_NAME_RE.search(block)
        if name_match is None:
            continue
        name = (name_match.group(1) or name_match.group(2) or "").strip().lower()
        if name not in SHELL_TOOL_NAMES:
            continue
        command_match = TOOL_COMMAND_RE.search(block)
        if command_match is None:
            continue
        command = command_match.group(1).strip()
        if command:
            return command
    return None


def call_chat_model(
    *,
    api_base: str,
    api_key: str,
    model: str,
    reasoning_effort: str,
    messages: Sequence[Mapping[str, Any]],
    max_tokens: int,
    timeout_seconds: float,
    max_attempts: int,
    tools: Sequence[Mapping[str, Any]] | None = None,
    tool_choice: Any = None,
    tool_call_name: str = "",
) -> dict[str, Any]:
    started = time.monotonic()
    response, attempts = chat_completion_raw_http_with_retries(
        api_base=api_base,
        api_key=api_key,
        payload={
            "model": model,
            "messages": [dict(message) for message in messages],
            "reasoning_effort": reasoning_effort,
            "max_tokens": int(max_tokens),
            "temperature": 0.0,
            **({"tools": [dict(tool) for tool in tools]} if tools else {}),
            **({"tool_choice": tool_choice} if tool_choice is not None else {}),
        },
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        allow_reasoning_only_message=(
            str(model).rsplit("/", 1)[-1].startswith("deepseek-v4-")
        ),
    )
    text = message_content_text(response)
    usage = _response_usage(response)
    return {
        "text": text,
        "tool_call_arguments": (
            first_tool_call_arguments(response, name=tool_call_name)
            if tool_call_name
            else None
        ),
        "attempts": attempts,
        "usage": usage,
        "latency_seconds": round(time.monotonic() - started, 3),
        "diagnostics": chat_response_diagnostics(response, raw_text=text),
    }


def _response_usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    return dict(usage) if isinstance(usage, Mapping) else {}


@contextlib.contextmanager
def sanitized_executor_environment(api_key_env: str) -> Iterable[None]:
    """Prevent model-issued bash commands from reading provider credentials."""

    sensitive_keys = {
        api_key_env,
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_ORG_ID",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
    }
    saved = {key: os.environ.get(key) for key in sensitive_keys}
    try:
        for key in sensitive_keys:
            os.environ.pop(key, None)
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_react_agent(
    *,
    row: AgentDiffRow,
    bash: Any,
    api_base: str,
    api_key: str,
    api_key_env: str,
    model: str,
    reasoning_effort: str,
    max_iterations: int,
    max_model_tokens: int,
    provider_timeout_seconds: float,
    provider_max_attempts: int,
    command_timeout_seconds: int,
    observation_chars: int,
    prompt_profile: str,
) -> dict[str, Any]:
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": build_system_prompt(row.service, prompt_profile=prompt_profile),
        },
        {"role": "user", "content": row.question},
    ]
    trace: list[dict[str, Any]] = []
    usage_totals = UsageAccumulator()
    provider_call_count = 0
    model_latency_seconds = 0.0
    executed_action_count = 0
    premature_done_refusals = 0

    for iteration in range(1, max_iterations + 1):
        try:
            receipt = call_chat_model(
                api_base=api_base,
                api_key=api_key,
                model=model,
                reasoning_effort=reasoning_effort,
                messages=messages,
                max_tokens=max_model_tokens,
                timeout_seconds=provider_timeout_seconds,
                max_attempts=provider_max_attempts,
            )
        except Exception as exc:  # noqa: BLE001 - preserve prior paid usage.
            return {
                "completed": False,
                "iterations": iteration - 1,
                "trace": trace,
                "provider_call_count": provider_call_count,
                "model_latency_seconds": round(model_latency_seconds, 3),
                "usage": usage_totals.as_dict(),
                "completion_reason": "provider_error",
                "provider_error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        provider_call_count += 1
        model_latency_seconds += float(receipt["latency_seconds"])
        usage_totals.add_usage(receipt["usage"])
        text = str(receipt["text"])
        parsed = parse_react_message(text)
        trace_event: dict[str, Any] = {
            "iteration": iteration,
            "assistant_text": text,
            "assistant_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "provider_attempts": receipt["attempts"],
            "usage": receipt["usage"],
            "diagnostics": receipt["diagnostics"],
        }

        action = parsed["action"]
        done = parsed["done"]
        if parsed.get("action_encoding"):
            trace_event["action_encoding"] = parsed["action_encoding"]
        if done and not action and executed_action_count == 0:
            # An agent that has not run a single command has not observed the
            # environment, so whatever it is reporting is not an outcome it
            # checked. Recorded runs show this arriving as a question addressed
            # to a person who is not there. The loop says once what is true of
            # itself and asks again; the refusal is bounded so a model that
            # keeps declining still ends rather than spending the whole budget.
            if premature_done_refusals < MAX_PREMATURE_DONE_REFUSALS:
                premature_done_refusals += 1
                trace_event["format_error"] = "done_before_any_action"
                trace.append(trace_event)
                messages.append({"role": "assistant", "content": text})
                messages.append(
                    {"role": "user", "content": NO_OBSERVATION_YET_CORRECTION}
                )
                continue
        if action:
            with sanitized_executor_environment(api_key_env):
                tool_started = time.monotonic()
                tool_result = bash.execute(
                    str(action), timeout=int(command_timeout_seconds)
                )
                tool_latency = round(time.monotonic() - tool_started, 3)
            stdout = str(tool_result.get("stdout") or "")
            stderr = str(tool_result.get("stderr") or "")
            exit_code = int(tool_result.get("exit_code") or 0)
            observation = stdout.strip() or "(empty output)"
            if exit_code != 0:
                observation = f"{stdout}\n[stderr]: {stderr}".strip()
            observation = observation[:observation_chars]
            trace_event.update(
                {
                    "action": str(action),
                    "tool_exit_code": exit_code,
                    "tool_status": tool_result.get("status"),
                    "tool_latency_seconds": tool_latency,
                    "observation": observation,
                    "workspace_path": tool_result.get("workspace_path"),
                }
            )
            executed_action_count += 1
            trace.append(trace_event)
            messages.append({"role": "assistant", "content": text})
            messages.append(
                {
                    "role": "user",
                    "content": f"<observation>\n{observation}\n</observation>",
                }
            )
            continue

        if done:
            trace_event["done"] = str(done)
            trace.append(trace_event)
            return {
                "completed": True,
                "iterations": iteration,
                "trace": trace,
                "provider_call_count": provider_call_count,
                "model_latency_seconds": round(model_latency_seconds, 3),
                "usage": usage_totals.as_dict(),
                "completion_reason": "done_tag",
            }

        trace_event["format_error"] = "missing_action_or_done"
        trace.append(trace_event)
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": FORMAT_CORRECTION})

    return {
        "completed": False,
        "iterations": max_iterations,
        "trace": trace,
        "provider_call_count": provider_call_count,
        "model_latency_seconds": round(model_latency_seconds, 3),
        "usage": usage_totals.as_dict(),
        "completion_reason": "max_iterations_exhausted",
    }


def execute_row(
    *,
    row: AgentDiffRow,
    args: argparse.Namespace,
    api_key: str,
    private_raw_dir: Path,
    stop_event: threading.Event,
) -> dict[str, Any]:
    if stop_event.is_set():
        return {
            "schema": "papc.agent_diff.hf_test45_baseline_private_row.v1",
            "public_key": row.public_key,
            "ordinal": row.ordinal,
            "service": row.service,
            "status": "skipped_after_stop",
        }
    ensure_agent_diff_sdk_on_path(Path(args.agent_diff_sdk_path))
    from agent_diff import AgentDiff, BashExecutorProxy  # type: ignore

    client = AgentDiff(base_url=args.agent_diff_base_url)
    env = None
    run = None
    started = time.monotonic()
    row_result: dict[str, Any] = {
        "schema": "papc.agent_diff.hf_test45_baseline_private_row.v1",
        "public_key": row.public_key,
        "ordinal": row.ordinal,
        "service": row.service,
        "test_id": row.test_id,
        "test_name": row.test_name,
        "metadata": row.metadata,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "status": "started",
    }
    try:
        env = client.init_env(
            templateService=row.info["service"],
            templateName=row.info["seed_template"],
            impersonateUserId=row.info["impersonate_user_id"],
            ttlSeconds=int(args.env_ttl_seconds),
        )
        run = client.start_run(envId=env.environmentId)
        bash = BashExecutorProxy(
            env.environmentId,
            base_url=client.base_url,
            api_key=client.api_key,
            workspace_dir=str(private_raw_dir / "workspaces"),
        )
        agent_trace = run_react_agent(
            row=row,
            bash=bash,
            api_base=args.api_base,
            api_key=api_key,
            api_key_env=args.api_key_env,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_iterations=args.max_iterations,
            max_model_tokens=args.max_model_tokens,
            provider_timeout_seconds=args.provider_timeout_seconds,
            provider_max_attempts=args.provider_max_attempts,
            command_timeout_seconds=args.command_timeout_seconds,
            observation_chars=args.observation_chars,
            prompt_profile=args.prompt_profile,
        )
        # Evaluation is a separate benchmark service call. Preserve measured
        # model usage even if that service fails after the agent has finished.
        row_result["agent"] = agent_trace
        provider_error = agent_trace.get("provider_error")
        if isinstance(provider_error, Mapping):
            raise RuntimeError(
                "agent_provider_error:"
                f"{provider_error.get('type')}:{provider_error.get('message')}"
            )
        client.evaluate_run(runId=run.runId, expectedOutput=row.answer)
        result = client.get_results_for_run(runId=run.runId)
        evaluation = scoring.apply_registered_scorer_corrections(
            test_id=row.test_id,
            evaluation={
                "passed": bool(result.passed),
                "score": result.score,
                "failures": list(result.failures),
                "diff": result.diff,
                "result_status": result.status,
            },
        )
        row_result.update(
            {
                "status": "evaluated",
                "agent": agent_trace,
                "evaluation": evaluation,
            }
        )
    except Exception as exc:  # noqa: BLE001 - row-level private error capture.
        error_type = type(exc).__name__
        error_message = str(exc)
        row_result.update(
            {
                "status": "error",
                "error_type": error_type,
                "error_message": error_message,
            }
        )
        if should_stop_after_row_error(
            error_type,
            error_message,
            stop_on_first_infrastructure_error=bool(
                getattr(args, "stop_on_first_infrastructure_error", False)
            ),
        ):
            stop_event.set()
    finally:
        if env is not None:
            try:
                client.delete_env(envId=env.environmentId)
                row_result["env_cleanup"] = "deleted"
            except Exception as cleanup_exc:  # noqa: BLE001
                row_result["env_cleanup"] = "delete_failed"
                row_result["env_cleanup_error"] = str(cleanup_exc)
        row_result["elapsed_seconds"] = round(time.monotonic() - started, 3)

    row_path = private_raw_dir / f"row_{row.ordinal:03d}_{row.public_key}.json"
    write_json_append_only(row_path, row_result)
    return row_result


def is_fatal_provider_error(error_type: str, error_message: str) -> bool:
    text = f"{error_type}:{error_message}".lower()
    return any(
        marker in text
        for marker in (
            "authentication",
            "unauthorized",
            "forbidden",
            "insufficient_quota",
            "billing",
            "invalid_api_key",
        )
    )


def is_infrastructure_error(error_type: str, error_message: str) -> bool:
    text = f"{error_type}:{error_message}".lower()
    return any(
        marker in text
        for marker in (
            "timeouterror",
            "timed out",
            "connectionerror",
            "connection reset",
            "connection refused",
            "remote end closed connection",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
        )
    )


def should_stop_after_row_error(
    error_type: str,
    error_message: str,
    *,
    stop_on_first_infrastructure_error: bool,
) -> bool:
    return is_fatal_provider_error(error_type, error_message) or (
        stop_on_first_infrastructure_error
        and is_infrastructure_error(error_type, error_message)
    )


def execute_platform_smoke(
    rows: Sequence[AgentDiffRow],
    *,
    base_url: str,
    count: int,
) -> list[dict[str, Any]]:
    ensure_agent_diff_sdk_on_path(AGENT_DIFF_SDK_PATH)
    from agent_diff import AgentDiff  # type: ignore

    client = AgentDiff(base_url=base_url)
    results: list[dict[str, Any]] = []
    for row in rows[: max(count, 0)]:
        env = None
        started = time.monotonic()
        item: dict[str, Any] = {
            "public_key": row.public_key,
            "ordinal": row.ordinal,
            "service": row.service,
            "status": "started",
        }
        try:
            env = client.init_env(
                templateService=row.info["service"],
                templateName=row.info["seed_template"],
                impersonateUserId=row.info["impersonate_user_id"],
                ttlSeconds=600,
            )
            run = client.start_run(envId=env.environmentId)
            client.evaluate_run(runId=run.runId, expectedOutput=row.answer)
            result = client.get_results_for_run(runId=run.runId)
            item.update(
                {
                    "status": "evaluated_without_agent",
                    "passed": bool(result.passed),
                    "score_type": type(result.score).__name__,
                }
            )
        except Exception as exc:  # noqa: BLE001
            item.update(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:200],
                }
            )
        finally:
            if env is not None:
                try:
                    client.delete_env(envId=env.environmentId)
                    item["env_cleanup"] = "deleted"
                except Exception as exc:  # noqa: BLE001
                    item["env_cleanup"] = "delete_failed"
                    item["env_cleanup_error"] = str(exc)[:200]
            item["elapsed_seconds"] = round(time.monotonic() - started, 3)
        results.append(item)
    return results


def ensure_agent_diff_sdk_on_path(sdk_path: Path) -> None:
    sdk = str(sdk_path)
    if sdk not in sys.path:
        sys.path.insert(0, sdk)


def build_preflight_report(
    *,
    rows: Sequence[AgentDiffRow],
    args: argparse.Namespace,
    run_id: str,
    blockers: Sequence[str],
    platform_smoke: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    service_counts = Counter(row.service for row in rows)
    seed_counts = Counter(
        f"{row.info.get('service')}:{row.info.get('seed_template')}" for row in rows
    )
    report = {
        "schema": "papc.agent_diff.hf_test45_baseline_preflight.v1",
        "created_at": now_iso(),
        "run_id": run_id,
        "decision": "ready" if not blockers else "blocked",
        "blockers": list(blockers),
        "execute": bool(args.execute),
        "provider_calls": 0,
        "dataset": {
            "source": "official_agent_diff_hf_local_test_jsonl",
            "split": "test",
            "expected_rows": EXPECTED_TEST_ROWS,
            "loaded_rows": len(rows),
            "selected_ordinals": [row.ordinal for row in rows],
            "service_counts": dict(sorted(service_counts.items())),
            "seed_counts": dict(sorted(seed_counts.items())),
        },
        "model_plan": {
            "agent_role": args.agent_role,
            "prompt_profile": args.prompt_profile,
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "api_base_supplied": bool(args.api_base),
            "api_key_env": args.api_key_env,
        },
        "execution_contract": {
            "benchmark": "Agent-Diff",
            "task_count": len(rows),
            "max_concurrency": args.max_concurrency,
            "max_iterations": getattr(args, "max_iterations", 40),
            "max_model_tokens": getattr(args, "max_model_tokens", 1600),
            "provider_timeout_seconds": getattr(
                args, "provider_timeout_seconds", 120.0
            ),
            "provider_max_attempts": getattr(args, "provider_max_attempts", 3),
            "command_timeout_seconds": getattr(
                args, "command_timeout_seconds", 30
            ),
            "observation_chars": getattr(args, "observation_chars", 12000),
            "env_ttl_seconds": getattr(args, "env_ttl_seconds", 7200),
            "stop_on_first_infrastructure_error": bool(
                getattr(args, "stop_on_first_infrastructure_error", False)
            ),
            "scorer_name": scoring.SCORER_NAME,
            "scorer_correction_schema": scoring.SCHEMA,
            "agent_loop": "official_react_xml_bash_proxy",
            "prompt_profile": args.prompt_profile,
            "api_docs_injected": False,
            "tools_required_injected": False,
            "answer_visible_to_model": False,
            "evaluate_with_hidden_answer_after_agent": True,
        },
        "output_boundary": {
            "public_contains_row_prompts": False,
            "public_contains_expected_answers": False,
            "private_raw_rows_append_only": True,
        },
    }
    if platform_smoke is not None:
        report["platform_smoke"] = list(platform_smoke)
    report["public_leakage"] = public_leakage_audit(report)
    return report


def build_execute_summary(
    *,
    rows: Sequence[AgentDiffRow],
    results: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    run_id: str,
    started_at: str,
    elapsed_seconds: float,
    private_root_name: str,
) -> dict[str, Any]:
    evaluated = [result for result in results if result.get("status") == "evaluated"]
    errors = [result for result in results if result.get("status") != "evaluated"]
    service_counts = Counter(row.service for row in rows)
    pass_count = sum(
        1
        for result in evaluated
        if bool(_mapping(result.get("evaluation")).get("passed")) is True
    )
    scores = [
        score
        for score in (
            score_number(_mapping(result.get("evaluation")).get("score"))
            for result in evaluated
        )
        if score is not None
    ]
    by_service: dict[str, dict[str, Any]] = {}
    for service in sorted(service_counts):
        service_results = [
            result for result in results if result.get("service") == service
        ]
        service_eval = [
            result for result in service_results if result.get("status") == "evaluated"
        ]
        service_scores = [
            score
            for score in (
                score_number(_mapping(result.get("evaluation")).get("score"))
                for result in service_eval
            )
            if score is not None
        ]
        by_service[service] = {
            "rows": service_counts[service],
            "evaluated": len(service_eval),
            "passed": sum(
                1
                for result in service_eval
                if bool(_mapping(result.get("evaluation")).get("passed")) is True
            ),
            "mean_score": round(sum(service_scores) / len(service_scores), 6)
            if service_scores
            else None,
            "errors": len(service_results) - len(service_eval),
        }

    usage_totals = UsageAccumulator()
    provider_call_count = 0
    model_latency_seconds = 0.0
    iterations = 0
    for result in results:
        agent = _mapping(result.get("agent"))
        usage_totals.add_usage(_mapping(agent.get("usage")))
        provider_call_count += int(agent.get("provider_call_count") or 0)
        model_latency_seconds += float(agent.get("model_latency_seconds") or 0.0)
        iterations += int(agent.get("iterations") or 0)

    summary = {
        "schema": "papc.agent_diff.hf_test45_cheap_baseline_public.v1",
        "created_at": now_iso(),
        "started_at": started_at,
        "run_id": run_id,
        "decision": "completed" if len(evaluated) == len(rows) else "incomplete",
        "blockers": summarize_blockers(errors),
        "dataset": {
            "source": "official_agent_diff_hf_local_test_jsonl",
            "split": "test",
            "expected_rows": EXPECTED_TEST_ROWS,
            "loaded_rows": len(rows),
            "selected_ordinals": [row.ordinal for row in rows],
            "service_counts": dict(sorted(service_counts.items())),
        },
        "model_plan": {
            "agent_role": args.agent_role,
            "prompt_profile": args.prompt_profile,
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "api_base_supplied": bool(args.api_base),
            "api_key_env": args.api_key_env,
        },
        "execution_contract": {
            "max_concurrency": args.max_concurrency,
            "scorer_name": scoring.SCORER_NAME,
            "scorer_correction_schema": scoring.SCHEMA,
            "agent_loop": "official_react_xml_bash_proxy",
            "prompt_profile": args.prompt_profile,
            "max_iterations": args.max_iterations,
            "max_model_tokens": args.max_model_tokens,
            "provider_timeout_seconds": args.provider_timeout_seconds,
            "provider_max_attempts": args.provider_max_attempts,
            "command_timeout_seconds": args.command_timeout_seconds,
            "observation_chars": args.observation_chars,
            "env_ttl_seconds": args.env_ttl_seconds,
            "stop_on_first_infrastructure_error": bool(
                args.stop_on_first_infrastructure_error
            ),
            "api_docs_injected": False,
            "tools_required_injected": False,
            "answer_visible_to_model": False,
            "evaluate_with_hidden_answer_after_agent": True,
        },
        "aggregate": {
            "rows": len(rows),
            "evaluated": len(evaluated),
            "passed": pass_count,
            "pass_rate": round(pass_count / len(evaluated), 6) if evaluated else None,
            "mean_score": round(sum(scores) / len(scores), 6) if scores else None,
            "errors": len(errors),
            "by_service": by_service,
        },
        "agent_resource": {
            "provider_calls": provider_call_count,
            "usage": usage_totals.as_dict(),
            "model_latency_seconds": round(model_latency_seconds, 3),
            "iterations": iterations,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "max_concurrency": args.max_concurrency,
        },
        "private_artifacts": {
            "raw_root_label": private_root_name,
            "raw_rows": len(results),
        },
        "public_leakage": {},
    }
    summary["public_leakage"] = public_leakage_audit(summary)
    if not summary["public_leakage"]["pass"]:
        summary["decision"] = "blocked_public_leakage"
        summary["blockers"].append("public_leakage_audit_failed")
    return summary


def summarize_blockers(errors: Sequence[Mapping[str, Any]]) -> list[str]:
    if not errors:
        return []
    counts = Counter(
        str(error.get("error_type") or error.get("status")) for error in errors
    )
    return [f"{key}:{value}" for key, value in sorted(counts.items())]


def score_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, Mapping):
        for key in ("score", "value", "overall", "percent"):
            candidate = value.get(key)
            if isinstance(candidate, int | float) and not isinstance(candidate, bool):
                return float(candidate)
    return None


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def public_leakage_audit(payload: Mapping[str, Any]) -> dict[str, Any]:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    markers = [marker for marker in PRIVATE_MARKERS if marker in text]
    for key in (
        "OPENAI_API_KEY",
        "AGENT_DIFF_API_KEY",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        value = os.environ.get(key)
        if value and value in text:
            markers.append(f"secret_value:{key}")
    return {
        "pass": not markers,
        "marker_count": len(markers),
        "markers": markers[:5],
    }


def write_json_append_only(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    with os.fdopen(os.open(path, flags, 0o600), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def run_id_default() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def validate_args(args: argparse.Namespace, rows: Sequence[AgentDiffRow]) -> list[str]:
    blockers: list[str] = []
    if not args.row_ordinals and len(rows) != EXPECTED_TEST_ROWS:
        blockers.append("official_test_row_count_not_45")
    if (
        not args.row_ordinals
        and args.max_rows is not None
        and args.max_rows != EXPECTED_TEST_ROWS
    ):
        blockers.append("max_rows_not_official_45")
    if args.max_concurrency < 1 or args.max_concurrency > DEFAULT_MAX_CONCURRENCY:
        blockers.append("max_concurrency_out_of_range_1_to_10")
    if args.prompt_profile not in PROMPT_PROFILE_CHOICES:
        blockers.append(f"unknown_prompt_profile:{args.prompt_profile}")
    if (
        args.agent_role == "strong_verify_before_done_full_agent"
        and args.prompt_profile != "verify_before_done"
    ):
        blockers.append("verify_role_requires_verify_prompt_profile")
    if (
        args.prompt_profile == "verify_before_done"
        and args.agent_role != "strong_verify_before_done_full_agent"
    ):
        blockers.append("verify_prompt_profile_requires_verify_role")
    if not args.api_base:
        blockers.append("api_base_missing")
    if args.execute and not os.environ.get(args.api_key_env):
        blockers.append("api_key_env_missing")
    unknown_services = sorted({row.service for row in rows} - set(SERVICE_CONFIG))
    if unknown_services:
        blockers.append("unknown_service:" + ",".join(unknown_services))
    return blockers


def select_rows(
    all_rows: Sequence[AgentDiffRow], *, max_rows: int | None, row_ordinals: str | None
) -> list[AgentDiffRow]:
    if not row_ordinals:
        return list(all_rows[:max_rows] if max_rows is not None else all_rows)
    requested = []
    for chunk in row_ordinals.split(","):
        stripped = chunk.strip()
        if not stripped:
            continue
        requested.append(int(stripped))
    by_ordinal = {row.ordinal: row for row in all_rows}
    missing = [ordinal for ordinal in requested if ordinal not in by_ordinal]
    if missing:
        raise RuntimeError(f"row_ordinals_not_found:{missing}")
    return [by_ordinal[ordinal] for ordinal in requested]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--agent-diff-sdk-path", type=Path, default=AGENT_DIFF_SDK_PATH)
    parser.add_argument("--agent-diff-base-url", default="http://localhost:8020")
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", ""))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument(
        "--agent-role",
        default=DEFAULT_AGENT_ROLE,
        choices=(
            "cheap_baseline_full_agent",
            "strong_baseline_full_agent",
            "strong_verify_before_done_full_agent",
        ),
    )
    parser.add_argument(
        "--prompt-profile",
        default=DEFAULT_PROMPT_PROFILE,
        choices=PROMPT_PROFILE_CHOICES,
    )
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    parser.add_argument("--max-rows", type=int, default=EXPECTED_TEST_ROWS)
    parser.add_argument(
        "--row-ordinals",
        default=None,
        help="Comma-separated official row ordinals for targeted retry runs.",
    )
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--max-model-tokens", type=int, default=1600)
    parser.add_argument("--provider-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--provider-max-attempts", type=int, default=3)
    parser.add_argument("--command-timeout-seconds", type=int, default=30)
    parser.add_argument("--observation-chars", type=int, default=12000)
    parser.add_argument("--env-ttl-seconds", type=int, default=7200)
    parser.add_argument(
        "--stop-on-first-infrastructure-error",
        action="store_true",
        help="Stop scheduling provider work after the first timeout or transport error.",
    )
    parser.add_argument("--platform-smoke-count", type=int, default=0)
    parser.add_argument("--run-id", default=run_id_default())
    parser.add_argument("--public-output-dir", type=Path, default=DEFAULT_PUBLIC_OUTPUT)
    parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE_ROOT)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    all_rows = load_official_test_rows(args.dataset_path)
    rows = select_rows(all_rows, max_rows=args.max_rows, row_ordinals=args.row_ordinals)
    blockers = validate_args(args, rows)
    platform_smoke = None
    if args.platform_smoke_count:
        platform_smoke = execute_platform_smoke(
            rows,
            base_url=args.agent_diff_base_url,
            count=args.platform_smoke_count,
        )
        if any(item.get("status") == "error" for item in platform_smoke):
            blockers.append("platform_smoke_error")

    if not args.execute or blockers:
        report = build_preflight_report(
            rows=rows,
            args=args,
            run_id=args.run_id,
            blockers=blockers,
            platform_smoke=platform_smoke,
        )
        role_slug = args.agent_role.removesuffix("_full_agent")
        output = (
            args.public_output_dir
            / f"agent_diff_hf_test45_{role_slug}_preflight_{args.run_id}.json"
        )
        write_json_append_only(output, report)
        print(json.dumps({"output": str(output), "decision": report["decision"]}))
        return 0 if not blockers else 2

    api_key = os.environ[args.api_key_env]
    started_at = now_iso()
    started = time.monotonic()
    role_slug = args.agent_role.removesuffix("_full_agent")
    private_root_name = f"agent_diff_hf_test45_{role_slug}_private_{args.run_id}"
    private_root = args.private_root / private_root_name
    private_raw_dir = private_root / "raw_rows"
    private_raw_dir.mkdir(parents=True, exist_ok=False)
    stop_event = threading.Event()
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(args.max_concurrency)
    ) as pool:
        futures = [
            pool.submit(
                execute_row,
                row=row,
                args=args,
                api_key=api_key,
                private_raw_dir=private_raw_dir,
                stop_event=stop_event,
            )
            for row in rows
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(dict(future.result()))

    results.sort(key=lambda item: int(item.get("ordinal") or 0))
    private_summary = {
        "schema": "papc.agent_diff.hf_test45_baseline_private_summary.v1",
        "created_at": now_iso(),
        "run_id": args.run_id,
        "rows": results,
    }
    write_json_append_only(private_root / "private_summary.json", private_summary)
    public_summary = build_execute_summary(
        rows=rows,
        results=results,
        args=args,
        run_id=args.run_id,
        started_at=started_at,
        elapsed_seconds=time.monotonic() - started,
        private_root_name=private_root_name,
    )
    public_output = (
        args.public_output_dir
        / f"agent_diff_hf_test45_{role_slug}_public_{args.run_id}.json"
    )
    write_json_append_only(public_output, public_summary)
    print(
        json.dumps(
            {
                "output": str(public_output),
                "decision": public_summary["decision"],
                "aggregate": public_summary["aggregate"],
            },
            sort_keys=True,
        )
    )
    return 0 if public_summary["decision"] == "completed" else 2


if __name__ == "__main__":
    # Keep temporary files private even if the upstream SDK creates a workspace
    # before we pass an explicit private workspace root.
    tempfile.tempdir = str(DEFAULT_PRIVATE_ROOT)
    raise SystemExit(main())
