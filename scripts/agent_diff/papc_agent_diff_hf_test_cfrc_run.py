#!/usr/bin/env python3
# ruff: noqa: E402
"""Run Agent-Diff official HF test45 with an authority-lifted CFRC arm.

This is a method runner, not a baseline or prompt-retry harness.  The cheap
model runs first in the live environment.  When its next visible command is a
certified pre-write mutating carrier, the runner freezes that pending carrier
and asks the strong model for a typed residual graph.  The strong model never
executes bash directly; the compiler projects accepted graph nodes into a small
set of curl templates. A declared terminal with open obligations is another
commitment frontier; read-only binding receipts are folded into the same
contract before re-proposal. A refused graph runs nothing and closes the
method's authority: CFRC reports its failure value and claims no completion,
while the surrounding run is free to let the suspended cheap model carry on.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import copy
import dataclasses
import hashlib
import json
import os
import re
import shlex
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, unquote_plus, urljoin, urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from papc.certspec.contract_evidence import (  # noqa: E402
    EnvironmentContext,
    environment_context_evidence,
    operation_grammar,
)
from papc.certspec.provider import UsageAccumulator  # noqa: E402
from papc.certspec.residual_closure import (  # noqa: E402
    open_obligations,
    unrealized_write_fields,
)
from papc.run import MODEL_POOL_ARMS  # noqa: E402
from scripts.agent_diff import (
    papc_agent_diff_hf_test_baseline_run as baseline,  # noqa: E402
)
from scripts.agent_diff import papc_agent_diff_scoring as scoring  # noqa: E402

DEFAULT_CHEAP_MODEL = "gpt-5.4-mini"
DEFAULT_STRONG_MODEL = "gpt-5.5"
# A run is on-protocol when its (cheap, strong) pair is one the unified entry
# registers as a named arm. Pinning the pair itself, rather than one model
# family, keeps the method code model-agnostic while still refusing ad hoc
# model mixes.
REGISTERED_MODEL_POOLS = frozenset(MODEL_POOL_ARMS.values())
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_STRONG_REASONING_EFFORT = DEFAULT_REASONING_EFFORT
DEFAULT_PROVIDER_TIMEOUT_SECONDS = 300.0
DEFAULT_MAX_STRONG_MODEL_TOKENS = 2400
DEFAULT_MAX_GRAPH_NODES = 16
DEFAULT_MAX_MUTATING_NODES = 12
DEFAULT_MIN_FRONTIER_OBSERVATIONS = 1
DEFAULT_MAX_STRONG_RESIDUAL_ROUNDS = 2
MAX_IDENTICAL_FRONTIER_ATTEMPTS = 2
MAX_EXACT_CONTRACT_PROVIDER_CALLS = 2
# A replica-witness failure used to buy a third sample on the same contract. It
# bought nothing: the cheap model's next iteration emits another suspended write,
# which produces no observation, so the certificate re-freezes byte identical --
# same sha256, same evidence set, same obligations -- and the successor is asked
# the same question a third time. Across the recorded runs the third sample never
# produced a live write; the two it made were compiler-rejected and
# witness-failed, on rows that pass either way.
MAX_EXACT_CONTRACT_PROVIDER_CALLS_AFTER_WITNESS_FAILURE = 2
MAX_READONLY_BINDING_PROBES = 2
# Concurrency is a scheduling choice, not part of the arm definition, so the
# registered default stays where the recorded operating points ran while the
# ceiling is the value every surface shares online.
MAX_ONLINE_CONCURRENCY = 14
STICKY_COMPILER_CAPABILITY_REJECTS = frozenset(
    {
        "box_content_upload_not_compiler_executable",
        "box_multi_write_unsupported_family",
    }
)
METHOD_ROLE = "cfrc_authority_lifted_prewrite_hybrid"
PRIVATE_SCHEMA = "papc.agent_diff.hf_test45_cfrc_private_row.v1"

MUTATING_HTTP_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
READ_HTTP_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
FORBIDDEN_GRAPH_KEYS = frozenset(
    {
        "bash",
        "cmd",
        "command",
        "commands",
        "script",
        "shell",
        "subprocess",
    }
)
HIDDEN_MARKERS = (
    "answer",
    "assertions",
    "expectedOutput",
    "expected_output",
    "fail label",
    "score",
    "task id",
    "test_id",
    "testName",
    "test_name",
    "tools_required",
)


SERVICE_OPERATION_GRAMMAR: dict[str, dict[str, dict[str, Any]]] = {
    "box": {
        "box.user.read": {"methods": ["GET"], "markers": ["/users/me"]},
        "box.search.read": {"methods": ["GET"], "markers": ["/search"]},
        "box.folder": {
            "effects": ["entity.create", "entity.rename", "entity.update", "entity.delete", "entity.move"],
            "methods": ["GET", "POST", "PUT", "DELETE"],
            "markers": ["/folders"],
        },
        "box.file": {
            "effects": ["entity.create", "entity.rename", "entity.update", "entity.delete", "entity.move", "access.grant"],
            "methods": ["GET", "POST", "PUT", "DELETE"],
            "markers": ["/files"],
        },
        "box.comment": {
            "effects": ["message.post", "entity.delete", "entity.update"],
            "methods": ["GET", "POST", "PUT", "DELETE"],
            "markers": ["/comments"],
        },
        "box.task": {
            "effects": ["entity.create", "entity.update", "entity.delete", "membership.add"],
            "methods": ["GET", "POST", "PUT", "DELETE"],
            "markers": ["/tasks"],
        },
        "box.hub": {
            "effects": ["entity.create", "entity.update", "membership.add"],
            "methods": ["GET", "POST", "PUT"],
            "markers": ["/hubs", "/hub_items"],
        },
        "box.collection": {
            "effects": ["entity.update"],
            "methods": ["GET", "PUT"],
            "markers": ["/collections"],
        },
    },
    "calendar": {
        "calendar.calendars": {
            "effects": ["entity.create", "entity.update", "entity.rename", "entity.delete"],
            "methods": ["GET", "POST", "PUT", "PATCH", "DELETE"],
            "markers": ["/calendars"],
        },
        "calendar.calendar_list": {
            "effects": ["entity.create", "entity.update", "visibility.set", "entity.delete"],
            "methods": ["GET", "POST", "PUT", "PATCH", "DELETE"],
            "markers": ["/users/me/calendarList"],
        },
        "calendar.events": {
            "effects": ["entity.create", "entity.update", "entity.rename", "entity.delete", "entity.move", "membership.add"],
            "methods": ["GET", "POST", "PUT", "PATCH", "DELETE"],
            "markers": ["/events"],
        },
        "calendar.acl": {
            "effects": ["access.grant", "access.revoke", "entity.update"],
            "methods": ["GET", "POST", "PUT", "PATCH", "DELETE"],
            "markers": ["/acl", "/calendars"],
        },
        "calendar.freebusy.read": {
            "methods": ["POST"],
            "markers": ["/freeBusy"],
            "read_only": True,
        },
        "calendar.settings.read": {
            "methods": ["GET"],
            "markers": ["/users/me/settings", "/colors"],
        },
    },
    "linear": {
        "linear.graphql.query": {
            "methods": ["POST"],
            "markers": ["/graphql"],
            "read_only": True,
        },
        "linear.graphql.mutation": {
            "effects": ["entity.create", "entity.update", "entity.rename", "entity.delete", "entity.move", "message.post", "membership.add", "membership.remove", "access.grant"],
            "methods": ["POST"],
            "markers": ["/graphql"],
        },
    },
    "slack": {
        "slack.users.read": {
            "methods": ["GET", "POST"],
            "markers": ["users.", "auth.test"],
            "read_only": True,
        },
        "slack.conversations.read": {
            "methods": ["GET", "POST"],
            "markers": [
                "conversations.list",
                "conversations.info",
                "conversations.members",
                "conversations.history",
                "conversations.replies",
            ],
            "read_only": True,
        },
        "slack.search.read": {
            "methods": ["GET", "POST"],
            "markers": ["search.messages"],
            "read_only": True,
        },
        "slack.reactions.read": {
            "methods": ["GET", "POST"],
            "markers": ["reactions.get"],
            "read_only": True,
        },
        "slack.chat": {
            "effects": ["message.post", "entity.update", "entity.delete"],
            "methods": ["POST"],
            "markers": ["chat.postMessage", "chat.update", "chat.delete"],
        },
        "slack.conversations.write": {
            "effects": ["entity.create", "entity.rename", "membership.add", "membership.remove", "entity.update", "entity.delete"],
            "methods": ["POST"],
            "markers": [
                "conversations.archive",
                "conversations.create",
                "conversations.invite",
                "conversations.join",
                "conversations.kick",
                "conversations.leave",
                "conversations.open",
                "conversations.rename",
                "conversations.setPurpose",
                "conversations.setTopic",
                "conversations.unarchive",
            ],
        },
        "slack.reactions": {
            # A reaction is the family that realizes a mark. It keeps the
            # older units so no graph that already relied on them is refused,
            # but only this family carries annotation.add, which is what makes
            # a marking obligation distinguishable from a posting one.
            "effects": ["annotation.add", "message.post", "entity.update"],
            "methods": ["POST"],
            "markers": ["reactions.add", "reactions.remove"],
        },
        "slack.pins": {
            "effects": ["entity.update"],
            "methods": ["POST"],
            "markers": ["pins.add", "pins.remove"],
        },
    },
}

SLACK_READ_ENDPOINTS = (
    "auth.test",
    "conversations.history",
    "conversations.info",
    "conversations.list",
    "conversations.members",
    "conversations.replies",
    "search.messages",
    "reactions.get",
    "users.info",
    "users.list",
    "users.lookupByEmail",
)


CFRC_STRONG_SYSTEM_PROMPT = """You are a PAPC/CFRC residual proposer.

You are not an executor. You do not run commands and you do not explore.
The runner will reject raw shell, curl commands, code snippets, hidden fields,
wrong-service actions, unknown operation families, and over-budget graphs.

Submit the graph by calling the submit_residual_graph tool. If that tool is
not offered, return the same object as text and nothing else.

JSON schema:
{
  "nodes": [
    {
      "node_id": "n1",
      "operation_family": "<one public grammar family>",
      "authority": "ell_write" | "ell_bounded_local_redo",
      "evidence_refs": ["e1"],
      "claimed_effects": ["what this node makes true"],
      "arguments": {
        "method": "GET|POST|PUT|PATCH|DELETE",
        "path": "/relative/path or Slack method",
        "params": {},
        "json": {}
      }
    }
  ],
  "action": "complete_residual|bind_evidence|decline"
}

Rules:
- Use only public schema grammar, the user request, accepted-prefix
  observations, and receipt-backed effects in K as evidence.
- Preserve the cheap prefix. Do not restart the task.
- "K" lists effects already realized in this environment, each with its receipt.
  They are done. Do not propose a node that would realize any of them again.
- Each write must change only the fields the request asks for. Do not set extra
  fields the request did not ask for; the environment scores changed fields with
  a strict-subset check, so any unrequested field change fails the task.
- Include every write the task still requires in this one graph; do not split a
  required write off for the cheap model to finish later. A cheap verification
  pass may still run after your writes, so make each write correct and complete.
- Every proposal declares one action. "bind_evidence" carries only read nodes.
  "complete_residual" may carry reads followed by writes and claims the whole
  remaining residual; every write graph must use it. "decline" carries no nodes
  and states that no admissible residual exists. A later node may consume an id
  an earlier node returns as {{n1.id}}, so needing a server-issued id first is
  not a reason to leave the residual open; chain the writes in one graph.
- Infer the residual actions from the frozen contract and visible evidence; the
  runner does not reveal or endorse the cheap model's suspended raw candidate.
- Include every write needed to complete the residual task, plus optional reads
  only when they are needed before a write.
- If the request has multiple create/update/delete/share/invite requirements,
  include all remaining writes; a graph containing only one convenient next
  action is under-complete.
- If the visible evidence is insufficient to safely choose a write, output only
  the read nodes needed to complete the binding. The runner will execute those
  typed reads and return the observations to you in a newly frozen reached-state
  contract; the cheap model does not regain write authority at this frontier.
- An empty filtered search does not discharge a clause about an explicitly
  named resource. Broaden the search or list the containing collection before
  deciding it is absent; do not declare complete_residual while that named
  clause is merely unresolved or silently omit it from the graph.
- For Calendar, bind relative and temporal references to the service_context
  Current Date/Time; do not use the real wall-clock date unless it appears in
  the task or observations. Keep a timed event within the single named local
  day unless the request explicitly says it crosses midnight.
- When a Calendar request explicitly says to use a named person's timezone,
  GET /calendar/v3/calendars/{email} to obtain that calendar's timeZone. A
  freeBusy response is rendered in the query's timezone and does not prove the
  person's timezone. Bind start.timeZone and end.timeZone to the timezone
  receipt; the GET and event write may share one graph by using a placeholder.
- For Calendar calendarList visibility, change only the field the request
  names: to hide a calendar from the list, PATCH the calendarList entry with
  exactly {"hidden": true} and change no other field; to show or unhide, use
  {"hidden": false}; to select or deselect a calendar, use {"selected": true}
  or {"selected": false}. Do not also set "selected" when only hide or show is
  requested, and do not set "hidden" when only select or deselect is requested.
- For an existing Calendar access grant, update or delete it using the exact ACL
  rule id the calendar's ACL list returns (GET .../acl, items[].id). Delete or
  update by that exact id; do not POST a duplicate grant and do not shorten or
  synthesize the rule id.
- "O" states the open obligations. Every effect it lists must be realized by a
  node in this graph. An effect is realized by the operation the public grammar
  says produces it, not by a message describing it.
- When "O" carries a quantifier, the obligation ranges over a set, so one
  instance does not discharge it. When "O" lists unopened enumerated
  containers, the target set is not known until those listings are complete.
  Gather every missing listing in one read-only graph: enumerate all the
  containers "O" names in that single pass, not one container per proposal.
  After one read-only pass the receipts are in "E"; the next proposal must
  either deliver the write graph grounded in those receipts or decline.
- For Linear, use path "/graphql" and json {"query": "...", "variables": {...}}.
- For Linear issue relations, IssueRelationType accepts exactly blocks,
  duplicate, related, or similar; there is no blocked_by value. To state that
  issue A is blocked by issue B, create the relation with issueId = B and
  type = "blocks" so that B blocks A.
- For Slack, path may be a method such as "chat.postMessage".
- Before reactions.remove, bind the exact channel, message timestamp, and
  reaction name to a receipt whose message.reactions contains that reaction.
  Text search alone does not prove which matching message owns the reaction.
  If search returns multiple candidates, batch one reactions.get read for each
  candidate in a single read-only graph, then remove only the receipt-confirmed
  reaction in the next graph. A reactions.get node may also directly precede
  reactions.remove in the same graph when both use the exact same channel and
  timestamp; Checked Execution will reject the graph on the replica before any
  live write if that probe does not show the reaction.
- For a Slack direct message or group direct message, the reliable pattern is
  conversations.open with "users" as a comma-separated list of user ids (one
  user opens a DM; two or more open a group conversation), then
  chat.postMessage to the returned {{nX.channel.id}}. conversations.create
  creates a named channel; it does not open a DM or group conversation.
- Creating a calendar also places it on the creator's calendar list already
  selected, so do not POST the same calendar to /users/me/calendarList again
  (that duplicate insert fails). To change color, visibility, or selection of
  a listed calendar, PATCH /users/me/calendarList/{calendarId} directly.
- For a global smallest/largest choice across Box folders, a size-sorted
  folder listing proves only that folder's local endpoints. Read the concrete
  size of every local endpoint across all enumerated folders in the same
  read-only graph, then compare those receipt-backed candidates and propose
  the writes in the next proposal.
- Cover the whole request: every person, resource, or set the request names,
  and every conditional or dependent clause it states, is an obligation the
  graph must realize or ground with a read node first. Do not drop a named
  invitee, recipient, or follow-up message.
- When a write target is determined by a field of an earlier read's response,
  put the read and the write in the same graph and bind the target with a
  placeholder; end with a read-only graph only when a choice among the read
  results cannot be expressed by a placeholder filter.
- Later nodes may reference prior JSON response fields with placeholders such
  as "{{n1.id}}" or "{{n2.data.issueCreate.issue.id}}". A list is indexed
  numerically ("{{n3.items[0].id}}") or filtered by field equality
  ("{{n3.items[?(@.scope.value=='ada@test.com')].id}}", with && between
  conditions); no other filter syntax resolves.
"""


@dataclasses.dataclass(frozen=True)
class CommandClassification:
    kind: str
    service: str | None
    methods: tuple[str, ...]
    operation_family: str | None
    url: str | None
    path: str | None
    body_keys: tuple[str, ...]
    command_text: str
    command_sha256: str
    reasons: tuple[str, ...]
    sub_requests: tuple["CommandClassification", ...] = ()


@dataclasses.dataclass(frozen=True)
class CompiledCommand:
    node_id: str
    operation_family: str
    method: str
    url: str
    command: str
    mutating: bool
    authority: str


class GraphReject(ValueError):
    """Raised when the strong graph cannot be projected safely."""


def _split_curl_invocations(command: str) -> tuple[str, ...]:
    """Split one action into its individual curl requests, in order."""

    starts = [
        match.start()
        for match in re.finditer(r"(?<![A-Za-z0-9_])curl\b", command)
    ]
    if len(starts) < 2:
        return (command,) if starts else ()
    bounds = starts + [len(command)]
    return tuple(
        command[bounds[index] : bounds[index + 1]].strip()
        for index in range(len(starts))
    )


def classify_action_command(
    command: str, expected_service: str | None = None
) -> CommandClassification:
    cleaned = command.strip()
    urls = _extract_urls(cleaned)
    url = urls[0] if urls else None
    path = _path_for_url(url) if url else None
    service = _service_from_url(url) or expected_service
    methods = _extract_http_methods(cleaned)
    if not methods:
        methods = ("GET",) if "curl" in cleaned.lower() else ()
    body_keys = tuple(sorted(_extract_body_keys(cleaned)))
    reasons: list[str] = []

    operation_family = _infer_operation_family(
        service=service,
        path=path,
        command=cleaned,
        methods=methods,
    )
    kind = "unknown"
    sub_commands = _split_curl_invocations(cleaned)
    if len(sub_commands) > 1:
        # A compound command can read with its first curl and mutate with a
        # later one. Classifying only the first request would license an
        # untracked write and let the residual executor duplicate it. Refusing
        # the action instead discarded work the cheap model had legitimately
        # done, so each request is classified on its own: the action carries
        # the most privileged kind among them, and every write it performed is
        # recorded in the reached state.
        parts = tuple(
            classify_action_command(part, expected_service)
            for part in sub_commands
        )
        aggregate_kind = "read"
        if any(part.kind == "mutating" for part in parts):
            aggregate_kind = "mutating"
        elif any(part.kind == "unknown" for part in parts):
            aggregate_kind = "unknown"
        return CommandClassification(
            kind=aggregate_kind,
            service=service,
            methods=tuple(methods),
            operation_family=operation_family,
            url=url,
            path=path,
            body_keys=body_keys,
            command_text=cleaned,
            command_sha256=hashlib.sha256(cleaned.encode("utf-8")).hexdigest(),
            reasons=("compound_command_decomposed",),
            sub_requests=parts,
        )
    if service and operation_family:
        grammar = SERVICE_OPERATION_GRAMMAR.get(service, {}).get(operation_family, {})
        read_only = bool(grammar.get("read_only"))
        if read_only:
            kind = "read"
            reasons.append("operation_family_read_only")
        elif any(method in MUTATING_HTTP_METHODS for method in methods):
            kind = "mutating"
            reasons.append("mutating_http_method")
        elif methods and all(method in READ_HTTP_METHODS for method in methods):
            kind = "read"
            reasons.append("read_http_method")
    elif any(method in MUTATING_HTTP_METHODS for method in methods):
        kind = "unknown"
        reasons.append("mutating_method_unknown_service_or_family")
    elif "curl" in cleaned.lower() and methods:
        kind = "read"
        reasons.append("curl_default_read")
    else:
        reasons.append("not_a_curl_http_action")

    return CommandClassification(
        kind=kind,
        service=service,
        methods=tuple(methods),
        operation_family=operation_family,
        url=url,
        path=path,
        body_keys=body_keys,
        command_text=cleaned,
        command_sha256=hashlib.sha256(cleaned.encode("utf-8")).hexdigest(),
        reasons=tuple(reasons),
    )


def _extract_urls(command: str) -> list[str]:
    matches = re.findall(r"""https?://[^\s'"<>]+""", command)
    return [match.rstrip(");") for match in matches]


def _path_for_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    return path


def _service_from_url(url: str | None) -> str | None:
    if not url:
        return None
    for service, cfg in baseline.SERVICE_CONFIG.items():
        base = str(cfg["base_url"])
        if url.startswith(base):
            return service
    host = urlparse(url).netloc
    if "slack.com" in host:
        return "slack"
    if "box.com" in host:
        return "box"
    if "googleapis.com" in host:
        return "calendar"
    if "linear.app" in host:
        return "linear"
    return None


def _extract_http_methods(command: str) -> tuple[str, ...]:
    methods = [
        method.upper()
        for method in re.findall(r"(?:-X|--request)\s+['\"]?([A-Za-z]+)['\"]?", command)
    ]
    if not methods and re.search(r"(?:^|\s)(?:-G|--get)(?:\s|$)", command):
        methods.append("GET")
    if not methods and re.search(r"(?:\s-d\s|--data|--json)", command):
        methods.append("POST")
    return tuple(methods)


def _extract_body_keys(command: str) -> set[str]:
    keys: set[str] = set()
    for parsed in _extract_json_payloads(command):
        keys.update(str(key) for key in parsed.keys())
    return keys


def _infer_operation_family(
    *,
    service: str | None,
    path: str | None,
    command: str,
    methods: Sequence[str],
) -> str | None:
    if not service:
        return None
    haystack = f"{path or ''}\n{command}"
    if service == "linear":
        if "graphql" not in haystack.lower():
            return None
        if _linear_command_is_mutation(command):
            return "linear.graphql.mutation"
        return "linear.graphql.query"
    if service == "slack":
        endpoint = _slack_endpoint_from_text(haystack)
        if endpoint:
            if endpoint in SLACK_READ_ENDPOINTS:
                if endpoint.startswith("users."):
                    return "slack.users.read"
                if endpoint.startswith("conversations."):
                    return "slack.conversations.read"
                if endpoint.startswith("reactions."):
                    return "slack.reactions.read"
                return "slack.search.read"
            for family, spec in SERVICE_OPERATION_GRAMMAR["slack"].items():
                if any(marker in endpoint for marker in spec["markers"]):
                    return family
        return None
    for family, spec in SERVICE_OPERATION_GRAMMAR.get(service, {}).items():
        if any(marker in haystack for marker in spec["markers"]):
            if service == "calendar" and family == "calendar.calendars":
                # More specific Calendar families should win when present.
                if "/events" in haystack or "/acl" in haystack:
                    continue
            return family
    if any(method in MUTATING_HTTP_METHODS for method in methods):
        return None
    return None


def _slack_endpoint_from_text(text: str) -> str | None:
    candidates = re.findall(r"(?:^|[/\s'\"])([a-z]+(?:\.[A-Za-z0-9_]+)+)", text)
    if not candidates:
        return None
    return candidates[0]


def _linear_command_is_mutation(command: str) -> bool:
    for payload in _extract_json_payloads(command):
        query = str(payload.get("query") or "").lstrip()
        if query.startswith("mutation"):
            return True
    return bool(
        re.search(
            r"""(?:\\?["'])?query(?:\\?["'])?\s*:\s*"""
            r"""(?:\\?["'])\s*mutation\b""",
            command,
            re.IGNORECASE,
        )
    )


def _extract_json_payloads(command: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for match in re.findall(
        r"(?:-d|--data|--data-raw|--data-binary|--json)\s+(['\"])(.*?)\1",
        command,
        re.DOTALL,
    ):
        raw = match[1]
        with contextlib.suppress(Exception):
            parsed = json.loads(raw)
            if isinstance(parsed, Mapping):
                payloads.append(dict(parsed))
    return payloads


def should_open_frontier(
    classification: CommandClassification,
    *,
    service: str,
    prefix_events: Sequence[Mapping[str, Any]],
    min_observations: int = DEFAULT_MIN_FRONTIER_OBSERVATIONS,
) -> bool:
    if classification.kind != "mutating":
        return False
    if classification.service != service:
        return False
    if not classification.operation_family:
        return False
    if _zero_observation_frontier_allowed(classification):
        return True
    successful_visible_reads = sum(
        1
        for event in prefix_events
        if event.get("classification_kind") == "read"
        and int(event.get("tool_exit_code") or 0) == 0
    )
    return successful_visible_reads >= min_observations


def _zero_observation_frontier_allowed(
    classification: CommandClassification,
) -> bool:
    if (
        classification.service == "calendar"
        and classification.operation_family == "calendar.events"
        and _calendar_event_datetime_missing_timezone(classification.command_text)
    ):
        return True
    return False


def _calendar_event_datetime_missing_timezone(command: str) -> bool:
    for payload in _extract_json_payloads(command):
        for key in ("start", "end"):
            value = payload.get(key)
            if (
                isinstance(value, Mapping)
                and value.get("dateTime")
                and not value.get("timeZone")
            ):
                return True
    return False


def build_frontier_certificate(
    *,
    row: baseline.AgentDiffRow,
    prefix_events: Sequence[Mapping[str, Any]],
    max_strong_calls: int,
    realized_commitments: Sequence[Mapping[str, Any]] = (),
    frontier_kind: str = "pre_write_pending_mutation",
) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    for index, event in enumerate(prefix_events, start=1):
        observation = compact_observation_for_evidence(
            str(event.get("observation_full") or event.get("observation") or "")
        )
        entry = {
            "ref": f"e{index}",
            "kind": event.get("classification_kind") or "unknown",
            "operation_family": event.get("operation_family"),
            "action_sha256": event.get("action_sha256"),
            "observation": observation,
        }
        request = _mapping(event.get("request"))
        method = str(request.get("method") or "").upper()
        path = str(request.get("path") or "")
        if entry["kind"] == "read" and method in READ_HTTP_METHODS and path.startswith(
            "/"
        ):
            # The request path binds an observation to the container or entity
            # that produced it. Keep only method+path: the original curl,
            # headers, tokens, and suspended write never enter the contract.
            entry["request"] = {"method": method, "path": path}
        evidence.append(entry)
    # The whole certificate is resent on every proposal, so evidence carries a
    # total budget as well as a per-entry one. Older observations demote to a
    # short head first: the newest reads carry the bindings a residual write
    # is about to use.
    total_evidence_chars = sum(len(entry["observation"]) for entry in evidence)
    for entry in evidence:
        if total_evidence_chars <= EVIDENCE_TOTAL_CHAR_BUDGET:
            break
        observation = str(entry["observation"])
        if len(observation) <= _EVIDENCE_DEMOTED_OBSERVATION_CHARS:
            continue
        entry["observation"] = observation[:_EVIDENCE_DEMOTED_OBSERVATION_CHARS]
        entry["observation_demoted"] = True
        total_evidence_chars -= len(observation) - len(entry["observation"])
    cfg = baseline.SERVICE_CONFIG[row.service]
    return {
        "schema": "papc.agent_diff.cfrc_frontier_certificate.v1",
        "service": row.service,
        "service_context": {
            "name": cfg["name"],
            "base_url": cfg["base_url"],
            "extra": cfg["extra"],
        },
        "frontier_kind": frontier_kind,
        "req": row.question,
        "K": [dict(item) for item in realized_commitments],
        "O": open_obligations(request=row.question, receipts=prefix_events),
        "E": evidence,
        "environment_context": environment_context_evidence(
            reached_state_environment(row.service)
        ),
        "B_call": {"strong_proposal_calls": max_strong_calls},
        "public_schema": operation_grammar(SERVICE_OPERATION_GRAMMAR[row.service]),
    }


# What Checked Execution returned. The method has exactly two outcomes once a
# frontier exists: it realized the residual suffix, or it refused. Anything the
# surrounding run does afterwards is recorded separately, so no handback can be
# read as a method completion.
CFRC_NO_FRONTIER = "cfrc_no_frontier"
CFRC_RESIDUAL_COMPLETED = "cfrc_residual_completed"
CFRC_REFUSED = "cfrc_refused"

# Who holds control after CFRC returned. Outside the method boundary.
POST_CONTROL_NONE = "none"
POST_CONTROL_CHEAP = "cheap_continuation"

# Whether one admitted graph is the entire residual suffix on a surface. A
# Linear residual is a single GraphQL mutation, so an admitted graph realizes
# it outright; the REST surfaces reach the same effect through several calls
# and a verification read. This is a property of the API shape, declared once
# here rather than decided inside the control flow, and it does not vary with
# the request.
SURFACE_WRITE_IS_ATOMIC = {
    "linear": True,
    "box": False,
    "slack": False,
    "calendar": False,
}


def reached_state_environment(service: str) -> EnvironmentContext:
    """Environment facts the successor inherits, as the surface states them.

    The service context is already given to the cheap model in its system
    prompt, so it is part of the reached state rather than a hint. A strong
    successor that never sees it would resolve relative references against
    the wrong clock. It is read from the surface's own declaration for every
    service, so no service receives a hand-written note.
    """

    stated = str(baseline.SERVICE_CONFIG.get(service, {}).get("extra") or "").strip()
    return EnvironmentContext(observed_at=stated or None)


# Lexical markers that decide which service a rule block governs. A block that
# names no service, or names more than one, is generic and is always sent; only
# a block that names exactly one other service is dropped, and the compiler
# already raises GraphReject("wrong_service") for every action such a block
# could describe. Any new rule must be worded so its service is named here,
# otherwise it will simply stay in the always-sent set.
_STRONG_PROMPT_SERVICE_MARKERS: dict[str, tuple[str, ...]] = {
    "calendar": (
        "Calendar",
        "calendar",
        "freeBusy",
        "calendarList",
        "/calendar/v3",
        "ACL",
    ),
    "slack": (
        "Slack",
        "reactions.",
        "conversations.",
        "chat.postMessage",
    ),
    "linear": (
        "Linear",
        "/graphql",
        "IssueRelationType",
    ),
    "box": ("Box",),
}


def _split_strong_prompt(prompt: str) -> tuple[str, tuple[str, ...]]:
    """Split the frozen prompt into its header and its "- " rule blocks."""

    marker = "\nRules:\n"
    head, found, rest = prompt.partition(marker)
    if not found:
        return prompt, ()
    blocks: list[str] = []
    current: list[str] = []
    for line in rest.splitlines(keepends=True):
        if line.startswith("- "):
            if current:
                blocks.append("".join(current))
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append("".join(current))
    return head + marker, tuple(blocks)


def _strong_prompt_block_services(block: str) -> frozenset[str]:
    return frozenset(
        service
        for service, markers in _STRONG_PROMPT_SERVICE_MARKERS.items()
        if any(marker in block for marker in markers)
    )


_STRONG_PROMPT_HEADER, _STRONG_PROMPT_BLOCKS = _split_strong_prompt(
    CFRC_STRONG_SYSTEM_PROMPT
)
_STRONG_PROMPT_BLOCK_SERVICES: tuple[frozenset[str], ...] = tuple(
    _strong_prompt_block_services(block) for block in _STRONG_PROMPT_BLOCKS
)


def service_scoped_strong_system_prompt(service: str) -> str:
    """Drop only the rule blocks that govern a single other service.

    The certificate carries exactly one service, and compile_typed_graph
    refuses any node outside it, so the dropped rules describe actions the
    admissible space never contained. The header, the schema, every generic
    block and every cross-service block are sent unchanged.
    """

    if not _STRONG_PROMPT_BLOCKS:
        return CFRC_STRONG_SYSTEM_PROMPT
    kept = [
        block
        for block, services in zip(
            _STRONG_PROMPT_BLOCKS, _STRONG_PROMPT_BLOCK_SERVICES, strict=True
        )
        if len(services) != 1 or service in services
    ]
    return _STRONG_PROMPT_HEADER + "".join(kept)


def build_strong_messages(
    *,
    certificate: Mapping[str, Any],
    repair_observation: str | None = None,
    prior_tool_arguments: str | None = None,
    live_receipts: Sequence[Mapping[str, Any]] = (),
    service_scoped_prompt: bool = False,
) -> list[dict[str, Any]]:
    native_tool_repair = False
    if prior_tool_arguments and repair_observation:
        try:
            native_tool_repair = isinstance(
                json.loads(prior_tool_arguments), Mapping
            )
        except json.JSONDecodeError:
            # A malformed argument string cannot be reconstructed as an
            # assistant tool call. Keep the historical text-repair path for
            # exactly that transport failure.
            native_tool_repair = False
    payload = {
        "certificate": certificate,
        "live_receipts": [dict(item) for item in live_receipts],
        # A real tool call gets its rejection through the matching tool-result
        # channel below. Providers that ignored the tool offer retain the
        # historical text field, so the fallback request is unchanged.
        "repair_observation": None if native_tool_repair else repair_observation,
        "required_top_level_fields": [
            "nodes",
            "action",
        ],
        "node_required_fields": [
            "node_id",
            "operation_family",
            "arguments",
            "authority",
            "evidence_refs",
            "claimed_effects",
        ],
    }
    system_prompt = CFRC_STRONG_SYSTEM_PROMPT
    if service_scoped_prompt:
        # The payload is untouched: what the successor may read does not
        # change. Only the rule blocks that govern services the compiler
        # would reject for this certificate are dropped.
        system_prompt = service_scoped_strong_system_prompt(
            str(certificate.get("service") or "")
        )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
        },
    ]
    if native_tool_repair:
        tool_call_id = "call_cfrc_previous_residual_graph"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": RESIDUAL_GRAPH_TOOL["function"]["name"],
                                "arguments": prior_tool_arguments,
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(
                        {
                            "status": "rejected",
                            "reason": repair_observation,
                            "instruction": (
                                "Submit a repaired residual graph for the same "
                                "frozen contract. Do not execute or restart."
                            ),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ]
        )
    return messages


# The three outcomes the setting allows a successor at a frontier: realize the
# residual, gather the evidence a later graph needs, or report that no legal
# suffix exists. Naming them in the carrier is what keeps the fourth shape --
# a write that does not claim closure -- out of the successor's reach, rather
# than leaving the compiler to reject it after the call is already spent.
RESIDUAL_GRAPH_ACTION_COMPLETE = "complete_residual"
RESIDUAL_GRAPH_ACTION_BIND = "bind_evidence"
RESIDUAL_GRAPH_ACTION_DECLINE = "decline"
RESIDUAL_GRAPH_ACTIONS: tuple[str, ...] = (
    RESIDUAL_GRAPH_ACTION_COMPLETE,
    RESIDUAL_GRAPH_ACTION_BIND,
    RESIDUAL_GRAPH_ACTION_DECLINE,
)
SUCCESSOR_DECLINED_REJECT = "successor_declined"

RESIDUAL_GRAPH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_residual_graph",
        "description": (
            "Submit the residual completion graph. This is the only action "
            "available here: do not run commands and do not explore. Choose "
            "one action: complete_residual with the nodes that realize the "
            "whole remaining residual, bind_evidence with read-only nodes "
            "that only gather bindings, or decline with no nodes when no "
            "admissible residual exists."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "node_id": {"type": "string"},
                            "operation_family": {"type": "string"},
                            "authority": {
                                "type": "string",
                                "enum": ["ell_write", "ell_bounded_local_redo"],
                            },
                            "evidence_refs": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "claimed_effects": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "arguments": {
                                "type": "object",
                                "properties": {
                                    "method": {
                                        "type": "string",
                                        "enum": [
                                            "GET",
                                            "POST",
                                            "PUT",
                                            "PATCH",
                                            "DELETE",
                                        ],
                                    },
                                    "path": {"type": "string"},
                                    "params": {"type": "object"},
                                    "json": {"type": "object"},
                                },
                                "required": ["method", "path"],
                                "additionalProperties": False,
                            },
                        },
                        "required": [
                            "node_id",
                            "operation_family",
                            "authority",
                            "evidence_refs",
                            "claimed_effects",
                            "arguments",
                        ],
                        "additionalProperties": False,
                    },
                },
                "action": {
                    "type": "string",
                    "enum": list(RESIDUAL_GRAPH_ACTIONS),
                    "description": (
                        "complete_residual: the nodes realize the whole "
                        "remaining residual, and a write node requires this "
                        "action. A later node may consume an id an earlier one "
                        "returns as {{n1.id}}, so needing a server-issued id "
                        "is not a reason to leave the residual open. "
                        "bind_evidence: read-only nodes that only gather the "
                        "bindings a later graph will need. "
                        "decline: no admissible residual exists for this "
                        "contract; submit no nodes."
                    ),
                },
                "residual_complete": {
                    "type": "boolean",
                    "description": (
                        "Deprecated in favour of action, and read only when "
                        "action is absent. True means the graph realizes the "
                        "whole remaining residual."
                    ),
                },
            },
            "required": ["nodes", "action"],
            "additionalProperties": False,
        },
    },
}
RESIDUAL_GRAPH_TOOL_CHOICE: dict[str, Any] = {
    "type": "function",
    "function": {"name": "submit_residual_graph"},
}


def residual_graph_tool(service: str) -> dict[str, Any]:
    """Return the proposal carrier narrowed to one public service grammar.

    The compiler accepts a closed set of operation-family names.  Leaving that
    field as arbitrary text lets a proposal spend its only repair round merely
    translating a provider-flavoured endpoint name into the PAPC public
    grammar.  The reached-state contract already fixes the service, so exposing
    exactly that service and its grammar is a projection of existing authority,
    not a benchmark hint or a model-specific correction.
    """

    grammar = SERVICE_OPERATION_GRAMMAR.get(service)
    if not grammar:
        raise ValueError(f"unsupported residual graph service: {service}")
    tool = copy.deepcopy(RESIDUAL_GRAPH_TOOL)
    properties = tool["function"]["parameters"]["properties"]
    properties["nodes"]["items"]["properties"]["operation_family"][
        "enum"
    ] = sorted(grammar)
    return tool


def parse_typed_graph(text: str) -> dict[str, Any]:
    raw = text.strip()
    if re.match(r"^(curl|bash|sh|python|node)\b", raw, re.IGNORECASE):
        raise GraphReject("raw_shell_not_json")
    candidate = _extract_json_object(raw)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise GraphReject(f"invalid_json:{exc.msg}") from exc
    if not isinstance(parsed, Mapping):
        raise GraphReject("graph_not_object")
    graph = _decode_transport_encoded_fields(dict(parsed))
    _reject_forbidden_graph_content(graph)
    return graph


# Structurally typed proposal fields, and the type each one is declared to
# carry. A tool-call transport that re-encodes a nested value hands back the
# right content under the wrong Python type, and the compiler would reject a
# well-formed graph for the encoding rather than for anything the successor
# decided.
_STRUCTURAL_NODE_FIELDS: dict[str, type] = {
    "arguments": dict,
    "evidence_refs": list,
    "claimed_effects": list,
}


def _decode_transport_encoded_fields(graph: dict[str, Any]) -> dict[str, Any]:
    """Undo one layer of transport re-encoding on structurally typed fields.

    The proposal is carried by a native tool call, and the endpoint in front of
    the successor does not enforce the tool's JSON Schema; the runner already
    supplies contract constants for the same reason. One way that shows up is a
    field the schema declares as an array or an object arriving as a string
    holding its JSON. The content is the successor's, unchanged; only its
    encoding is the transport's. Decoding is refused unless the decoded value
    is exactly the declared type, so a genuinely malformed field still reaches
    the compiler and is still rejected on its merits.

    ``residual_complete`` gets the same treatment for a sharper reason: it is
    read for truth, and Python considers the string ``"false"`` true. Left
    alone, a re-encoded refusal to claim closure would read as a claim of it,
    which is the one direction this check must never fail in.
    """

    decoded_candidates = _decoded_as(graph.get("candidates"), list)
    if decoded_candidates is not None:
        graph["candidates"] = decoded_candidates
    candidates = graph.get("candidates")
    containers: list[dict[str, Any]] = [graph]
    if isinstance(candidates, list):
        for index, candidate in enumerate(candidates):
            if isinstance(candidate, Mapping):
                # Rebind the copy into the list, so decoding a candidate's
                # nodes reaches the graph the compiler will actually read.
                candidates[index] = dict(candidate)
                containers.append(candidates[index])
    for container in containers:
        nodes = _decoded_as(container.get("nodes"), list)
        if nodes is not None:
            container["nodes"] = nodes
        nodes = container.get("nodes")
        if not isinstance(nodes, list):
            continue
        for index, node in enumerate(nodes):
            if not isinstance(node, Mapping):
                continue
            decoded_node = dict(node)
            for field, declared in _STRUCTURAL_NODE_FIELDS.items():
                decoded = _decoded_as(decoded_node.get(field), declared)
                if decoded is not None:
                    decoded_node[field] = decoded
            nodes[index] = decoded_node
    flag = graph.get("residual_complete")
    if isinstance(flag, str):
        lowered = flag.strip().lower()
        if lowered in {"true", "false"}:
            graph["residual_complete"] = lowered == "true"
    return _apply_declared_action(graph)


def _apply_declared_action(graph: dict[str, Any]) -> dict[str, Any]:
    """Resolve the declared action into the closure claim the compiler reads.

    The action is the successor's own statement of which of the three outcomes
    it is taking, so it decides the closure claim rather than sitting beside a
    separate boolean that could contradict it. A proposal that names no action
    keeps the older boolean, which is what a provider that ignored the tool and
    answered as text still sends.
    """

    action = graph.get("action")
    if not isinstance(action, str):
        return graph
    action = action.strip().lower()
    if action not in RESIDUAL_GRAPH_ACTIONS:
        return graph
    graph["action"] = action
    if action == RESIDUAL_GRAPH_ACTION_DECLINE:
        raise GraphReject(SUCCESSOR_DECLINED_REJECT)
    graph["residual_complete"] = action == RESIDUAL_GRAPH_ACTION_COMPLETE
    return graph


def _decoded_as(value: Any, declared: type) -> Any | None:
    """Return ``value`` decoded from JSON when it is a re-encoded ``declared``."""

    if not isinstance(value, str):
        return None
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, declared) else None


def _bind_graph_contract_constants(
    graph: Mapping[str, Any], *, service: str
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Bind constants already fixed by the frozen reached-state contract.

    ``service`` is an input to residual completion, not a successor decision.
    Older text proposals may still echo it, so an explicit value is preserved
    and the compiler continues to reject a mismatch.  When it is absent, the
    runner supplies the frozen value instead of spending a repair call asking
    the model to copy a constant through a proxy that does not enforce JSON
    Schema ``required`` fields.
    """

    bound = dict(graph)
    if str(bound.get("service") or "").strip():
        return bound, ()
    bound["service"] = service
    return bound, ("service",)


def _extract_json_object(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise GraphReject("missing_json_object")
    return text[start : end + 1]


def _reject_forbidden_graph_content(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if key_text in FORBIDDEN_GRAPH_KEYS:
                raise GraphReject(f"forbidden_raw_execution_key:{key_text}")
            if key_text in {"task_id", "test_id", "tools_required", "answer"}:
                raise GraphReject(f"hidden_field_key:{key_text}")
            _reject_forbidden_graph_content(item)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _reject_forbidden_graph_content(item)
        return
    if isinstance(value, str):
        lowered = value.lower()
        if re.search(r"(^|\n)\s*(curl|bash|sh|python|node)\b", lowered):
            raise GraphReject("raw_shell_string")
        for marker in HIDDEN_MARKERS:
            if marker.lower() in lowered:
                raise GraphReject(f"hidden_marker:{marker}")


_RESPONSE_REFERENCE = re.compile(r"\{\{\s*(?P<body>[^{}]+?)\s*\}\}")
_RESPONSE_REFERENCE_HEAD = re.compile(r"^([A-Za-z0-9_-]+)")


def _response_reference_node_ids(value: Any) -> tuple[list[str], list[str]]:
    """Collect the node ids a value's ``{{...}}`` references read.

    Returns the referenced ids and any brace expression the runtime resolver
    could not parse. Admission uses both, so a reference that cannot be
    satisfied is refused before the graph reaches an environment rather than
    stopping the runtime partway through a suffix that has already written.
    """

    referenced: list[str] = []
    malformed: list[str] = []

    def walk(item: Any) -> None:
        if isinstance(item, Mapping):
            for entry in item.values():
                walk(entry)
            return
        if isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            for entry in item:
                walk(entry)
            return
        if not isinstance(item, str):
            return
        bodies = _RESPONSE_REFERENCE.findall(item)
        for body in bodies:
            head = _RESPONSE_REFERENCE_HEAD.match(body.strip())
            if head is None:
                malformed.append(body[:78])
            else:
                referenced.append(head.group(1))
        # A brace pair the grammar above cannot even see, because it nests or
        # leaves a brace unbalanced, is malformed too: the runtime fails closed
        # on any surviving pair, so admission must refuse it here.
        if item.count("{{") != len(bodies):
            malformed.append(item[:78])

    walk(value)
    return referenced, malformed


def compile_typed_graph(
    graph: Mapping[str, Any],
    *,
    service: str,
    max_graph_nodes: int,
    max_mutating_nodes: int,
) -> list[CompiledCommand]:
    selected = _select_graph_candidate(graph)
    if selected.get("service") != service:
        raise GraphReject("wrong_service")
    nodes_value = selected.get("nodes")
    if not isinstance(nodes_value, Sequence) or isinstance(
        nodes_value, (str, bytes, bytearray)
    ):
        raise GraphReject("nodes_not_sequence")
    nodes = [dict(node) for node in nodes_value if isinstance(node, Mapping)]
    if len(nodes) != len(nodes_value):
        raise GraphReject("node_not_object")
    if not nodes:
        raise GraphReject("empty_graph")
    if len(nodes) > max_graph_nodes:
        raise GraphReject("graph_node_budget_exceeded")

    compiled: list[CompiledCommand] = []
    seen_node_ids: set[str] = set()
    mutating_count = 0
    for index, node in enumerate(nodes, start=1):
        node_id = str(node.get("node_id") or f"n{index}")
        dependencies = _string_list(node.get("dependencies"))
        if any(dep not in seen_node_ids for dep in dependencies):
            raise GraphReject("dependency_not_closed")
        command = _compile_node(node, service=service, node_id=node_id)
        # Every ``{{nX...}}`` the node will ask the runtime to resolve must name
        # a node that already ran. Declared dependencies do not cover this: the
        # proposal schema no longer carries them, so without this check a
        # reference to a missing or later node is only discovered mid-execution,
        # after the earlier mutating nodes have already touched live state.
        referenced, malformed = _response_reference_node_ids(
            (command.command, _mapping(node.get("arguments")))
        )
        if malformed:
            raise GraphReject("malformed_response_reference")
        unresolvable = [ref for ref in referenced if ref not in seen_node_ids]
        if unresolvable:
            raise GraphReject(f"response_reference_not_closed:{unresolvable[0]}")
        if command.mutating:
            mutating_count += 1
        if mutating_count > max_mutating_nodes:
            raise GraphReject("mutating_node_budget_exceeded")
        compiled.append(command)
        seen_node_ids.add(node_id)
    return compiled


def _select_graph_candidate(graph: Mapping[str, Any]) -> dict[str, Any]:
    candidates = graph.get("candidates")
    if not isinstance(candidates, Sequence) or isinstance(
        candidates, (str, bytes, bytearray)
    ):
        return dict(graph)
    valid_candidates = [dict(item) for item in candidates if isinstance(item, Mapping)]
    if not valid_candidates:
        raise GraphReject("empty_candidate_set")

    def score(candidate: Mapping[str, Any]) -> tuple[int, int, int, int]:
        nodes = [
            dict(node)
            for node in candidate.get("nodes", [])
            if isinstance(node, Mapping)
        ]
        complete = int(bool(candidate.get("claimed_effects") or nodes))
        authority_rank = _authority_rank(str(candidate.get("authority") or "ell_write"))
        mutating = sum(
            1
            for node in nodes
            if str(_mapping(node.get("arguments")).get("method") or "").upper()
            in MUTATING_HTTP_METHODS
        )
        return (complete, -authority_rank, -mutating, -len(nodes))

    return max(valid_candidates, key=score)


def _compile_node(
    node: Mapping[str, Any],
    *,
    service: str,
    node_id: str,
) -> CompiledCommand:
    operation_family = str(node.get("operation_family") or "")
    grammar = SERVICE_OPERATION_GRAMMAR.get(service, {}).get(operation_family)
    if grammar is None:
        raise GraphReject(f"unknown_operation_family:{operation_family}")
    arguments = _mapping(node.get("arguments"))
    if not arguments:
        raise GraphReject("arguments_not_object")
    if any(key in arguments for key in ("command", "bash", "shell", "script")):
        raise GraphReject("raw_execution_argument")
    if service == "calendar" and operation_family == "calendar.events":
        arguments = _calendar_event_arguments_with_default_timezone(arguments)
    if service == "slack" and operation_family == "slack.reactions":
        arguments = _slack_reaction_arguments(arguments)

    method = str(arguments.get("method") or _default_method(grammar)).upper()
    allowed_methods = set(str(item).upper() for item in grammar.get("methods", []))
    if method not in allowed_methods:
        raise GraphReject(f"method_not_allowed:{method}")
    url = _normalize_node_url(arguments, service=service)
    path_marker_text = _path_marker_text(url, service)
    if not any(str(marker) in path_marker_text for marker in grammar["markers"]):
        raise GraphReject("path_marker_not_allowed")

    box_content_upload = service == "box" and _box_content_upload_url(
        method=method,
        url=url,
    )

    if service == "linear":
        query = str(_mapping(arguments.get("json")).get("query") or "")
        if operation_family == "linear.graphql.mutation" and "mutation" not in query:
            raise GraphReject("linear_mutation_missing_mutation_query")
        if operation_family == "linear.graphql.query" and "mutation" in query:
            raise GraphReject("linear_query_contains_mutation")

    mutating = method in MUTATING_HTTP_METHODS and not bool(grammar.get("read_only"))
    authority = str(node.get("authority") or "ell_write")
    command = (
        _box_content_upload_template(method=method, url=url, arguments=arguments)
        if box_content_upload
        else _curl_template(method=method, url=url, arguments=arguments)
    )
    return CompiledCommand(
        node_id=node_id,
        operation_family=operation_family,
        method=method,
        url=url,
        command=command,
        mutating=mutating,
        authority=authority,
    )


def _box_content_upload_url(*, method: str, url: str) -> bool:
    """Detect a Box file-content upload requiring the multipart projection.

    This catches the upload host directly plus the ``/files/content`` upload
    path in any base form, in addition to the version-upload shape. The
    multipart compiler below accepts only new text-file uploads; all other
    shapes remain a fail-closed capability boundary.
    """
    parsed = urlparse(url)
    if parsed.netloc.lower() == "upload.box.com":
        return True
    path = parsed.path
    if "/files/content" in path or "/2.0/files/content" in path:
        return True
    return method == "POST" and path.rstrip("/").endswith("/content")


def _box_content_upload_template(
    *,
    method: str,
    url: str,
    arguments: Mapping[str, Any],
) -> str:
    """Compile a structured Box text upload into a fixed multipart command."""

    if method != "POST" or urlparse(url).path.rstrip("/") != "/2.0/files/content":
        raise GraphReject("box_content_upload_not_compiler_executable")
    if _mapping(arguments.get("params")):
        raise GraphReject("box_content_upload_params_not_supported")
    body = _mapping(arguments.get("json"))
    attributes = _mapping(body.get("attributes"))
    parent = _mapping(attributes.get("parent"))
    file_content = body.get("file")
    if (
        not isinstance(attributes.get("name"), str)
        or not str(attributes.get("name") or "").strip()
        or not isinstance(parent.get("id"), (str, int))
        or not str(parent.get("id") or "").strip()
        or not isinstance(file_content, str)
        or "\x00" in file_content
        or len(file_content) > 100_000
    ):
        raise GraphReject("box_content_upload_arguments_invalid")
    attributes_json = json.dumps(
        attributes,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return " ".join(
        [
            "printf",
            "%s",
            shlex.quote(file_content),
            "|",
            "curl",
            "-sS",
            "-X",
            "POST",
            "https://upload.box.com/api/2.0/files/content",
            "-F",
            shlex.quote(f"attributes={attributes_json}"),
            "-F",
            shlex.quote("file=@-;filename=payload.txt;type=text/plain"),
        ]
    )


def _default_method(grammar: Mapping[str, Any]) -> str:
    methods = [str(method).upper() for method in grammar.get("methods", [])]
    return "GET" if "GET" in methods else methods[0]


def _normalize_node_url(arguments: Mapping[str, Any], *, service: str) -> str:
    raw = str(
        arguments.get("url")
        or arguments.get("path")
        or arguments.get("endpoint")
        or ("/graphql" if service == "linear" else "")
    ).strip()
    if not raw:
        raise GraphReject("missing_url_or_path")
    if service == "slack":
        if raw.startswith("/api/"):
            raw = raw.removeprefix("/api/")
        elif raw.startswith("api/"):
            raw = raw.removeprefix("api/")
        if re.match(r"^[A-Za-z]+(?:\.[A-Za-z0-9_]+)+$", raw):
            raw = "/" + raw
    base = str(baseline.SERVICE_CONFIG[service]["base_url"])
    if service == "calendar":
        raw = _canonical_calendar_path(raw)
    if raw.startswith("http://") or raw.startswith("https://"):
        if service == "calendar":
            raw = raw.replace(
                "https://www.googleapis.com/calendar/v3/calendar/v3/",
                "https://www.googleapis.com/calendar/v3/",
            )
        if not raw.startswith(base):
            raise GraphReject("url_outside_service_base")
        return raw
    if service == "box" and raw.startswith("/2.0/"):
        return "https://api.box.com" + raw
    if service == "linear":
        return base
    if raw.startswith("/"):
        if (
            service == "box"
            and base.endswith("/2.0")
            and raw.startswith(
                (
                    "/files",
                    "/folders",
                    "/comments",
                    "/tasks",
                    "/hubs",
                    "/hub_items",
                    "/collections",
                    "/search",
                    "/users",
                )
            )
        ):
            return base + raw
        return urljoin(base.rstrip("/") + "/", raw.lstrip("/"))
    return urljoin(base.rstrip("/") + "/", raw)


def _canonical_calendar_path(raw: str) -> str:
    if raw.startswith("/calendar/v3/"):
        return raw.removeprefix("/calendar/v3")
    if raw.startswith("calendar/v3/"):
        return raw.removeprefix("calendar/v3")
    if raw.startswith("/v3/"):
        return raw.removeprefix("/v3")
    if raw.startswith("v3/"):
        return raw.removeprefix("v3")
    return raw


def _path_marker_text(url: str, service: str) -> str:
    # A response filter may contain ``?`` inside ``{{...}}``. Mask the whole
    # placeholder before URL parsing so that character cannot turn the suffix
    # (for example ``/events``) into a fake query string during admission.
    parsed = urlparse(re.sub(r"\{\{.*?\}\}", "bound", url))
    if service == "slack":
        return parsed.path.rsplit("/", 1)[-1]
    if service == "linear":
        return "/graphql"
    return parsed.path


def _curl_template(*, method: str, url: str, arguments: Mapping[str, Any]) -> str:
    parts = ["curl", "-sS", "-X", method, shlex.quote(url)]
    params = _mapping(arguments.get("params"))
    body = arguments.get("json")
    if body is None:
        body = arguments.get("body")
    if body is None:
        body = arguments.get("data")
    if method == "GET" and params:
        parts.insert(2, "-G")
        for key, value in sorted(params.items()):
            parts.extend(["--data-urlencode", shlex.quote(f"{key}={value}")])
    elif _is_slack_url(url) and params and (body is None or body == {}):
        parts.extend(
            ["-H", shlex.quote("Content-type: application/x-www-form-urlencoded")]
        )
        for key, value in sorted(params.items()):
            parts.extend(["--data-urlencode", shlex.quote(f"{key}={value}")])
    elif body is not None:
        parts.extend(["-H", shlex.quote("Content-Type: application/json")])
        parts.extend(["-d", shlex.quote(json.dumps(body, separators=(",", ":")))])
    return " ".join(parts)


def _is_slack_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc.endswith("slack.com") and parsed.path.startswith("/api/")


def _slack_reaction_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(arguments)
    params = dict(_mapping(arguments.get("params")))
    if "reaction" in params and "name" not in params:
        params["name"] = params.pop("reaction")
    if "name" in params:
        params["name"] = str(params["name"]).strip(":")
    if params:
        normalized["params"] = params
    return normalized


def _calendar_event_arguments_with_default_timezone(
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = dict(arguments)
    body_key = next(
        (
            key
            for key in ("json", "body", "data")
            if isinstance(arguments.get(key), Mapping)
        ),
        None,
    )
    if body_key is None:
        return normalized
    body = dict(_mapping(arguments.get(body_key)))
    changed = False
    for key in ("start", "end"):
        value = body.get(key)
        if not isinstance(value, Mapping):
            continue
        time_value = dict(value)
        if time_value.get("dateTime") and not time_value.get("timeZone"):
            time_value["timeZone"] = "America/Los_Angeles"
            body[key] = time_value
            changed = True
    if changed:
        normalized[body_key] = body
    return normalized


def _authority_rank(authority: str) -> int:
    ranks = {
        "ell_msg": 0,
        "ell_write": 1,
        "ell_reversible_repair": 2,
        "ell_bounded_repair": 3,
        "ell_bounded_local_redo": 4,
    }
    return ranks.get(authority, 9)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [str(item) for item in value]
    return []


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _usage_dict_sum(items: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    accumulator = UsageAccumulator()
    for item in items:
        accumulator.add_usage(item)
    return accumulator.as_dict()


def _source_binding() -> dict[str, dict[str, str]]:
    paths = {
        "runner": Path(__file__).resolve(),
        "provider_client": Path(str(baseline.__file__)).resolve(),
        "scorer": Path(str(scoring.__file__)).resolve(),
    }
    return {
        label: {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for label, path in paths.items()
    }


def _public_command_digest(command: CompiledCommand) -> dict[str, Any]:
    return {
        "node_id": command.node_id,
        "operation_family": command.operation_family,
        "method": command.method,
        "mutating": command.mutating,
        "authority": command.authority,
        "command_sha256": hashlib.sha256(command.command.encode("utf-8")).hexdigest(),
    }


# The untruncated observation kept for contract evidence, and the budget one
# evidence entry may spend. The cheap transcript keeps its own tighter cap; the
# frozen contract must not lose entities to that cap, because a residual write
# can only bind targets the evidence still shows.
OBSERVATION_FULL_CHAR_CAP = 200_000
EVIDENCE_OBSERVATION_CHAR_CAP = 24_000
EVIDENCE_TOTAL_CHAR_BUDGET = 96_000
_EVIDENCE_DEMOTED_OBSERVATION_CHARS = 2_000
_EVIDENCE_SCALAR_CHAR_CAP = 160
# Keys that carry entity identity or bindable state rather than free text.
# The second compaction level keeps only these, so no identifier is ever
# shortened; free-text keys are the ones dropped.
_EVIDENCE_IDENTITY_KEYS = frozenset(
    {
        "id",
        "identifier",
        "name",
        "summary",
        "title",
        "email",
        "login",
        "type",
        "kind",
        "role",
        "status",
        "etag",
        "primary",
        "hidden",
        "selected",
        "deleted",
        "ts",
        "channel",
        "user",
        "users",
        "items",
        "entries",
        "matches",
        "messages",
        "channels",
        "scope",
        "value",
        "accessRole",
        # Binding-relevant listing state: admission checks read these from
        # compacted evidence, so compaction must not strip them.
        "size",
        "total_count",
        "offset",
        "direction",
        "sort",
    }
)


def _evidence_identity_key(key: str) -> bool:
    lowered = key.lower()
    return (
        lowered in _EVIDENCE_IDENTITY_KEYS
        or lowered.endswith("_id")
        or lowered.startswith("is_")
    )


def _compact_json_value(value: Any, *, identity_only: bool) -> Any:
    if isinstance(value, Mapping):
        # Containers always survive so unknown collection keys keep their
        # rows; identity-only pruning drops scalar free text, never structure
        # and never an identity value.
        return {
            key: _compact_json_value(value[key], identity_only=identity_only)
            for key in value
            if not identity_only
            or isinstance(value[key], (Mapping, list, tuple))
            or _evidence_identity_key(str(key))
        }
    if isinstance(value, (list, tuple)):
        return [
            _compact_json_value(item, identity_only=identity_only) for item in value
        ]
    if isinstance(value, str) and len(value) > _EVIDENCE_SCALAR_CHAR_CAP:
        return value[:_EVIDENCE_SCALAR_CHAR_CAP]
    return value


def compact_observation_for_evidence(
    observation: str,
    char_cap: int = EVIDENCE_OBSERVATION_CHAR_CAP,
) -> str:
    """A listing observation with every row kept and long free text dropped.

    Truncating a listing by characters silently removes the tail entities, and
    a successor that cannot see an entity re-creates it. The first level
    shortens long free-text values; the second keeps only identity and state
    keys, so identifiers are never cut. A payload that still overflows is cut
    at the budget, which keeps strictly more rows than cutting raw text.
    """

    if len(observation) <= char_cap:
        return observation
    text = observation.strip()
    if not text.startswith(("{", "[")):
        return observation[:char_cap]
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return observation[:char_cap]
    compacted = ""
    for identity_only in (False, True):
        compacted = json.dumps(
            _compact_json_value(parsed, identity_only=identity_only),
            separators=(",", ":"),
            ensure_ascii=False,
        )
        if len(compacted) <= char_cap:
            return compacted
    return compacted[:char_cap]


_OBSERVATION_OPEN = "<observation>\n"
_OBSERVATION_CLOSE = "\n</observation>"


def compact_cheap_observation_history(
    messages: list[dict[str, str]],
    *,
    char_cap: int = _EVIDENCE_DEMOTED_OBSERVATION_CHARS,
) -> int:
    """Carry read observations forward in the form the contract records them.

    An observation is read in full on the turn that must act on it. Once the
    prefix has acted, the reached-state contract holds that observation as
    compacted evidence, so re-quoting the raw payload on every later turn
    duplicates in the transcript what the certificate already carries
    canonically. This rewrites every observation except the newest one into
    that same evidence form, in place: same message count, same roles, same
    order, identity and state keys preserved by the shared compaction.
    """

    rewritten = 0
    newest_seen = False
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "")
        if not (
            content.startswith(_OBSERVATION_OPEN)
            and content.endswith(_OBSERVATION_CLOSE)
        ):
            continue
        if not newest_seen:
            newest_seen = True
            continue
        body = content[len(_OBSERVATION_OPEN) : -len(_OBSERVATION_CLOSE)]
        compacted = compact_observation_for_evidence(body, char_cap)
        if len(compacted) < len(body):
            message["content"] = (
                _OBSERVATION_OPEN + compacted + _OBSERVATION_CLOSE
            )
            rewritten += 1
    return rewritten


def execute_bash_observation(
    *,
    bash: Any,
    command: str,
    api_key_env: str,
    timeout_seconds: int,
    observation_chars: int,
) -> dict[str, Any]:
    execution_command, transport_normalizations = (
        normalize_execution_command(command)
    )
    with baseline.sanitized_executor_environment(api_key_env):
        started = time.monotonic()
        result = bash.execute(
            execution_command,
            timeout=int(timeout_seconds),
        )
        latency = round(time.monotonic() - started, 3)
    stdout = str(result.get("stdout") or "")
    stderr = str(result.get("stderr") or "")
    exit_code = int(result.get("exit_code") or 0)
    observation = stdout.strip() or "(empty output)"
    if exit_code != 0:
        observation = f"{stdout}\n[stderr]: {stderr}".strip()
    return {
        "exit_code": exit_code,
        "status": result.get("status"),
        "latency_seconds": latency,
        "observation": observation[:observation_chars],
        "observation_full": observation[:OBSERVATION_FULL_CHAR_CAP],
        "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
        "execution_command_sha256": hashlib.sha256(
            execution_command.encode("utf-8")
        ).hexdigest(),
        "transport_normalizations": transport_normalizations,
        "workspace_path": result.get("workspace_path"),
    }


_CURL_FORM_DATA_FLAGS = {
    "-d",
    "--data",
    "--data-ascii",
    "--data-binary",
    "--data-raw",
    "--data-urlencode",
}
_SHELL_CONTROL_TOKENS = {
    "&",
    "&&",
    "(",
    ")",
    ";",
    "<",
    ">",
    ">>",
    "|",
    "||",
}


def _without_false_form_field(
    value: str,
    *,
    field: str,
) -> tuple[str, bool]:
    """Remove one explicit false field while preserving the other form pairs."""

    stripped = value.lstrip()
    if stripped.startswith(("{", "[")):
        return value, False
    pairs = value.split("&")
    kept: list[str] = []
    removed = False
    for pair in pairs:
        key, separator, raw_value = pair.partition("=")
        if (
            separator
            and unquote_plus(key).strip() == field
            and unquote_plus(raw_value).strip().casefold() == "false"
        ):
            removed = True
            continue
        kept.append(pair)
    return "&".join(kept), removed


def normalize_execution_command(command: str) -> tuple[str, list[str]]:
    """Apply candidate-independent transport equivalences before execution.

    Slack's public ``conversations.create`` API defaults ``is_private`` to
    false. Some compatible sandboxes coerce the non-empty form string
    ``"false"`` to true. Omitting that explicit default is semantically
    equivalent at the public API boundary and avoids depending on a sandbox's
    host-language truthiness.
    """

    classification = classify_action_command(command)
    if (
        classification.service != "slack"
        or classification.path != "/api/conversations.create"
    ):
        return command, []
    try:
        tokens = shlex.split(command)
    except ValueError:
        return command, []
    if any(token in _SHELL_CONTROL_TOKENS for token in tokens):
        return command, []

    normalized: list[str] = []
    changed = False
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in _CURL_FORM_DATA_FLAGS and index + 1 < len(tokens):
            value, removed = _without_false_form_field(
                tokens[index + 1],
                field="is_private",
            )
            if removed:
                changed = True
                if value:
                    normalized.extend((token, value))
                index += 2
                continue
        matched_flag = next(
            (
                flag
                for flag in _CURL_FORM_DATA_FLAGS
                if token.startswith(flag + "=")
            ),
            None,
        )
        if matched_flag is not None:
            value, removed = _without_false_form_field(
                token.removeprefix(matched_flag + "="),
                field="is_private",
            )
            if removed:
                changed = True
                if value:
                    normalized.append(f"{matched_flag}={value}")
                index += 1
                continue
        normalized.append(token)
        index += 1

    if not changed:
        return command, []
    return shlex.join(normalized), [
        "slack_conversations_create_omit_explicit_default_false"
    ]


def _transport_trace_fields(tool: Mapping[str, Any]) -> dict[str, Any]:
    normalizations = list(
        _sequence(tool.get("transport_normalizations"))
    )
    if not normalizations:
        return {}
    return {
        "execution_command_sha256": str(
            tool.get("execution_command_sha256") or ""
        ),
        "transport_normalizations": normalizations,
    }


def _calendar_acl_put_not_found_repair_command(
    command: CompiledCommand,
    observation: str,
    *,
    resolved_command: str | None = None,
) -> CompiledCommand | None:
    if command.operation_family != "calendar.acl" or command.method != "PUT":
        return None
    lowered = observation.lower()
    if "aclnotfound" not in lowered and "aclrule not found" not in lowered:
        return None
    resolved_url = command.url
    if resolved_command:
        resolved_urls = _extract_urls(resolved_command)
        if resolved_urls:
            resolved_url = resolved_urls[0]
    if "/acl/" not in resolved_url:
        return None
    body = _curl_json_body(resolved_command or command.command)
    if body is None:
        return None
    repair_url = resolved_url.rsplit("/acl/", 1)[0] + "/acl"
    repair_command = _curl_template(
        method="POST",
        url=repair_url,
        arguments={"json": body},
    )
    return dataclasses.replace(
        command,
        method="POST",
        url=repair_url,
        command=repair_command,
    )


def _calendar_acl_delete_not_found_is_success(
    command: CompiledCommand,
    observation: str,
    *,
    certificate: Mapping[str, Any] | None = None,
    resolved_command: str | None = None,
) -> bool:
    """Decide whether an ``AclRule not found`` DELETE 404 is idempotent success.

    A 404 is genuinely idempotent only when the visible ACL evidence does NOT
    prove the target rule still exists.  If a visible ACL-list rule matches the
    DELETE target's scope/user (e.g. the emitted rule id is a shortened
    scope-suffix of the full id the ACL list returned), the 404 means the
    emitted rule id was wrong.  That is a real failure that must be handed back
    to the cheap model to repair, so this returns ``False``.
    """
    if command.operation_family != "calendar.acl" or command.method != "DELETE":
        return False
    lowered = observation.lower()
    if "aclnotfound" not in lowered and "aclrule not found" not in lowered:
        return False
    target_url = command.url
    if resolved_command:
        resolved_urls = _extract_urls(resolved_command)
        if resolved_urls:
            target_url = resolved_urls[0]
    cert = certificate or {}
    # Idempotent success needs POSITIVE proof of prior absence: a visible ACL
    # list that was read and does not contain the target rule. Without any
    # visible ACL list we cannot prove absence, so a removal that 404s is a real
    # failure (the emitted rule id was likely wrong) -> hand back to cheap to
    # read the ACL list and delete the correct id.
    if not _visible_acl_rule_ids(cert):
        return False
    if _calendar_acl_visible_rule_matches_delete_target(target_url, cert):
        return False
    return True


def _calendar_acl_rule_id_from_url(url: str | None) -> str:
    if not url:
        return ""
    path = urlparse(url).path
    if "/acl/" not in path:
        return ""
    rule_id, _, _ = path.rsplit("/acl/", 1)[1].partition("?")
    return rule_id.strip("/")


def _calendar_id_from_acl_url(url: str | None) -> str:
    if not url:
        return ""
    match = re.search(r"/calendars/([^/]+)/acl", urlparse(url).path)
    return match.group(1) if match else ""


def _acl_scope_value(rule_id: str) -> str:
    for prefix in ("user:", "group:", "domain:"):
        index = rule_id.find(prefix)
        if index != -1:
            return rule_id[index + len(prefix) :]
    if "@" in rule_id:
        return rule_id
    return ""


def _visible_acl_rule_ids(certificate: Mapping[str, Any]) -> set[str]:
    ids: set[str] = set()
    for event in _sequence(certificate.get("E")):
        observation = str(_mapping(event).get("observation") or "")
        parsed = parse_observation_json(observation)
        if not parsed:
            continue
        items = parsed.get("items")
        if not isinstance(items, Sequence) or isinstance(
            items, (str, bytes, bytearray)
        ):
            continue
        for item in items:
            if not isinstance(item, Mapping):
                continue
            rule_id = item.get("id")
            if not isinstance(rule_id, str) or not rule_id:
                continue
            if _looks_like_acl_rule(item, rule_id):
                ids.add(rule_id)
    return ids


def _looks_like_acl_rule(item: Mapping[str, Any], rule_id: str) -> bool:
    if str(item.get("kind") or "") == "calendar#aclRule":
        return True
    if "scope" in item or "role" in item:
        return True
    return ":" in rule_id


def _calendar_acl_visible_rule_matches_delete_target(
    target_url: str,
    certificate: Mapping[str, Any],
) -> bool:
    emitted_rule_id = _calendar_acl_rule_id_from_url(target_url)
    if not emitted_rule_id:
        return False
    visible_ids = _visible_acl_rule_ids(certificate)
    if not visible_ids:
        return False
    for visible in visible_ids:
        if visible == emitted_rule_id or visible.endswith(":" + emitted_rule_id):
            return True
    cal_id = _calendar_id_from_acl_url(target_url)
    scope_value = _acl_scope_value(emitted_rule_id)
    if cal_id and scope_value:
        for visible in visible_ids:
            if (
                visible.startswith(cal_id + ":")
                and _acl_scope_value(visible) == scope_value
            ):
                return True
    return False


def _calendar_acl_canonicalize_rule_id(
    command_text: str,
    certificate: Mapping[str, Any],
) -> str:
    """Rewrite a calendar ACL command URL to the full visible ACL rule id.

    If the emitted ``.../acl/{ruleId}`` is an unambiguous scope-suffix match of
    exactly one visible ACL-list id (e.g. short ``user:x@test.com`` matches the
    visible ``cal_vault:user:x@test.com``), replace the URL's rule id with the
    full visible id.  Only the URL is rewritten (never the request body) and
    only on a single unambiguous match; otherwise the command is unchanged.
    """
    urls = _extract_urls(command_text)
    if not urls:
        return command_text
    url = urls[0]
    if "/acl/" not in url:
        return command_text
    prefix, _, tail = url.partition("/acl/")
    rule_part, query_sep, query = tail.partition("?")
    emitted = rule_part.strip("/")
    if not emitted:
        return command_text
    visible_ids = _visible_acl_rule_ids(certificate)
    matches = sorted(
        {
            visible
            for visible in visible_ids
            if visible == emitted or visible.endswith(":" + emitted)
        }
    )
    if len(matches) > 1:
        cal_id = _calendar_id_from_acl_url(url)
        if cal_id:
            scoped = [
                visible for visible in matches if visible.startswith(cal_id + ":")
            ]
            if len(scoped) == 1:
                matches = scoped
    if len(matches) != 1:
        return command_text
    full = matches[0]
    if full == emitted:
        return command_text
    new_url = prefix + "/acl/" + full + (("?" + query) if query_sep else "")
    return command_text.replace(url, new_url, 1)


def _curl_json_body(command: str) -> Any | None:
    with contextlib.suppress(ValueError):
        parts = shlex.split(command)
        for index, part in enumerate(parts[:-1]):
            if part == "-d":
                return json.loads(parts[index + 1])
    return None


def run_cfrc_agent(
    *,
    row: baseline.AgentDiffRow,
    bash: Any,
    api_base: str,
    api_key: str,
    api_key_env: str,
    cheap_model: str,
    cheap_reasoning_effort: str,
    strong_model: str,
    strong_reasoning_effort: str,
    max_iterations: int,
    max_model_tokens: int,
    max_strong_model_tokens: int,
    provider_timeout_seconds: float,
    provider_max_attempts: int,
    command_timeout_seconds: int,
    observation_chars: int,
    max_graph_nodes: int,
    max_mutating_nodes: int,
    min_frontier_observations: int,
    max_strong_residual_rounds: int = DEFAULT_MAX_STRONG_RESIDUAL_ROUNDS,
    max_readonly_binding_probes: int = MAX_READONLY_BINDING_PROBES,
    service_scoped_strong_prompt: bool = False,
    compact_cheap_history: bool = False,
    witness_environment_factory: Any | None = None,
) -> dict[str, Any]:
    if cheap_reasoning_effort != DEFAULT_REASONING_EFFORT:
        raise ValueError("Agent-Diff requires cheap reasoning effort=high")
    if strong_reasoning_effort != DEFAULT_STRONG_REASONING_EFFORT:
        raise ValueError("Agent-Diff requires strong reasoning effort=high")
    compacted_observation_messages = 0
    cheap_messages: list[dict[str, str]] = [
        {"role": "system", "content": baseline.build_system_prompt(row.service)},
        {"role": "user", "content": row.question},
    ]
    prefix_events: list[dict[str, Any]] = []
    # Effects the method has already realized. They travel with the
    # contract so a later frontier preserves them instead of redoing them.
    realized_commitments: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    cheap_usage = UsageAccumulator()
    strong_usage = UsageAccumulator()
    cheap_provider_calls = 0
    strong_provider_calls = 0
    cheap_latency = 0.0
    strong_latency = 0.0
    strong_frontier_calls = 0
    allow_additional_frontier = False
    readonly_binding_probes = 0
    frontier_certificate_attempts: Counter[str] = Counter()
    frontier_certificate_provider_calls: Counter[str] = Counter()
    witness_retry_contracts: set[str] = set()
    frontier_locked_after_compiled_write = False
    executed_mutating_count = 0
    done_obligation_guard_used = False
    # What CFRC itself returned, and separately what the harness did after it
    # returned. The method boundary ends at the first of these: a refusal is a
    # refusal whether or not the surrounding run lets the cheap model carry on.
    method_decision = CFRC_NO_FRONTIER
    post_frontier_control = POST_CONTROL_CHEAP
    run_environment = {
        "environment_context": environment_context_evidence(
            reached_state_environment(row.service)
        )
    }

    # Commands whose successful writes shaped the live environment, in order.
    # A replica witness replays exactly these to stand at the same reached
    # state before it tries a candidate suffix.
    witness_replay_commands: list[str] = []

    def _propose_residual(cert: Mapping[str, Any]) -> dict[str, Any]:
        nonlocal strong_provider_calls, strong_latency
        remaining_proposal_calls = _remaining_exact_contract_provider_calls(
            certificate=cert,
            provider_calls=frontier_certificate_provider_calls,
            witness_retry_contracts=witness_retry_contracts,
        )
        if remaining_proposal_calls < 1:
            raise RuntimeError("exact_certificate_provider_call_budget_exhausted")
        common = dict(
            row=row,
            bash=bash,
            api_base=api_base,
            api_key=api_key,
            api_key_env=api_key_env,
            strong_model=strong_model,
            strong_reasoning_effort=strong_reasoning_effort,
            max_strong_model_tokens=max_strong_model_tokens,
            provider_timeout_seconds=provider_timeout_seconds,
            provider_max_attempts=provider_max_attempts,
            command_timeout_seconds=command_timeout_seconds,
            observation_chars=observation_chars,
            certificate=cert,
        )
        witness_bash_factory = None
        if witness_environment_factory is not None and not witness_replay_commands:
            # A replica is only faithful while the live prefix has written
            # nothing: server-generated ids from replayed writes differ
            # between environments, so a candidate bound to a live id
            # would false-fail its witness. With realized writes present,
            # skip the witness rather than judge against a divergent
            # replica.
            def witness_bash_factory():
                return witness_environment_factory(())

        residual_result = propose_compile_execute_residual(
            **common,
            max_graph_nodes=max_graph_nodes,
            max_mutating_nodes=max_mutating_nodes,
            max_proposal_rounds=remaining_proposal_calls,
            service_scoped_strong_prompt=service_scoped_strong_prompt,
            witness_bash_factory=witness_bash_factory,
        )
        _record_exact_contract_provider_result(
            certificate=cert,
            provider_call_count=int(
                residual_result["strong_provider_call_count"]
            ),
            status=str(residual_result.get("status") or ""),
            provider_calls=frontier_certificate_provider_calls,
            witness_retry_contracts=witness_retry_contracts,
        )
        strong_provider_calls += int(residual_result["strong_provider_call_count"])
        strong_latency += float(residual_result["strong_model_latency_seconds"])
        strong_usage.add_usage(_mapping(residual_result["strong_usage"]))
        realized_commitments.extend(
            realized_commitments_from_residual_runtime(residual_result)
        )
        witness_replay_commands.extend(
            _mutating_resolved_commands_from_residual(residual_result)
        )
        return residual_result

    def _terminal_completion(iteration: int) -> dict[str, Any]:
        """Return a checked whole-suffix completion from the terminal cut."""

        return {
            "completed": True,
            "completion_reason": "compiled_suffix_executed",
            "method_decision": CFRC_RESIDUAL_COMPLETED,
            "post_frontier_control": POST_CONTROL_NONE,
            "iterations": iteration,
            "trace": trace,
            "cheap_provider_call_count": cheap_provider_calls,
            "strong_provider_call_count": strong_provider_calls,
            "cheap_model_latency_seconds": round(cheap_latency, 3),
            "strong_model_latency_seconds": round(strong_latency, 3),
            "cheap_usage": cheap_usage.as_dict(),
            "compacted_observation_message_count": (
                compacted_observation_messages
            ),
            "strong_usage": strong_usage.as_dict(),
        }

    def _run_terminal_frontier(iteration: int) -> dict[str, Any] | None:
        """Complete open obligations after cheap declares itself finished.

        A graph that realized no write binds state but discharges no write
        obligation, and it leaves the reached state exactly as the contract
        described it. Whatever its nodes did observe is therefore new evidence
        about that same state: the receipts fold into E and the successor is
        re-run on a newly frozen reached-state contract, whether the graph ran
        to completion or failed partway. Exhausting that bounded probe loop is
        an explicit refusal, never a successful completion.

        Returning ``None`` is the method's failure value at this frontier. The
        cheap model has already declared itself finished, so there is no
        continuation to resume: the row ends at the reached state the prefix
        had already realized, recorded as a refusal rather than a completion.
        Because nothing follows this frontier, proposal budget left unspent
        here is abandoned, so a zero-write outcome may not end it early.
        """

        nonlocal strong_frontier_calls, readonly_binding_probes
        nonlocal executed_mutating_count, method_decision
        nonlocal post_frontier_control

        certificate = build_frontier_certificate(
            row=row,
            prefix_events=prefix_events,
            max_strong_calls=max_strong_residual_rounds,
            realized_commitments=realized_commitments,
            frontier_kind="declared_terminal_open_residual",
        )
        if _remaining_write_obligation_count(certificate) < 1:
            return None

        strong_frontier_calls += 1
        readonly_binding_probes = 0
        residual = _propose_residual(certificate)
        active_terminal_event: dict[str, Any] = {
            "phase": "strong_terminal_residual",
            "iteration": iteration,
            "frontier": {
                "opened": True,
                "frontier_trigger": "cheap_declared_completion",
                "frontier_call_index": strong_frontier_calls,
                "certificate": certificate,
                "residual": residual,
            },
        }
        trace.append(active_terminal_event)

        while True:
            status = str(residual.get("status") or "")
            mutating_command_count = int(
                residual.get("mutating_command_count") or 0
            )
            # Receipts a zero-write graph produced against the live interface.
            # A candidate that was refused before it ran, or that failed on its
            # replica, reached nothing live and so re-grounds nothing.
            grounding_receipts = (
                _prefix_events_from_residual_runtime(residual)
                if mutating_command_count == 0
                else []
            )
            if grounding_receipts:
                prefix_events.extend(grounding_receipts)
                probe_budget_open = (
                    readonly_binding_probes < max_readonly_binding_probes
                )
                if (
                    strong_frontier_calls < max_strong_residual_rounds
                    and probe_budget_open
                ):
                    readonly_binding_probes += 1
                    strong_frontier_calls += 1
                    certificate = build_frontier_certificate(
                        row=row,
                        prefix_events=prefix_events,
                        max_strong_calls=max_strong_residual_rounds,
                        realized_commitments=realized_commitments,
                        frontier_kind="declared_terminal_open_residual",
                    )
                    residual = _propose_residual(certificate)
                    active_terminal_event = {
                        "phase": "strong_terminal_residual",
                        "iteration": iteration,
                        "frontier": {
                            "opened": True,
                            "frontier_trigger": "cheap_declared_completion",
                            "frontier_call_index": strong_frontier_calls,
                            "certificate": certificate,
                            "residual": residual,
                        },
                    }
                    trace.append(active_terminal_event)
                    continue
                if not probe_budget_open:
                    active_terminal_event["frontier"]["readonly_probe_guard"] = {
                        "reason": "readonly_binding_probe_cap_reached",
                        "readonly_binding_probes": readonly_binding_probes,
                        "max_readonly_binding_probes": max_readonly_binding_probes,
                    }
            if status == "compiled_suffix_executed" and mutating_command_count == 0:
                active_terminal_event["refusal"] = {
                    "policy": "open_write_obligations_after_read_only_suffix",
                    "reason": "terminal_binding_probe_budget_exhausted",
                }
                method_decision = CFRC_REFUSED
                post_frontier_control = POST_CONTROL_NONE
                return None

            if status == "compiled_suffix_executed":
                executed_mutating_count += mutating_command_count
                if bool(residual.get("residual_complete")) or SURFACE_WRITE_IS_ATOMIC.get(
                    row.service, False
                ):
                    return _terminal_completion(iteration)
                active_terminal_event["refusal"] = {
                    "policy": "whole_suffix_required",
                    "reason": "terminal_suffix_not_complete",
                }
                method_decision = CFRC_REFUSED
                post_frontier_control = POST_CONTROL_NONE
                return None

            active_terminal_event["refusal"] = {
                "policy": "no_action_under_method_authority",
                "reason": str(residual.get("public_status") or status),
            }
            method_decision = CFRC_REFUSED
            post_frontier_control = POST_CONTROL_NONE
            return None

    for iteration in range(1, max_iterations + 1):
        if compact_cheap_history:
            compacted_observation_messages += compact_cheap_observation_history(
                cheap_messages
            )
        receipt = baseline.call_chat_model(
            api_base=api_base,
            api_key=api_key,
            model=cheap_model,
            reasoning_effort=cheap_reasoning_effort,
            messages=cheap_messages,
            max_tokens=max_model_tokens,
            timeout_seconds=provider_timeout_seconds,
            max_attempts=provider_max_attempts,
        )
        cheap_provider_calls += 1
        cheap_latency += float(receipt["latency_seconds"])
        cheap_usage.add_usage(_mapping(receipt["usage"]))
        text = str(receipt["text"])
        parsed = baseline.parse_react_message(text)
        event: dict[str, Any] = {
            "phase": "cheap_prefix",
            "iteration": iteration,
            "assistant_text": text,
            "assistant_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "provider_attempts": receipt["attempts"],
            "usage": receipt["usage"],
            "diagnostics": receipt["diagnostics"],
        }

        action = parsed["action"]
        done = parsed["done"]
        if action:
            classification = classify_action_command(str(action), row.service)
            event.update(
                {
                    "action": str(action),
                    "action_sha256": hashlib.sha256(
                        str(action).encode("utf-8")
                    ).hexdigest(),
                    "classification": dataclasses.asdict(classification),
                }
            )
            if classification.kind == "unknown":
                # An unclassified shell fragment has no method authority.  In
                # particular, a wrapper such as ``<![CDATA[ ... ]]>`` can
                # report a shell error while commands between the two bad
                # wrapper lines still run.  Never use execution as a parser.
                observation = (
                    "Action rejected before execution: submit exactly one "
                    "supported curl HTTP action, without shell wrappers or "
                    "multi-command scripts. No command was run."
                )
                event.update(
                    {
                        "tool_exit_code": None,
                        "tool_status": "rejected_before_execution",
                        "tool_latency_seconds": 0.0,
                        "observation": observation,
                        "workspace_path": None,
                    }
                )
                trace.append(event)
                cheap_messages.append({"role": "assistant", "content": text})
                cheap_messages.append(
                    {
                        "role": "user",
                        "content": f"<observation>\n{observation}\n</observation>",
                    }
                )
                continue
            if (
                not frontier_locked_after_compiled_write
                and strong_frontier_calls < max_strong_residual_rounds
                and (strong_frontier_calls == 0 or allow_additional_frontier)
                and should_open_frontier(
                    classification,
                    service=row.service,
                    prefix_events=prefix_events,
                    min_observations=min_frontier_observations,
                )
                and _reserve_exact_contract_provider_budget(
                    certificate=(
                        certificate := build_frontier_certificate(
                            row=row,
                            prefix_events=prefix_events,
                            max_strong_calls=max_strong_residual_rounds,
                            realized_commitments=realized_commitments,
                        )
                    ),
                    provider_calls=frontier_certificate_provider_calls,
                    witness_retry_contracts=witness_retry_contracts,
                    event=event,
                )
                and _reserve_frontier_certificate_attempt(
                    certificate=certificate,
                    attempts=frontier_certificate_attempts,
                    event=event,
                )
            ):
                strong_frontier_calls += 1
                allow_additional_frontier = False
                readonly_binding_probes = 0
                residual = _propose_residual(certificate)
                event["frontier"] = {
                    "opened": True,
                    "frontier_call_index": strong_frontier_calls,
                    "certificate": certificate,
                    "residual": residual,
                }
                trace.append(event)
                active_event = event

                # A read-only binding probe re-plans via STRONG (with the probe
                # observations folded into certificate evidence E) before control
                # is ever handed back to the cheap model.  Control returns to
                # cheap only on a write-bearing residual, an abstain/error, or
                # when either the bounded strong round budget or the read-only
                # probe streak budget is exhausted: consecutive read-only
                # residuals on one frontier carry no realized effect, so their
                # streak is capped independently of the total round budget.
                while True:
                    status = residual["status"]
                    mutating_command_count = int(
                        residual.get("mutating_command_count") or 0
                    )
                    if (
                        status == "compiled_suffix_executed"
                        and mutating_command_count == 0
                    ):
                        prefix_events.extend(
                            _prefix_events_from_residual_runtime(residual)
                        )
                        probe_budget_open = (
                            readonly_binding_probes < max_readonly_binding_probes
                        )
                        if (
                            strong_frontier_calls < max_strong_residual_rounds
                            and probe_budget_open
                        ):
                            readonly_binding_probes += 1
                            strong_frontier_calls += 1
                            certificate = build_frontier_certificate(
                                row=row,
                                prefix_events=prefix_events,
                                max_strong_calls=max_strong_residual_rounds,
                                realized_commitments=realized_commitments,
                            )
                            residual = _propose_residual(certificate)
                            active_event = {
                                "phase": "strong_residual_replan",
                                "iteration": iteration,
                                "frontier": {
                                    "opened": True,
                                    "frontier_call_index": strong_frontier_calls,
                                    "certificate": certificate,
                                    "residual": residual,
                                },
                            }
                            trace.append(active_event)
                            continue
                        method_decision = CFRC_REFUSED
                        post_frontier_control = POST_CONTROL_CHEAP
                        if not probe_budget_open:
                            active_event["frontier"]["readonly_probe_guard"] = {
                                "reason": "readonly_binding_probe_cap_reached",
                                "readonly_binding_probes": (
                                    readonly_binding_probes
                                ),
                                "max_readonly_binding_probes": (
                                    max_readonly_binding_probes
                                ),
                            }
                        allow_additional_frontier = (
                            _certificate_allows_additional_frontier(certificate)
                        )
                        cheap_messages.append({"role": "assistant", "content": text})
                        cheap_messages.append(
                            {
                                "role": "user",
                                "content": _read_probe_observation_for_cheap(
                                    residual,
                                    certificate=certificate,
                                ),
                            }
                        )
                        break

                    if status == "compiled_suffix_executed":
                        executed_mutating_count += mutating_command_count
                        frontier_locked_after_compiled_write = True
                        # An explicitly closed, witnessed suffix is terminal.
                        # Letting the suspended cheap model write again can
                        # undo the Checked Execution result. A suffix that ran
                        # writes without closing every open obligation is not a
                        # completion: the method reports it as a refusal and
                        # returns control, carrying what it realized in ``K``.
                        terminal = bool(
                            residual.get("residual_complete")
                        ) or SURFACE_WRITE_IS_ATOMIC.get(row.service, False)
                        method_decision = (
                            CFRC_RESIDUAL_COMPLETED if terminal else CFRC_REFUSED
                        )
                        if not terminal:
                            post_frontier_control = POST_CONTROL_CHEAP
                            cheap_messages.append(
                                {"role": "assistant", "content": text}
                            )
                            cheap_messages.append(
                                {
                                    "role": "user",
                                    "content": _residual_observation_for_cheap(
                                        residual,
                                        certificate=certificate,
                                    ),
                                }
                            )
                            break
                        post_frontier_control = POST_CONTROL_NONE
                        return {
                            "completed": True,
                            "completion_reason": "compiled_suffix_executed",
                            "method_decision": method_decision,
                            "post_frontier_control": post_frontier_control,
                            "iterations": iteration,
                            "trace": trace,
                            "cheap_provider_call_count": cheap_provider_calls,
                            "strong_provider_call_count": strong_provider_calls,
                            "cheap_model_latency_seconds": round(cheap_latency, 3),
                            "strong_model_latency_seconds": round(strong_latency, 3),
                            "cheap_usage": cheap_usage.as_dict(),
                            "compacted_observation_message_count": (
                                compacted_observation_messages
                            ),
                            "strong_usage": strong_usage.as_dict(),
                        }

                    if status in {
                        "compiler_rejected_before_write",
                        "compiler_rejected_after_repair",
                        # A declared decline reaches the same place: no write
                        # authority was taken and the reason is stated.
                        "successor_declined",
                    }:
                        # A candidate that fails admission changes nothing. The
                        # method returns its failure value with the reason
                        # stated and takes no write authority; ending the
                        # incumbent run is a deployment choice outside the
                        # method, so control simply returns to the cheap model.
                        method_decision = CFRC_REFUSED
                        post_frontier_control = POST_CONTROL_CHEAP
                        allow_additional_frontier = (
                            _certificate_allows_additional_frontier(certificate)
                        )
                        active_event["refusal"] = {
                            "policy": "no_action_under_method_authority",
                            "reason": residual.get("public_status"),
                        }
                        cheap_messages.append({"role": "assistant", "content": text})
                        cheap_messages.append(
                            {
                                "role": "user",
                                "content": _rejected_fallback_observation_for_cheap(
                                    str(residual.get("public_status") or "rejected"),
                                    certificate=certificate,
                                ),
                            }
                        )
                        break

                    method_decision = CFRC_REFUSED
                    post_frontier_control = POST_CONTROL_CHEAP
                    if status == "replica_witness_failed_after_repair":
                        # Nothing ran on live state: the candidate failed on
                        # its replica. Report a refusal, not a live failure,
                        # and keep the frontier available for a later write.
                        allow_additional_frontier = (
                            _certificate_allows_additional_frontier(certificate)
                        )
                        active_event["refusal"] = {
                            "policy": "no_action_under_method_authority",
                            "reason": str(residual.get("public_status") or ""),
                        }
                        cheap_messages.append({"role": "assistant", "content": text})
                        cheap_messages.append(
                            {
                                "role": "user",
                                "content": _rejected_fallback_observation_for_cheap(
                                    str(residual.get("public_status") or ""),
                                    certificate=certificate,
                                ),
                            }
                        )
                        break
                    if _read_only_residual_left_state_untouched(residual):
                        # A suffix of reads that failed on live state wrote
                        # nothing, so the reached state the contract describes
                        # is still the reached state. Locking here was the one
                        # case where the lock outlived its reason: it denied a
                        # later write the frontier, and the cheap model then
                        # carried the whole task alone from a state the method
                        # had already read. The receipts those reads did return
                        # are new evidence, so the next write is intercepted
                        # against a contract that now contains them. Any
                        # dispatched write, successful or not, leaves the answer
                        # unknown and still locks below.
                        allow_additional_frontier = (
                            _certificate_allows_additional_frontier(certificate)
                        )
                        active_event["refusal"] = {
                            "policy": "no_action_under_method_authority",
                            "reason": str(residual.get("public_status") or ""),
                            "read_only_residual_kept_frontier": True,
                        }
                    else:
                        frontier_locked_after_compiled_write = True
                    cheap_messages.append({"role": "assistant", "content": text})
                    cheap_messages.append(
                        {
                            "role": "user",
                            "content": _residual_failure_observation_for_cheap(
                                residual,
                                certificate=certificate,
                            ),
                        }
                    )
                    break

                continue

            tool = execute_bash_observation(
                bash=bash,
                command=str(action),
                api_key_env=api_key_env,
                timeout_seconds=command_timeout_seconds,
                observation_chars=observation_chars,
            )
            event.update(
                {
                    "tool_exit_code": tool["exit_code"],
                    "tool_status": tool["status"],
                    "tool_latency_seconds": tool["latency_seconds"],
                    "observation": tool["observation"],
                    "workspace_path": tool["workspace_path"],
                    **_transport_trace_fields(tool),
                }
            )
            prefix_events.append(
                _prefix_event_from_tool(
                    action=str(action),
                    classification=classification,
                    tool=tool,
                )
            )
            if classification.kind == "mutating":
                executed_mutating_count += 1
                prefix_observation_full = str(
                    tool.get("observation_full") or tool.get("observation") or ""
                )
                if int(tool["exit_code"]) == 0 and not semantic_tool_failure(
                    prefix_observation_full
                ):
                    # A write the cheap prefix already performed is part of the
                    # reached state; a later frontier must carry it in ``K`` so
                    # the successor neither redoes it nor counts it as open.
                    # A compound action can perform more than one write. Each
                    # of them is part of the reached state, so each is carried
                    # in ``K``; recording only the action as a whole let the
                    # successor redo a write the prefix had already made.
                    written = [
                        part
                        for part in classification.sub_requests
                        if part.kind == "mutating"
                    ] or [classification]
                    for part in written:
                        realized_commitments.append(
                            {
                                "operation_family": part.operation_family,
                                "method": ",".join(part.methods),
                                "url": part.url,
                                "command_sha256": part.command_sha256,
                                "receipt": prefix_observation_full[:600],
                            }
                        )
                    witness_replay_commands.append(str(action))
            trace.append(event)
            cheap_messages.append({"role": "assistant", "content": text})
            cheap_messages.append(
                {
                    "role": "user",
                    "content": f"<observation>\n{tool['observation']}\n</observation>",
                }
            )
            continue

        if done:
            if (
                not done_obligation_guard_used
                and executed_mutating_count == 0
                and _visible_write_obligation_requires_action(row)
            ):
                done_obligation_guard_used = True
                event["done_guard"] = "visible_write_obligation_no_mutation"
                trace.append(event)
                cheap_messages.append({"role": "assistant", "content": text})
                cheap_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "<observation>\n"
                            "The visible request contains create, update, post, "
                            "invite, membership, reaction, delete, or removal "
                            "requirements, but no mutating API action has been "
                            "executed in this environment yet. Continue the task "
                            "and only use <done> after the requested visible writes "
                            "are complete.\n"
                            "Original visible request:\n"
                            f"{row.question}\n"
                            f"{_cheap_continuation_guidance(run_environment)}"
                            "</observation>"
                        ),
                    }
                )
                continue
            event["done"] = str(done)
            trace.append(event)
            if (
                method_decision == CFRC_NO_FRONTIER
                and strong_frontier_calls < max_strong_residual_rounds
            ):
                terminal_result = _run_terminal_frontier(iteration)
                if terminal_result is not None:
                    return terminal_result
            return {
                "completed": True,
                "completion_reason": "cheap_done",
                "method_decision": method_decision,
                "post_frontier_control": post_frontier_control,
                "iterations": iteration,
                "trace": trace,
                "cheap_provider_call_count": cheap_provider_calls,
                "strong_provider_call_count": strong_provider_calls,
                "cheap_usage": cheap_usage.as_dict(),
                "compacted_observation_message_count": (
                    compacted_observation_messages
                ),
                "strong_usage": strong_usage.as_dict(),
                "cheap_model_latency_seconds": round(cheap_latency, 3),
                "strong_model_latency_seconds": round(strong_latency, 3),
            }

        event["format_error"] = "missing_action_or_done"
        trace.append(event)
        cheap_messages.append({"role": "assistant", "content": text})
        cheap_messages.append(
            {
                "role": "user",
                "content": (
                    "Please respond using either <action>one bash command</action> "
                    "or <done>brief summary</done>."
                ),
            }
        )

    return {
        "completed": False,
        "completion_reason": "max_iterations_exhausted",
        "method_decision": method_decision,
        "post_frontier_control": post_frontier_control,
        "exhausted_iterations": True,
        "iterations": max_iterations,
        "trace": trace,
        "cheap_provider_call_count": cheap_provider_calls,
        "strong_provider_call_count": strong_provider_calls,
        "cheap_usage": cheap_usage.as_dict(),
        "compacted_observation_message_count": (
            compacted_observation_messages
        ),
        "strong_usage": strong_usage.as_dict(),
        "cheap_model_latency_seconds": round(cheap_latency, 3),
        "strong_model_latency_seconds": round(strong_latency, 3),
    }


def _residual_observation_for_cheap(
    residual: Mapping[str, Any],
    *,
    certificate: Mapping[str, Any],
) -> str:
    observations: list[str] = []
    for attempt in residual.get("attempts", []):
        attempt_map = _mapping(attempt)
        for runtime_event in attempt_map.get("runtime", []):
            runtime_map = _mapping(runtime_event)
            observations.append(str(runtime_map.get("observation") or ""))
    joined = (
        "\n\n".join(observations) if observations else str(residual["public_status"])
    )
    return (
        "<observation>\n"
        "A compiler-selected CFRC residual suffix has been executed in the same "
        "environment. Use the visible result below to verify any remaining work "
        "and finish. Do not redo completed writes unless verification shows a "
        "remaining mismatch.\n"
        f"{_cheap_continuation_guidance(certificate)}"
        f"{joined}\n"
        "</observation>"
    )


def _read_probe_observation_for_cheap(
    residual: Mapping[str, Any],
    *,
    certificate: Mapping[str, Any],
) -> str:
    observations: list[str] = []
    for attempt in residual.get("attempts", []):
        attempt_map = _mapping(attempt)
        for runtime_event in attempt_map.get("runtime", []):
            runtime_map = _mapping(runtime_event)
            observations.append(str(runtime_map.get("observation") or ""))
    joined = (
        "\n\n".join(observations) if observations else str(residual["public_status"])
    )
    return (
        "<observation>\n"
        "A CFRC read-only binding probe has been executed in the same environment. "
        "No CFRC writes were executed. Use the visible read results below to bind "
        "IDs, times, users, channels, or files, then continue the original task. "
        "Do not use <done> until every requested write, permission change, invite, "
        "rename, delete, update, or message has actually been completed.\n"
        f"{_cheap_continuation_guidance(certificate)}"
        f"{joined}\n"
        "</observation>"
    )


def _residual_failure_observation_for_cheap(
    residual: Mapping[str, Any],
    *,
    certificate: Mapping[str, Any],
) -> str:
    observations: list[str] = []
    for attempt in residual.get("attempts", []):
        attempt_map = _mapping(attempt)
        compiler_reject = attempt_map.get("compiler_reject")
        if compiler_reject:
            observations.append(f"compiler_reject: {compiler_reject}")
        for runtime_event in attempt_map.get("runtime", []):
            runtime_map = _mapping(runtime_event)
            observations.append(str(runtime_map.get("observation") or ""))
    joined = (
        "\n\n".join(observations) if observations else str(residual["public_status"])
    )
    return (
        "<observation>\n"
        "CFRC residual attempt did not complete cleanly. Use the visible failure "
        "below to continue from the current environment; if the suspended write "
        "is still needed, repair or perform the smallest remaining API action "
        "yourself.\n"
        f"{_cheap_continuation_guidance(certificate)}"
        f"{joined}\n"
        "</observation>"
    )


def _rejected_fallback_observation_for_cheap(
    observation: str,
    *,
    certificate: Mapping[str, Any],
) -> str:
    # What the refusal actually did, stated plainly. Reporting the suspended
    # action as executed leaves the cheap model believing a write happened that
    # never ran, so it moves on and the request keeps an obligation nobody
    # discharged.
    return (
        "<observation>\n"
        f"{observation}\n\n"
        "The proposed residual was rejected before any write ran, and the "
        "suspended action was not executed either. Nothing in the environment "
        "changed. Continue from the state you last observed, perform the write "
        "yourself if the request still needs it, and only use <done> after all "
        "requested effects are complete.\n"
        f"{_cheap_continuation_guidance(certificate)}"
        "</observation>"
    )


def _cheap_continuation_guidance(certificate: Mapping[str, Any]) -> str:
    """State what the contract recorded, not how to satisfy the request.

    Only request-independent reached-state environment facts are passed on.
    Listing receipts cannot establish whether the retained request requires
    exhaustive inspection, so they contribute no continuation instruction.
    """

    lines: list[str] = []
    context = _mapping(certificate.get("environment_context"))
    observed_at = str(context.get("observed_at") or "")
    if observed_at:
        lines.append(
            "Environment context stated by the service at handoff: "
            f"{observed_at} Resolve relative and temporal references against "
            "it rather than the real wall clock."
        )
    return "".join(f"{line}\n" for line in lines)


def _visible_write_obligation_requires_action(row: baseline.AgentDiffRow) -> bool:
    """Whether the visible request still owes at least one state change.

    Counted with the shared mutation vocabulary rather than a per-service
    phrase list, so the guard fires on the same evidence everywhere.
    """

    return _residual_write_obligation_count({"req": row.question}) > 0


def semantic_tool_failure(observation: str) -> bool:
    stripped = observation.strip()
    lowered = stripped.lower()
    if not stripped:
        return False
    if lowered in {"not found", "not_found"}:
        return True
    if "not_found" in lowered:
        return True
    with contextlib.suppress(Exception):
        parsed = json.loads(stripped)
        if isinstance(parsed, Mapping):
            if parsed.get("ok") is False:
                return True
            if parsed.get("errors"):
                return True
            if parsed.get("error") and parsed.get("ok") is not True:
                return True
    return False


def _keeps_the_enclosing_quote_closed(rendered: str) -> str:
    """Render a substituted value so it cannot close the quote around it.

    The command was assembled by :func:`_curl_template` with ``shlex.quote``,
    so every placeholder sits inside single quotes. Substituting a receipt
    value verbatim therefore hands the shell whatever apostrophes that receipt
    happened to contain: a Slack message reading ``pricing isn't terrible``
    closes the quote early and the node dies with ``unexpected EOF while
    looking for matching '``. That refusal is the method failing on its own
    renderer rather than on the residual, and it was observed on five rows of
    the recorded old-pool run. Escaping the apostrophe the way ``shlex`` itself
    does keeps the projection quoting-closed by construction.
    """

    return rendered.replace("'", "'\"'\"'")


def resolve_response_placeholders(
    text: str,
    node_outputs: Mapping[str, Mapping[str, Any]],
    *,
    shell_quoted: bool = False,
) -> tuple[str, list[str]]:
    """Replace every ``{{...}}`` reference with a prior node's response value.

    Any brace expression that does not parse or does not resolve is reported
    as unresolved, so a malformed reference stops the runtime instead of
    reaching the shell as literal braces.

    ``shell_quoted`` says the text is a rendered command whose placeholders sit
    inside single quotes, so substituted values are escaped for that context.
    Structured payloads resolve with it off, because there the value is a datum
    rather than a shell token and escaping would corrupt it.
    """

    unresolved: list[str] = []

    def _render(value: str) -> str:
        return _keeps_the_enclosing_quote_closed(value) if shell_quoted else value

    def replace_json_value(match: re.Match[str]) -> str:
        body = match.group("body").replace("'\"'\"'", "'")
        found, value = _evaluate_placeholder(body, node_outputs)
        if not found:
            unresolved.append(match.group(0))
            return match.group(0)
        return _render(json.dumps(value, separators=(",", ":")))

    def replace(match: re.Match[str]) -> str:
        body = match.group("body").replace("'\"'\"'", "'")
        found, value = _evaluate_placeholder(body, node_outputs)
        if not found:
            unresolved.append(match.group(0))
            return match.group(0)
        return _render(str(value))

    resolved = re.sub(
        r'"\{\{\s*(?P<body>[^{}]+?)\s*\}\}"',
        replace_json_value,
        text,
    )
    resolved = re.sub(
        r"\{\{\s*(?P<body>[^{}]+?)\s*\}\}",
        replace,
        resolved,
    )
    # Fail closed on any surviving brace pair, whatever its body looks like:
    # a reference the grammar above could not even parse (nested braces,
    # unbalanced quotes) must stop the runtime, not reach the shell.
    for match in re.finditer(r"\{\{", resolved):
        closing = resolved.find("}}", match.start())
        if 0 <= closing <= match.start() + 78:
            fragment = resolved[match.start() : closing + 2]
        else:
            fragment = resolved[match.start() : match.start() + 80]
        unresolved.append(fragment)
    deduplicated: list[str] = []
    for reference in unresolved:
        stripped = reference.strip('"')
        if stripped not in deduplicated:
            deduplicated.append(stripped)
    return resolved, deduplicated


def resolve_structured_response_placeholders(
    value: Any,
    node_outputs: Mapping[str, Mapping[str, Any]],
) -> tuple[Any, list[str]]:
    """Resolve response references without flattening typed JSON values.

    The shell command and the receipt checker must read the same write.  The
    command resolver substitutes placeholders before execution; resolving the
    structured arguments here keeps receipt validation from comparing a
    successful echo against the literal ``{{node.field}}`` source text.
    """

    if isinstance(value, Mapping):
        resolved_mapping: dict[str, Any] = {}
        unresolved: list[str] = []
        for key, item in value.items():
            resolved_item, item_unresolved = resolve_structured_response_placeholders(
                item,
                node_outputs,
            )
            resolved_mapping[str(key)] = resolved_item
            unresolved.extend(item_unresolved)
        return resolved_mapping, list(dict.fromkeys(unresolved))
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        resolved_sequence: list[Any] = []
        unresolved = []
        for item in value:
            resolved_item, item_unresolved = resolve_structured_response_placeholders(
                item,
                node_outputs,
            )
            resolved_sequence.append(resolved_item)
            unresolved.extend(item_unresolved)
        return resolved_sequence, list(dict.fromkeys(unresolved))
    if not isinstance(value, str):
        return value, []

    whole = re.fullmatch(r"\{\{\s*(?P<body>[^{}]+?)\s*\}\}", value)
    if whole:
        found, resolved_value = _evaluate_placeholder(
            whole.group("body").replace("'\"'\"'", "'"),
            node_outputs,
        )
        if found:
            return resolved_value, []
        return value, [value]
    return resolve_response_placeholders(value, node_outputs)


_PLACEHOLDER_FIELD = re.compile(r"^\.([A-Za-z0-9_@-]+)")
_PLACEHOLDER_INDEX = re.compile(r"^\[([0-9]+)\]")
_PLACEHOLDER_FILTER = re.compile(r"^\[\?\((.*?)\)\]")
_FILTER_CONDITION = re.compile(
    r"^@((?:\.[A-Za-z0-9_@-]+)+)\s*==\s*(?:'([^']*)'|\"([^\"]*)\"|([0-9]+))$"
)


def _evaluate_placeholder(
    body: str,
    node_outputs: Mapping[str, Mapping[str, Any]],
) -> tuple[bool, Any]:
    """Evaluate one ``{{...}}`` body against prior node outputs.

    Returns ``(True, value)`` only when the whole expression parses and every
    step resolves to a present, non-null value.
    """

    cursor = body.strip()
    head = re.match(r"^([A-Za-z0-9_-]+)", cursor)
    if not head:
        return False, None
    node_id = head.group(1)
    if node_id not in node_outputs:
        return False, None
    current: Any = node_outputs[node_id]
    cursor = cursor[head.end() :]
    while cursor:
        field = _PLACEHOLDER_FIELD.match(cursor)
        if field:
            if (
                field.group(1) == "length"
                and not isinstance(current, (str, bytes, bytearray, Mapping))
                and isinstance(current, Sequence)
            ):
                current = len(current)
                cursor = cursor[field.end() :]
                continue
            if not isinstance(current, Mapping):
                return False, None
            current = current.get(field.group(1))
            if current is None:
                return False, None
            cursor = cursor[field.end() :]
            continue
        index = _PLACEHOLDER_INDEX.match(cursor)
        if index:
            if isinstance(current, (str, bytes, bytearray)) or not isinstance(
                current, Sequence
            ):
                return False, None
            position = int(index.group(1))
            if position >= len(current):
                return False, None
            current = current[position]
            cursor = cursor[index.end() :]
            continue
        list_filter = _PLACEHOLDER_FILTER.match(cursor)
        if list_filter:
            if isinstance(current, (str, bytes, bytearray)) or not isinstance(
                current, Sequence
            ):
                return False, None
            matched = _filter_matches(current, list_filter.group(1))
            if matched is None or not matched:
                return False, None
            cursor = cursor[list_filter.end() :]
            follow_index = _PLACEHOLDER_INDEX.match(cursor)
            if follow_index:
                position = int(follow_index.group(1))
                if position >= len(matched):
                    return False, None
                current = matched[position]
                cursor = cursor[follow_index.end() :]
                continue
            current = matched[0]
            continue
        return False, None
    return True, current


_FILTER_CLAUSE = re.compile(
    r"^@((?:\.[A-Za-z0-9_@-]+)+)\s*==\s*"
    r"(?:'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"|([0-9]+))\s*"
)


def _filter_matches(items: Sequence[Any], conditions: str) -> list[Any] | None:
    """List items whose fields equal every ``@.path=='value'`` condition.

    Conditions are consumed lexically, so a quoted literal may contain the
    ``&&`` separator; malformed condition text yields ``None`` rather than a
    partial parse.
    """

    parsed: list[tuple[str, str]] = []
    cursor = conditions.strip()
    while cursor:
        clause = _FILTER_CLAUSE.match(cursor)
        if not clause:
            return None
        path = clause.group(1).lstrip(".")
        expected = next(
            group for group in clause.groups()[1:] if group is not None
        )
        parsed.append((path, expected))
        cursor = cursor[clause.end() :]
        if cursor.startswith("&&"):
            cursor = cursor[2:].lstrip()
        elif cursor:
            return None
    if not parsed:
        return None
    return [
        item
        for item in items
        if isinstance(item, Mapping)
        and all(
            _filter_value_matches(
                _lookup_response_path(item, path),
                expected,
            )
            for path, expected in parsed
        )
    ]


def _filter_value_matches(actual: Any, expected: str) -> bool:
    """Match exact scalars plus ISO datetimes denoting the same instant."""

    if str(actual) == expected:
        return True
    if not isinstance(actual, str) or "T" not in actual or "T" not in expected:
        return False
    try:
        actual_datetime = datetime.fromisoformat(actual.replace("Z", "+00:00"))
        expected_datetime = datetime.fromisoformat(
            expected.replace("Z", "+00:00")
        )
    except ValueError:
        return False
    if actual_datetime.tzinfo is None or expected_datetime.tzinfo is None:
        return False
    return actual_datetime == expected_datetime


def _lookup_response_path(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if not part:
            continue
        if isinstance(current, Mapping):
            current = current.get(part)
            continue
        if isinstance(current, Sequence) and not isinstance(
            current, (str, bytes, bytearray)
        ):
            if part == "length":
                current = len(current)
                continue
            with contextlib.suppress(ValueError, IndexError):
                current = current[int(part)]
                continue
        return None
    return current


def parse_observation_json(observation: str) -> dict[str, Any] | None:
    text = observation.strip()
    starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
    if not starts:
        return None
    snippet = text[min(starts) :]
    with contextlib.suppress(Exception):
        parsed = json.loads(snippet)
        if isinstance(parsed, Mapping):
            return dict(parsed)
    return None


def _prefix_event_from_tool(
    *,
    action: str,
    classification: CommandClassification,
    tool: Mapping[str, Any],
) -> dict[str, Any]:
    observation = str(tool.get("observation") or "")
    event = {
        "action": action,
        "action_sha256": hashlib.sha256(action.encode("utf-8")).hexdigest(),
        "classification_kind": classification.kind,
        "operation_family": classification.operation_family,
        "tool_exit_code": int(tool.get("exit_code") or 0),
        "observation": observation,
        "observation_full": str(tool.get("observation_full") or observation),
    }
    method = next(
        (
            candidate
            for candidate in classification.methods
            if candidate in READ_HTTP_METHODS
        ),
        None,
    )
    if classification.kind == "read" and method and classification.path:
        event["request"] = {
            "method": method,
            "path": classification.path,
        }
    return event


def _prefix_events_from_residual_runtime(
    residual: Mapping[str, Any],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for attempt in residual.get("attempts", []):
        attempt_map = _mapping(attempt)
        for runtime in attempt_map.get("runtime", []):
            runtime_map = _mapping(runtime)
            if bool(runtime_map.get("mutating")):
                continue
            command_sha256 = str(runtime_map.get("command_sha256") or "")
            event = {
                "action": f"compiled_read:{command_sha256}",
                "action_sha256": command_sha256,
                "classification_kind": "read",
                "operation_family": str(runtime_map.get("operation_family") or ""),
                "tool_exit_code": int(runtime_map.get("tool_exit_code") or 0),
                "observation": str(runtime_map.get("observation") or ""),
                "observation_full": str(
                    runtime_map.get("observation_full")
                    or runtime_map.get("observation")
                    or ""
                ),
            }
            resolved_command = str(runtime_map.get("resolved_command") or "")
            if resolved_command:
                classification = classify_action_command(resolved_command)
                method = next(
                    (
                        candidate
                        for candidate in classification.methods
                        if candidate in READ_HTTP_METHODS
                    ),
                    None,
                )
                if method and classification.path:
                    event["request"] = {
                        "method": method,
                        "path": classification.path,
                    }
            events.append(event)
    return events


def realized_commitments_from_residual_runtime(
    residual: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Effects the method already realized, as commitments a later proposal keeps.

    A write that succeeded is part of the reached state from the moment it
    returns. Dropping it means the next frontier reads a contract that never
    mentions it and proposes the same write again, which is how one requested
    calendar becomes three.
    """

    commitments: list[dict[str, Any]] = []
    for attempt in residual.get("attempts", []):
        for runtime in _mapping(attempt).get("runtime", []):
            runtime_map = _mapping(runtime)
            if not bool(runtime_map.get("mutating")):
                continue
            if int(runtime_map.get("tool_exit_code") or 0) != 0:
                continue
            if bool(runtime_map.get("semantic_failure")):
                # An interface that answered ok:false wrote nothing; carrying
                # it as a commitment would tell the next frontier the effect
                # exists when no receipt says so.
                continue
            commitments.append(
                {
                    "operation_family": str(runtime_map.get("operation_family") or ""),
                    "method": str(runtime_map.get("method") or ""),
                    "url": runtime_map.get("url"),
                    "command_sha256": str(runtime_map.get("command_sha256") or ""),
                    "receipt": str(runtime_map.get("observation") or "")[:600],
                }
            )
    return commitments


def realized_commitments_from_prefix_events(
    prefix_events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Writes the cheap prefix already realized, as contract ``K`` entries.

    A frontier that opens after an executed write must carry that write, or
    the successor re-proposes it and obligation accounting keeps it open.
    """

    commitments: list[dict[str, Any]] = []
    for event in prefix_events:
        event_map = _mapping(event)
        if event_map.get("classification_kind") != "mutating":
            continue
        if int(event_map.get("tool_exit_code") or 0) != 0:
            continue
        observation = str(
            event_map.get("observation_full") or event_map.get("observation") or ""
        )
        if semantic_tool_failure(observation):
            continue
        commitments.append(
            {
                "operation_family": event_map.get("operation_family"),
                "method": "",
                "url": None,
                "command_sha256": str(event_map.get("action_sha256") or ""),
                "receipt": observation[:600],
            }
        )
    return commitments


def _mutating_resolved_commands_from_residual(
    residual: Mapping[str, Any],
) -> list[str]:
    """Resolved commands of the writes this residual realized, in order."""

    commands: list[str] = []
    for attempt in residual.get("attempts", []):
        for runtime in _mapping(attempt).get("runtime", []):
            runtime_map = _mapping(runtime)
            if not bool(runtime_map.get("mutating")):
                continue
            if int(runtime_map.get("tool_exit_code") or 0) != 0:
                continue
            if bool(runtime_map.get("semantic_failure")):
                continue
            resolved = str(runtime_map.get("resolved_command") or "")
            if resolved:
                commands.append(resolved)
    return commands


def _certificate_allows_additional_frontier(
    certificate: Mapping[str, Any],
) -> bool:
    """Whether the contract still reports residual work worth re-proposing.

    Listing receipts alone cannot establish request-scoped residual work.
    This legacy route can only use its explicit residual-write projection,
    net of the writes ``K`` already carries receipts for.
    """

    return _remaining_write_obligation_count(certificate) >= 2


def _certificate_sha256(certificate: Mapping[str, Any]) -> str:
    payload = json.dumps(
        certificate,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _remaining_exact_contract_provider_calls(
    *,
    certificate: Mapping[str, Any],
    provider_calls: Counter[str],
    witness_retry_contracts: set[str],
) -> int:
    """Return the remaining samples justified by this exact contract.

    One initial proposal and one feedback-conditioned repair are the ordinary
    budget. A replica-witness failure can earn one final independent sample
    because it supplied execution evidence that compilation alone could not.
    Reads change ``E`` and therefore receive a fresh budget under a new hash.
    """

    certificate_sha256 = _certificate_sha256(certificate)
    limit = (
        MAX_EXACT_CONTRACT_PROVIDER_CALLS_AFTER_WITNESS_FAILURE
        if certificate_sha256 in witness_retry_contracts
        else MAX_EXACT_CONTRACT_PROVIDER_CALLS
    )
    return max(0, limit - provider_calls[certificate_sha256])


def _record_exact_contract_provider_result(
    *,
    certificate: Mapping[str, Any],
    provider_call_count: int,
    status: str,
    provider_calls: Counter[str],
    witness_retry_contracts: set[str],
) -> None:
    certificate_sha256 = _certificate_sha256(certificate)
    provider_calls[certificate_sha256] += max(0, int(provider_call_count))
    if status in {"compiler_rejected_before_write", "successor_declined"}:
        # These statuses are reserved for sticky capability-boundary rejects
        # and for a successor that declared no admissible residual. Re-sampling
        # the byte-identical contract cannot change compiler support, and it
        # cannot change a decline either, so let cheap continue until new
        # receipts change the certificate instead of paying for the same
        # refusal again.
        provider_calls[certificate_sha256] = max(
            provider_calls[certificate_sha256],
            MAX_EXACT_CONTRACT_PROVIDER_CALLS,
        )
    if status.startswith("replica_witness_failed"):
        witness_retry_contracts.add(certificate_sha256)


def _reserve_exact_contract_provider_budget(
    *,
    certificate: Mapping[str, Any],
    provider_calls: Counter[str],
    witness_retry_contracts: set[str],
    event: dict[str, Any],
) -> bool:
    remaining = _remaining_exact_contract_provider_calls(
        certificate=certificate,
        provider_calls=provider_calls,
        witness_retry_contracts=witness_retry_contracts,
    )
    if remaining:
        return True
    certificate_sha256 = _certificate_sha256(certificate)
    max_provider_calls = (
        MAX_EXACT_CONTRACT_PROVIDER_CALLS_AFTER_WITNESS_FAILURE
        if certificate_sha256 in witness_retry_contracts
        else MAX_EXACT_CONTRACT_PROVIDER_CALLS
    )
    event["frontier_guard"] = {
        "reason": "exact_certificate_provider_call_cap",
        "certificate_sha256": certificate_sha256,
        "prior_provider_calls": provider_calls[certificate_sha256],
        "max_provider_calls": max_provider_calls,
        "control": "resume_cheap_without_reproposal",
    }
    return False


def _reserve_frontier_certificate_attempt(
    *,
    certificate: Mapping[str, Any],
    attempts: Counter[str],
    event: dict[str, Any],
) -> bool:
    """Reserve one of two samples for an exact reached-state contract."""

    certificate_sha256 = _certificate_sha256(certificate)
    prior_attempts = attempts[certificate_sha256]
    if prior_attempts >= MAX_IDENTICAL_FRONTIER_ATTEMPTS:
        event["frontier_guard"] = {
            "reason": "identical_certificate_attempt_cap",
            "certificate_sha256": certificate_sha256,
            "prior_attempts": prior_attempts,
            "max_identical_attempts": MAX_IDENTICAL_FRONTIER_ATTEMPTS,
            "control": "resume_cheap_without_reproposal",
        }
        return False
    attempts[certificate_sha256] += 1
    return True


def _is_sticky_compiler_capability_reject(reason: str) -> bool:
    return reason in STICKY_COMPILER_CAPABILITY_REJECTS


def _declared_terminal_decline(
    *,
    rejection: str,
    certificate: Mapping[str, Any] | None,
    graph: Any,
) -> bool:
    """Whether an empty graph on a terminal frontier is a decline, not a reject.

    ``{"nodes": [], "residual_complete": true}`` on a declared-terminal frontier
    is the successor saying it holds no residual for this contract. Re-asking it
    on the byte-identical certificate is asking the same question twice: the
    only new input a repair round carries is the refusal, and the refusal here is
    the compiler observing the emptiness the successor itself declared.

    Narrow on purpose. The pre-write frontier is excluded because it has no
    observed fire and therefore no validation, and ``residual_complete`` false is
    excluded because that case is load-bearing twice in the recorded corpus.
    """

    if rejection != "empty_graph":
        return False
    if not isinstance(certificate, Mapping):
        return False
    if certificate.get("frontier_kind") != "declared_terminal_open_residual":
        return False
    if not isinstance(graph, Mapping):
        return False
    return bool(graph.get("residual_complete"))


def _execute_compiled_nodes(
    compiled: Sequence[CompiledCommand],
    *,
    bash: Any,
    api_key_env: str,
    command_timeout_seconds: int,
    observation_chars: int,
    certificate: Mapping[str, Any],
    arguments_by_node: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    """Run one compiled suffix in order against one environment.

    Placeholders resolve against the responses this same environment returned,
    so the identical graph can execute on a replica and on the live state with
    each execution binding its own ids. Receipt checking reads the full
    response body; the character-capped observation exists only for
    transcripts.
    """

    runtime_events: list[dict[str, Any]] = []
    runtime_failed = False
    node_outputs: dict[str, dict[str, Any]] = {}
    for command in compiled:
        resolved_command, unresolved_refs = resolve_response_placeholders(
            command.command,
            node_outputs,
            shell_quoted=True,
        )
        if unresolved_refs:
            runtime_events.append(
                {
                    **_public_command_digest(command),
                    "tool_exit_code": 1,
                    "tool_status": "compiler_unresolved_response_ref",
                    "tool_latency_seconds": 0.0,
                    "observation": "unresolved_response_ref:"
                    + ",".join(unresolved_refs),
                    "semantic_failure": True,
                    "workspace_path": None,
                }
            )
            runtime_failed = True
            break
        if command.operation_family == "calendar.acl" and command.method in {
            "DELETE",
            "PUT",
            "PATCH",
        }:
            resolved_command = _calendar_acl_canonicalize_rule_id(
                resolved_command, certificate
            )
        tool = execute_bash_observation(
            bash=bash,
            command=resolved_command,
            api_key_env=api_key_env,
            timeout_seconds=command_timeout_seconds,
            observation_chars=observation_chars,
        )
        full_observation = str(tool.get("observation_full") or tool["observation"])
        resolved_arguments, unresolved_argument_refs = (
            resolve_structured_response_placeholders(
                arguments_by_node.get(command.node_id),
                node_outputs,
            )
        )
        runtime_event = {
            **_public_command_digest(command),
            "tool_exit_code": tool["exit_code"],
            "tool_status": tool["status"],
            "tool_latency_seconds": tool["latency_seconds"],
            "observation": tool["observation"],
            "observation_full": full_observation,
            "resolved_command": resolved_command,
            "semantic_failure": semantic_tool_failure(full_observation),
            "workspace_path": tool["workspace_path"],
            **_transport_trace_fields(tool),
        }
        if unresolved_argument_refs:
            runtime_event["unresolved_argument_refs"] = list(
                unresolved_argument_refs
            )
            runtime_event["semantic_failure"] = True
        if command.mutating and int(tool["exit_code"]) == 0:
            # Checked Execution binds completion to matching receipts. A
            # write whose own receipt does not report the fields it set
            # discharged nothing, whatever the status code said.
            unrealized = unrealized_write_fields(
                arguments=_mapping(resolved_arguments),
                receipt=full_observation,
            )
            if unrealized:
                runtime_event["unrealized_write_fields"] = list(unrealized)
                runtime_event["semantic_failure"] = True
        runtime_events.append(runtime_event)
        if int(tool["exit_code"]) != 0 or bool(runtime_event["semantic_failure"]):
            if _calendar_acl_delete_not_found_is_success(
                command,
                full_observation,
                certificate=certificate,
                resolved_command=resolved_command,
            ):
                runtime_event["semantic_failure"] = False
                runtime_event["repair_reason"] = (
                    "calendar_acl_delete_not_found_idempotent_success"
                )
                continue
            repair_command = _calendar_acl_put_not_found_repair_command(
                command,
                full_observation,
                resolved_command=resolved_command,
            )
            if repair_command is None:
                runtime_failed = True
                break
            repair_tool = execute_bash_observation(
                bash=bash,
                command=repair_command.command,
                api_key_env=api_key_env,
                timeout_seconds=command_timeout_seconds,
                observation_chars=observation_chars,
            )
            repair_observation_full = str(
                repair_tool.get("observation_full") or repair_tool["observation"]
            )
            repair_event = {
                **_public_command_digest(repair_command),
                "tool_exit_code": repair_tool["exit_code"],
                "tool_status": repair_tool["status"],
                "tool_latency_seconds": repair_tool["latency_seconds"],
                "observation": repair_tool["observation"],
                "observation_full": repair_observation_full,
                "semantic_failure": semantic_tool_failure(repair_observation_full),
                "workspace_path": repair_tool["workspace_path"],
                "repair_of": command.node_id,
                "repair_reason": "calendar_acl_put_not_found",
                **_transport_trace_fields(repair_tool),
            }
            runtime_events.append(repair_event)
            if int(repair_tool["exit_code"]) != 0 or bool(
                repair_event["semantic_failure"]
            ):
                runtime_failed = True
                break
            parsed_repair_output = parse_observation_json(repair_observation_full)
            if parsed_repair_output is not None:
                node_outputs[command.node_id] = parsed_repair_output
            continue
        parsed_output = parse_observation_json(full_observation)
        if parsed_output is not None:
            node_outputs[command.node_id] = parsed_output
    return runtime_events, runtime_failed


def _witness_failure_observation(
    witness_events: Sequence[Mapping[str, Any]],
) -> str:
    """What the replica run showed, framed as re-proposal input.

    The successor receives the failed node and its observation. Runtime
    internals stay out of this feedback so re-proposal remains source-bound.
    """

    if not witness_events:
        return "replica_witness_failed"
    last = _mapping(witness_events[-1])
    return (
        "replica_witness_failed:"
        f"{last.get('node_id')}:"
        f"{str(last.get('observation') or '')[:2000]}"
    )


def propose_compile_execute_residual(
    *,
    row: baseline.AgentDiffRow,
    bash: Any,
    api_base: str,
    api_key: str,
    api_key_env: str,
    strong_model: str,
    strong_reasoning_effort: str,
    max_strong_model_tokens: int,
    provider_timeout_seconds: float,
    provider_max_attempts: int,
    command_timeout_seconds: int,
    observation_chars: int,
    max_graph_nodes: int,
    max_mutating_nodes: int,
    certificate: Mapping[str, Any],
    max_proposal_rounds: int = 2,
    live_receipts: Sequence[Mapping[str, Any]] = (),
    service_scoped_strong_prompt: bool = False,
    witness_bash_factory: Any | None = None,
) -> dict[str, Any]:
    if strong_reasoning_effort != DEFAULT_STRONG_REASONING_EFFORT:
        raise ValueError("Agent-Diff requires strong reasoning effort=high")
    if max_proposal_rounds < 1:
        raise ValueError("max_proposal_rounds_must_be_positive")
    strong_usage = UsageAccumulator()
    strong_calls = 0
    strong_latency = 0.0
    attempts: list[dict[str, Any]] = []
    # The public service grammar is fixed for this residual, including repairs.
    proposal_tools = [
        residual_graph_tool(str(certificate.get("service") or row.service))
    ]

    for proposal_round in range(1, max_proposal_rounds + 1):
        repair_observation = None
        prior_tool_arguments = None
        if attempts:
            # A refused graph is as much a reason to re-propose as a failed
            # call: both tell the successor its suffix does not settle the
            # contract, and only the refusal says why before anything runs.
            repair_observation = str(
                attempts[-1].get("runtime_observation")
                or attempts[-1].get("compiler_reject")
                or ""
            )
            if attempts[-1].get("proposal_carrier") == "tool_call":
                prior_tool_arguments = str(
                    attempts[-1].get("assistant_text") or ""
                )
        receipt = baseline.call_chat_model(
            api_base=api_base,
            api_key=api_key,
            model=strong_model,
            reasoning_effort=strong_reasoning_effort,
            messages=build_strong_messages(
                certificate=certificate,
                repair_observation=repair_observation,
                prior_tool_arguments=prior_tool_arguments,
                live_receipts=live_receipts,
                service_scoped_prompt=service_scoped_strong_prompt,
            ),
            max_tokens=max_strong_model_tokens,
            timeout_seconds=provider_timeout_seconds,
            max_attempts=provider_max_attempts,
            # The graph is offered as the one action available rather than
            # described and hoped for. A successor reached through an agent
            # session arrives with its own tools already in context; asked for
            # a graph in prose with no tool of its own, it reached for those
            # instead and answered with what it had found, and seventy-five of
            # its proposals carried no object for the parser to read. Offering
            # the action it is supposed to take is what redirects it. The shape
            # moved here from the system prompt rather than being added on top,
            # so a caller that already produced graphs sends about the same.
            tools=proposal_tools,
            tool_choice=RESIDUAL_GRAPH_TOOL_CHOICE,
            tool_call_name=RESIDUAL_GRAPH_TOOL["function"]["name"],
        )
        strong_calls += 1
        strong_latency += float(receipt["latency_seconds"])
        strong_usage.add_usage(_mapping(receipt["usage"]))
        # A provider that ignores the offered tool can still return a graph in
        # text, so both supported proposal carriers reach the same compiler.
        tool_call_arguments = receipt.get("tool_call_arguments")
        text = str(tool_call_arguments or receipt["text"])
        attempt: dict[str, Any] = {
            "proposal_round": proposal_round,
            "reasoning_effort": strong_reasoning_effort,
            "proposal_carrier": (
                "tool_call" if tool_call_arguments else "text"
            ),
            "assistant_text": text,
            "assistant_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "usage": receipt["usage"],
            "diagnostics": receipt["diagnostics"],
            "provider_attempts": receipt["attempts"],
        }
        graph: Any = None
        try:
            graph = parse_typed_graph(text)
            graph, contract_bound_fields = _bind_graph_contract_constants(
                graph, service=row.service
            )
            if contract_bound_fields:
                attempt["contract_bound_fields"] = list(contract_bound_fields)
            compiled = compile_typed_graph(
                graph,
                service=row.service,
                max_graph_nodes=max_graph_nodes,
                max_mutating_nodes=max_mutating_nodes,
            )
            # Compiling the graph is Executable Residual Completion. Checked
            # Execution then admits only a closed graph that settles the
            # reached-state contract.
            admission_verdicts: list[str] = []
            for check in (
                _whole_suffix_reject_reason,
                _box_extremum_reject_reason,
                _calendar_named_day_reject_reason,
                _calendar_person_timezone_reject_reason,
                _slack_conversation_shape_reject_reason,
                _slack_reaction_binding_reject_reason,
                _conditional_create_reject_reason,
                _box_multi_write_reject_reason,
                _unbound_write_target_reason,
                _undercomplete_graph_reason,
            ):
                reason = check(
                    graph=graph, compiled=compiled, certificate=certificate
                )
                if not reason:
                    continue
                admission_verdicts.append(reason)
                break
            if admission_verdicts:
                attempt["admission_verdicts"] = list(admission_verdicts)
                raise GraphReject(admission_verdicts[0])
        except GraphReject as exc:
            rejection = str(exc)
            attempt["compiler_reject"] = rejection
            attempts.append(attempt)
            if rejection == SUCCESSOR_DECLINED_REJECT:
                # A declared decline is an answer, not a malformed graph. The
                # repair round carries the refusal as its only new input, and
                # here the refusal would only restate what the successor just
                # said, so the residual is handed back without spending it.
                return _residual_status(
                    status="successor_declined",
                    public_status=SUCCESSOR_DECLINED_REJECT,
                    attempts=attempts,
                    strong_calls=strong_calls,
                    strong_latency=strong_latency,
                    strong_usage=strong_usage.as_dict(),
                )
            if _is_sticky_compiler_capability_reject(
                rejection
            ) or _declared_terminal_decline(
                rejection=rejection,
                certificate=certificate,
                graph=graph,
            ):
                return _residual_status(
                    status="compiler_rejected_before_write",
                    public_status=f"compiler_reject:{rejection}",
                    attempts=attempts,
                    strong_calls=strong_calls,
                    strong_latency=strong_latency,
                    strong_usage=strong_usage.as_dict(),
                )
            if proposal_round < max_proposal_rounds:
                # Bounded re-proposal: the successor gets one more attempt on
                # the same frozen contract, with the refusal as its only new
                # input. Returning here instead would hand the residual back
                # before the method had used the budget it already holds.
                continue
            return _residual_status(
                status="compiler_rejected_after_repair",
                public_status=f"compiler_reject:{exc}",
                attempts=attempts,
                strong_calls=strong_calls,
                strong_latency=strong_latency,
                strong_usage=strong_usage.as_dict(),
            )

        attempt["compiled"] = [_public_command_digest(command) for command in compiled]
        # What each node asked the interface to set, kept so a receipt can be
        # read against the write that produced it.
        arguments_by_node = {
            str(node.get("node_id")): _mapping(node.get("arguments"))
            for node in _sequence(_select_graph_candidate(graph).get("nodes"))
            if isinstance(node, Mapping)
        }
        if witness_bash_factory is not None and any(
            command.mutating for command in compiled
        ):
            # Replica witness: run the whole candidate suffix against a fresh
            # replica of the reached state first. A failure there authorizes
            # nothing and touches nothing live, so it is a legal reason to
            # re-propose; only a witnessed graph reaches the live environment.
            # Every round gets its own replica, because a failed witness has
            # already written into the previous one. A read-only graph skips
            # the replica: reads change nothing, so live failure is already a
            # safe re-proposal input.
            with witness_bash_factory() as witness_bash:
                witness_events, witness_failed = _execute_compiled_nodes(
                    compiled,
                    bash=witness_bash,
                    api_key_env=api_key_env,
                    command_timeout_seconds=command_timeout_seconds,
                    observation_chars=observation_chars,
                    certificate=certificate,
                    arguments_by_node=arguments_by_node,
                )
            attempt["witness_runtime"] = witness_events
            if witness_failed:
                attempt["runtime"] = []
                attempts.append(attempt)
                if proposal_round < max_proposal_rounds:
                    attempt["runtime_observation"] = _witness_failure_observation(
                        witness_events
                    )
                    continue
                return _residual_status(
                    status="replica_witness_failed_after_repair",
                    public_status="replica_witness_failed_after_repair",
                    attempts=attempts,
                    strong_calls=strong_calls,
                    strong_latency=strong_latency,
                    strong_usage=strong_usage.as_dict(),
                )
        runtime_events, runtime_failed = _execute_compiled_nodes(
            compiled,
            bash=bash,
            api_key_env=api_key_env,
            command_timeout_seconds=command_timeout_seconds,
            observation_chars=observation_chars,
            certificate=certificate,
            arguments_by_node=arguments_by_node,
        )
        attempt["runtime"] = runtime_events
        attempts.append(attempt)
        if not runtime_failed:
            return _residual_status(
                status="compiled_suffix_executed",
                public_status="compiled_suffix_executed",
                attempts=attempts,
                strong_calls=strong_calls,
                strong_latency=strong_latency,
                strong_usage=strong_usage.as_dict(),
                mutating_command_count=executed_mutating_count(attempts),
                residual_complete=bool(graph.get("residual_complete")),
            )
        realized_writes = executed_mutating_count(attempts)
        if realized_writes > 0:
            # A failed suffix that already changed state cannot be safely
            # re-authored from the old contract: doing so can duplicate the
            # writes whose receipts just arrived. Freeze the partial reached
            # state and fail closed; a later experiment may capture a new
            # contract from those receipts, but this proposal call may not.
            return _residual_status(
                status="compiled_runtime_failed_after_partial_write",
                public_status="compiled_runtime_failed_after_partial_write",
                attempts=attempts,
                strong_calls=strong_calls,
                strong_latency=strong_latency,
                strong_usage=strong_usage.as_dict(),
                mutating_command_count=realized_writes,
            )
        if proposal_round < max_proposal_rounds:
            attempt["runtime_observation"] = runtime_events[-1]["observation"]
            continue
        return _residual_status(
            status="compiled_runtime_failed_after_repair",
            public_status="compiled_runtime_failed_after_repair",
            attempts=attempts,
            strong_calls=strong_calls,
            strong_latency=strong_latency,
            strong_usage=strong_usage.as_dict(),
            mutating_command_count=realized_writes,
        )

    return _residual_status(
        status="compiled_runtime_failed_after_repair",
        public_status="compiled_runtime_failed_after_repair",
        attempts=attempts,
        strong_calls=strong_calls,
        strong_latency=strong_latency,
        strong_usage=strong_usage.as_dict(),
        mutating_command_count=executed_mutating_count(attempts),
    )


def executed_mutating_count(attempts: Sequence[Mapping[str, Any]]) -> int:
    """Writes this residual actually performed, across every proposal round.

    A round that fails at runtime has still changed the environment, and a
    re-proposal cannot undo it. Counting only the last round's compiled graph
    reports zero writes after ten of them succeeded, which reads to the caller
    as a read-only residual and licenses a replan that writes them all again.
    """

    return sum(
        1
        for attempt in attempts
        for runtime in _mapping(attempt).get("runtime", [])
        if bool(_mapping(runtime).get("mutating"))
        and int(_mapping(runtime).get("tool_exit_code") or 0) == 0
    )


def _read_only_residual_left_state_untouched(residual: Mapping[str, Any]) -> bool:
    """Whether a failed residual provably changed nothing on live state.

    Three things have to hold together. No write may have succeeded, and none
    may even have been sent, because a write that errored leaves the reached
    state unknown rather than unchanged. And at least one read has to have
    returned a receipt, so that a rebuilt contract differs from the one that
    just failed; without that the next frontier would ask the successor the
    same question against the same evidence.
    """

    return (
        int(residual.get("mutating_command_count") or 0) == 0
        and int(residual.get("mutating_dispatch_count") or 0) == 0
        and int(residual.get("settled_read_count") or 0) > 0
    )


def dispatched_mutating_count(attempts: Sequence[Mapping[str, Any]]) -> int:
    """Writes this residual *sent*, whether or not they reported success.

    A write that returns an error may still have changed the environment: the
    status code is the interface's account of what happened, not proof that
    nothing did. Anything that decides whether the reached state is still the
    one the contract describes has to count dispatches, not successes, because
    a failed write leaves the answer unknown rather than negative.
    """

    return sum(
        1
        for attempt in attempts
        for runtime in _mapping(attempt).get("runtime", [])
        if bool(_mapping(runtime).get("mutating"))
    )


def settled_read_count(attempts: Sequence[Mapping[str, Any]]) -> int:
    """Reads that returned a receipt, and so could enter a rebuilt contract."""

    return sum(
        1
        for attempt in attempts
        for runtime in _mapping(attempt).get("runtime", [])
        if not bool(_mapping(runtime).get("mutating"))
        and int(_mapping(runtime).get("tool_exit_code") or 0) == 0
        and not bool(_mapping(runtime).get("semantic_failure"))
    )


def _residual_status(
    *,
    status: str,
    public_status: str,
    attempts: Sequence[Mapping[str, Any]],
    strong_calls: int,
    strong_latency: float,
    strong_usage: Mapping[str, Any],
    mutating_command_count: int = 0,
    residual_complete: bool = False,
) -> dict[str, Any]:
    return {
        "status": status,
        "public_status": public_status,
        "attempts": list(attempts),
        "strong_provider_call_count": strong_calls,
        "strong_model_latency_seconds": round(strong_latency, 3),
        "strong_usage": dict(strong_usage),
        "mutating_command_count": mutating_command_count,
        # Derived here rather than at the call sites so every status carries
        # the same account of what reached the live environment.
        "mutating_dispatch_count": dispatched_mutating_count(attempts),
        "settled_read_count": settled_read_count(attempts),
        "residual_complete": residual_complete,
    }


def _undercomplete_graph_reason(
    *,
    graph: Mapping[str, Any],
    compiled: Sequence[CompiledCommand],
    certificate: Mapping[str, Any],
) -> str | None:
    """Reject a single-action graph that leaves stated residual work open.

    This legacy route only exposes an explicit residual-write count. A listing
    receipt cannot become a coverage claim without a frozen request-scope
    predicate, so it is not consulted here.
    """

    mutating_count = sum(1 for command in compiled if command.mutating)
    if mutating_count != 1:
        return None
    if _remaining_write_obligation_count(certificate) >= 3:
        return "undercomplete_residual_write_coverage"
    return None


def _whole_suffix_reject_reason(
    *,
    graph: Mapping[str, Any],
    compiled: Sequence[CompiledCommand],
    certificate: Mapping[str, Any],
) -> str | None:
    """Require every authority-bearing graph to claim whole-residual closure."""

    del certificate
    if any(command.mutating for command in compiled) and not bool(
        graph.get("residual_complete")
    ):
        return "whole_suffix_not_declared_complete"
    return None


def _remaining_write_obligation_count(certificate: Mapping[str, Any]) -> int:
    """Stated residual writes not yet covered by a realized commitment.

    The request-side count alone over-rejects late frontiers: once ``K``
    carries receipts for most of the requested writes, a one-write graph is
    exactly the residual, not an under-complete one. Both sides of the
    subtraction are lexical approximations, so the caller keeps a gap of two
    before rejecting; tightening to a gap of one would revive the infinite
    false-reject loop this subtraction removed.
    """

    realized = sum(
        1
        for item in _sequence(certificate.get("K"))
        if str(_mapping(item).get("receipt") or "").strip()
    )
    return max(0, _residual_write_obligation_count(certificate) - realized)


def _box_extremum_reject_reason(
    *,
    graph: Mapping[str, Any],
    compiled: Sequence[CompiledCommand],
    certificate: Mapping[str, Any],
) -> str | None:
    """Bind smallest/largest writes to receipt-derived global extrema.

    Box folder listings sorted by size establish only *local* candidates. A
    graph may write once every local candidate has a concrete size receipt and
    its chosen target is a global extremum across those candidates. Nothing in
    this check depends on a task id, object id, or expected answer.
    """

    if str(graph.get("service") or certificate.get("service") or "") != "box":
        return None
    request = str(certificate.get("req") or "").lower()
    required = {
        label
        for label in ("smallest", "largest")
        if re.search(rf"\b{label}\b", request)
    }
    if not required:
        return None

    candidates = _box_sorted_extremum_candidates(certificate)
    relevant_ids = set().union(*(candidates[label] for label in required))
    if not relevant_ids:
        # Without a complete sorted listing, this checker has no receipt-backed
        # scope predicate and therefore makes no claim.
        return None

    sizes = _box_receipted_file_sizes(certificate)
    missing = sorted(relevant_ids - sizes.keys())
    if missing:
        return "box_extremum_candidate_size_missing:" + ",".join(missing)

    selected = _select_graph_candidate(graph)
    nodes = [
        dict(node)
        for node in _sequence(selected.get("nodes"))
        if isinstance(node, Mapping)
    ]
    targets: dict[str, set[str]] = {label: set() for label in required}
    for node, command in zip(nodes, compiled, strict=False):
        if not command.mutating or command.operation_family != "box.file":
            continue
        descriptor = json.dumps(
            {
                "claimed_effects": node.get("claimed_effects"),
                "json": _mapping(node.get("arguments")).get("json"),
            },
            ensure_ascii=False,
            sort_keys=True,
        ).lower()
        file_targets = set(re.findall(r"/files/([0-9]+)", command.url))
        for label in required:
            if re.search(rf"\b{label}\b", descriptor):
                targets[label].update(file_targets)

    for label in sorted(required):
        if not targets[label]:
            return f"box_extremum_write_missing:{label}"
        candidate_sizes = {
            file_id: sizes[file_id] for file_id in candidates[label]
        }
        extremum = (
            min(candidate_sizes.values())
            if label == "smallest"
            else max(candidate_sizes.values())
        )
        expected = {
            file_id
            for file_id, size in candidate_sizes.items()
            if size == extremum
        }
        if not targets[label].issubset(expected):
            return (
                f"box_extremum_target_mismatch:{label}:expected="
                + ",".join(sorted(expected))
            )
    return None


def _box_sorted_extremum_candidates(
    certificate: Mapping[str, Any],
) -> dict[str, set[str]]:
    candidates = {"smallest": set(), "largest": set()}
    for event in _sequence(certificate.get("E")):
        event_map = _mapping(event)
        request = _mapping(event_map.get("request"))
        if str(request.get("method") or "").upper() != "GET":
            continue
        path = str(request.get("path") or "")
        parsed_path = urlparse("https://box.invalid" + path)
        if not re.search(r"/folders/[^/]+/items$", parsed_path.path):
            continue
        query = parse_qs(parsed_path.query)
        if [value.lower() for value in query.get("sort", [])] != ["size"]:
            continue
        direction = str(query.get("direction", ["ASC"])[0]).upper()
        if direction not in {"ASC", "DESC"}:
            continue
        observation = parse_observation_json(
            str(event_map.get("observation") or "")
        )
        if observation is None:
            continue
        entries = [
            _mapping(item)
            for item in _sequence(observation.get("entries"))
            if isinstance(item, Mapping)
        ]
        with contextlib.suppress(TypeError, ValueError):
            total_count = int(observation.get("total_count"))
            offset = int(observation.get("offset") or 0)
            if offset != 0 or len(entries) < total_count:
                continue
            file_ids = [
                str(item.get("id"))
                for item in entries
                if str(item.get("type") or "") == "file" and item.get("id")
            ]
            if not file_ids:
                continue
            if direction == "ASC":
                candidates["smallest"].add(file_ids[0])
                candidates["largest"].add(file_ids[-1])
            else:
                candidates["smallest"].add(file_ids[-1])
                candidates["largest"].add(file_ids[0])
    return candidates


def _box_receipted_file_sizes(
    certificate: Mapping[str, Any],
) -> dict[str, float]:
    sizes: dict[str, float] = {}

    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            item_type = str(value.get("type") or "")
            file_id = str(value.get("id") or "")
            raw_size = value.get("size")
            if (
                item_type in {"", "file"}
                and file_id
                and not isinstance(raw_size, bool)
                and isinstance(raw_size, (int, float))
            ):
                sizes[file_id] = float(raw_size)
            for nested in value.values():
                collect(nested)
            return
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for nested in value:
                collect(nested)

    for event in _sequence(certificate.get("E")):
        parsed = parse_observation_json(
            str(_mapping(event).get("observation") or "")
        )
        if parsed is not None:
            collect(parsed)
    return sizes


_NAMED_WEEKDAY = re.compile(
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
_EXPLICIT_OVERNIGHT = re.compile(
    r"\b(?:overnight|cross(?:es|ing)?\s+midnight|"
    r"until\s+the\s+next\s+(?:day|morning)|"
    r"through\s+the\s+next\s+(?:day|morning))\b",
    re.IGNORECASE,
)


def _calendar_named_day_reject_reason(
    *,
    graph: Mapping[str, Any],
    compiled: Sequence[CompiledCommand],
    certificate: Mapping[str, Any],
) -> str | None:
    """Keep a timed event inside a requested named local day."""

    if (
        str(graph.get("service") or certificate.get("service") or "")
        != "calendar"
    ):
        return None
    request = str(certificate.get("req") or "")
    if not _NAMED_WEEKDAY.search(request) or _EXPLICIT_OVERNIGHT.search(request):
        return None

    selected = _select_graph_candidate(graph)
    nodes = [
        dict(node)
        for node in _sequence(selected.get("nodes"))
        if isinstance(node, Mapping)
    ]
    for node, command in zip(nodes, compiled, strict=False):
        if (
            not command.mutating
            or command.operation_family != "calendar.events"
            or command.method not in {"POST", "PUT", "PATCH"}
        ):
            continue
        body = _mapping(_mapping(node.get("arguments")).get("json"))
        start = str(_mapping(body.get("start")).get("dateTime") or "")
        end = str(_mapping(body.get("end")).get("dateTime") or "")
        start_date = re.match(r"^\d{4}-\d{2}-\d{2}(?=T)", start)
        end_date = re.match(r"^\d{4}-\d{2}-\d{2}(?=T)", end)
        if start_date and end_date and start_date.group() != end_date.group():
            return f"calendar_named_day_cross_midnight:{command.node_id}"
    return None


_EMAIL_ADDRESS = re.compile(
    r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+"
)
_PERSON_TIMEZONE_PHRASE = re.compile(
    r"\b(?:his|her|their)\s+(?:local\s+)?(?:time\s*zone|timezone)\b",
    re.IGNORECASE,
)


def _calendar_person_timezone_targets(request: str) -> set[str]:
    """Emails nearest an explicit his/her/their-timezone clause."""

    email_matches = list(_EMAIL_ADDRESS.finditer(request))
    targets: set[str] = set()
    for phrase in _PERSON_TIMEZONE_PHRASE.finditer(request):
        nearby = [
            candidate
            for candidate in email_matches
            if candidate.start() >= phrase.start() - 320
            and candidate.end() <= phrase.end() + 100
        ]
        if not nearby:
            continue
        closest = min(
            nearby,
            key=lambda candidate: min(
                abs(candidate.start() - phrase.end()),
                abs(candidate.end() - phrase.start()),
            ),
        )
        targets.add(closest.group().casefold())
    return targets


def _calendar_receipted_email_timezones(
    certificate: Mapping[str, Any],
) -> dict[str, set[str]]:
    bound: dict[str, set[str]] = {}

    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            direct_emails = {
                text.casefold()
                for key in ("email", "id")
                if isinstance((text := value.get(key)), str)
                and _EMAIL_ADDRESS.fullmatch(text)
            }
            direct_timezones = {
                text
                for key, raw in value.items()
                if str(key).casefold() in {"timezone", "time_zone", "tz"}
                and isinstance(raw, str)
                and (text := raw.strip())
            }
            for email in direct_emails:
                bound.setdefault(email, set()).update(direct_timezones)
            for nested in value.values():
                collect(nested)
            return
        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            for nested in value:
                collect(nested)

    for event in _sequence(certificate.get("E")):
        parsed = parse_observation_json(
            str(_mapping(event).get("observation") or "")
        )
        if parsed is not None:
            collect(parsed)
    return bound


def _calendar_timezone_placeholder_is_bound(
    *,
    timezone_value: str,
    target_email: str,
    nodes_by_id: Mapping[str, Mapping[str, Any]],
) -> bool:
    match = re.fullmatch(
        r"\{\{([A-Za-z0-9_-]+)\.(?:timeZone|timezone)\}\}",
        timezone_value,
    )
    if not match:
        return False
    producer = _mapping(nodes_by_id.get(match.group(1)))
    arguments = _mapping(producer.get("arguments"))
    path = unquote_plus(str(arguments.get("path") or "")).casefold()
    return (
        str(arguments.get("method") or "GET").upper() == "GET"
        and target_email in path
        and "/calendars/" in path
        and "/calendarlist/" not in path
    )


def _calendar_person_timezone_reject_reason(
    *,
    graph: Mapping[str, Any],
    compiled: Sequence[CompiledCommand],
    certificate: Mapping[str, Any],
) -> str | None:
    """Bind a named person's local timezone to their calendar receipt."""

    if str(graph.get("service") or certificate.get("service") or "") != "calendar":
        return None
    targets = _calendar_person_timezone_targets(
        str(certificate.get("req") or "")
    )
    if not targets:
        return None

    selected = _select_graph_candidate(graph)
    nodes = [
        dict(node)
        for node in _sequence(selected.get("nodes"))
        if isinstance(node, Mapping)
    ]
    nodes_by_id = {
        str(node.get("node_id") or ""): node
        for node in nodes
    }
    receipt_timezones = _calendar_receipted_email_timezones(certificate)
    for node, command in zip(nodes, compiled, strict=False):
        if (
            not command.mutating
            or command.operation_family != "calendar.events"
            or command.method not in {"POST", "PUT", "PATCH"}
        ):
            continue
        arguments = _mapping(node.get("arguments"))
        body = _mapping(arguments.get("json"))
        rendered = json.dumps(
            {
                "body": body,
                "claimed_effects": _string_list(node.get("claimed_effects")),
            },
            ensure_ascii=False,
        ).casefold()
        for target_email in sorted(targets):
            if target_email not in rendered:
                continue
            start_timezone = str(
                _mapping(body.get("start")).get("timeZone") or ""
            )
            end_timezone = str(
                _mapping(body.get("end")).get("timeZone") or ""
            )
            if (
                start_timezone
                and start_timezone == end_timezone
                and _calendar_timezone_placeholder_is_bound(
                    timezone_value=start_timezone,
                    target_email=target_email,
                    nodes_by_id=nodes_by_id,
                )
            ):
                continue
            bound = receipt_timezones.get(target_email, set())
            if not bound:
                return f"calendar_person_timezone_unbound:{command.node_id}"
            if (
                not start_timezone
                or not end_timezone
                or start_timezone not in bound
                or end_timezone not in bound
            ):
                return f"calendar_person_timezone_mismatch:{command.node_id}"
    return None


def _slack_conversation_shape_reject_reason(
    *,
    graph: Mapping[str, Any],
    compiled: Sequence[CompiledCommand],
    certificate: Mapping[str, Any],
) -> str | None:
    """Require Slack's public DM-opening operation before posting.

    ``conversations.create`` creates a named channel, and a user id is not a
    conversation id for ``chat.postMessage``. Both single-user and group DMs
    must be opened with ``conversations.open`` and then referenced by the
    returned channel id.
    """

    del compiled
    if str(graph.get("service") or certificate.get("service") or "") != "slack":
        return None
    selected = _select_graph_candidate(graph)
    for node in _sequence(selected.get("nodes")):
        if not isinstance(node, Mapping):
            continue
        node_map = _mapping(node)
        arguments = _mapping(node_map.get("arguments"))
        path = str(arguments.get("path") or "").lstrip("/")
        values = {
            **_mapping(arguments.get("params")),
            **_mapping(arguments.get("json")),
        }
        node_id = str(node_map.get("node_id") or "unknown")
        if path == "conversations.create" and values.get("users"):
            return f"slack_dm_requires_conversations_open:{node_id}"
        channel = str(values.get("channel") or "")
        if path == "chat.postMessage" and re.fullmatch(
            r"U[A-Za-z0-9_]+", channel
        ):
            return f"slack_dm_requires_conversations_open:{node_id}"
    return None


def _slack_receipted_reactions(
    certificate: Mapping[str, Any],
) -> set[tuple[str, str, str]]:
    confirmed: set[tuple[str, str, str]] = set()

    def collect_message(
        message: Mapping[str, Any],
        *,
        channel: str,
    ) -> None:
        timestamp = str(message.get("ts") or "")
        if not channel or not timestamp:
            return
        for reaction in _sequence(message.get("reactions")):
            reaction_name = str(_mapping(reaction).get("name") or "")
            if reaction_name:
                confirmed.add(
                    (channel, timestamp, reaction_name.strip(":").casefold())
                )

    def collect(value: Any, inherited_channel: str = "") -> None:
        if isinstance(value, Mapping):
            raw_channel = value.get("channel")
            channel = inherited_channel
            if isinstance(raw_channel, str):
                channel = raw_channel
            elif isinstance(raw_channel, Mapping):
                channel = str(raw_channel.get("id") or inherited_channel)
            message = value.get("message")
            if isinstance(message, Mapping):
                collect_message(message, channel=channel)
            if value.get("ts") and value.get("reactions"):
                collect_message(value, channel=channel)
            for nested in value.values():
                collect(nested, channel)
            return
        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            for nested in value:
                collect(nested, inherited_channel)

    for event in _sequence(certificate.get("E")):
        observation = str(_mapping(event).get("observation") or "")
        parsed = parse_observation_json(observation)
        if parsed is not None:
            collect(parsed)
            continue
        # Shell loops often prefix each reactions.get receipt with its target
        # timestamp.  The receipt is still trusted JSON; parse each complete
        # line instead of discarding the whole multi-record observation.
        for line in observation.splitlines():
            parsed_line = parse_observation_json(line)
            if parsed_line is not None:
                collect(parsed_line)
    return confirmed


def _slack_reaction_binding_reject_reason(
    *,
    graph: Mapping[str, Any],
    compiled: Sequence[CompiledCommand],
    certificate: Mapping[str, Any],
) -> str | None:
    """Require a receipt that the selected message owns the reaction."""

    if str(graph.get("service") or certificate.get("service") or "") != "slack":
        return None
    confirmed = _slack_receipted_reactions(certificate)
    selected = _select_graph_candidate(graph)
    nodes = [
        dict(node)
        for node in _sequence(selected.get("nodes"))
        if isinstance(node, Mapping)
    ]
    for index, (node, command) in enumerate(
        zip(nodes, compiled, strict=False)
    ):
        arguments = _mapping(node.get("arguments"))
        path = str(arguments.get("path") or "").lstrip("/")
        if path != "reactions.remove":
            continue
        values = {
            **_mapping(arguments.get("params")),
            **_mapping(arguments.get("json")),
        }
        channel = str(values.get("channel") or "")
        timestamp = str(values.get("timestamp") or values.get("ts") or "")
        reaction = str(
            values.get("name") or values.get("reaction") or ""
        ).strip(":").casefold()
        if not reaction:
            return f"slack_reaction_name_missing:{command.node_id}"
        same_target_probe = False
        for prior in nodes[:index]:
            prior_arguments = _mapping(prior.get("arguments"))
            if (
                str(prior_arguments.get("path") or "").lstrip("/")
                != "reactions.get"
            ):
                continue
            prior_values = {
                **_mapping(prior_arguments.get("params")),
                **_mapping(prior_arguments.get("json")),
            }
            if (
                str(prior_values.get("channel") or "") == channel
                and str(
                    prior_values.get("timestamp")
                    or prior_values.get("ts")
                    or ""
                )
                == timestamp
            ):
                same_target_probe = True
                break
        if (
            (channel, timestamp, reaction) not in confirmed
            and not same_target_probe
        ):
            return f"slack_reaction_target_unconfirmed:{command.node_id}"
    return None


_CONDITIONAL_CREATE_REUSE = re.compile(
    r"\bif\s+needed\b|"
    r"\bif\s+(?:it|one|that)\s+(?:does(?:n['’]t)?|does\s+not)\s+"
    r"(?:already\s+)?exist|"
    r"\b(?:do\s+not|don['’]t|avoid)\s+duplicat",
    re.IGNORECASE,
)


def _conditional_create_reject_reason(
    *,
    graph: Mapping[str, Any],
    compiled: Sequence[CompiledCommand],
    certificate: Mapping[str, Any],
) -> str | None:
    """Reuse a receipt-visible named resource when creation is conditional."""

    if not _CONDITIONAL_CREATE_REUSE.search(
        str(certificate.get("req") or "")
    ):
        return None

    existing: dict[str, set[str]] = {}

    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            resource_id = str(value.get("id") or "")
            kind = str(value.get("kind") or "")
            if resource_id and kind in {
                "calendar#calendar",
                "calendar#calendarListEntry",
            }:
                resource_name = str(value.get("summary") or "").strip()
                if resource_name:
                    normalized = re.sub(r"\s+", " ", resource_name.casefold())
                    existing.setdefault(normalized, set()).add(resource_id)
            for nested in value.values():
                collect(nested)
            return
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for nested in value:
                collect(nested)

    for event in _sequence(certificate.get("E")):
        parsed = parse_observation_json(
            str(_mapping(event).get("observation") or "")
        )
        if parsed is not None:
            collect(parsed)

    selected = _select_graph_candidate(graph)
    nodes = [
        dict(node)
        for node in _sequence(selected.get("nodes"))
        if isinstance(node, Mapping)
    ]
    for node, command in zip(nodes, compiled, strict=False):
        if (
            not command.mutating
            or command.method != "POST"
            or command.operation_family != "calendar.calendars"
        ):
            continue
        claimed = " ".join(_string_list(node.get("claimed_effects"))).lower()
        if "create" not in claimed:
            continue
        arguments = _mapping(node.get("arguments"))
        values = {
            **_mapping(arguments.get("params")),
            **_mapping(arguments.get("json")),
        }
        proposed = str(values.get("summary") or "").strip()
        if not proposed:
            continue
        normalized = re.sub(r"\s+", " ", proposed.casefold())
        ids = sorted(existing.get(normalized, set()))
        if ids:
            return (
                "conditional_create_existing_resource:"
                f"{command.node_id}:{ids[0]}"
            )
    return None


def _box_multi_write_reject_reason(
    *,
    graph: Mapping[str, Any],
    compiled: Sequence[CompiledCommand],
    certificate: Mapping[str, Any],
) -> str | None:
    if str(graph.get("service") or certificate.get("service") or "") != "box":
        return None
    mutating = [command for command in compiled if command.mutating]
    if len(mutating) <= 1:
        return None
    selected = _select_graph_candidate(graph)
    nodes = [
        dict(node)
        for node in _sequence(selected.get("nodes"))
        if isinstance(node, Mapping)
    ]
    visible_text = _visible_evidence_text(certificate)
    for node, command in zip(nodes, compiled, strict=False):
        if not command.mutating:
            continue
        target_ids = _box_target_ids_for_node(node, command)
        for target_id in target_ids:
            if not _resource_id_is_visible(target_id, visible_text):
                return "box_multi_write_unbound_target"
    return None


def _visible_evidence_text(certificate: Mapping[str, Any]) -> str:
    # Include the request text: an id named in the request is a bound target even
    # if the cheap prefix never echoed it into an observation (F2 false-reject
    # seam). Callers must still check identifier boundaries, not substrings.
    parts: list[str] = [str(certificate.get("req") or "")]
    for event in _sequence(certificate.get("E")):
        event_map = _mapping(event)
        parts.append(str(event_map.get("observation") or ""))
    return "\n".join(parts)


def _resource_id_is_visible(resource_id: str, visible_text: str) -> bool:
    # In particular, id 1234 is not grounded by a receipt for id 12345. Keep
    # sentence-ending punctuation and URL separators usable, but do not accept
    # one part of a dotted handle (including a message timestamp) as the ID.
    return re.search(
        rf"(?<![\w@+\-])(?<![\w@+\-]\.)"
        rf"{re.escape(resource_id)}(?![\w@+\-]|\.[\w@+\-])",
        visible_text,
    ) is not None


def _unbound_write_target_reason(
    *,
    graph: Mapping[str, Any],
    compiled: Sequence[CompiledCommand],
    certificate: Mapping[str, Any],
) -> str | None:
    """Reject a residual whose write targets a concrete resource id that is not
    bound in the request or visible evidence E and is not a prior-node
    placeholder.

    This is the service-independent generalization of the Box multi-write target
    binding lower bound: any mutating node whose write targets a concrete,
    server-issued resource id must have that id visible.  The check is
    deliberately conservative -- when a clean target id cannot be extracted for a
    node/service (e.g. a create with no target, a well-known alias, or a
    ``{{n1.id}}`` placeholder produced by a prior node), it does not reject.
    """
    service = str(graph.get("service") or certificate.get("service") or "")
    selected = _select_graph_candidate(graph)
    nodes = [
        dict(node)
        for node in _sequence(selected.get("nodes"))
        if isinstance(node, Mapping)
    ]
    visible_text = _visible_evidence_text(certificate)
    for node, command in zip(nodes, compiled, strict=False):
        if not command.mutating:
            continue
        for target_id in _write_target_ids_for_node(service, node, command):
            if not _resource_id_is_visible(target_id, visible_text):
                return f"unbound_write_target:{service}"
    return None


def _write_target_ids_for_node(
    service: str,
    node: Mapping[str, Any],
    command: CompiledCommand,
) -> set[str]:
    if service == "box":
        return _box_target_ids_for_node(node, command)
    if service == "calendar":
        return _calendar_write_target_ids(command.url)
    if service == "slack":
        return _slack_write_target_ids(node)
    return set()


def _calendar_write_target_ids(url: str) -> set[str]:
    path = urlparse(url).path
    ids: set[str] = set()
    for pattern in (r"/events/([^/?]+)", r"/calendarList/([^/?]+)"):
        for match in re.findall(pattern, path):
            decoded = unquote(match)
            if _is_concrete_resource_id(decoded):
                ids.add(decoded)
    return ids


def _slack_write_target_ids(node: Mapping[str, Any]) -> set[str]:
    arguments = _mapping(node.get("arguments"))
    ids: set[str] = set()
    for source in (_mapping(arguments.get("params")), _mapping(arguments.get("json"))):
        value = source.get("channel")
        if isinstance(value, str) and _is_concrete_resource_id(value):
            ids.add(value)
    return ids


def _is_concrete_resource_id(value: str) -> bool:
    if not value or "{{" in value or "}}" in value:
        return False
    if value in {"primary", "me", "self", "default"}:
        return False
    # Only treat opaque, server-issued handles (with a digit or ``@``) as ids
    # that must be bound; short human labels chosen from the request are ignored
    # so the guard cannot false-reject legitimate writes.
    return len(value) >= 4 and bool(re.search(r"[0-9@]", value))


def _box_target_ids_for_node(
    node: Mapping[str, Any],
    command: CompiledCommand,
) -> set[str]:
    ids = set(re.findall(r"/(?:files|folders)/([0-9]+)", command.url))
    arguments = _mapping(node.get("arguments"))
    ids.update(_box_semantic_target_ids(arguments))
    return ids


def _box_semantic_target_ids(value: Any, key_path: tuple[str, ...] = ()) -> set[str]:
    if isinstance(value, Mapping):
        ids: set[str] = set()
        for key, item in value.items():
            ids.update(_box_semantic_target_ids(item, (*key_path, str(key).lower())))
        return ids
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        ids = set()
        for item in value:
            ids.update(_box_semantic_target_ids(item, key_path))
        return ids
    if not key_path:
        return set()
    if not _box_key_path_denotes_target_id(key_path):
        return set()
    text = str(value) if isinstance(value, int) else value
    if isinstance(text, str) and re.fullmatch(r"[0-9]{4,}", text):
        return {text}
    return set()


def _box_key_path_denotes_target_id(key_path: tuple[str, ...]) -> bool:
    leaf = key_path[-1]
    if leaf in {
        "file_id",
        "folder_id",
        "parent_id",
        "item_id",
        "target_id",
        "source_id",
        "destination_id",
        "source_file_id",
        "source_folder_id",
        "destination_folder_id",
    }:
        return True
    if leaf != "id" or len(key_path) < 2:
        return False
    return key_path[-2] in {
        "file",
        "folder",
        "item",
        "parent",
        "source",
        "destination",
        "target",
    }




# Ordinary English verbs for changing state. This is not a per-service phrase
# list: the same vocabulary is counted on every surface, so no service can be
# given a hand-tuned trigger under the guise of an obligation count.
_RESIDUAL_WRITE_INTENT = (
    r"\bcreate\b",
    r"\badd\b",
    r"\binvite\b",
    r"\bupdate\b",
    r"\bdelete\b",
    r"\bremove\b",
    r"\bpost\b",
    r"\breact\b",
    r"\brename\b",
    r"\bmove\b",
    r"\bcopy\b",
    r"\bupload\b",
    r"\bgive\b.+\baccess\b",
    r"\bset\b.+\bup\b",
    r"\bshare\b",
    r"\bwatch(?:ed|ing)?\b",
)


def _residual_write_obligation_count(certificate: Mapping[str, Any]) -> int:
    """How many distinct state changes the original request still owes."""

    request = str(certificate.get("req") or "").lower()
    lexical_count = sum(
        len(re.findall(pattern, request)) for pattern in _RESIDUAL_WRITE_INTENT
    )
    canonical_effect_count = len(
        {
            str(effect)
            for effect in _sequence(_mapping(certificate.get("O")).get("effects"))
            if str(effect)
        }
    )
    return max(lexical_count, canonical_effect_count)


def _effective_max_iterations(row: baseline.AgentDiffRow, configured: int) -> int:
    # The raised global iteration cap covers the long-horizon shapes that
    # previously needed a per-service extension, matching how the graph-node
    # cap was generalized.
    del row
    return max(configured, 60)


def _effective_max_graph_nodes(row: baseline.AgentDiffRow, configured: int) -> int:
    # The raised global graph-node cap now covers the long-horizon and lifecycle
    # shapes that previously needed per-request phrase-budget extensions.
    return configured


def build_witness_environment_factory(
    *,
    row: baseline.AgentDiffRow,
    agent_diff_base_url: str,
    env_ttl_seconds: int,
    api_key_env: str,
    command_timeout_seconds: int,
    observation_chars: int,
    workspace_dir: str | None = None,
) -> Any:
    """Replica reached-state environments for witness execution.

    The returned factory takes the ordered commands whose writes produced the
    live reached state, yields a bash executor on a fresh environment seeded
    identically and replayed through those commands, and deletes the
    environment afterwards. The replica sees no request text and no scorer;
    it only re-runs commands the live environment already accepted.
    """

    from agent_diff import AgentDiff, BashExecutorProxy  # type: ignore

    @contextlib.contextmanager
    def factory(replay_commands: Sequence[str]):
        client = AgentDiff(base_url=agent_diff_base_url)
        env = client.init_env(
            templateService=row.info["service"],
            templateName=row.info["seed_template"],
            impersonateUserId=row.info["impersonate_user_id"],
            ttlSeconds=int(env_ttl_seconds),
        )
        try:
            with contextlib.ExitStack() as stack:
                if workspace_dir is None:
                    workspace = stack.enter_context(
                        tempfile.TemporaryDirectory(prefix="agent_diff_witness_")
                    )
                else:
                    workspace = workspace_dir
                bash = BashExecutorProxy(
                    env.environmentId,
                    base_url=client.base_url,
                    api_key=client.api_key,
                    workspace_dir=workspace,
                )
                for command in replay_commands:
                    execute_bash_observation(
                        bash=bash,
                        command=str(command),
                        api_key_env=api_key_env,
                        timeout_seconds=command_timeout_seconds,
                        observation_chars=observation_chars,
                    )
                yield bash
        finally:
            with contextlib.suppress(Exception):
                client.delete_env(envId=env.environmentId)

    return factory


def execute_row(
    *,
    row: baseline.AgentDiffRow,
    args: argparse.Namespace,
    api_key: str,
    private_raw_dir: Path,
    stop_event: threading.Event,
) -> dict[str, Any]:
    if stop_event.is_set():
        return {
            "schema": PRIVATE_SCHEMA,
            "public_key": row.public_key,
            "ordinal": row.ordinal,
            "service": row.service,
            "status": "skipped_after_stop",
        }
    baseline.ensure_agent_diff_sdk_on_path(Path(args.agent_diff_sdk_path))
    from agent_diff import AgentDiff, BashExecutorProxy  # type: ignore

    client = AgentDiff(base_url=args.agent_diff_base_url)
    env = None
    started = time.monotonic()
    row_result: dict[str, Any] = {
        "schema": PRIVATE_SCHEMA,
        "public_key": row.public_key,
        "ordinal": row.ordinal,
        "service": row.service,
        "test_id": row.test_id,
        "test_name": row.test_name,
        "metadata": row.metadata,
        "method_role": METHOD_ROLE,
        "cheap_model": args.cheap_model,
        "cheap_reasoning_effort": args.cheap_reasoning_effort,
        "strong_model": args.strong_model,
        "strong_reasoning_effort": args.strong_reasoning_effort,
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
        effective_max_iterations = _effective_max_iterations(row, args.max_iterations)
        effective_max_graph_nodes = _effective_max_graph_nodes(
            row, args.max_graph_nodes
        )
        agent_trace = run_cfrc_agent(
            row=row,
            bash=bash,
            api_base=args.api_base,
            api_key=api_key,
            api_key_env=args.api_key_env,
            cheap_model=args.cheap_model,
            cheap_reasoning_effort=args.cheap_reasoning_effort,
            strong_model=args.strong_model,
            strong_reasoning_effort=args.strong_reasoning_effort,
            max_iterations=effective_max_iterations,
            max_model_tokens=args.max_model_tokens,
            max_strong_model_tokens=args.max_strong_model_tokens,
            provider_timeout_seconds=args.provider_timeout_seconds,
            provider_max_attempts=args.provider_max_attempts,
            command_timeout_seconds=args.command_timeout_seconds,
            observation_chars=args.observation_chars,
            max_graph_nodes=effective_max_graph_nodes,
            max_mutating_nodes=args.max_mutating_nodes,
            min_frontier_observations=args.min_frontier_observations,
            max_strong_residual_rounds=args.max_strong_residual_rounds,
            max_readonly_binding_probes=args.max_readonly_binding_probes,
            service_scoped_strong_prompt=bool(
                getattr(args, "service_scoped_strong_prompt", False)
            ),
            compact_cheap_history=bool(
                getattr(args, "compact_cheap_history", False)
            ),
            witness_environment_factory=(
                None
                if getattr(args, "no_replica_witness", False)
                else build_witness_environment_factory(
                    row=row,
                    agent_diff_base_url=args.agent_diff_base_url,
                    env_ttl_seconds=args.env_ttl_seconds,
                    api_key_env=args.api_key_env,
                    command_timeout_seconds=args.command_timeout_seconds,
                    observation_chars=args.observation_chars,
                    workspace_dir=str(private_raw_dir / "witness_workspaces"),
                )
            ),
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
                "effective_max_graph_nodes": effective_max_graph_nodes,
                "effective_max_iterations": effective_max_iterations,
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
        if baseline.should_stop_after_row_error(
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
    baseline.write_json_append_only(row_path, row_result)
    return row_result


def build_preflight_report(
    *,
    rows: Sequence[baseline.AgentDiffRow],
    args: argparse.Namespace,
    run_id: str,
    blockers: Sequence[str],
    platform_smoke: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    service_counts = Counter(row.service for row in rows)
    report: dict[str, Any] = {
        "schema": "papc.agent_diff.hf_test45_cfrc_preflight.v1",
        "created_at": baseline.now_iso(),
        "run_id": run_id,
        "decision": "ready" if not blockers else "blocked",
        "blockers": list(blockers),
        "execute": bool(args.execute),
        "provider_calls": 0,
        "dataset": {
            "source": "official_agent_diff_hf_local_test_jsonl",
            "split": "test",
            "expected_rows": baseline.EXPECTED_TEST_ROWS,
            "loaded_rows": len(rows),
            "selected_ordinals": [row.ordinal for row in rows],
            "service_counts": dict(sorted(service_counts.items())),
        },
        "model_plan": {
            "method_role": METHOD_ROLE,
            "cheap_model": args.cheap_model,
            "cheap_reasoning_effort": args.cheap_reasoning_effort,
            "residual_strong_model": args.strong_model,
            "residual_strong_reasoning_effort": args.strong_reasoning_effort,
            "api_base_supplied": bool(args.api_base),
            "api_key_env": args.api_key_env,
        },
        "execution_contract": {
            "benchmark": "Agent-Diff",
            "task_count": len(rows),
            "max_concurrency": args.max_concurrency,
            "registered_default_max_concurrency": (
                baseline.DEFAULT_MAX_CONCURRENCY
            ),
            "scorer_name": scoring.SCORER_NAME,
            "scorer_correction_schema": scoring.SCHEMA,
            "identical_certificate_frontier_attempt_cap": (
                MAX_IDENTICAL_FRONTIER_ATTEMPTS
            ),
            "exact_contract_provider_call_cap": (
                MAX_EXACT_CONTRACT_PROVIDER_CALLS
            ),
            "exact_contract_provider_call_cap_after_witness_failure": (
                MAX_EXACT_CONTRACT_PROVIDER_CALLS_AFTER_WITNESS_FAILURE
            ),
            "readonly_binding_probe_streak_cap": int(
                getattr(
                    args,
                    "max_readonly_binding_probes",
                    MAX_READONLY_BINDING_PROBES,
                )
            ),
            "service_scoped_strong_prompt": bool(
                getattr(args, "service_scoped_strong_prompt", False)
            ),
            "compact_cheap_history": bool(
                getattr(args, "compact_cheap_history", False)
            ),
            "max_strong_residual_frontiers": int(
                getattr(
                    args,
                    "max_strong_residual_rounds",
                    DEFAULT_MAX_STRONG_RESIDUAL_ROUNDS,
                )
            ),
            "strong_completion_limit_field": "max_tokens",
            "reasoning_inclusive_completion_caps_enabled": False,
            "strong_completion_token_cap": int(
                getattr(
                    args,
                    "max_strong_model_tokens",
                    DEFAULT_MAX_STRONG_MODEL_TOKENS,
                )
            ),
            "strong_reasoning_effort_policy": {
                "configured": args.strong_reasoning_effort,
                "mode": "fixed_high",
            },
            "sticky_compiler_capability_rejects": sorted(
                STICKY_COMPILER_CAPABILITY_REJECTS
            ),
            "sticky_capability_reproposal_requires_certificate_change": True,
            "method_type": "authority_lifted_cfrc",
            "frontier_policy": (
                "open_obligation_pre_write_or_declared_terminal"
            ),
            "terminal_frontier": True,
            "stop_on_first_infrastructure_error": bool(
                getattr(args, "stop_on_first_infrastructure_error", False)
            ),
            "prefix_source": "online_same_environment_cheap_prefix",
            "strong_role": "typed_residual_graph_proposer_only",
            "compiler": "typed_graph_projection_v1",
            "public_schema_visible": True,
            "typed_graph_projection": True,
            "scorer_retry": False,
            "api_docs_injected": False,
            "tools_required_injected": False,
            "answer_visible_to_model": False,
            "task_id_visible_to_model": False,
            "evaluate_with_hidden_answer_after_agent": True,
            "max_graph_nodes": args.max_graph_nodes,
            "max_mutating_nodes": args.max_mutating_nodes,
            "shape_specific_budget_extensions": {
                "linear_max_iterations": max(args.max_iterations, 60),
            },
        },
        "output_boundary": {
            "public_contains_row_prompts": False,
            "public_contains_expected_answers": False,
            "public_contains_tools_required": False,
            "public_contains_task_ids": False,
            "private_raw_rows_append_only": True,
        },
        "source_binding": _source_binding(),
    }
    if platform_smoke is not None:
        report["platform_smoke"] = list(platform_smoke)
    report["public_leakage"] = baseline.public_leakage_audit(report)
    return report


def build_execute_summary(
    *,
    rows: Sequence[baseline.AgentDiffRow],
    results: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    run_id: str,
    started_at: str,
    elapsed_seconds: float,
    private_root_name: str,
    patch_provenance: Mapping[str, Any] | None = None,
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
            baseline.score_number(_mapping(result.get("evaluation")).get("score"))
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
                baseline.score_number(_mapping(result.get("evaluation")).get("score"))
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

    cheap_usage_items = []
    strong_usage_items = []
    cheap_calls = 0
    strong_calls = 0
    cheap_latency = 0.0
    strong_latency = 0.0
    method_decisions: Counter[str] = Counter()
    for result in evaluated:
        agent = _mapping(result.get("agent"))
        cheap_usage_items.append(_mapping(agent.get("cheap_usage")))
        strong_usage_items.append(_mapping(agent.get("strong_usage")))
        cheap_calls += int(agent.get("cheap_provider_call_count") or 0)
        strong_calls += int(agent.get("strong_provider_call_count") or 0)
        cheap_latency += float(agent.get("cheap_model_latency_seconds") or 0.0)
        strong_latency += float(agent.get("strong_model_latency_seconds") or 0.0)
        method_decisions[str(agent.get("method_decision") or "unknown")] += 1

    summary: dict[str, Any] = {
        "schema": "papc.agent_diff.hf_test45_cfrc_public.v1",
        "created_at": baseline.now_iso(),
        "started_at": started_at,
        "private_reached_state_replica": False,
        "incremental_live_receipt_grounding": True,
        "run_id": run_id,
        "decision": "completed" if len(evaluated) == len(rows) else "incomplete",
        "blockers": baseline.summarize_blockers(errors),
        "dataset": {
            "source": "official_agent_diff_hf_local_test_jsonl",
            "split": "test",
            "expected_rows": baseline.EXPECTED_TEST_ROWS,
            "loaded_rows": len(rows),
            "selected_ordinals": [row.ordinal for row in rows],
            "service_counts": dict(sorted(service_counts.items())),
        },
        "model_plan": {
            "method_role": METHOD_ROLE,
            "cheap_model": args.cheap_model,
            "cheap_reasoning_effort": args.cheap_reasoning_effort,
            "residual_strong_model": args.strong_model,
            "residual_strong_reasoning_effort": args.strong_reasoning_effort,
            "api_base_supplied": bool(args.api_base),
            "api_key_env": args.api_key_env,
        },
        "execution_contract": {
            "max_concurrency": args.max_concurrency,
            "registered_default_max_concurrency": (
                baseline.DEFAULT_MAX_CONCURRENCY
            ),
            "scorer_name": scoring.SCORER_NAME,
            "scorer_correction_schema": scoring.SCHEMA,
            "identical_certificate_frontier_attempt_cap": (
                MAX_IDENTICAL_FRONTIER_ATTEMPTS
            ),
            "exact_contract_provider_call_cap": (
                MAX_EXACT_CONTRACT_PROVIDER_CALLS
            ),
            "exact_contract_provider_call_cap_after_witness_failure": (
                MAX_EXACT_CONTRACT_PROVIDER_CALLS_AFTER_WITNESS_FAILURE
            ),
            "readonly_binding_probe_streak_cap": int(
                getattr(
                    args,
                    "max_readonly_binding_probes",
                    MAX_READONLY_BINDING_PROBES,
                )
            ),
            "service_scoped_strong_prompt": bool(
                getattr(args, "service_scoped_strong_prompt", False)
            ),
            "compact_cheap_history": bool(
                getattr(args, "compact_cheap_history", False)
            ),
            "max_strong_residual_frontiers": int(
                getattr(
                    args,
                    "max_strong_residual_rounds",
                    DEFAULT_MAX_STRONG_RESIDUAL_ROUNDS,
                )
            ),
            "strong_completion_limit_field": "max_tokens",
            "reasoning_inclusive_completion_caps_enabled": False,
            "strong_completion_token_cap": int(
                getattr(
                    args,
                    "max_strong_model_tokens",
                    DEFAULT_MAX_STRONG_MODEL_TOKENS,
                )
            ),
            "strong_reasoning_effort_policy": {
                "configured": args.strong_reasoning_effort,
                "mode": "fixed_high",
            },
            "sticky_compiler_capability_rejects": sorted(
                STICKY_COMPILER_CAPABILITY_REJECTS
            ),
            "sticky_capability_reproposal_requires_certificate_change": True,
            "method_type": "authority_lifted_cfrc",
            "frontier_policy": (
                "open_obligation_pre_write_or_declared_terminal"
            ),
            "terminal_frontier": True,
            "stop_on_first_infrastructure_error": bool(
                getattr(args, "stop_on_first_infrastructure_error", False)
            ),
            "prefix_source": "online_same_environment_cheap_prefix",
            "strong_role": "typed_residual_graph_proposer_only",
            "compiler": "typed_graph_projection_v1",
            "public_schema_visible": True,
            "typed_graph_projection": True,
            "scorer_retry": False,
            "api_docs_injected": False,
            "tools_required_injected": False,
            "answer_visible_to_model": False,
            "task_id_visible_to_model": False,
            "evaluate_with_hidden_answer_after_agent": True,
            "max_graph_nodes": args.max_graph_nodes,
            "max_mutating_nodes": args.max_mutating_nodes,
            "shape_specific_budget_extensions": {
                "linear_max_iterations": max(args.max_iterations, 60),
            },
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
        "method_stats": {
            "decision_counts": dict(sorted(method_decisions.items())),
        },
        "agent_resource": {
            "provider_calls": cheap_calls + strong_calls,
            "cheap_provider_calls": cheap_calls,
            "strong_provider_calls": strong_calls,
            "cheap_usage": _usage_dict_sum(cheap_usage_items),
            "strong_usage": _usage_dict_sum(strong_usage_items),
            "model_latency_seconds": round(cheap_latency + strong_latency, 3),
            "cheap_model_latency_seconds": round(cheap_latency, 3),
            "strong_model_latency_seconds": round(strong_latency, 3),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "max_concurrency": args.max_concurrency,
        },
        "acceptance": {
            "target_pass_count": 40,
            "meets_target": pass_count >= 40 and len(evaluated) == len(rows),
            "requires_no_row_errors": len(errors) == 0,
            "public_leakage_required": True,
            "eligible_to_update_current_evidence": (
                pass_count >= 40
                and len(evaluated) == len(rows)
                and len(errors) == 0
            ),
        },
        "private_artifacts": {
            "raw_root_label": private_root_name,
            # A patch run's results describe the merged base denominator, while
            # its own private root holds only the rows it re-ran.
            "raw_rows": (
                len(_sequence(patch_provenance.get("patched_ordinals")))
                if patch_provenance is not None
                else len(results)
            ),
        },
        "source_binding": _source_binding(),
        "public_leakage": {},
    }
    if patch_provenance is not None:
        summary["patch_provenance"] = dict(patch_provenance)
    summary["public_leakage"] = baseline.public_leakage_audit(summary)
    if not summary["public_leakage"]["pass"]:
        summary["decision"] = "blocked_public_leakage"
        summary["blockers"].append("public_leakage_audit_failed")
        summary["acceptance"]["meets_target"] = False
        summary["acceptance"]["eligible_to_update_current_evidence"] = False
    return summary


PATCH_PROVENANCE_SCHEMA = "papc.agent_diff.infrastructure_patch.v1"


@dataclasses.dataclass(frozen=True)
class InfrastructurePatchPlan:
    """Exactly the rows a base run lost to provider infrastructure.

    A relay that answers ``503`` leaves a row with no evaluation, so the base
    run's denominator is short through no fault of the method. Re-running only
    those rows recovers the denominator. The selection is derived from the base
    run's own row files rather than supplied by the caller, because choosing
    which rows to re-run by hand would be selection on outcome.
    """

    base_private_root_label: str
    base_run_id: str | None
    base_rows: tuple[dict[str, Any], ...]
    infrastructure_ordinals: tuple[int, ...]
    blockers: tuple[str, ...]


def load_base_run_rows(base_private_root: Path) -> list[dict[str, Any]]:
    raw_dir = base_private_root / "raw_rows"
    if not raw_dir.is_dir():
        raise RuntimeError(f"patch_base_raw_rows_not_found:{raw_dir}")
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(raw_dir.glob("*.json"))
    ]
    return sorted(
        (dict(row) for row in rows if isinstance(row, Mapping)),
        key=lambda row: int(row.get("ordinal") or 0),
    )


def _base_run_id(base_private_root: Path) -> str | None:
    summary_path = base_private_root / "private_summary.json"
    with contextlib.suppress(OSError, ValueError):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        run_id = _mapping(payload).get("run_id")
        return str(run_id) if run_id else None
    return None


def _requested_ordinals(row_ordinals: str) -> set[int]:
    return {int(chunk.strip()) for chunk in row_ordinals.split(",") if chunk.strip()}


def _ordinal_csv(ordinals: Iterable[int]) -> str:
    return ",".join(str(int(ordinal)) for ordinal in ordinals)


def build_infrastructure_patch_plan(
    base_private_root: Path, *, row_ordinals: str | None
) -> InfrastructurePatchPlan:
    base_rows = load_base_run_rows(base_private_root)
    infrastructure_ordinals = tuple(
        sorted(
            int(row.get("ordinal") or 0)
            for row in base_rows
            if row.get("status") == "error"
        )
    )
    blockers: list[str] = []
    if not infrastructure_ordinals:
        blockers.append("patch_base_has_no_infrastructure_rows")
    if row_ordinals and _requested_ordinals(row_ordinals) != set(
        infrastructure_ordinals
    ):
        # An explicit selection may narrow nothing and add nothing. A row the
        # base run evaluated already has an outcome, so re-running it would
        # re-roll that outcome instead of recovering a lost one.
        blockers.append("patch_selection_is_not_the_base_infrastructure_set")
    return InfrastructurePatchPlan(
        base_private_root_label=base_private_root.name,
        base_run_id=_base_run_id(base_private_root),
        base_rows=tuple(base_rows),
        infrastructure_ordinals=infrastructure_ordinals,
        blockers=tuple(blockers),
    )


def merge_infrastructure_patch(
    *,
    plan: InfrastructurePatchPlan,
    patch_results: Sequence[Mapping[str, Any]],
    all_rows: Sequence[baseline.AgentDiffRow],
) -> tuple[list[baseline.AgentDiffRow], list[dict[str, Any]], dict[str, Any]]:
    """The base denominator with this run's rows substituted for the lost ones.

    Both runs were really paid for, so the merged rows carry the base run's
    usage for every row it evaluated plus this run's usage for the rows it
    re-ran; a row that errors again stays an error and is named in the
    provenance rather than retried.
    """

    merged: dict[int, dict[str, Any]] = {
        int(row.get("ordinal") or 0): dict(row) for row in plan.base_rows
    }
    for result in patch_results:
        merged[int(result.get("ordinal") or 0)] = dict(result)
    merged_results = [merged[ordinal] for ordinal in sorted(merged)]
    merged_rows = baseline.select_rows(
        all_rows, max_rows=None, row_ordinals=_ordinal_csv(sorted(merged))
    )
    provenance = {
        "schema": PATCH_PROVENANCE_SCHEMA,
        # The label, not the absolute path: a private root path would trip the
        # public leakage audit that every public summary must pass.
        "base_private_root": plan.base_private_root_label,
        "base_run_id": plan.base_run_id,
        "patched_ordinals": list(plan.infrastructure_ordinals),
        "patch_reason": "infrastructure_error_only",
        "rows_still_error_after_patch": sorted(
            int(result.get("ordinal") or 0)
            for result in patch_results
            if result.get("status") != "evaluated"
        ),
    }
    return merged_rows, merged_results, provenance


def validate_args(
    args: argparse.Namespace, rows: Sequence[baseline.AgentDiffRow]
) -> list[str]:
    blockers: list[str] = []
    # A patch run's row set is the base run's infrastructure set, so the
    # official-45 shape is checked against the merged denominator instead.
    subset_authorized = bool(args.row_ordinals) or bool(
        getattr(args, "patch_infra_from", None)
    )
    if not subset_authorized and len(rows) != baseline.EXPECTED_TEST_ROWS:
        blockers.append("official_test_row_count_not_45")
    if (
        not subset_authorized
        and args.max_rows is not None
        and args.max_rows != baseline.EXPECTED_TEST_ROWS
    ):
        blockers.append("max_rows_not_official_45")
    # Concurrency is a scheduling choice, not part of the arm definition: any
    # value up to the shared online ceiling is legal, so a diagnostic can yield
    # provider capacity to a concurrent online run and a full run can take the
    # capacity back. The report records both the value used and the registered
    # default, so a readout can always see whether a run scheduled itself below
    # or above the reference rate.
    if args.max_concurrency < 1:
        blockers.append("max_concurrency_below_1")
    if args.max_concurrency > MAX_ONLINE_CONCURRENCY:
        blockers.append(f"max_concurrency_over_{MAX_ONLINE_CONCURRENCY}")
    if (
        args.execute
        and float(
            getattr(
                args,
                "provider_timeout_seconds",
                DEFAULT_PROVIDER_TIMEOUT_SECONDS,
            )
        )
        < DEFAULT_PROVIDER_TIMEOUT_SECONDS
    ):
        blockers.append("provider_timeout_below_300")
    if (args.cheap_model, args.strong_model) not in REGISTERED_MODEL_POOLS:
        blockers.append("model_pool_not_registered")
    if args.cheap_reasoning_effort != DEFAULT_REASONING_EFFORT:
        blockers.append("cheap_reasoning_not_high")
    if args.strong_reasoning_effort != DEFAULT_STRONG_REASONING_EFFORT:
        blockers.append("strong_reasoning_not_high")
    if int(
        getattr(
            args,
            "max_strong_model_tokens",
            DEFAULT_MAX_STRONG_MODEL_TOKENS,
        )
    ) < 1:
        blockers.append("max_strong_model_tokens_invalid")
    if args.max_graph_nodes < 1:
        blockers.append("max_graph_nodes_invalid")
    if args.max_mutating_nodes < 1:
        blockers.append("max_mutating_nodes_invalid")
    if not args.api_base:
        blockers.append("api_base_missing")
    if args.execute and not os.environ.get(args.api_key_env):
        blockers.append("api_key_env_missing")
    unknown_services = sorted(
        {row.service for row in rows} - set(SERVICE_OPERATION_GRAMMAR)
    )
    if unknown_services:
        blockers.append("unknown_service:" + ",".join(unknown_services))
    return blockers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path", type=Path, default=baseline.DEFAULT_DATASET_PATH
    )
    parser.add_argument(
        "--agent-diff-sdk-path", type=Path, default=baseline.AGENT_DIFF_SDK_PATH
    )
    parser.add_argument("--agent-diff-base-url", default="http://localhost:8020")
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", ""))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--cheap-model", default=DEFAULT_CHEAP_MODEL)
    parser.add_argument("--cheap-reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--strong-model", default=DEFAULT_STRONG_MODEL)
    parser.add_argument(
        "--strong-reasoning-effort",
        default=DEFAULT_STRONG_REASONING_EFFORT,
        choices=(DEFAULT_STRONG_REASONING_EFFORT,),
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=baseline.DEFAULT_MAX_CONCURRENCY
    )
    parser.add_argument("--max-rows", type=int, default=baseline.EXPECTED_TEST_ROWS)
    parser.add_argument(
        "--row-ordinals",
        default=None,
        help="Comma-separated official row ordinals for targeted diagnostic runs.",
    )
    parser.add_argument(
        "--patch-infra-from",
        type=Path,
        default=None,
        help=(
            "Private root of a base run whose infrastructure errors this run "
            "re-runs. Only rows the base run failed to evaluate are eligible, "
            "and the public summary reports the merged base denominator."
        ),
    )
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--max-model-tokens", type=int, default=1600)
    parser.add_argument(
        "--max-strong-model-tokens",
        type=int,
        default=DEFAULT_MAX_STRONG_MODEL_TOKENS,
    )
    parser.add_argument(
        "--provider-timeout-seconds",
        type=float,
        default=DEFAULT_PROVIDER_TIMEOUT_SECONDS,
    )
    parser.add_argument("--provider-max-attempts", type=int, default=3)
    parser.add_argument("--command-timeout-seconds", type=int, default=30)
    parser.add_argument("--observation-chars", type=int, default=12000)
    parser.add_argument("--env-ttl-seconds", type=int, default=7200)
    parser.add_argument(
        "--stop-on-first-infrastructure-error",
        action="store_true",
        help=(
            "Stop scheduling provider work after the first timeout or "
            "transport error."
        ),
    )
    parser.add_argument("--max-graph-nodes", type=int, default=DEFAULT_MAX_GRAPH_NODES)
    parser.add_argument(
        "--max-mutating-nodes", type=int, default=DEFAULT_MAX_MUTATING_NODES
    )
    parser.add_argument(
        "--min-frontier-observations",
        type=int,
        default=DEFAULT_MIN_FRONTIER_OBSERVATIONS,
    )
    parser.add_argument(
        "--max-strong-residual-rounds",
        type=int,
        default=DEFAULT_MAX_STRONG_RESIDUAL_ROUNDS,
    )
    parser.add_argument(
        "--exact-contract-provider-call-cap",
        type=int,
        default=None,
        help=(
            "How many samples one frozen contract may buy. The default of two "
            "is an initial proposal plus one feedback-conditioned repair. "
            "Setting it to one keeps the initial proposal and drops the repair, "
            "which is the operating point to use when the surface has quality "
            "headroom over the strong reference and the repair rounds are the "
            "largest remaining cost. B_call reports whatever is set here, so "
            "the successor is told the budget it actually has."
        ),
    )
    parser.add_argument(
        "--max-readonly-binding-probes",
        type=int,
        default=MAX_READONLY_BINDING_PROBES,
    )
    parser.add_argument(
        "--service-scoped-strong-prompt",
        action="store_true",
        help=(
            "Drop the strong system prompt rule blocks that only govern "
            "services other than the certificate's own. The compiler "
            "already rejects any action outside that service, so the "
            "dropped blocks cannot change what the successor may emit."
        ),
    )
    parser.add_argument(
        "--compact-cheap-history",
        action="store_true",
        help=(
            "Re-send already-read observations in the compacted evidence "
            "form the reached-state contract records, instead of re-quoting "
            "the raw payload on every later prefix turn. The newest "
            "observation is always sent in full."
        ),
    )
    parser.add_argument(
        "--no-replica-witness",
        action="store_true",
        help=(
            "Skip the replica reached-state witness before live execution. "
            "The default runs every admitted graph on a replayed replica "
            "environment first, so only a witnessed suffix touches live state."
        ),
    )
    parser.add_argument("--platform-smoke-count", type=int, default=0)
    parser.add_argument("--run-id", default=baseline.run_id_default())
    parser.add_argument(
        "--public-output-dir", type=Path, default=baseline.DEFAULT_PUBLIC_OUTPUT
    )
    parser.add_argument(
        "--private-root", type=Path, default=baseline.DEFAULT_PRIVATE_ROOT
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cap = getattr(args, "exact_contract_provider_call_cap", None)
    if cap is not None:
        # Bound once at entry so every budget site, and the contract the
        # successor is shown, report the same number.
        module = sys.modules[__name__]
        module.MAX_EXACT_CONTRACT_PROVIDER_CALLS = max(1, int(cap))
        module.MAX_EXACT_CONTRACT_PROVIDER_CALLS_AFTER_WITNESS_FAILURE = max(
            1, int(cap)
        )
    all_rows = baseline.load_official_test_rows(args.dataset_path)
    rows = baseline.select_rows(
        all_rows, max_rows=args.max_rows, row_ordinals=args.row_ordinals
    )
    patch_plan = (
        build_infrastructure_patch_plan(
            args.patch_infra_from, row_ordinals=args.row_ordinals
        )
        if args.patch_infra_from is not None
        else None
    )
    if patch_plan is not None and not patch_plan.blockers:
        rows = baseline.select_rows(
            all_rows,
            max_rows=None,
            row_ordinals=_ordinal_csv(patch_plan.infrastructure_ordinals),
        )
    blockers = validate_args(args, rows)
    if patch_plan is not None:
        blockers.extend(patch_plan.blockers)
    platform_smoke = None
    if args.platform_smoke_count:
        platform_smoke = baseline.execute_platform_smoke(
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
        output = (
            args.public_output_dir
            / f"agent_diff_hf_test45_cfrc_preflight_{args.run_id}.json"
        )
        baseline.write_json_append_only(output, report)
        print(json.dumps({"output": str(output), "decision": report["decision"]}))
        return 0 if not blockers else 2

    api_key = os.environ[args.api_key_env]
    started_at = baseline.now_iso()
    started = time.monotonic()
    private_root_name = f"agent_diff_hf_test45_cfrc_private_{args.run_id}"
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
        "schema": "papc.agent_diff.hf_test45_cfrc_private_summary.v1",
        "created_at": baseline.now_iso(),
        "run_id": args.run_id,
        "rows": results,
    }
    baseline.write_json_append_only(
        private_root / "private_summary.json", private_summary
    )
    summary_rows: Sequence[baseline.AgentDiffRow] = rows
    summary_results: Sequence[Mapping[str, Any]] = results
    patch_provenance: dict[str, Any] | None = None
    if patch_plan is not None:
        summary_rows, summary_results, patch_provenance = merge_infrastructure_patch(
            plan=patch_plan, patch_results=results, all_rows=all_rows
        )
    public_summary = build_execute_summary(
        rows=summary_rows,
        results=summary_results,
        args=args,
        run_id=args.run_id,
        started_at=started_at,
        elapsed_seconds=time.monotonic() - started,
        private_root_name=private_root_name,
        patch_provenance=patch_provenance,
    )
    public_output = (
        args.public_output_dir / f"agent_diff_hf_test45_cfrc_public_{args.run_id}.json"
    )
    baseline.write_json_append_only(public_output, public_summary)
    print(
        json.dumps(
            {
                "output": str(public_output),
                "decision": public_summary["decision"],
                "aggregate": public_summary["aggregate"],
                "acceptance": public_summary["acceptance"],
            },
            sort_keys=True,
        )
    )
    return 0 if public_summary["decision"] == "completed" else 2


if __name__ == "__main__":
    tempfile.tempdir = str(baseline.DEFAULT_PRIVATE_ROOT)
    raise SystemExit(main())
