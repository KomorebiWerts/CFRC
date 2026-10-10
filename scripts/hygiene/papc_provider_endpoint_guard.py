#!/usr/bin/env python3
"""Provider endpoint guard for paid PAPC experiment launches."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

SCHEMA = "papc.provider_endpoint_guard.v1"
PASS_DECISION = "provider_endpoint_guard_passed"
BLOCKED_DECISION = "blocked_provider_endpoint_guard"
DEFAULT_OUTPUT = Path("outputs/papc/provider_endpoint_guard_20260526.json")
DEFAULT_REQUIRED_HOST = "127.0.0.1"
LOCAL_PROVIDER_HOSTS = ("127.0.0.1", "localhost")
REQUIRED_LOCAL_NO_PROXY_HOSTS = ("localhost", "127.0.0.1")
PROXY_ENV_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
RUNNER_PATTERNS = (
    "papc_agent_diff_hf_test_baseline_run.py --execute",
    "papc_agent_diff_hf_test_cfrc_run.py --execute",
    "papc_mtrouter_agent_diff_lightweight.py --mode execute --execute",
    "papc_tau2_official_run_with_evaluator_patch.py",
    "papc_tau2_cfrc_clean_run.py --execute",
    "papc_tau2_airline_residual_run.py --execute",
    "papc_toolsandbox_base129_true_online_run.py --execute",
    "papc_toolsandbox_base129_official_baseline_accounting_run.py --execute",
    "papc_state_bench_true_live_cfrc_run.py --execute",
)

HttpGet = Callable[[str, str, float], Mapping[str, Any]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", ""))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--require-host", action="append", default=[DEFAULT_REQUIRED_HOST]
    )
    parser.add_argument("--models-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--execute-models-check", action="store_true")
    parser.add_argument("--max-active-runner-count", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output_json),
                "decision": report["decision"],
                "blockers": report["blockers"],
                "models_check_ok": report["models_check"].get("ok"),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["decision"] == PASS_DECISION else 2


def build_report(
    args: argparse.Namespace,
    *,
    ps_text: str | None = None,
    http_get: HttpGet | None = None,
) -> dict[str, Any]:
    api_key_present = bool(os.environ.get(args.api_key_env))
    api_host = urlparse(args.api_base).hostname if args.api_base else ""
    active_runners = active_runner_lines(
        current_ps_text() if ps_text is None else ps_text
    )
    blockers = env_blockers(
        api_base=args.api_base,
        api_host=api_host or "",
        api_key_present=api_key_present,
        required_hosts=args.require_host,
        active_runner_count=len(active_runners),
        max_active_runner_count=args.max_active_runner_count,
    )
    models_check = {"attempted": False, "ok": None}
    if args.execute_models_check and not blockers:
        models_check = models_endpoint_check(
            api_base=args.api_base,
            api_key=os.environ.get(args.api_key_env, ""),
            timeout_seconds=args.models_timeout_seconds,
            http_get=http_get or http_get_json,
        )
        if models_check.get("ok") is not True:
            blockers.append("models_endpoint_check_failed")
    decision = PASS_DECISION if not blockers else BLOCKED_DECISION
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "api_spend_performed": False,
        "generation_performed": False,
        "models_check": models_check,
        "api_base_host": api_host,
        "api_key_env_supplied": bool(args.api_key_env),
        "api_key_present": api_key_present,
        "required_hosts": list(args.require_host),
        "proxy_env_set": [name for name in PROXY_ENV_VARS if os.environ.get(name)],
        "no_proxy_tokens": sorted(no_proxy_tokens()),
        "active_runner_count": len(active_runners),
        "active_runner_lines": active_runners,
        "max_active_runner_count": args.max_active_runner_count,
        "blockers": sorted(set(blockers)),
    }


def env_blockers(
    *,
    api_base: str,
    api_host: str,
    api_key_present: bool,
    required_hosts: Sequence[str],
    active_runner_count: int,
    max_active_runner_count: int,
) -> list[str]:
    blockers: list[str] = []
    if not api_base:
        blockers.append("api_base_missing")
    if not api_key_present:
        blockers.append("api_key_env_missing")
    if required_hosts and api_host not in set(required_hosts):
        blockers.append(f"api_host_not_allowed:{api_host or 'missing'}")
    if api_host in set(required_hosts):
        for name in PROXY_ENV_VARS:
            if os.environ.get(name):
                blockers.append(f"proxy_env_must_be_unset:{name}")
        tokens = no_proxy_tokens()
        for host in required_no_proxy_hosts(required_hosts):
            if host not in tokens:
                blockers.append(f"no_proxy_missing:{host}")
    if active_runner_count > max_active_runner_count:
        blockers.append("active_runner_count_exceeds_limit")
    return blockers


def models_endpoint_check(
    *,
    api_base: str,
    api_key: str,
    timeout_seconds: float,
    http_get: HttpGet,
) -> dict[str, Any]:
    started = time.monotonic()
    url = provider_check_url(api_base)
    try:
        response = http_get(url, api_key, timeout_seconds)
    except Exception as exc:
        return {
            "attempted": True,
            "ok": False,
            "latency_seconds": round(time.monotonic() - started, 3),
            "error_type": type(exc).__name__,
            "error_message": truncate(str(exc), 240),
        }
    return {
        "attempted": True,
        "ok": int(response.get("status") or 0) == 200,
        "latency_seconds": round(time.monotonic() - started, 3),
        "url_path": urlparse(url).path,
        "status": response.get("status"),
        "body_prefix": truncate(str(response.get("body_prefix") or ""), 120),
    }


def provider_check_url(api_base: str) -> str:
    parsed = urlparse(api_base.rstrip("/"))
    if parsed.hostname in set(LOCAL_PROVIDER_HOSTS):
        return urlunparse((parsed.scheme, parsed.netloc, "/healthz", "", "", ""))
    return api_base.rstrip("/") + "/models"


def required_no_proxy_hosts(required_hosts: Sequence[str]) -> tuple[str, ...]:
    hosts = {host for host in required_hosts if host}
    hosts.update(REQUIRED_LOCAL_NO_PROXY_HOSTS)
    return tuple(sorted(hosts))


def http_get_json(url: str, api_key: str, timeout_seconds: float) -> dict[str, Any]:
    request = Request(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            body = response.read(4096).decode("utf-8", errors="replace")
            return {"status": response.status, "body_prefix": body[:240]}
    except HTTPError as exc:
        body = exc.read(4096).decode("utf-8", errors="replace")
        return {"status": exc.code, "body_prefix": body[:240]}
    except URLError as exc:
        raise RuntimeError(str(exc.reason)) from exc


def current_ps_text() -> str:
    user = os.environ.get("USER") or ""
    command = ["ps", "-f", "-u", user] if user else ["ps", "-ef"]
    return subprocess.check_output(command, text=True)


def active_runner_lines(ps_text: str) -> list[str]:
    rows: list[str] = []
    for line in ps_text.splitlines():
        if "papc_provider_endpoint_guard.py" in line:
            continue
        if "pytest" in line or "/tests/" in line or "tests/" in line:
            continue
        if any(pattern in line for pattern in RUNNER_PATTERNS):
            rows.append(line.strip())
    return rows


def no_proxy_tokens() -> set[str]:
    return {
        token.strip()
        for value in (os.environ.get("NO_PROXY", ""), os.environ.get("no_proxy", ""))
        for token in value.split(",")
        if token.strip()
    }


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 14)] + "...[truncated]"


if __name__ == "__main__":
    raise SystemExit(main())
