#!/usr/bin/env python3
"""Unified CFRC entry point.

    papc-run <surface> <mode> [runner args...]
    papc-run points [<name> --run-id <id> [--execute] [--concurrency N]]
    papc-run cost <artifact...>

Each (surface, mode) pair forwards its trailing arguments to one runner script.
Without ``--execute`` a runner only writes a provider-free preflight.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Model pools reported in the paper: (cheap prefix model, strong successor).
# Reasoning effort is fixed at high for every pool. The ``openai/`` prefix is a
# LiteLLM routing hint; ``-native`` pools spell the id for surfaces that call
# the provider SDK directly.
MODEL_POOL_ARMS: dict[str, tuple[str, str]] = {
    "gpt54mini-gpt55-high": ("gpt-5.4-mini", "gpt-5.5"),
    "gpt56-luna-sol-high": ("gpt-5.6-luna", "gpt-5.6-sol"),
    "gpt56luna-claude-sonnet5-high": ("gpt-5.6-luna", "openai/claude-sonnet-5"),
    "gpt56luna-claude-sonnet5-native": ("gpt-5.6-luna", "claude-sonnet-5"),
    "gpt56luna-gemini37flash-high": ("gpt-5.6-luna", "gemini-3.7-flash-high"),
    "claudehaiku45-gpt56sol-high": (
        "openai/claude-haiku-4-5-20251001",
        "gpt-5.6-sol",
    ),
}
MODEL_POOL_FLAG = "--model-pool-arm"
EFFORT_FLAGS = ("--cheap-reasoning-effort", "--strong-reasoning-effort")

# (surface, mode) -> runner script, relative to the repository root.
SURFACE_RUNNERS: dict[tuple[str, str], str] = {
    ("tau2", "retail"): "scripts/tau2/papc_tau2_cfrc_clean_run.py",
    ("tau2", "airline"): "scripts/tau2/papc_tau2_airline_residual_run.py",
    ("tau2", "baseline"): "scripts/tau2/papc_tau2_official_run_with_evaluator_patch.py",
    ("tau2", "airline-baseline"): "scripts/tau2/papc_tau2_airline_official_baseline_run.py",
    ("tau2", "claude-baseline"): "scripts/tau2/papc_tau2_claude_official_baseline_run.py",
    ("state", "method"): "scripts/state_bench/live/papc_state_bench_true_live_cfrc_run.py",
    ("state", "official-score"): "scripts/state_bench/live/papc_state_bench_official_score_run.py",
    ("state", "score"): "scripts/state_bench/papc_state_bench_true_live_score_aggregate.py",
    ("state", "accounting"): "scripts/state_bench/papc_state_bench_runtime_accounting.py",
    ("agent-diff", "method"): "scripts/agent_diff/papc_agent_diff_hf_test_cfrc_run.py",
    ("agent-diff", "baseline"): "scripts/agent_diff/papc_agent_diff_hf_test_baseline_run.py",
    ("toolsandbox", "method"): "scripts/toolsandbox/papc_toolsandbox_base129_true_online_run.py",
    ("toolsandbox", "baseline"): (
        "scripts/toolsandbox/papc_toolsandbox_base129_official_baseline_accounting_run.py"
    ),
}
SURFACES = sorted({surface for surface, _ in SURFACE_RUNNERS})


def modes_for(surface: str) -> list[str]:
    return sorted(mode for s, mode in SURFACE_RUNNERS if s == surface)


def usage() -> str:
    lines = ["Usage: papc-run <surface> <mode> [runner args...]", "", "Surfaces and modes:"]
    lines += [f"  {s:<12} {', '.join(modes_for(s))}" for s in SURFACES]
    lines += [
        "",
        "  points [<name> --run-id <id> [--execute] [--concurrency N]]",
        "               list registered operating points, or print one launch command",
        "  cost <artifact...>",
        "               cache-neutral agent-side cost of run artifacts",
        "",
        f"Model pools ({MODEL_POOL_FLAG}): " + ", ".join(MODEL_POOL_ARMS),
        "Without --execute a runner writes a provider-free preflight.",
    ]
    return "\n".join(lines)


def resolve(surface: str, mode: str) -> Path | None:
    rel = SURFACE_RUNNERS.get((surface, mode))
    return REPO_ROOT / rel if rel else None


def runner_environment() -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(REPO_ROOT / "src"), str(REPO_ROOT)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def expand_model_pool_arm(args: list[str]) -> list[str]:
    """Replace ``--model-pool-arm <arm>`` with explicit model and effort flags."""

    if MODEL_POOL_FLAG not in args:
        return list(args)
    index = args.index(MODEL_POOL_FLAG)
    if index + 1 >= len(args):
        raise ValueError(f"{MODEL_POOL_FLAG} requires an arm name")
    arm = args[index + 1]
    if arm not in MODEL_POOL_ARMS:
        raise ValueError(f"unknown model-pool arm {arm!r}; valid: {', '.join(MODEL_POOL_ARMS)}")
    rest = args[:index] + args[index + 2 :]
    if {"--cheap-model", "--strong-model"} & set(rest):
        raise ValueError(f"{MODEL_POOL_FLAG} cannot be combined with explicit model flags")
    cheap, strong = MODEL_POOL_ARMS[arm]
    expanded = [*rest, "--cheap-model", cheap, "--strong-model", strong]
    for flag in EFFORT_FLAGS:
        if flag not in rest:
            expanded += [flag, "high"]
        elif rest[rest.index(flag) + 1 : rest.index(flag) + 2] != ["high"]:
            raise ValueError(f"{flag} must be high for a registered model-pool arm")
    return expanded


def _flag_value(args: list[str], flag: str) -> str | None:
    if flag not in args:
        return None
    index = args.index(flag)
    if index + 1 >= len(args):
        raise ValueError(f"{flag} requires a value")
    return args[index + 1]


def points_command(argv: list[str]) -> int:
    from papc.certspec.operating_points import (
        OPERATING_POINTS,
        UnsupportedLaunch,
        format_points,
        launch_argv,
    )

    if not argv:
        print(format_points())
        return 0
    point = OPERATING_POINTS.get(argv[0])
    if point is None:
        print(f"unknown operating point {argv[0]!r}; valid: {', '.join(OPERATING_POINTS)}", file=sys.stderr)
        return 2
    rest = argv[1:]
    try:
        run_id = _flag_value(rest, "--run-id")
        concurrency = _flag_value(rest, "--concurrency")
        if not run_id:
            raise ValueError("--run-id is required to name the artifacts")
        command = launch_argv(
            point,
            run_id=run_id,
            execute="--execute" in rest,
            concurrency=int(concurrency) if concurrency else None,
        )
    except (UnsupportedLaunch, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(shlex.join(command))
    for note in point.notes:
        print(f"# {note}", file=sys.stderr)
    return 0


def _run_script(script: Path, args: list[str]) -> int:
    command = [sys.executable, str(script), *args]
    return subprocess.run(command, cwd=str(REPO_ROOT), env=runner_environment()).returncode


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return 0
    if argv[0] == "points":
        return points_command(argv[1:])
    if argv[0] == "cost":
        return _run_script(REPO_ROOT / "scripts/papc_cost_readout.py", argv[1:])
    surface = argv[0]
    if surface not in SURFACES:
        print(f"unknown surface {surface!r}\n\n{usage()}", file=sys.stderr)
        return 2
    if len(argv) < 2 or resolve(surface, argv[1]) is None:
        print(f"surface {surface!r} needs a mode: {', '.join(modes_for(surface))}", file=sys.stderr)
        return 2
    try:
        args = expand_model_pool_arm(argv[2:])
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return _run_script(resolve(surface, argv[1]), args)


if __name__ == "__main__":
    raise SystemExit(main())
