#!/usr/bin/env python3
"""Print the cache-neutral agent-side cost of PAPC run artifacts.

Reads one or more public run artifacts from any of the five surfaces and prints
one table row each: surface, cost pool, reported quality, cheap and strong
dollars at frozen public prices with no cache discount, the total, the same-pool
official cheap single-run denominator, and the multiple of that denominator.

This is a read-only offline tool. It never calls a provider and never writes to
the artifacts it reads.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from papc.certspec.cost_readout import (  # noqa: E402
    CHEAP_ANCHORS,
    CostReadout,
    RoleUsage,
    format_latency_table,
    format_table,
    read_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="papc_cost_readout.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "artifacts",
        nargs="+",
        help="run artifact JSON paths (shell globs expand to several paths)",
    )
    parser.add_argument(
        "--sum-shards",
        action="store_true",
        help=(
            "also print one summed row over every artifact given; only valid "
            "when they are shards of one run, meaning same surface and pool"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit the readouts as JSON instead of a table",
    )
    parser.add_argument(
        "--anchors",
        action="store_true",
        help="print the ten registered cheap anchors and exit",
    )
    parser.add_argument(
        "--latency",
        action="store_true",
        help=(
            "print summed provider latency per role and per denominator row "
            "instead of cost; wall clock is deliberately omitted because it "
            "only reports the concurrency the run happened to use"
        ),
    )
    parser.add_argument(
        "--against",
        metavar="CANDIDATE",
        help=(
            "read the positional artifacts as same-configuration repeats of a "
            "parent and report how far this candidate sits from their mean, in "
            "units of the parent's own standard error; a difference under two "
            "of those may not be called cheaper or dearer"
        ),
    )
    return parser


def report_against_parent(
    parent: Sequence[CostReadout], candidate: CostReadout
) -> str:
    """State a candidate's distance from its parent in the parent's own units.

    Three separate conclusions were drawn on this project from differences
    smaller than the parent's spread, twice calling a candidate dearer and once
    calling one cheaper. A percentage hides that; a multiple of the standard
    error cannot.
    """

    costs = [readout.agent_side_usd for readout in parent]
    n = len(costs)
    mean = sum(costs) / n
    lines = [
        f"parent  n={n}  mean ${mean:.6f}",
        "  shots: " + ", ".join(f"${c:.6f}" for c in costs),
    ]
    if n < 2:
        lines.append(
            "  one shot names no spread, so no candidate can be judged against it"
        )
        lines.append(
            f"candidate ${candidate.agent_side_usd:.6f}  "
            f"{100 * (candidate.agent_side_usd / mean - 1):+.2f}% (undecidable)"
        )
        return "\n".join(lines)
    variance = sum((c - mean) ** 2 for c in costs) / (n - 1)
    sd = variance**0.5
    se = sd / n**0.5
    delta = candidate.agent_side_usd - mean
    lines.append(f"  SD ${sd:.6f} ({100 * sd / mean:.1f}%)   SE ${se:.6f} ({100 * se / mean:.1f}%)")
    lines.append(
        f"candidate ${candidate.agent_side_usd:.6f}  "
        f"{100 * (delta / mean):+.2f}%  = {delta / se:+.2f} SE"
    )
    verdict = (
        "clears two standard errors"
        if abs(delta) >= 2 * se
        else "under two standard errors, so it is undecided and may not be "
        "reported as cheaper or dearer"
    )
    lines.append(f"  {verdict}")
    return "\n".join(lines)


def sum_shards(readouts: Sequence[CostReadout]) -> CostReadout:
    """Fold same-run shards into one readout.

    Dollars are summed, not repriced, so the fold is exact for shards that were
    each priced with the same model pair.
    """

    surfaces = {readout.surface for readout in readouts}
    pools = {readout.pool for readout in readouts}
    if len(surfaces) != 1 or len(pools) != 1:
        raise SystemExit(
            f"--sum-shards needs one surface and one pool, got "
            f"surfaces={sorted(surfaces)} pools={sorted(pools)}"
        )
    first = readouts[0]
    cheap = _sum_usage(readout.cheap for readout in readouts)
    strong = _sum_usage(readout.strong for readout in readouts)
    cheap_usd = sum(readout.cheap_usd for readout in readouts)
    strong_usd = sum(readout.strong_usd for readout in readouts)
    agent_side_usd = cheap_usd + strong_usd
    anchor = CHEAP_ANCHORS.get((first.surface, first.pool))
    anchor_usd = anchor.agent_side_usd if anchor is not None else None
    multiple = agent_side_usd / anchor_usd if anchor_usd else None
    reported_parts = [
        readout.reported_agent_side_usd
        for readout in readouts
        if readout.reported_agent_side_usd is not None
    ]
    reported = sum(reported_parts) if len(reported_parts) == len(readouts) else None
    return CostReadout(
        artifact=f"<sum of {len(readouts)} artifacts>",
        surface=first.surface,
        pool=first.pool,
        cheap_model=first.cheap_model,
        strong_model=first.strong_model,
        quality_label=None,
        cheap=cheap,
        strong=strong,
        cheap_usd=cheap_usd,
        strong_usd=strong_usd,
        agent_side_usd=agent_side_usd,
        anchor_usd=anchor_usd,
        multiple=multiple,
        within_two_x=multiple <= 2.0 if multiple is not None else None,
        reported_agent_side_usd=reported,
        discrepancy_usd=agent_side_usd - reported if reported is not None else None,
    )


def _sum_usage(usages: object) -> RoleUsage:
    calls = 0
    calls_seen = False
    input_tokens = 0
    output_tokens = 0
    for usage in usages:  # type: ignore[union-attr]
        if usage.calls is not None:
            calls += usage.calls
            calls_seen = True
        input_tokens += usage.input_tokens
        output_tokens += usage.output_tokens
    return RoleUsage(
        calls=calls if calls_seen else None,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _print_anchors() -> None:
    rows = [("surface", "pool", "quality", "agent-side $", "source")]
    for key in sorted(CHEAP_ANCHORS):
        anchor = CHEAP_ANCHORS[key]
        rows.append(
            (
                anchor.surface,
                anchor.pool,
                anchor.quality_label,
                f"{anchor.agent_side_usd:.6f}",
                anchor.source_note,
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(4)]
    for row in rows:
        head = "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row[:4]))
        print(f"{head}  {row[4]}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.anchors:
        _print_anchors()
        return 0

    readouts: list[CostReadout] = []
    failures: list[str] = []
    for artifact in args.artifacts:
        try:
            readouts.append(read_artifact(artifact))
        except ValueError as error:
            failures.append(str(error))

    if args.against:
        if failures:
            for failure in failures:
                print(f"unreadable: {failure}", file=sys.stderr)
            return 1
        try:
            candidate = read_artifact(args.against)
        except ValueError as error:
            print(f"unreadable candidate: {error}", file=sys.stderr)
            return 1
        print(report_against_parent(readouts, candidate))
        return 0

    if args.sum_shards and readouts:
        readouts.append(sum_shards(readouts))

    if args.as_json:
        print(json.dumps([dataclasses.asdict(r) for r in readouts], indent=2))
    elif readouts:
        print(
            format_latency_table(readouts)
            if getattr(args, "latency", False)
            else format_table(readouts)
        )

    for failure in failures:
        print(f"unreadable: {failure}", file=sys.stderr)
    if not readouts:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
