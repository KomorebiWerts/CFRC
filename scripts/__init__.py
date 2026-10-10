"""Compatibility imports for categorized PAPC scripts.

New code should import scripts from their category package, for example
``scripts.state_bench.live.papc_state_bench_true_live_cfrc_run``.
The lazy fallback keeps older ``from scripts import papc_g...`` imports working
while the repository moves away from a flat scripts directory.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from types import ModuleType

_CATEGORIES = (
    "theory",
    "hygiene",
    "agent_diff",
    "tau2",
    "state_bench",
    "state_bench.common",
    "state_bench.live",
    "toolsandbox",
)


def __getattr__(name: str) -> ModuleType:
    if not name.startswith("papc_"):
        raise AttributeError(name)
    for category in _CATEGORIES:
        module_name = f"{__name__}.{category}.{name}"
        if importlib.util.find_spec(module_name) is None:
            continue
        module = importlib.import_module(module_name)
        globals()[name] = module
        sys.modules[f"{__name__}.{name}"] = module
        return module
    raise AttributeError(name)
