"""Shared fail-closed runtime environment for external-router live paths."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
from collections.abc import Mapping
from typing import Any

from papc.paths import DATA_ROOT

CANONICAL_ROUTER_RUNTIME_EXECUTABLE = (
    f"{DATA_ROOT}/papc/upstream_envs/"
    "external_routers_py312/bin/python"
)
CANONICAL_ROUTER_RUNTIME_PREFIX = (
    f"{DATA_ROOT}/papc/upstream_envs/external_routers_py312"
)
ROUTER_RUNTIME_PACKAGES: Mapping[str, str] = {
    "azure-core": "1.38.0",
    "azure-identity": "1.25.1",
    "joblib": "1.5.3",
    "lightgbm": "4.6.0",
    "numpy": "1.26.4",
    "onnxruntime": "1.28.0",
    "openai": "2.16.0",
    "pandas": "2.3.3",
    "python-dotenv": "1.2.1",
    "pyyaml": "6.0.3",
    "scikit-learn": "1.8.0",
    "scipy": "1.14.1",
    "sentence-transformers": "5.6.1",
    "tenacity": "9.1.2",
    "tiktoken": "0.12.0",
    "tokenizers": "0.22.2",
    "torch": "2.11.0",
    "transformers": "5.14.1",
}
ROUTER_RUNTIME_MODULES = (
    "papc.baselines.routellm_mf_upstream",
    "papc.baselines.frugalgpt_upstream",
    "papc.baselines.automix_upstream",
    "papc.baselines.mtrouter_upstream",
    "papc.baselines.opensquilla_upstream",
)
FROZEN_ROUTER_RUNTIME_ENVIRONMENT = {
    "schema": "papc.external_router.runtime_environment.v1",
    "python": "3.12.3",
    "executable": CANONICAL_ROUTER_RUNTIME_EXECUTABLE,
    "prefix": CANONICAL_ROUTER_RUNTIME_PREFIX,
    "packages": dict(ROUTER_RUNTIME_PACKAGES),
    "router_modules": list(ROUTER_RUNTIME_MODULES),
}


class RouterRuntimeEnvironmentError(RuntimeError):
    """The current process is not the frozen external-router runtime."""


def router_runtime_environment_record() -> dict[str, Any]:
    """Inspect the active interpreter, dependencies, and all five adapters."""

    modules = [
        importlib.import_module(module_name).__name__
        for module_name in ROUTER_RUNTIME_MODULES
    ]
    return {
        "schema": "papc.external_router.runtime_environment.v1",
        "python": platform.python_version(),
        "executable": sys.executable,
        "prefix": sys.prefix,
        "packages": {
            package_name: importlib.metadata.version(package_name)
            for package_name in ROUTER_RUNTIME_PACKAGES
        },
        "router_modules": modules,
    }


def validate_router_runtime_environment() -> str:
    """Fail closed before any live path may read a provider secret."""

    try:
        observed = router_runtime_environment_record()
    except (
        ImportError,
        importlib.metadata.PackageNotFoundError,
    ) as exc:
        raise RouterRuntimeEnvironmentError(
            "external_router_runtime_dependency_missing"
        ) from exc
    if observed != FROZEN_ROUTER_RUNTIME_ENVIRONMENT:
        raise RouterRuntimeEnvironmentError(
            "external_router_runtime_environment_mismatch"
        )
    return FROZEN_ROUTER_RUNTIME_ENVIRONMENT_SHA256


def _digest(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


FROZEN_ROUTER_RUNTIME_ENVIRONMENT_SHA256 = _digest(
    FROZEN_ROUTER_RUNTIME_ENVIRONMENT
)
