"""Offline ToolSandbox adapter skeleton for PAPC residual completion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from papc.certspec.toolsandbox.residual_certificate import (
    ToolSandboxResidualCertificate,
    certificate_from_runtime_digest,
)
from papc.certspec.toolsandbox.residual_projection import project_residual_suffix
from papc.certspec.toolsandbox.state_write_contract import (
    DEPENDENCY_COMPLETION_UNIT,
    MINEFIELD_SIDE_EFFECT_UNIT,
)


@dataclass(frozen=True)
class ToolSandboxOfflineAdapter:
    """Adapter that turns sanitized ToolSandbox digests into PAPC receipts."""

    surface: str = "toolsandbox"

    def certificate(self, digest: Mapping[str, Any]) -> ToolSandboxResidualCertificate:
        return certificate_from_runtime_digest(digest)

    def project(
        self,
        digest: Mapping[str, Any],
        proposal: Mapping[str, Any],
    ) -> dict[str, Any]:
        return project_residual_suffix(digest=digest, proposal=proposal)

    def deterministic_dependency_completion(self) -> dict[str, Any]:
        return {
            "action": "execute",
            "tool_calls": [
                {
                    "effect_family": "dependency_completion",
                    "effect_units": [DEPENDENCY_COMPLETION_UNIT],
                }
            ],
        }

    def deterministic_minefield_violation(self) -> dict[str, Any]:
        return {
            "action": "execute",
            "tool_calls": [
                {
                    "effect_units": [
                        DEPENDENCY_COMPLETION_UNIT,
                        MINEFIELD_SIDE_EFFECT_UNIT,
                    ]
                }
            ],
        }

    def separable_dependency_and_minefield(self) -> dict[str, Any]:
        return {
            "action": "execute",
            "tool_calls": [
                {"effect_units": [DEPENDENCY_COMPLETION_UNIT]},
                {"effect_units": [MINEFIELD_SIDE_EFFECT_UNIT]},
            ],
        }

    def message_only_completion(self) -> dict[str, Any]:
        return {
            "action": "execute",
            "assistant_message": "Done.",
            "tool_calls": [],
        }


__all__ = ["ToolSandboxOfflineAdapter"]
