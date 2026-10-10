"""tau2 retail adapter surface for PAPC experiments.

Core PAPC modules should avoid importing tau2-specific projection details
unless they are explicitly running tau2 experiments or audits.
"""

from __future__ import annotations

from papc.certspec.structured_certificate import register_target_type

register_target_type("tau2", "retail", "commerce_workflow_target_set")
