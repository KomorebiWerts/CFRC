"""Static import isolation for the protected baseline tree."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

_FORBIDDEN_PREFIX = "papc.certspec"


@dataclass(frozen=True)
class ImportViolation:
    path: str
    line: int
    target: str


def find_certspec_imports(paths: Iterable[Path]) -> tuple[ImportViolation, ...]:
    """Return direct, relative, and literal dynamic imports of method code."""

    violations: list[ImportViolation] = []
    for root in paths:
        files = (root,) if root.is_file() else sorted(root.rglob("*.py"))
        for path in files:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                targets = _forbidden_targets(node)
                violations.extend(
                    ImportViolation(
                        path=str(path),
                        line=int(getattr(node, "lineno", 0)),
                        target=target,
                    )
                    for target in targets
                )
    return tuple(violations)


def _forbidden_targets(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(
            alias.name
            for alias in node.names
            if _is_forbidden_absolute(alias.name)
        )
    if isinstance(node, ast.ImportFrom):
        module = str(node.module or "")
        targets: list[str] = []
        if node.level:
            if module == "certspec" or module.startswith("certspec."):
                targets.append("." * node.level + module)
            if not module:
                targets.extend(
                    "." * node.level + alias.name
                    for alias in node.names
                    if alias.name == "certspec"
                )
        elif _is_forbidden_absolute(module):
            targets.append(module)
        elif module == "papc":
            targets.extend(
                f"papc.{alias.name}"
                for alias in node.names
                if alias.name == "certspec"
            )
        return tuple(targets)
    if isinstance(node, ast.Call) and node.args:
        function = node.func
        is_dynamic_import = (
            isinstance(function, ast.Name)
            and function.id in {"__import__", "import_module"}
        ) or (
            isinstance(function, ast.Attribute)
            and function.attr == "import_module"
        )
        target = node.args[0]
        if (
            is_dynamic_import
            and isinstance(target, ast.Constant)
            and isinstance(target.value, str)
            and _is_forbidden_absolute(target.value)
        ):
            return (target.value,)
    if isinstance(node, ast.Attribute):
        target = _attribute_target(node)
        if _is_forbidden_absolute(target):
            return (target,)
    return ()


def _attribute_target(node: ast.Attribute) -> str:
    parts = [node.attr]
    value: ast.AST = node.value
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return ".".join(reversed(parts))


def _is_forbidden_absolute(target: str) -> bool:
    return target == _FORBIDDEN_PREFIX or target.startswith(
        _FORBIDDEN_PREFIX + "."
    )
