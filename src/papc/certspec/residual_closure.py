"""Residual closure: derive ``O`` from the retained request and trusted evidence.

Closure reads the retained request because the setting says it must: an open
obligation is what the request still owes at the reached state, and no amount
of reached-state evidence alone can say what was asked for. What closure may
not do is read the request with per-task or per-surface patterns, because a
pattern written for one request shape is an answer to that request rather than
a projection of it.

The vocabulary below is therefore one table applied identically to every
surface. It turns request wording into typed effect units; it never names an
operation, an argument value, or a task. Which operation realizes an effect is
declared by the public interface record, not here.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from papc.certspec.cfrc_ir import ResidualObligation

_TOKEN = re.compile(r"[A-Za-z0-9_.@#-]+")

# Effect units. These name what becomes true, not how to make it true.
MEMBERSHIP_ADD = "membership.add"
MEMBERSHIP_REMOVE = "membership.remove"
CONTAINER_CREATE = "container.create"
ENTITY_CREATE = "entity.create"
ENTITY_RENAME = "entity.rename"
ENTITY_UPDATE = "entity.update"
ENTITY_DELETE = "entity.delete"
ENTITY_MOVE = "entity.move"
MESSAGE_POST = "message.post"
ACCESS_GRANT = "access.grant"
ACCESS_REVOKE = "access.revoke"
VISIBILITY_SET = "visibility.set"
ANNOTATION_ADD = "annotation.add"

# One lexicon for every surface. A surface may not extend or override it: a
# per-surface entry would be a statement about that surface's tasks rather
# than about the language the request is written in.
_EFFECT_LEXICON: tuple[tuple[str, str], ...] = (
    (r"\binvit(?:e|ing|ation)\b", MEMBERSHIP_ADD),
    (r"\bjoin(?:ed|ing)?\b", MEMBERSHIP_ADD),
    (r"\bmember(?:ship)?\b(?![^.]*\bcomment\b)", MEMBERSHIP_ADD),
    (r"\badd\w*\b[^.]*\bto\b[^.]*\b(?:channel|conversation|group|team|space)\b", MEMBERSHIP_ADD),
    (r"\bremov(?:e|ed|ing)\b(?=[^.]*\bfrom\b)", MEMBERSHIP_REMOVE),
    (r"\bkick(?:ed|ing)?\b", MEMBERSHIP_REMOVE),
    (r"\bcreat(?:e|ed|ing)\b", ENTITY_CREATE),
    (r"\bmake\b", ENTITY_CREATE),
    (r"\bset\b\s+\bup\b", ENTITY_CREATE),
    (r"\bnew\b", ENTITY_CREATE),
    (r"\brenam(?:e|ed|ing)\b", ENTITY_RENAME),
    (r"\bcorrect(?:ed|ing)?\b", ENTITY_RENAME),
    (r"\bfix(?:ed|ing)?\b", ENTITY_RENAME),
    (r"\bmisspell\w*\b", ENTITY_RENAME),
    (r"\btypo\w*\b", ENTITY_RENAME),
    (r"\bupdat(?:e|ed|ing)\b", ENTITY_UPDATE),
    (r"\bchang(?:e|ed|ing)\b", ENTITY_UPDATE),
    (r"\bedit(?:ed|ing)?\b", ENTITY_UPDATE),
    (r"\bdelet(?:e|ed|ing)\b", ENTITY_DELETE),
    (r"\bmov(?:e|ed|ing)\b", ENTITY_MOVE),
    (r"\bpost(?:ed|ing)?\b", MESSAGE_POST),
    (r"\bsend\b|\bsent\b", MESSAGE_POST),
    (r"\bcomment(?:ed|ing)?\b", MESSAGE_POST),
    (r"\bshar(?:e|ed|ing)\b", ACCESS_GRANT),
    (r"\bgiv(?:e|ing)\b[^.]*\baccess\b", ACCESS_GRANT),
    (r"\bgrant(?:ed|ing)?\b", ACCESS_GRANT),
    (r"\brevok(?:e|ed|ing)\b", ACCESS_REVOKE),
    (r"\bunshar(?:e|ed|ing)\b", ACCESS_REVOKE),
    (r"\bhid(?:e|den|ing)\b", VISIBILITY_SET),
    (r"\bunhide\b", VISIBILITY_SET),
    (r"\bdeselect\w*\b", VISIBILITY_SET),
    (r"\bselected\b(?=[^.]*\bcalendar\b)|\bvisible\b", VISIBILITY_SET),
    # Marking an item is its own effect. Requests state it as a reaction, a
    # thumbs up, or a written-out shortcode, and none of those is a message:
    # folding them into message.post left the mark absent from O, so a
    # continuation that only posted read as having covered the request. The
    # shortcode has to start with a letter, or a clock time supplies one.
    (r"\breact(?:ion|ions|ed|ing)?\b", ANNOTATION_ADD),
    (r"\bthumbs[- ]?up\b", ANNOTATION_ADD),
    (r"\bemoji\b", ANNOTATION_ADD),
    (r":[a-z][a-z0-9_+\-]+:", ANNOTATION_ADD),
)

# Quantifiers that widen an obligation from one target to a set. A quantified
# obligation cannot be discharged from a partial listing, because the target
# set is not known until the listing is complete.
_QUANTIFIERS: tuple[tuple[str, str], ...] = (
    (r"\ball\b", "all"),
    (r"\bevery\b", "all"),
    (r"\beach\b", "all"),
    (r"\bany\b", "any"),
    (r"\bboth\b", "all"),
    (r"\bsmallest\b|\blargest\b|\bbiggest\b", "extremum"),
    (r"\bnewest\b|\boldest\b|\blatest\b|\bearliest\b", "extremum"),
)


@dataclass(frozen=True)
class RequestScope:
    """What the retained request names, and nothing beyond it."""

    effect_units: tuple[str, ...]
    referenced_tokens: frozenset[str]
    quantifiers: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "effect_units": list(self.effect_units),
            "referenced_tokens": sorted(self.referenced_tokens),
            "quantifiers": list(self.quantifiers),
        }


def request_scope(request: str) -> RequestScope:
    """Project the retained request into typed effects and named references."""

    text = str(request or "")
    lowered = text.lower()
    effects: list[str] = []
    for pattern, unit in _EFFECT_LEXICON:
        if unit in effects:
            continue
        if re.search(pattern, lowered):
            effects.append(unit)
    quantifiers: list[str] = []
    for pattern, name in _QUANTIFIERS:
        if name in quantifiers:
            continue
        if re.search(pattern, lowered):
            quantifiers.append(name)
    return RequestScope(
        effect_units=tuple(effects),
        referenced_tokens=frozenset(_TOKEN.findall(lowered)),
        quantifiers=tuple(quantifiers),
    )


def grounded_targets(
    scope: RequestScope,
    *,
    evidence_tokens: Sequence[str],
) -> tuple[str, ...]:
    """Names the request uses that trusted evidence also reports.

    A token the request mentions but no receipt reports is not a target: the
    request may be describing an outcome rather than naming an entity. A token
    a receipt reports but the request never mentions is not a target either,
    because nothing asked for it.
    """

    grounded = {
        token
        for token in (str(item).strip().lower() for item in evidence_tokens)
        if token and token in scope.referenced_tokens
    }
    return tuple(sorted(grounded))


def residual_obligations(
    *,
    scope: RequestScope,
    targets: Sequence[str],
    realized_effects: Sequence[str] = (),
) -> tuple[ResidualObligation, ...]:
    """Pair each still-open requested effect with the targets it names.

    An effect already realized in the accepted prefix is not reopened: closure
    reports what remains, and a commitment that has been kept belongs in ``K``.
    """

    realized = {str(item) for item in realized_effects}
    obligations: list[ResidualObligation] = []
    for unit in scope.effect_units:
        if unit in realized:
            continue
        obligations.append(
            ResidualObligation(
                obligation_id=f"{unit}",
                required_unit_tokens=tuple(str(item) for item in targets),
                effect_families=(unit,),
                provenance_refs=("retained_request",),
                full_discharge_required=True,
            )
        )
    return tuple(obligations)


def write_footprint(arguments: Mapping[str, Any] | None) -> frozenset[str]:
    """Field names one proposed write would set.

    Only leaf assignments count. A container key names where a field lives, not
    a field the write changes.
    """

    if not isinstance(arguments, Mapping):
        return frozenset()
    fields: set[str] = set()

    def walk(value: Any, prefix: str = "") -> None:
        if isinstance(value, Mapping):
            for key in value:
                walk(value[key], f"{prefix}{key}.")
            return
        if prefix:
            fields.add(prefix.rstrip("."))

    body = arguments.get("json")
    walk(body if isinstance(body, Mapping) else {})
    params = arguments.get("params")
    walk(params if isinstance(params, Mapping) else {})
    return frozenset(fields)


# An entity reference is long enough to identify something. A version segment
# or an ordinal in a path names where a call goes, not what it acts on.
_ID_LIKE = re.compile(r"^(?=.*\d)[A-Za-z0-9_@.-]{6,}$")
_API_VERSION = re.compile(r"^v\d+$")
_MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_CREATING_METHODS = frozenset({"POST"})
# Methods that act on an entity that already exists, so the interface answers
# with that entity rather than a fresh one.
_REPLACING_METHODS = frozenset({"PUT", "PATCH"})
# Arguments whose value distinguishes one created entity from another.
# The value a creation gives its new entity. A container reference is not a
# name: two writes into the same container are two writes, not one entity.
_NAMING_ARGUMENTS = ("name", "summary", "title", "displayname")

# Fields an interface computes for itself. A receipt reporting its own value
# for one of these is the interface working, not the write being refused.
_SERVER_DERIVED_FIELDS = frozenset(
    {"status", "state", "etag", "updated", "created", "id", "kind", "version"}
)


def is_mutating(arguments: Mapping[str, Any] | None) -> bool:
    if not isinstance(arguments, Mapping):
        return False
    return str(arguments.get("method") or "").upper() in _MUTATING_METHODS


def write_targets(arguments: Mapping[str, Any] | None) -> frozenset[str]:
    """Entity references a write acts on.

    Field names describe the shape of a call and belong to the interface
    record. What a strict environment compares is which entities changed, so
    closure tracks the references a write names rather than the fields it
    sets.
    """

    if not isinstance(arguments, Mapping):
        return frozenset()
    targets: set[str] = set()
    for segment in str(arguments.get("path") or "").split("/"):
        segment = segment.strip()
        if segment and not _API_VERSION.match(segment) and _ID_LIKE.match(segment):
            targets.add(segment.lower())

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key in value:
                walk(value[key])
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
            return
        text = str(value).strip()
        if text and _ID_LIKE.match(text):
            targets.add(text.lower())

    for key in ("json", "params"):
        walk(arguments.get(key))
    return frozenset(targets)


def creates_entity(arguments: Mapping[str, Any] | None) -> bool:
    """Whether a write brings a new entity into existence rather than editing one."""

    if not isinstance(arguments, Mapping):
        return False
    return str(arguments.get("method") or "").upper() in _CREATING_METHODS


def ungrounded_write_targets(
    *,
    targets: frozenset[str],
    evidence_tokens: Sequence[str],
) -> tuple[str, ...]:
    """References a write acts on that no trusted receipt reports.

    Editing an entity the reached state never showed means the successor chose
    a target it could not have verified. Creation is exempt: a new entity has
    no prior receipt by definition.
    """

    seen = {str(item).strip().lower() for item in evidence_tokens if str(item).strip()}
    return tuple(sorted(target for target in targets if target not in seen))


def creation_name(arguments: Mapping[str, Any] | None) -> str | None:
    """The value that distinguishes one created entity from another."""

    if not isinstance(arguments, Mapping):
        return None
    for key in ("json", "params"):
        body = arguments.get(key)
        if not isinstance(body, Mapping):
            continue
        for field in body:
            if str(field).replace("_", "").lower() in _NAMING_ARGUMENTS:
                value = body[field]
                if isinstance(value, str) and value.strip():
                    return value.strip().lower()
    return None


def duplicate_creations(
    creations: Sequence[tuple[str, str | None]],
) -> tuple[str, ...]:
    """Creations that would bring the same entity into existence twice.

    A request that names several outcomes licenses several creations. What it
    never licenses is the same outcome twice, which is how an unbounded
    proposal turns one requested calendar into three.
    """

    seen: dict[tuple[str, str], int] = {}
    for family, name in creations:
        if not name:
            continue
        key = (str(family), name)
        seen[key] = seen.get(key, 0) + 1
    return tuple(
        f"{family}:{name}x{count}"
        for (family, name), count in sorted(seen.items())
        if count > 1
    )


def unopened_enumerated_containers(
    receipts: Sequence[Mapping[str, Any]],
    *,
    container_type: str = "folder",
) -> tuple[str, ...]:
    """Containers a listing reported that no later call opened.

    A quantified obligation ranges over a set the request does not enumerate,
    so it cannot be discharged while the reached state still shows a container
    whose contents were never read. This is a diff over receipts, not a guess
    about what the request meant: the listing named the container, and no
    subsequent call asked for its items.
    """

    listed: set[str] = set()
    opened: set[str] = set()
    for receipt in receipts:
        action = str(receipt.get("action") or "")
        for match in re.finditer(r"/([A-Za-z0-9_-]+)/items\b", action):
            opened.add(match.group(1).lower())
        payload = receipt.get("observation")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (TypeError, ValueError):
                payload = None
        if not isinstance(payload, Mapping):
            continue
        for entry in payload.get("entries") or ():
            if not isinstance(entry, Mapping):
                continue
            if str(entry.get("type") or "") != container_type:
                continue
            identifier = entry.get("id")
            if identifier:
                listed.add(str(identifier).lower())
    return tuple(sorted(listed - opened))


def open_obligations(
    *,
    request: str,
    receipts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """The ``O`` record a frozen contract carries to its successor.

    Effects come from the retained request, and the coverage gap comes from
    receipts. The gap is reported only under a quantifier, because a request
    that names one target says nothing about containers it never mentioned.
    """

    scope = request_scope(request)
    record: dict[str, Any] = {
        "effects": list(scope.effect_units),
        "quantifiers": list(scope.quantifiers),
    }
    if scope.quantifiers:
        gap = unopened_enumerated_containers(receipts)
        if gap:
            record["unopened_enumerated_containers"] = list(gap[:8])
    return record


def _leaf_assignments(value: Any, prefix: str = "") -> dict[str, Any]:
    leaves: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key in value:
            leaves.update(_leaf_assignments(value[key], f"{prefix}{key}."))
        return leaves
    if prefix:
        leaves[prefix.rstrip(".")] = value
    return leaves


def _lookup(payload: Any, path: str) -> tuple[bool, Any]:
    current = payload
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def unrealized_write_fields(
    *,
    arguments: Mapping[str, Any] | None,
    receipt: Any,
) -> tuple[str, ...]:
    """Fields a write set that its own receipt reports differently.

    An interface that accepts a write echoes the entity it wrote. A field the
    receipt reports with another value was not realized, whatever the status
    code said, so treating exit zero as discharge lets an obligation close on a
    call that changed nothing.

    A field the receipt omits counts only when the write replaced or updated an
    existing entity. Such an interface answers with the entity it just wrote, so
    a field missing from that answer was not applied to anything.
    """

    if not isinstance(arguments, Mapping):
        return ()
    body = arguments.get("json")
    if not isinstance(body, Mapping):
        return ()
    if isinstance(receipt, str):
        try:
            receipt = json.loads(receipt)
        except (TypeError, ValueError):
            return ()
    if not isinstance(receipt, Mapping):
        return ()
    mismatched: list[str] = []
    for path, wrote in _leaf_assignments(body).items():
        if isinstance(wrote, (list, tuple)) or wrote is None:
            continue
        if path.rsplit(".", 1)[-1].lower() in _SERVER_DERIVED_FIELDS:
            continue
        present, reported = _lookup(receipt, path)
        if not present:
            if str(arguments.get("method") or "").upper() in _REPLACING_METHODS:
                mismatched.append(path)
            continue
        if isinstance(reported, (Mapping, list, tuple)):
            continue
        if not _contradicts(wrote=wrote, reported=reported):
            continue
        mismatched.append(path)
    return tuple(sorted(mismatched))


_EVIDENCE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*")


def _evidence_normalised(text: str) -> str:
    """One comparable surface for a body of evidence or a closing message."""

    return str(text or "").lower().replace(",", "").replace("$", " ")


def ungrounded_values(*, draft: str, evidence: Sequence[str]) -> tuple[str, ...]:
    """The values a closing message states that nothing in hand supports.

    A closing message may only report what the request asked for and what the
    executed actions returned. A token carrying a digit is a value rather than
    prose -- an identifier, an amount, a date, a flight -- so it has to be
    traceable to one of those two sources. One that is not is a value the
    message invented, and no reached state can witness it.

    Prose is not checked: a message is free to choose its own words. Nor is a
    short token, which cannot be read as an identifier and collides with
    ordinary numbering. Matching is deliberately lenient, since the cost of a
    missed match is one closing call that did not need to run, while the cost of
    a wrong acceptance is a message that misreports what was done.
    """

    haystack = " ".join(_evidence_normalised(item) for item in evidence)
    if not haystack.strip():
        return ()
    haystack_numbers = {
        value for value in (_as_number(token) for token in _EVIDENCE_TOKEN.findall(haystack))
        if value is not None
    }
    ungrounded: list[str] = []
    for token in _EVIDENCE_TOKEN.findall(_evidence_normalised(draft)):
        candidate = token.strip("._/-")
        if len(candidate) < 3 or not any(char.isdigit() for char in candidate):
            continue
        if any(separator in candidate for separator in "-_/"):
            # A separator joins a number to a word: a window, a duration, a
            # policy horizon. That is the message choosing how to say something,
            # not a value it claims to have read, and evidence records the same
            # thing in whatever form its own schema uses.
            continue
        if candidate in haystack:
            continue
        number = _as_number(candidate)
        if number is not None and number in haystack_numbers:
            continue
        ungrounded.append(candidate)
    return tuple(sorted(set(ungrounded)))


def _as_number(token: str) -> float | None:
    try:
        return float(token)
    except (TypeError, ValueError):
        return None


def leaf_assignments(body: Any) -> dict[str, Any]:
    """Public name for the leaf projection a write body is read with.

    A caller that has to compare written values against something other than a
    receipt still has to agree with this module on what one write assigned, so
    the projection is exported rather than restated.
    """

    return _leaf_assignments(body)


def contradicts(*, wrote: Any, reported: Any) -> bool:
    """Public name for the test between a written value and a reported one."""

    return _contradicts(wrote=wrote, reported=reported)


def _contradicts(*, wrote: Any, reported: Any) -> bool:
    """Whether a receipt value disagrees with what the write asked for.

    An interface may normalize what it stores: it resolves a name to an
    identifier, drops a sigil, or reports a state it derived. A normalized
    echo is the write taking effect, so only a value that shares no form with
    the requested one counts as a contradiction.
    """

    want = str(wrote).strip().lower().lstrip("#@")
    got = str(reported).strip().lower().lstrip("#@")
    if want == got:
        return False
    if not want or not got:
        return True
    if want in got or got in want:
        return False
    # A resolved identifier keeps no textual trace of the name it replaced, so
    # a receipt that reports one cannot be read as a refusal of the write.
    return not any(char.isdigit() for char in got)
