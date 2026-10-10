"""Deterministic live-prefix frontier monitor for ToolSandbox.

The first ToolSandbox method arm only admits terminal/pre-write
``toolsandbox.reminder.add`` candidates.  The monitor reads the live visible
prefix and coarse tool traces, never task ids, gold outcomes, or scorer state.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

SCHEMA = "papc.toolsandbox.online_frontier_monitor.v1"

REMINDER_EFFECT_UNIT = "toolsandbox.reminder.add"
REMINDER_TEMPLATE_ID = "toolsandbox.add_reminder.v1"
REMINDER_LOCATION_TEMPLATE_ID = "toolsandbox.add_reminder.with_location.v1"
REMINDER_MUTATING_TOOL = "add_reminder"
WIFI_OFF_EFFECT_UNIT = "toolsandbox.setting.wifi"
WIFI_OFF_TEMPLATE_ID = "toolsandbox.set_wifi_status.v1"
WIFI_MUTATING_TOOL = "set_wifi_status"
LOW_BATTERY_EFFECT_UNIT = "toolsandbox.setting.low_battery"
CELLULAR_OFF_EFFECT_UNIT = "toolsandbox.setting.cellular"
CELLULAR_OFF_TEMPLATE_ID = "toolsandbox.set_cellular_service_status.v1"
CELLULAR_MUTATING_TOOL = "set_cellular_service_status"
WEATHER_TEMP_F_EFFECT_UNIT = "toolsandbox.weather.answer"
WEATHER_TEMP_F_TEMPLATE_ID = "toolsandbox.weather.current_temperature_f.v1"
STOCK_ANSWER_EFFECT_UNIT = "toolsandbox.stock.answer"
STOCK_TEMPLATE_ID = "toolsandbox.stock.symbol.v1"
STOCK_READ_TOOL = "search_stock"
MESSAGE_ANSWER_EFFECT_UNIT = "toolsandbox.message.search.answer"
MESSAGE_TEMPLATE_ID = "toolsandbox.message.recency.v1"
MESSAGE_SEARCH_TOOL = "search_messages"
CURRENT_TIMESTAMP_TOOL = "get_current_timestamp"
LOW_BATTERY_MUTATING_TOOL = "set_low_battery_mode_status"
LOCATION_SERVICE_MUTATING_TOOL = "set_location_service_status"
LOCATION_SERVICE_EFFECT_UNIT = "toolsandbox.setting.location_service"
REMINDER_TOOL_SEQUENCE = (
    "get_current_timestamp",
    "timestamp_to_datetime_info",
    "timestamp_to_datetime_info",
    "datetime_info_to_timestamp",
    "add_reminder",
)
# Transcribed from the official signatures: add_reminder(content, reminder_timestamp,
# latitude=None, longitude=None) and datetime_info_to_timestamp(year, month, day,
# hour, minute, second). Stating the shape the tools actually accept is what keeps
# the successor from inventing a constraint of its own.
REMINDER_TIME_TOOL_ARG_KEY_TYPES: dict[str, list[dict[str, Any]]] = {
    "get_current_timestamp": [],
    "timestamp_to_datetime_info": [
        {"key": "timestamp", "type": "number", "required": True},
    ],
    "datetime_info_to_timestamp": [
        {"key": "year", "type": "number", "required": True},
        {"key": "month", "type": "number", "required": True},
        {"key": "day", "type": "number", "required": True},
        {"key": "hour", "type": "number", "required": True},
        {"key": "minute", "type": "number", "required": True},
        {"key": "second", "type": "number", "required": True},
    ],
}
REMINDER_ADD_TOOL_ARG_KEY_TYPES: list[dict[str, Any]] = [
    {"key": "content", "type": "string", "required": True},
    {"key": "reminder_timestamp", "type": "number", "required": True},
    {"key": "latitude", "type": "number", "required": False},
    {"key": "longitude", "type": "number", "required": False},
]
# Every clause below restates the official tool contract. None of it names a
# value, a place, or a time drawn from any scenario.
REMINDER_SCHEMA_INSTRUCTIONS: tuple[str, ...] = (
    "add_reminder.content is the task the user wants to be reminded of. The time"
    " belongs in reminder_timestamp and the place belongs in latitude and"
    " longitude, so do not repeat either of them inside content.",
    "reminder_timestamp is any POSIX timestamp the tool accepts; it validates the"
    " type only and never requires a time later than now. When the visible"
    " request names a calendar date and a time of day, compute exactly that"
    " instant and submit it. Do not decline, shift it to another date, or ask for"
    " a new time because it is earlier than the current timestamp.",
    "Derive reminder_timestamp with the allowed time tools rather than writing a"
    " POSIX number yourself. datetime_info_to_timestamp requires all six of year,"
    " month, day, hour, minute and second.",
    "Read the current timestamp only when the request is relative, for example"
    " tomorrow or next Monday. When it gives an absolute date, build the"
    " timestamp from that date directly.",
)
SETTING_OFF_RE = re.compile(
    r"\b(turn|switch|shut|disable|set)\s+off\s+"
    r"(?P<target>wifi|wi-fi|wireless|cellular|cell|mobile(?: service)?)\b",
    re.IGNORECASE,
)
WEATHER_TEMP_F_RE = re.compile(
    r"\b(current\s+temp|current\s+temperature|temperature|temp)\b"
    r".{0,80}\b("
    r"fahrenheit|degrees?\s*f|°f|can't\s+read\s+celsius|cannot\s+read\s+celsius|"
    r"celsius"
    r")\b",
    re.IGNORECASE,
)
STOCK_SYMBOL_RE = re.compile(
    r"\b(stock\s+symbol|ticker\s+symbol|ticker)\b"
    r"|\bcode\b.{0,40}\bstock\b"
    r"|\bstock\b.{0,40}\bcode\b",
    re.IGNORECASE,
)
MESSAGE_SEARCH_INTENT_RE = re.compile(
    r"\b(find|search|get|show|read|tell|want)\b.{0,80}\b(messages?|texts?)\b"
    r"|\b(messages?|texts?)\b.{0,80}\b(find|search|get|show|read|tell|want)\b",
    re.IGNORECASE,
)
MESSAGE_SEARCH_NEGATION_RE = re.compile(
    r"\b(?:do\s+not|don't|never)\b.{0,60}\b(?:find|search|get|show|read)\b",
    re.IGNORECASE,
)
MESSAGE_NON_READ_RE = re.compile(
    r"\b(?:send|delete|remove|forward|reply\s+to|edit|modify|update)\b"
    r".{0,60}\b(?:messages?|texts?)\b",
    re.IGNORECASE,
)
MESSAGE_RECENCY_RE = re.compile(
    r"\b(?P<latest>most\s+recent(?:ly)?|latest|newest|last)\b"
    r"|\b(?P<oldest>oldest|earliest|first(?:\s+ever)?|least\s+recent|longest\s+ago)\b",
    re.IGNORECASE,
)
REMINDER_LOCATION_CUE_RE = re.compile(
    r"\b(?:at|near|in)\s+"
    r"(?!\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b)"
    r"(?:[a-z0-9'-]+\s+){0,8}"
    r"(?:foods?|market|store|park|bridge|airport|mall|restaurant|cafe|"
    r"ave|avenue|street|st\b|creek|road|rd\b|boulevard|blvd\b)",
    re.IGNORECASE,
)
ISSUE_RESOLUTION_RE = re.compile(
    r"\b(resolve|fix|handle)\b.{0,40}\b(issue|problem)\b",
    re.IGNORECASE,
)

_INTENT_RE = re.compile(
    r"("
    r"\b(remind me|please remind|set (?:a )?reminder|add (?:a )?reminder|"
    r"create (?:a )?reminder|add (?:a )?todo|create (?:a )?todo|remember to)\b"
    r"|提醒我|加(?:一个)?提醒|创建(?:一个)?提醒|设置(?:一个)?提醒"
    r")",
    re.IGNORECASE,
)
_NON_ADD_REMINDER_RE = re.compile(
    r"\b(modify|change|update|edit|delete|remove|cancel|find|search|show|list)\s+"
    r"(?:my\s+)?(?:a\s+)?reminders?\b",
    re.IGNORECASE,
)
_TIME_RE = re.compile(
    r"("
    r"\b(today|tomorrow|tonight|morning|afternoon|evening|noon|midnight|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b"
    r"|"
    r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b"
    r"|"
    r"\b\d{4}-\d{1,2}-\d{1,2}\b"
    r"|"
    r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b"
    r"|"
    r"明天|今天|今晚|上午|下午|晚上|中午|下周|周[一二三四五六日天]"
    r")",
    re.IGNORECASE,
)
_CONTENT_STOPWORDS = {
    "a",
    "add",
    "an",
    "at",
    "all",
    "any",
    "best",
    "can",
    "city",
    "country",
    "create",
    "current",
    "device",
    "device's",
    "do",
    "don",
    "don't",
    "for",
    "got",
    "have",
    "i",
    "if",
    "info",
    "information",
    "it",
    "just",
    "know",
    "location",
    "me",
    "more",
    "need",
    "needed",
    "not",
    "offset",
    "on",
    "one",
    "or",
    "out",
    "phone",
    "phone's",
    "last",
    "next",
    "please",
    "reminder",
    "reminders",
    "remind",
    "set",
    "services",
    "specific",
    "s",
    "that",
    "this",
    "the",
    "time",
    "timezone",
    "to",
    "tool",
    "tools",
    "use",
    "utc",
    "your",
    "you",
    "zone",
    "figure",
    "judgment",
    "upcoming",
}
_CONTENT_META_RE = re.compile(
    r"\b("
    r"time\s*zone|timezone|utc(?:\s+offset)?|city|country|location|"
    r"device(?:'s)?\s+current\s+time\s*zone|phone(?:'s)?\s+current\s+time\s*zone|"
    r"current\s+time\s*zone|current\s+timezone|use\s+your\s+tools?|do\s+your\s+best"
    r")\b",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z']+")


@dataclass(frozen=True)
class ToolSandboxReminderAddFrontierMonitor:
    """Enumerate only live pre-write reminder-add CFRC candidates."""

    public_ordinal: int = 0

    def typed_row_from_context(self, context: Mapping[str, Any]) -> Mapping[str, Any] | None:
        return self._typed_row_from_context(context)

    def _typed_row_from_context(
        self, context: Mapping[str, Any]
    ) -> Mapping[str, Any] | None:
        conversation = _conversation(context.get("conversation"))
        if not conversation:
            return None
        if _prefix_has_mutating_reminder_write(conversation):
            return None
        user_text = "\n".join(_user_text(message) for message in conversation)
        setting_row = _setting_off_typed_row(
            text=user_text,
            public_ordinal=self.public_ordinal,
            frontier_message_index=len(conversation),
        )
        if setting_row is not None:
            return setting_row
        message_row = _message_recency_typed_row(
            text=user_text,
            conversation=conversation,
            available_tool_names=context.get("available_tool_names"),
            public_ordinal=self.public_ordinal,
            frontier_message_index=len(conversation),
        )
        if message_row is not None:
            return message_row
        stock_row = _stock_symbol_typed_row(
            text=user_text,
            public_ordinal=self.public_ordinal,
            frontier_message_index=len(conversation),
        )
        if stock_row is not None:
            return stock_row
        weather_row = _weather_temp_f_typed_row(
            text=user_text,
            conversation=conversation,
            cheap_step_count=_cheap_step_count(context),
            public_ordinal=self.public_ordinal,
            frontier_message_index=len(conversation),
        )
        if weather_row is not None:
            return weather_row
        if not _has_reminder_intent(user_text):
            return None
        if not _has_time_carrier(user_text):
            return None
        if not _has_content_carrier(user_text):
            return None
        if _has_location_cue(user_text):
            return _location_reminder_typed_row(
                public_ordinal=self.public_ordinal,
                frontier_message_index=len(conversation),
                with_setting_repair=_location_setting_repair_available(context),
            )
        return _typed_row(
            public_ordinal=self.public_ordinal,
            frontier_message_index=len(conversation),
        )


def _typed_row(*, public_ordinal: int, frontier_message_index: int) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "public_ordinal": int(public_ordinal),
        "family": "online_live_frontier",
        "residual_closable": True,
        "residual_closability_basis": "online_visible_prefix_pre_write_reminder_add",
        "typed_suffix_blockers": [],
        "method_introduced_minefield_risk": "none",
        "frontier_binding": {
            "frontier_kind": "pre_write_terminal_prefix",
            "frontier_source": "online_live_prefix",
            "frontier_message_index": int(frontier_message_index),
            "frontier_tool_call_index": None,
            "authority_ceiling": "ell1",
            "effect_units": [REMINDER_EFFECT_UNIT],
        },
        "typed_suffix": {
            "template_id": REMINDER_TEMPLATE_ID,
            "suffix_tool_sequence": list(REMINDER_TOOL_SEQUENCE),
            "read_tool_names": sorted(set(REMINDER_TOOL_SEQUENCE[:-1])),
            "mutating_tool_name": REMINDER_MUTATING_TOOL,
            "mutating_arg_key_types": [
                {"key": "content", "type": "string"},
                {"key": "reminder_timestamp", "type": "number"},
            ],
            "optional_mutating_arg_key_types": [
                {"key": "latitude", "type": "number"},
                {"key": "longitude", "type": "number"},
            ],
            "tool_arg_key_types": {
                **REMINDER_TIME_TOOL_ARG_KEY_TYPES,
                REMINDER_MUTATING_TOOL: list(REMINDER_ADD_TOOL_ARG_KEY_TYPES),
            },
            "assistant_response_required": True,
            "domain_instructions": list(REMINDER_SCHEMA_INSTRUCTIONS),
            "fixed_argument_values_used": False,
            "requires_proposal_graph": True,
            "execution_verified": False,
        },
    }


def _location_reminder_typed_row(
    *,
    public_ordinal: int,
    frontier_message_index: int,
    with_setting_repair: bool,
) -> dict[str, Any]:
    setting_sequence = (
        [
            "get_low_battery_mode_status",
            LOW_BATTERY_MUTATING_TOOL,
            "get_wifi_status",
            WIFI_MUTATING_TOOL,
            # A place lookup also needs location service; without it the search
            # raises PermissionError and the place can never be resolved.
            "get_location_service_status",
            LOCATION_SERVICE_MUTATING_TOOL,
        ]
        if with_setting_repair
        else []
    )
    suffix_tool_sequence = [
        *setting_sequence,
        "search_location_around_lat_lon",
        "get_current_timestamp",
        "timestamp_to_datetime_info",
        "datetime_info_to_timestamp",
        REMINDER_MUTATING_TOOL,
    ]
    effect_units = [REMINDER_EFFECT_UNIT]
    effect_families = ["toolsandbox.reminder", "toolsandbox.location"]
    tool_arg_key_types: dict[str, list[dict[str, Any]]] = {
        # latitude/longitude are optional on the official tool; omitting them
        # anchors the search at the device's current location. Forcing them to
        # 0,0 anchors the query in the Gulf of Guinea and returns nothing.
        "search_location_around_lat_lon": [
            {"key": "location", "type": "string", "required": True},
            {"key": "latitude", "type": "number", "required": False},
            {"key": "longitude", "type": "number", "required": False},
        ],
        # latitude/longitude are Optional on the official add_reminder. Marking
        # them required turns an unresolvable place into a dead end: the residual
        # can no longer discharge the reminder obligation it *can* discharge.
        REMINDER_MUTATING_TOOL: list(REMINDER_ADD_TOOL_ARG_KEY_TYPES),
        **REMINDER_TIME_TOOL_ARG_KEY_TYPES,
    }
    setting_code = ""
    if with_setting_repair:
        effect_units.extend(
            [
                LOW_BATTERY_EFFECT_UNIT,
                WIFI_OFF_EFFECT_UNIT,
                LOCATION_SERVICE_EFFECT_UNIT,
            ]
        )
        effect_families.append("toolsandbox.setting")
        tool_arg_key_types.update(
            {
                LOW_BATTERY_MUTATING_TOOL: [
                    {"key": "on", "type": "boolean", "required": True},
                ],
                WIFI_MUTATING_TOOL: [
                    {"key": "on", "type": "boolean", "required": True},
                ],
                LOCATION_SERVICE_MUTATING_TOOL: [
                    {"key": "on", "type": "boolean", "required": True},
                ],
            }
        )
        setting_code = (
            "low_battery_on = get_low_battery_mode_status()\n"
            "if low_battery_on:\n"
            "    set_low_battery_mode_status(on=False)\n"
            "wifi_on = get_wifi_status()\n"
            "if not wifi_on:\n"
            "    set_wifi_status(on=True)\n"
            "location_on = get_location_service_status()\n"
            "if not location_on:\n"
            "    set_location_service_status(on=True)\n"
        )
    return {
        "schema": SCHEMA,
        "public_ordinal": int(public_ordinal),
        "family": "online_live_frontier",
        "residual_closable": True,
        "residual_closability_basis": "online_visible_prefix_pre_write_reminder_add_with_location",
        "typed_suffix_blockers": [],
        "method_introduced_minefield_risk": "none",
        "frontier_binding": {
            "frontier_kind": "pre_write_terminal_prefix",
            "frontier_source": "online_live_prefix",
            "frontier_message_index": int(frontier_message_index),
            "frontier_tool_call_index": None,
            "authority_ceiling": "ell1",
            "effect_units": effect_units,
        },
        "typed_suffix": {
            "template_id": REMINDER_LOCATION_TEMPLATE_ID,
            "effect_families": effect_families,
            "suffix_tool_sequence": suffix_tool_sequence,
            "read_tool_names": [
                "search_location_around_lat_lon",
                "get_current_timestamp",
                "timestamp_to_datetime_info",
                "datetime_info_to_timestamp",
                *(
                    [
                        "get_low_battery_mode_status",
                        "get_wifi_status",
                        "get_location_service_status",
                    ]
                    if with_setting_repair
                    else []
                ),
            ],
            "mutating_tool_name": REMINDER_MUTATING_TOOL,
            "mutating_tool_must_be_last": True,
            "mutating_arg_key_types": [
                {"key": "content", "type": "string"},
                {"key": "reminder_timestamp", "type": "number"},
            ],
            "optional_mutating_arg_key_types": [
                {"key": "latitude", "type": "number"},
                {"key": "longitude", "type": "number"},
            ],
            "tool_arg_key_types": tool_arg_key_types,
            "assistant_response_required": False,
            "assistant_response_template_allowed": True,
            "code_payload_allowed": True,
            "code_example_shape": {
                "decision": "propose",
                "authority_label": "ell1",
                "tool_sequence": suffix_tool_sequence,
                "code": setting_code
                + (
                    "locations = search_location_around_lat_lon(location='visible place')\n"
                    "loc = locations[0]\n"
                    "now = get_current_timestamp()\n"
                    "info = timestamp_to_datetime_info(timestamp=now)\n"
                    "target = datetime_info_to_timestamp(year=info['year'], month=info['month'], day=info['day'], hour=info['hour'], minute=info['minute'], second=0)\n"
                    "add_reminder(content='visible reminder content only', reminder_timestamp=target, latitude=loc['latitude'], longitude=loc['longitude'])"
                ),
            },
            "domain_instructions": [
                "This suffix is for reminders whose visible user text includes a place.",
                "Resolve the visible place with search_location_around_lat_lon, select the best matching result from its returned list, and pass that result's latitude and longitude to add_reminder.",
                "latitude and longitude are optional on add_reminder. If the location search returns no usable result after a reasonable attempt, still call add_reminder with content and reminder_timestamp and omit latitude and longitude; never invent coordinates and never pass 0. Discharging the reminder without the place beats leaving it undone.",
                *REMINDER_SCHEMA_INSTRUCTIONS,
                *(
                    [
                        "Before location search, read low-battery, Wi-Fi and location-service status in separate top-level statements; turn low-battery off only when on, and turn Wi-Fi and location service on only when off."
                    ]
                    if with_setting_repair
                    else []
                ),
                "Omit latitude and longitude in search_location_around_lat_lon so the search anchors at the device's current location; pass them only when the user gave precise coordinates. Never pass latitude=0 or longitude=0.",
                "For multi-turn clarification, combine visible user turns: one turn may contain the place and content, another may contain the time.",
            ],
            "fixed_argument_values_used": False,
            "requires_proposal_graph": True,
            "execution_verified": False,
        },
    }


def _location_setting_repair_available(context: Mapping[str, Any]) -> bool:
    available = {
        str(name)
        for name in context.get("available_tool_names") or ()
        if isinstance(name, str)
    }
    return {
        "get_low_battery_mode_status",
        LOW_BATTERY_MUTATING_TOOL,
        "get_wifi_status",
        WIFI_MUTATING_TOOL,
        "get_location_service_status",
        LOCATION_SERVICE_MUTATING_TOOL,
    }.issubset(available)


def _weather_temp_f_typed_row(
    *,
    text: str,
    conversation: Sequence[Mapping[str, Any]],
    cheap_step_count: int,
    public_ordinal: int,
    frontier_message_index: int,
) -> dict[str, Any] | None:
    if WEATHER_TEMP_F_RE.search(text or "") is None:
        return None
    if cheap_step_count <= 0 or not _prefix_has_wifi_failure(conversation):
        return None
    return {
        "schema": SCHEMA,
        "public_ordinal": int(public_ordinal),
        "family": "online_live_frontier",
        "residual_closable": True,
        "residual_closability_basis": "online_visible_prefix_after_wifi_failure_weather_temp_f_repair",
        "typed_suffix_blockers": [],
        "method_introduced_minefield_risk": "none",
        "frontier_binding": {
            "frontier_kind": "pre_write_terminal_prefix",
            "frontier_source": "online_live_prefix",
            "frontier_message_index": int(frontier_message_index),
            "frontier_tool_call_index": None,
            "authority_ceiling": "ell1",
            "effect_units": [WIFI_OFF_EFFECT_UNIT, WEATHER_TEMP_F_EFFECT_UNIT],
        },
        "typed_suffix": {
            "template_id": WEATHER_TEMP_F_TEMPLATE_ID,
            "effect_families": ["toolsandbox.setting", "toolsandbox.weather"],
            "suffix_tool_sequence": [
                WIFI_MUTATING_TOOL,
                "search_location_around_lat_lon",
                "search_weather_around_lat_lon",
                "unit_conversion",
            ],
            "read_tool_names": [
                "search_location_around_lat_lon",
                "search_weather_around_lat_lon",
                "unit_conversion",
            ],
            "mutating_tool_name": WIFI_MUTATING_TOOL,
            "mutating_tool_must_be_last": False,
            "mutating_arg_key_types": [
                {"key": "on", "type": "boolean"},
            ],
            "tool_arg_key_types": {
                "search_location_around_lat_lon": [
                    {"key": "location", "type": "string", "required": True},
                    {"key": "latitude", "type": "number", "required": False},
                    {"key": "longitude", "type": "number", "required": False},
                ],
                WIFI_MUTATING_TOOL: [
                    {"key": "on", "type": "boolean", "required": True},
                ],
                # All three are optional on the official tool; omitting lat/lon
                # anchors the query at the device's current location. days
                # defaults to 0, so that value comes from the signature rather
                # than from the request and needs no request grounding.
                "search_weather_around_lat_lon": [
                    {"key": "latitude", "type": "number", "required": False},
                    {"key": "longitude", "type": "number", "required": False},
                    {
                        "key": "days",
                        "type": "number",
                        "required": False,
                        "schema_default": 0,
                    },
                ],
                "unit_conversion": [
                    {"key": "amount", "type": "number", "required": True},
                    {"key": "from_unit", "type": "string", "required": True},
                    {"key": "to_unit", "type": "string", "required": True},
                ],
            },
            "assistant_response_required": True,
            "code_payload_allowed": False,
            "domain_instructions": [
                "This weather suffix is allowed only because the visible prefix already contains a Wi-Fi-disabled tool failure.",
                "First call set_wifi_status(on=True); do not repeat the failed weather/location read before repairing Wi-Fi.",
                "For a named landmark or general place, use search_location_around_lat_lon to resolve a canonical location; do not abstain solely because the user did not choose a sub-location.",
                "For landmark lookup with search_location_around_lat_lon, omit latitude and longitude so the search anchors at the device's current location; never pass latitude=0 or longitude=0, and do not hard-code landmark coordinates before lookup.",
                "For weather right now, omit days so search_weather_around_lat_lon uses its default of the current day. Its Celsius result may use temperature for current observations or average_temperature for forecast-style responses.",
                "Report the temperature you read from the weather result, converted with unit_conversion when the user cannot read Celsius. Never state a temperature no result returned.",
            ],
            "fixed_argument_values_used": False,
            "requires_proposal_graph": True,
            "execution_verified": False,
        },
    }


def _cheap_step_count(context: Mapping[str, Any]) -> int:
    try:
        return max(0, int(context.get("cheap_step_count") or 0))
    except (TypeError, ValueError):
        return 0


def _prefix_has_wifi_failure(conversation: Sequence[Mapping[str, Any]]) -> bool:
    for message in conversation:
        haystack = " ".join(
            str(value)
            for value in (
                message.get("content"),
                message.get("tool_call_exception"),
                message.get("name"),
                message.get("tool_name"),
            )
            if value is not None
        ).lower()
        compact = re.sub(r"[^a-z0-9]+", "", haystack)
        if (
            "wifiisnotenabled" in compact
            or "wifiisdisabled" in compact
            or "wi-fiisnotenabled" in haystack
            or "wi-fiisdisabled" in haystack
        ):
            return True
    return False


def _stock_symbol_typed_row(
    *,
    text: str,
    public_ordinal: int,
    frontier_message_index: int,
) -> dict[str, Any] | None:
    if STOCK_SYMBOL_RE.search(text or "") is None:
        return None
    if ISSUE_RESOLUTION_RE.search(text or "") is not None:
        return _stock_symbol_issue_resolution_typed_row(
            public_ordinal=public_ordinal,
            frontier_message_index=frontier_message_index,
        )
    return {
        "schema": SCHEMA,
        "public_ordinal": int(public_ordinal),
        "family": "online_live_frontier",
        "residual_closable": True,
        "residual_closability_basis": "online_visible_prefix_pre_write_stock_symbol_answer",
        "typed_suffix_blockers": [],
        "method_introduced_minefield_risk": "none",
        "frontier_binding": {
            "frontier_kind": "pre_write_terminal_prefix",
            "frontier_source": "online_live_prefix",
            "frontier_message_index": int(frontier_message_index),
            "frontier_tool_call_index": None,
            "authority_ceiling": "ell1",
            "effect_units": [STOCK_ANSWER_EFFECT_UNIT],
        },
        "typed_suffix": {
            "template_id": STOCK_TEMPLATE_ID,
            "effect_families": ["toolsandbox.stock"],
            "suffix_tool_sequence": [STOCK_READ_TOOL],
            "read_tool_names": [STOCK_READ_TOOL],
            "mutating_tool_name": STOCK_READ_TOOL,
            "mutating_tool_must_be_last": True,
            "mutating_arg_key_types": [
                {"key": "query", "type": "string"},
            ],
            "tool_arg_key_types": {
                STOCK_READ_TOOL: [
                    {"key": "query", "type": "string", "required": True},
                ],
            },
            "assistant_response_required": True,
            "assistant_response_template_allowed": True,
            "code_payload_allowed": False,
            "domain_instructions": [
                "Use exactly one search_stock call with the company name visible in the prefix as search_stock(query=...).",
                "Do not call setting tools for this read-only stock suffix unless the visible user request explicitly asks you to resolve an issue alone.",
                "Answer with the symbol field of the search_stock result. Never state a symbol you did not read from that result, even when you recognise the company.",
                "search_stock may legitimately return no result. Say so plainly in the final text instead of supplying a symbol.",
            ],
            "fixed_argument_values_used": False,
            "requires_proposal_graph": True,
            "execution_verified": False,
        },
    }


def _message_recency_typed_row(
    *,
    text: str,
    conversation: Sequence[Mapping[str, Any]],
    available_tool_names: Any,
    public_ordinal: int,
    frontier_message_index: int,
) -> dict[str, Any] | None:
    if MESSAGE_SEARCH_INTENT_RE.search(text or "") is None:
        return None
    if MESSAGE_SEARCH_NEGATION_RE.search(text or "") is not None:
        return None
    if MESSAGE_NON_READ_RE.search(text or "") is not None:
        return None
    recency = _last_message_recency(text)
    if recency is None:
        return None
    last_recency_index = _last_user_recency_index(conversation)
    last_search_index = _last_tool_call_index(conversation, MESSAGE_SEARCH_TOOL)
    if last_recency_index is None or (
        last_search_index is not None and last_recency_index <= last_search_index
    ):
        return None
    available = _available_tool_names(available_tool_names)
    if available is not None and MESSAGE_SEARCH_TOOL not in available:
        return None

    if available is not None and CURRENT_TIMESTAMP_TOOL not in available:
        return None
    suffix_tool_sequence = [CURRENT_TIMESTAMP_TOOL, MESSAGE_SEARCH_TOOL]
    comparison = ">" if recency == "latest" else "<"
    temporal_label = "most recent" if recency == "latest" else "oldest"
    code = (
        f"_current_timestamp = {CURRENT_TIMESTAMP_TOOL}()\n"
        + (
            f"messages = {MESSAGE_SEARCH_TOOL}("
            "creation_timestamp_upperbound=_current_timestamp)\n"
        )
        + "selected_content = 'not available'\n"
        + "selected_timestamp = None\n"
        + "for message in messages:\n"
        + (
            "    if selected_timestamp is None or "
            f"message['creation_timestamp'] {comparison} selected_timestamp:\n"
        )
        + "        selected_timestamp = message['creation_timestamp']\n"
        + "        selected_content = message['content']\n"
        + "print({'message_content': selected_content})"
    )
    return {
        "schema": SCHEMA,
        "public_ordinal": int(public_ordinal),
        "family": "online_live_frontier",
        "residual_closable": True,
        "residual_closability_basis": (
            "online_visible_prefix_pre_write_message_recency_answer"
        ),
        "typed_suffix_blockers": [],
        "method_introduced_minefield_risk": "none",
        "frontier_binding": {
            "frontier_kind": "pre_write_terminal_prefix",
            "frontier_source": "online_live_prefix",
            "frontier_message_index": int(frontier_message_index),
            "frontier_tool_call_index": None,
            "authority_ceiling": "ell1",
            "effect_units": [MESSAGE_ANSWER_EFFECT_UNIT],
        },
        "typed_suffix": {
            "template_id": MESSAGE_TEMPLATE_ID,
            "effect_families": ["toolsandbox.message"],
            "suffix_tool_sequence": suffix_tool_sequence,
            "read_tool_names": suffix_tool_sequence,
            "mutating_tool_name": MESSAGE_SEARCH_TOOL,
            "mutating_tool_must_be_last": True,
            "mutating_arg_key_types": [
                {"key": "creation_timestamp_upperbound", "type": "number"},
            ],
            "tool_arg_key_types": {
                CURRENT_TIMESTAMP_TOOL: [],
                MESSAGE_SEARCH_TOOL: [
                    {
                        "key": "creation_timestamp_upperbound",
                        "type": "number",
                        "required": True,
                    },
                ],
            },
            "assistant_response_allowed": False,
            "assistant_response_required": True,
            "assistant_response_template_allowed": True,
            "assistant_response_template_required": True,
            "code_payload_allowed": True,
            "code_example_shape": {
                "decision": "propose",
                "authority_label": "ell1",
                "tool_sequence": suffix_tool_sequence,
                "code": code,
                # A placeholder only. Wording that mirrors a scored reference
                # answer would be fitting the scorer, not completing the residual.
                "assistant_response_template": "{message_content}",
                "assistant_response_bindings": {
                    "message_content": {
                        "source": "code_output",
                        "output_index": -1,
                        "path": ["message_content"],
                        "fallback": "not available",
                    }
                },
            },
            "domain_instructions": [
                (
                    f"The visible user turns jointly request the {temporal_label} "
                    "message; this direction comes only from the visible prefix."
                ),
                (
                    "Read the current timestamp, pass it as "
                    "search_messages.creation_timestamp_upperbound, and choose the "
                    "row by comparing the returned creation_timestamp values; the "
                    "list order carries no meaning."
                ),
                (
                    "Quote the content field of the row you selected verbatim in "
                    "the final text. Never reword it and never supply content no "
                    "row returned."
                ),
                (
                    "If search_messages returns no rows, say so in the final text; "
                    "never invent a message."
                ),
            ],
            "fixed_argument_values_used": False,
            "requires_proposal_graph": True,
            "execution_verified": False,
        },
    }


def _last_message_recency(text: str) -> str | None:
    recency: str | None = None
    for match in MESSAGE_RECENCY_RE.finditer(text or ""):
        recency = "latest" if match.group("latest") is not None else "oldest"
    return recency


def _available_tool_names(value: Any) -> frozenset[str] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return frozenset()
    return frozenset(str(name) for name in value if isinstance(name, str))


def _stock_symbol_issue_resolution_typed_row(
    *,
    public_ordinal: int,
    frontier_message_index: int,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "public_ordinal": int(public_ordinal),
        "family": "online_live_frontier",
        "residual_closable": True,
        "residual_closability_basis": "online_visible_prefix_pre_write_stock_symbol_issue_resolution",
        "typed_suffix_blockers": [],
        "method_introduced_minefield_risk": "none",
        "frontier_binding": {
            "frontier_kind": "pre_write_terminal_prefix",
            "frontier_source": "online_live_prefix",
            "frontier_message_index": int(frontier_message_index),
            "frontier_tool_call_index": None,
            "authority_ceiling": "ell1",
            "effect_units": [
                STOCK_ANSWER_EFFECT_UNIT,
                LOW_BATTERY_EFFECT_UNIT,
                WIFI_OFF_EFFECT_UNIT,
            ],
        },
        "typed_suffix": {
            "template_id": STOCK_TEMPLATE_ID,
            "effect_families": ["toolsandbox.stock", "toolsandbox.setting"],
            "suffix_tool_sequence": [
                "get_low_battery_mode_status",
                LOW_BATTERY_MUTATING_TOOL,
                "get_wifi_status",
                WIFI_MUTATING_TOOL,
                STOCK_READ_TOOL,
            ],
            "read_tool_names": [
                "get_low_battery_mode_status",
                "get_wifi_status",
                STOCK_READ_TOOL,
            ],
            "mutating_tool_name": STOCK_READ_TOOL,
            "mutating_tool_must_be_last": True,
            "mutating_arg_key_types": [
                {"key": "query", "type": "string"},
            ],
            "tool_arg_key_types": {
                "get_low_battery_mode_status": [],
                LOW_BATTERY_MUTATING_TOOL: [
                    {"key": "on", "type": "boolean", "required": True},
                ],
                "get_wifi_status": [],
                WIFI_MUTATING_TOOL: [
                    {"key": "on", "type": "boolean", "required": True},
                ],
                STOCK_READ_TOOL: [
                    {"key": "query", "type": "string", "required": True},
                ],
            },
            "assistant_response_required": True,
            "assistant_response_template_allowed": True,
            "code_payload_allowed": True,
            "domain_instructions": [
                "The visible user request explicitly asks you to resolve issues alone before answering the stock-symbol question.",
                "Read low-battery and Wi-Fi status first, and write a setting only when its status blocks the search: turn low-battery off only when it is on, and Wi-Fi on only when it is off.",
                "Repair the settings before search_stock; the search cannot succeed while Wi-Fi is off.",
                "Answer with the symbol field of the search_stock result. Never state a symbol you did not read from that result, even when you recognise the company.",
                "search_stock may legitimately return no result. Say so plainly in the final text instead of supplying a symbol.",
            ],
            "fixed_argument_values_used": False,
            "requires_proposal_graph": True,
            "execution_verified": False,
        },
    }


def _setting_off_typed_row(
    *,
    text: str,
    public_ordinal: int,
    frontier_message_index: int,
) -> dict[str, Any] | None:
    match = SETTING_OFF_RE.search(text or "")
    if match is None:
        return None
    target = match.group("target").lower()
    if target in {"wifi", "wi-fi", "wireless"}:
        return _setting_typed_row(
            public_ordinal=public_ordinal,
            frontier_message_index=frontier_message_index,
            effect_unit=WIFI_OFF_EFFECT_UNIT,
            template_id=WIFI_OFF_TEMPLATE_ID,
            mutating_tool_name=WIFI_MUTATING_TOOL,
        )
    return _setting_typed_row(
        public_ordinal=public_ordinal,
        frontier_message_index=frontier_message_index,
        effect_unit=CELLULAR_OFF_EFFECT_UNIT,
        template_id=CELLULAR_OFF_TEMPLATE_ID,
        mutating_tool_name=CELLULAR_MUTATING_TOOL,
    )


def _setting_typed_row(
    *,
    public_ordinal: int,
    frontier_message_index: int,
    effect_unit: str,
    template_id: str,
    mutating_tool_name: str,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "public_ordinal": int(public_ordinal),
        "family": "online_live_frontier",
        "residual_closable": True,
        "residual_closability_basis": "online_visible_prefix_pre_write_setting_off",
        "typed_suffix_blockers": [],
        "method_introduced_minefield_risk": "none",
        "frontier_binding": {
            "frontier_kind": "pre_write_terminal_prefix",
            "frontier_source": "online_live_prefix",
            "frontier_message_index": int(frontier_message_index),
            "frontier_tool_call_index": None,
            "authority_ceiling": "ell1",
            "effect_units": [effect_unit],
        },
        "typed_suffix": {
            "template_id": template_id,
            "effect_families": ["toolsandbox.setting"],
            "suffix_tool_sequence": [mutating_tool_name],
            "read_tool_names": [],
            "mutating_tool_name": mutating_tool_name,
            "mutating_arg_key_types": [
                {"key": "on", "type": "boolean"},
            ],
            "tool_arg_key_types": {
                mutating_tool_name: [
                    {"key": "on", "type": "boolean", "required": True},
                ],
            },
            "assistant_response_required": True,
            "domain_instructions": [
                f"The visible request asks to switch this setting off, so call"
                f" {mutating_tool_name} once with the value that turns it off.",
                "Then finish with terminal text confirming the change you made."
                " The residual is not complete until the user has been told.",
            ],
            "fixed_argument_values_used": False,
            "requires_proposal_graph": True,
            "execution_verified": False,
        },
    }


def _conversation(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _user_text(message: Mapping[str, Any]) -> str:
    if str(message.get("role") or "") != "user":
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _has_reminder_intent(text: str) -> bool:
    if _NON_ADD_REMINDER_RE.search(text or ""):
        return False
    return bool(_INTENT_RE.search(text or ""))


def _has_time_carrier(text: str) -> bool:
    return bool(_TIME_RE.search(text or ""))


def _has_content_carrier(text: str) -> bool:
    cleaned = _INTENT_RE.sub(" ", text or "")
    cleaned = _TIME_RE.sub(" ", cleaned)
    cleaned = _CONTENT_META_RE.sub(" ", cleaned)
    tokens = [token.lower() for token in _WORD_RE.findall(cleaned)]
    return any(token not in _CONTENT_STOPWORDS for token in tokens)


def _has_location_cue(text: str) -> bool:
    return REMINDER_LOCATION_CUE_RE.search(text or "") is not None


def _prefix_has_mutating_reminder_write(
    conversation: Sequence[Mapping[str, Any]],
) -> bool:
    for message in conversation:
        for tool_name in _tool_names(message):
            if tool_name == REMINDER_MUTATING_TOOL:
                return True
        content = message.get("content")
        if isinstance(content, str) and "add_reminder(" in content:
            return True
    return False


def _last_tool_call_index(
    conversation: Sequence[Mapping[str, Any]],
    tool_name: str,
) -> int | None:
    last_index: int | None = None
    for index, message in enumerate(conversation):
        if tool_name in _tool_names(message):
            last_index = index
    return last_index


def _last_user_recency_index(
    conversation: Sequence[Mapping[str, Any]],
) -> int | None:
    last_index: int | None = None
    for index, message in enumerate(conversation):
        if _last_message_recency(_user_text(message)) is not None:
            last_index = index
    return last_index


def _tool_names(message: Mapping[str, Any]) -> tuple[str, ...]:
    names: list[str] = []
    for tool_call in message.get("tool_calls") or ():
        if not isinstance(tool_call, Mapping):
            continue
        function = tool_call.get("function")
        if isinstance(function, Mapping):
            names.append(str(function.get("name") or ""))
        else:
            names.append(str(tool_call.get("name") or tool_call.get("function_name") or ""))
    names.extend(_trace_tool_names(message.get("tool_trace")))
    return tuple(name for name in names if name)


def _trace_tool_names(value: Any) -> tuple[str, ...]:
    names: list[str] = []

    def collect(item: Any) -> None:
        if isinstance(item, str):
            try:
                collect(json.loads(item))
            except (TypeError, ValueError):
                for known_name in (
                    REMINDER_MUTATING_TOOL,
                    MESSAGE_SEARCH_TOOL,
                ):
                    if known_name in item:
                        names.append(known_name)
                return
            return
        if isinstance(item, Mapping):
            name = item.get("tool_name") or item.get("name")
            if isinstance(name, str) and name:
                names.append(name)
            for nested_key in ("tool_trace", "tool_traces"):
                if nested_key in item:
                    collect(item[nested_key])
            return
        if isinstance(item, Sequence) and not isinstance(
            item,
            (str, bytes, bytearray),
        ):
            for nested in item:
                collect(nested)

    collect(value)
    return tuple(names)


__all__ = [
    "MESSAGE_ANSWER_EFFECT_UNIT",
    "MESSAGE_SEARCH_TOOL",
    "MESSAGE_TEMPLATE_ID",
    "REMINDER_EFFECT_UNIT",
    "REMINDER_MUTATING_TOOL",
    "REMINDER_TEMPLATE_ID",
    "REMINDER_TOOL_SEQUENCE",
    "SCHEMA",
    "ToolSandboxReminderAddFrontierMonitor",
]
