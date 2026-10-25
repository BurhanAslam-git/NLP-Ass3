"""
Tool registry: every tool the LLM may call, defined in ONE place.

Each tool = name + description + JSON schema for its arguments + an async
handler. TOOL_SPECS (name/description/schema only) is sent to Ollama in the
"tools" field of /api/chat; the model reads them and decides whether to answer
directly or to return a structured tool call such as
    {"function": {"name": "calculate_fine", "arguments": {"due_date": "2026-10-18"}}}
The orchestrator (conversation/tool_orchestrator.py) then validates the
arguments with validate_args() and runs the handler with a timeout.

Assignment 3 primary tools: search_catalogue, calculate_fine, book_study_room.
Kept from Assignment 2: issue_book, renew_book, return_book, check_patron_loans.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from backend.app.domain.fines import calculate_fine
from backend.app.domain.records import records_db
from backend.app.domain.rooms import book_study_room

logger = logging.getLogger(__name__)

# Descriptions are kept short: every token here is re-read by the LLM on every turn.
_CARD = {"type": "string", "description": "Card number, e.g. P-78421"}
_BOOK = {"type": "string", "description": "Book title or ID, e.g. BK-1003"}

# name -> {"description", "parameters" (JSON schema), "handler"}
TOOLS: dict[str, dict[str, Any]] = {
    "search_catalogue": {
        "description": "Find books in the live catalogue: shelf location, status, due date.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Title words or subject"},
                "author": {"type": "string"},
                "available_only": {"type": "boolean"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
        "handler": records_db.search_catalogue,
    },
    "calculate_fine": {
        "description": "Overdue fine for one late item (PKR 20/day, max PKR 200).",
        "parameters": {
            "type": "object",
            "properties": {
                "due_date": {"type": "string", "description": "YYYY-MM-DD"},
                "return_date": {"type": "string", "description": "YYYY-MM-DD, default today"},
                "item_type": {"type": "string", "description": "book or reference"},
            },
            "required": ["due_date"],
        },
        "handler": calculate_fine,
    },
    "book_study_room": {
        "description": "Study rooms 301-306: list free times, book (max 3 h), cancel, or list my bookings.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["list_availability", "book", "cancel", "my_bookings"]},
                "card_number": _CARD,
                "room_id": {"type": "string", "description": "301-306"},
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "start_time": {"type": "string", "description": "HH:MM 24h"},
                "duration_minutes": {"type": "integer"},
                "booking_id": {"type": "string"},
            },
            "required": ["action"],
        },
        "handler": book_study_room,
    },
    "check_patron_loans": {
        "description": "A patron's current loans and due dates.",
        "parameters": {"type": "object", "properties": {"card_number": _CARD}, "required": ["card_number"]},
        "handler": lambda card_number: records_db.check_patron_loans(card_number=card_number),
    },
    "issue_book": {
        "description": "Check out a book to a patron.",
        "parameters": {
            "type": "object",
            "properties": {"card_number": _CARD, "book_title_or_id": _BOOK},
            "required": ["card_number", "book_title_or_id"],
        },
        "handler": lambda card_number, book_title_or_id: records_db.issue_book(card_number, book_title_or_id),
    },
    "renew_book": {
        "description": "Renew a patron's loan by 21 days.",
        "parameters": {
            "type": "object",
            "properties": {"card_number": _CARD, "book_title_or_id": _BOOK},
            "required": ["card_number", "book_title_or_id"],
        },
        "handler": lambda card_number, book_title_or_id: records_db.renew_book(card_number, book_title_or_id),
    },
    "return_book": {
        "description": "Return a borrowed book.",
        "parameters": {
            "type": "object",
            "properties": {"card_number": _CARD, "book_title_or_id": _BOOK},
            "required": ["card_number", "book_title_or_id"],
        },
        "handler": lambda card_number, book_title_or_id: records_db.return_book(card_number, book_title_or_id),
    },
}


def tool_specs() -> list[dict[str, Any]]:
    """The list Ollama receives in its "tools" field (OpenAI function format)."""
    return [
        {"type": "function", "function": {"name": name, "description": t["description"], "parameters": t["parameters"]}}
        for name, t in TOOLS.items()
    ]


# Kept for Assignment 2 code/tests that import LIBRARY_TOOLS.
LIBRARY_TOOLS = tool_specs()


def validate_args(name: str, args: Any) -> dict[str, Any]:
    """
    Check arguments against the tool's JSON schema. Returns cleaned arguments
    or raises ValueError with a message the LLM (and the user) can understand.
    Small models often send numbers/booleans as strings, so we convert those;
    unknown keys are dropped.
    """
    if name not in TOOLS:
        raise KeyError(f"Unknown tool '{name}'. Available tools: {', '.join(TOOLS)}.")
    if not isinstance(args, dict):
        raise ValueError("Tool arguments must be a JSON object.")
    schema = TOOLS[name]["parameters"]
    clean: dict[str, Any] = {}
    for key, value in args.items():
        prop = schema["properties"].get(key)
        if prop is None or value is None or value == "":
            continue
        kind = prop["type"]
        if kind == "integer":
            try:
                value = int(value)
            except (TypeError, ValueError):
                raise ValueError(f"'{key}' must be a whole number.")
        elif kind == "boolean":
            if isinstance(value, str):
                value = value.strip().lower() in ("true", "yes", "1")
            value = bool(value)
        else:
            value = str(value).strip()
        if "enum" in prop and value not in prop["enum"]:
            raise ValueError(f"'{key}' must be one of {prop['enum']}.")
        clean[key] = value
    missing = [r for r in schema["required"] if r not in clean]
    if missing:
        raise ValueError(f"Missing required argument(s): {', '.join(missing)}.")
    return clean


async def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate + run a tool (no timeout). The orchestrator adds the timeout."""
    try:
        args = validate_args(name, arguments)
    except (KeyError, ValueError) as exc:
        return {"success": False, "error": str(exc).strip("'\"")}
    logger.info("Executing tool %s(%s)", name, args)
    handler: Callable[..., Awaitable[dict[str, Any]]] = TOOLS[name]["handler"]
    return await handler(**args)
