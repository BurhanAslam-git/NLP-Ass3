"""
book_study_room tool — a small calendar for the group study rooms.

Facts from our documents:
- FAQ-005: 6 group study rooms on Level 3. 301–303 seat up to 6; 304–306 seat
  up to 10 and have a projector + whiteboard. Bookings up to 3 days in advance,
  max 3 hours per group per day. A patron card number is required.
- POL-009: opening hours Mon–Fri 08:00–22:00, Sat 09:00–18:00, Sunday closed.
  (Extended exam-week hours from FAQ-008 are not modelled.)

Bookings live in data/room_bookings.json. Every read-modify-write happens under
one asyncio.Lock and is saved atomically (temp file + os.replace), so two users
can never double-book the same slot.
"""

import asyncio
import json
import os
import re
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any

BOOKINGS_PATH = Path("data/room_bookings.json")

ROOMS = {
    "301": {"capacity": 6, "features": "whiteboard"},
    "302": {"capacity": 6, "features": "whiteboard"},
    "303": {"capacity": 6, "features": "whiteboard"},
    "304": {"capacity": 10, "features": "projector and whiteboard"},
    "305": {"capacity": 10, "features": "projector and whiteboard"},
    "306": {"capacity": 10, "features": "projector and whiteboard"},
}
# weekday() -> (open, close); Monday = 0 ... Sunday = 6 (closed)
OPENING_HOURS = {d: (time(8), time(22)) for d in range(5)} | {5: (time(9), time(18))}
MAX_MINUTES_PER_DAY = 180
ADVANCE_DAYS = 3
_CARD_RE = re.compile(r"^P-\d{5}$")

_lock = asyncio.Lock()


def _now() -> datetime:
    return datetime.now()  # separate function so tests can freeze time


def _load() -> list[dict[str, Any]]:
    if not BOOKINGS_PATH.exists():
        return []
    return json.loads(BOOKINGS_PATH.read_text(encoding="utf-8")).get("bookings", [])


def _save(bookings: list[dict[str, Any]]) -> None:
    tmp = BOOKINGS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps({"bookings": bookings}, indent=2), encoding="utf-8")
    os.replace(tmp, BOOKINGS_PATH)  # atomic: readers never see a half-written file


def _fail(message: str) -> dict[str, Any]:
    return {"success": False, "error": message}


def _minutes(t: time) -> int:
    return t.hour * 60 + t.minute


def _fmt(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _free_slots(day: str, room: str, bookings: list[dict], open_m: int, close_m: int) -> list[str]:
    """Free intervals in one room on one day, e.g. ['08:00-10:00', '12:00-22:00']."""
    busy = sorted(
        (_minutes(time.fromisoformat(b["start_time"])), _minutes(time.fromisoformat(b["end_time"])))
        for b in bookings if b["date"] == day and b["room_id"] == room
    )
    free, cursor = [], open_m
    for start, end in busy:
        if start > cursor:
            free.append(f"{_fmt(cursor)}-{_fmt(start)}")
        cursor = max(cursor, end)
    if cursor < close_m:
        free.append(f"{_fmt(cursor)}-{_fmt(close_m)}")
    return free


async def book_study_room(
    action: str,
    card_number: str | None = None,
    room_id: str | None = None,
    date: str | None = None,          # noqa: A002  (name matches the tool schema)
    start_time: str | None = None,
    duration_minutes: int | None = None,
    booking_id: str | None = None,
) -> dict[str, Any]:
    now = _now()
    card = (card_number or "").strip().upper()
    room = (room_id or "").strip().removeprefix("Room ").removeprefix("room ")

    async with _lock:
        bookings = _load()

        # ---------------------------------------------------------- my_bookings
        if action == "my_bookings":
            if not _CARD_RE.match(card):
                return _fail("A patron card number like P-78421 is required.")
            mine = [b for b in bookings if b["card_number"] == card and b["date"] >= now.date().isoformat()]
            return {"success": True, "card_number": card, "bookings": mine}

        # --------------------------------------------------------------- cancel
        if action == "cancel":
            match = next((b for b in bookings if b["booking_id"] == (booking_id or "").strip().upper()), None)
            if match is None:
                return _fail(f"No booking with id {booking_id!r}.")
            if card and match["card_number"] != card:
                return _fail("That booking belongs to a different patron card.")
            bookings.remove(match)
            _save(bookings)
            return {"success": True, "cancelled": match}

        # list_availability and book both need a valid day
        try:
            day = datetime.strptime(date, "%Y-%m-%d").date() if date else now.date()
        except ValueError:
            return _fail("date must be in YYYY-MM-DD format.")
        if day.weekday() not in OPENING_HOURS:
            return _fail(f"The library is closed on Sundays ({day.isoformat()}). Rooms can be booked Monday to Saturday.")
        open_t, close_t = OPENING_HOURS[day.weekday()]
        open_m, close_m = _minutes(open_t), _minutes(close_t)
        if room and room not in ROOMS:
            return _fail(f"Unknown room {room_id!r}. Rooms are {', '.join(ROOMS)} (Level 3).")

        # ---------------------------------------------------- list_availability
        if action == "list_availability":
            rooms = [room] if room else list(ROOMS)
            free = {r: _free_slots(day.isoformat(), r, bookings, open_m, close_m) for r in rooms}
            return {
                "success": True,
                "date": day.isoformat(),
                "opening_hours": f"{open_t:%H:%M}-{close_t:%H:%M}",
                "rooms": [{"room_id": r, **ROOMS[r], "free": free[r]} for r in rooms],
                "message": f"Free times on {day.isoformat()}: "
                           + "; ".join(f"room {r}: {', '.join(f) or 'fully booked'}" for r, f in free.items()),
            }

        # ----------------------------------------------------------------- book
        if action != "book":
            return _fail("action must be one of: list_availability, book, cancel, my_bookings.")
        if not _CARD_RE.match(card):
            return _fail("A patron card number like P-78421 is required to book a room.")
        if not room:
            return _fail("room_id is required (301-306).")
        if not start_time or duration_minutes is None:
            return _fail("start_time (HH:MM) and duration_minutes are required.")
        try:
            start = datetime.combine(day, time.fromisoformat(start_time))
        except ValueError:
            return _fail("start_time must be HH:MM (24-hour), e.g. 14:00.")
        if not 0 < duration_minutes <= MAX_MINUTES_PER_DAY:
            return _fail(f"Bookings can be 1 to {MAX_MINUTES_PER_DAY} minutes (max 3 hours per day).")
        end = start + timedelta(minutes=duration_minutes)

        if start < now:
            return _fail("That time is in the past.")
        if day > now.date() + timedelta(days=ADVANCE_DAYS):
            return _fail(f"Rooms can only be booked up to {ADVANCE_DAYS} days in advance.")
        if start.time() < open_t or end.date() != day or end.time() > close_t:
            return _fail(f"Outside opening hours. On {day:%A} the library is open {open_t:%H:%M}-{close_t:%H:%M}.")

        day_str, s_m, e_m = day.isoformat(), _minutes(start.time()), _minutes(end.time())
        used = sum(
            _minutes(time.fromisoformat(b["end_time"])) - _minutes(time.fromisoformat(b["start_time"]))
            for b in bookings if b["card_number"] == card and b["date"] == day_str
        )
        if used + duration_minutes > MAX_MINUTES_PER_DAY:
            return _fail(f"Card {card} already has {used} minutes booked on {day_str}; the limit is 3 hours per day.")
        for b in bookings:
            overlaps = (
                b["date"] == day_str
                and s_m < _minutes(time.fromisoformat(b["end_time"]))
                and _minutes(time.fromisoformat(b["start_time"])) < e_m
            )
            if overlaps and b["room_id"] == room:
                return _fail(
                    f"Room {room} is already booked {b['start_time']}-{b['end_time']} on {day_str}. "
                    f"Free times: {', '.join(_free_slots(day_str, room, bookings, open_m, close_m)) or 'none'}."
                )
            if overlaps and b["card_number"] == card:
                return _fail(f"Card {card} already has room {b['room_id']} at that time.")

        new_id = f"SR-{max([int(b['booking_id'][3:]) for b in bookings] + [0]) + 1:04d}"
        booking = {
            "booking_id": new_id,
            "room_id": room,
            "card_number": card,
            "date": day_str,
            "start_time": _fmt(s_m),
            "end_time": _fmt(e_m),
            "capacity": ROOMS[room]["capacity"],
        }
        bookings.append(booking)
        _save(bookings)
        return {"success": True, "booking": booking,
                "message": f"Room {room} booked for {day_str} {_fmt(s_m)}-{_fmt(e_m)} (booking {new_id})."}
