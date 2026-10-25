"""Assignment 3 tools: search_catalogue, calculate_fine, book_study_room + orchestrator failures."""

import asyncio
from datetime import datetime

import pytest

from backend.app.conversation import tool_orchestrator
from backend.app.conversation.tool_orchestrator import chat_with_tools, run_tool
from backend.app.domain import rooms
from backend.app.domain.fines import calculate_fine
from backend.app.domain.records import records_db
from backend.app.domain.tools import TOOLS, tool_specs, validate_args

# --------------------------------------------------------------------------- registry


def test_every_tool_has_name_description_and_schema():
    specs = tool_specs()
    names = {s["function"]["name"] for s in specs}
    assert {"search_catalogue", "calculate_fine", "book_study_room"} <= names
    for spec in specs:
        fn = spec["function"]
        assert fn["description"] and fn["parameters"]["type"] == "object"
        assert set(fn["parameters"]["required"]) <= set(fn["parameters"]["properties"])


def test_validate_args_coerces_and_rejects():
    assert validate_args("search_catalogue", {"query": "python", "limit": "3", "junk": 1}) == {
        "query": "python",
        "limit": 3,
    }
    with pytest.raises(ValueError, match="Missing required"):
        validate_args("calculate_fine", {})
    with pytest.raises(ValueError, match="must be one of"):
        validate_args("book_study_room", {"action": "delete_everything"})
    with pytest.raises(KeyError):
        validate_args("launch_rocket", {})


# --------------------------------------------------------------------------- search_catalogue


async def test_search_catalogue_finds_book_with_location_and_status():
    res = await records_db.search_catalogue("speech and language processing")
    assert res["count"] >= 1
    top = res["results"][0]
    assert top["id"] == "BK-1002" and "Stacks B" in top["location"] and top["status"]


async def test_search_catalogue_available_only_and_no_match():
    res = await records_db.search_catalogue("algorithms", available_only=True)
    assert all(b["status"] == "available" for b in res["results"])
    none = await records_db.search_catalogue("harry potter")
    assert none["count"] == 0 and "No catalogue items" in none["message"]


# --------------------------------------------------------------------------- calculate_fine (POL-002)


@pytest.mark.parametrize(
    "due, returned, expected",
    [
        ("2026-10-25", "2026-10-25", 0),     # on time
        ("2026-10-25", "2026-10-26", 20),    # 1 day late
        ("2026-10-18", "2026-10-25", 140),   # 7 days
        ("2026-10-01", "2026-10-25", 200),   # 24 days -> capped at 200
        ("2026-10-25", "2026-10-20", 0),     # returned early
    ],
)
async def test_calculate_fine_policy(due, returned, expected):
    res = await calculate_fine(due, returned)
    assert res["fine_pkr"] == expected
    assert res["policy"].startswith("POL-002")


async def test_calculate_fine_lost_and_bad_date():
    lost = await calculate_fine("2026-08-01", "2026-10-25")
    assert lost["declared_lost"] and "2400" in lost["lost_charges"]
    with pytest.raises(ValueError):
        await calculate_fine("25/10/2026")


# --------------------------------------------------------------------------- book_study_room


@pytest.fixture
def room_calendar(tmp_path, monkeypatch):
    monkeypatch.setattr(rooms, "BOOKINGS_PATH", tmp_path / "room_bookings.json")
    monkeypatch.setattr(rooms, "_now", lambda: datetime(2026, 10, 26, 10, 0))  # Monday 10:00


async def test_room_booking_flow(room_calendar):
    ok = await rooms.book_study_room("book", "P-78421", "304", "2026-10-26", "14:00", 120)
    assert ok["success"], ok
    booking_id = ok["booking"]["booking_id"]

    clash = await rooms.book_study_room("book", "P-50192", "304", "2026-10-26", "15:00", 60)
    assert not clash["success"] and "already booked" in clash["error"]

    avail = await rooms.book_study_room("list_availability", room_id="304", date="2026-10-26")
    assert avail["rooms"][0]["free"] == ["08:00-14:00", "16:00-22:00"]

    mine = await rooms.book_study_room("my_bookings", card_number="P-78421")
    assert [b["booking_id"] for b in mine["bookings"]] == [booking_id]

    cancelled = await rooms.book_study_room("cancel", card_number="P-78421", booking_id=booking_id)
    assert cancelled["success"]


@pytest.mark.parametrize(
    "args, error",
    [
        (("P-78421", "301", "2026-10-26", "09:00", 60), "past"),            # before "now"
        (("P-78421", "301", "2026-10-26", "21:30", 60), "opening hours"),   # ends after 22:00
        (("P-78421", "301", "2026-10-26", "12:00", 240), "max 3 hours"),    # too long
        (("P-78421", "301", "2026-10-31", "12:00", 60), "3 days"),          # Saturday, too far ahead
        (("P-78421", "301", "2026-11-01", "12:00", 60), "closed on Sundays"),
        (("P-78421", "999", "2026-10-26", "12:00", 60), "Unknown room"),
        (("", "301", "2026-10-26", "12:00", 60), "card number"),
    ],
)
async def test_room_booking_rejections(room_calendar, args, error):
    res = await rooms.book_study_room("book", *args)
    assert not res["success"] and error in res["error"]


async def test_room_daily_limit(room_calendar):
    assert (await rooms.book_study_room("book", "P-78421", "301", "2026-10-26", "11:00", 120))["success"]
    res = await rooms.book_study_room("book", "P-78421", "302", "2026-10-26", "15:00", 90)
    assert not res["success"] and "3 hours per day" in res["error"]


# --------------------------------------------------------------------------- orchestrator failures


async def test_run_tool_unknown_tool():
    result, ok, _ = await run_tool("launch_rocket", {})
    assert not ok and "Unknown tool" in result["error"]


async def test_run_tool_bad_arguments():
    result, ok, _ = await run_tool("calculate_fine", {"return_date": "2026-10-25"})
    assert not ok and "Missing required" in result["error"]


async def test_run_tool_timeout(monkeypatch):
    async def slow_tool(**_):
        await asyncio.sleep(10)
        return {"success": True}

    monkeypatch.setitem(TOOLS, "slow_tool", {"description": "slow", "handler": slow_tool,
                                             "parameters": {"type": "object", "properties": {}, "required": []}})
    monkeypatch.setattr(tool_orchestrator.settings, "tool_timeout_seconds", 0.2)
    result, ok, latency_ms = await run_tool("slow_tool", {})
    assert not ok and "took longer" in result["error"]
    assert latency_ms < 2000  # gave up quickly, did not hang


async def test_run_tool_exception(monkeypatch):
    async def broken_tool(**_):
        raise OSError("disk unplugged")

    monkeypatch.setitem(TOOLS, "broken_tool", {"description": "broken", "handler": broken_tool,
                                               "parameters": {"type": "object", "properties": {}, "required": []}})
    result, ok, _ = await run_tool("broken_tool", {})
    assert not ok and "disk unplugged" in result["error"]


async def test_run_tool_fills_card_from_conversation():
    result, ok, _ = await run_tool("check_patron_loans", {}, defaults={"card_number": "P-78421"})
    assert ok and result["card_number"] == "P-78421"


class FakeLLM:
    """Round 1: asks for a tool. Round 2: answers from the tool result."""

    def __init__(self):
        self.calls = []

    async def stream_chat(self, payload, tools=None, tool_calls_out=None):
        self.calls.append(tools is not None)
        if len(self.calls) == 1:
            tool_calls_out.append({"function": {"name": "calculate_fine",
                                                "arguments": {"due_date": "2026-10-18", "return_date": "2026-10-25"}}})
            return
        assert payload[-1]["role"] == "tool" and '"fine_pkr": 140' in payload[-1]["content"]
        for word in ["You ", "owe ", "PKR 140."]:
            yield word


async def test_orchestrator_loop_calls_tool_then_answers():
    llm, payload = FakeLLM(), [{"role": "user", "content": "fine for a book due 18 Oct?"}]
    events = [e async for e in chat_with_tools(llm, payload, "s1")]
    kinds = [k for k, _ in events]
    assert kinds.index("tool_call") < kinds.index("tool_result") < kinds.index("token")
    assert "".join(d["content"] for k, d in events if k == "token") == "You owe PKR 140."
    assert llm.calls == [True, True]  # tools offered in both rounds (max 2 tool rounds)
