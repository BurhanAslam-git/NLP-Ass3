"""
calculate_fine tool — overdue fines exactly as in POL-002 (Overdue Fines Policy).

Policy facts used (data/documents/POL-002_fine_policy.md):
- PKR 20 per item per day, for all loan types.
- Fines start from midnight on the day after the due date
  (returned 1 day late = 1 day of fine).
- Capped at PKR 200 per item.
- Not returned within 60 days of the due date => declared lost: replacement
  cost (market price or PKR 2,000, whichever is greater) + PKR 200 processing
  fee + accrued fines up to the cap.
- Unpaid fines above PKR 100 suspend borrowing privileges.
Dates are parsed with date.fromisoformat — no eval, no free-form maths.
"""

from datetime import date
from typing import Any

FINE_PER_DAY_PKR = 20
FINE_CAP_PKR = 200
LOST_AFTER_DAYS = 60
LOST_MIN_REPLACEMENT_PKR = 2000
LOST_PROCESSING_FEE_PKR = 200
SUSPENSION_ABOVE_PKR = 100


def _parse(value: str, field: str) -> date:
    try:
        return date.fromisoformat(value.strip())
    except (ValueError, AttributeError):
        raise ValueError(f"{field} must be a date in YYYY-MM-DD format, got {value!r}.")


async def calculate_fine(due_date: str, return_date: str | None = None, item_type: str = "book") -> dict[str, Any]:
    due = _parse(due_date, "due_date")
    returned = _parse(return_date, "return_date") if return_date else date.today()

    if item_type.strip().lower() in ("reference", "reference_only", "periodical"):
        return {
            "success": True,
            "fine_pkr": 0,
            "message": "Reference items and periodicals do not circulate, so no overdue fine applies (POL-006).",
        }

    days_overdue = max(0, (returned - due).days)
    uncapped = days_overdue * FINE_PER_DAY_PKR
    fine = min(uncapped, FINE_CAP_PKR)
    declared_lost = days_overdue > LOST_AFTER_DAYS

    if days_overdue == 0:
        breakdown = f"Returned on or before the due date ({due.isoformat()}): no fine."
    else:
        breakdown = f"{days_overdue} day(s) late x PKR {FINE_PER_DAY_PKR} = PKR {uncapped}"
        breakdown += f", capped at PKR {FINE_CAP_PKR}." if uncapped > FINE_CAP_PKR else "."

    result: dict[str, Any] = {
        "success": True,
        "due_date": due.isoformat(),
        "return_date": returned.isoformat(),
        "item_type": item_type,
        "days_overdue": days_overdue,
        "rate_pkr_per_day": FINE_PER_DAY_PKR,
        "cap_pkr": FINE_CAP_PKR,
        "fine_pkr": fine,
        "breakdown": breakdown,
        "borrowing_suspended": fine > SUSPENSION_ABOVE_PKR,
        "declared_lost": declared_lost,
        "policy": "POL-002 Overdue Fines Policy",
    }
    if declared_lost:
        minimum_total = LOST_MIN_REPLACEMENT_PKR + LOST_PROCESSING_FEE_PKR + fine
        result["lost_charges"] = (
            f"Over {LOST_AFTER_DAYS} days late, so the item is declared lost: replacement cost "
            f"(market price or PKR {LOST_MIN_REPLACEMENT_PKR}, whichever is greater) + PKR "
            f"{LOST_PROCESSING_FEE_PKR} processing fee + PKR {fine} fine = at least PKR {minimum_total}."
        )
    return result
