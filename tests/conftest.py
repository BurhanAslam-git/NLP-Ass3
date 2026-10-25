"""Shared test setup: tests must never change the real data files."""

import shutil

import pytest

from backend.app.domain import rooms
from backend.app.domain.records import records_db


@pytest.fixture(autouse=True, scope="session")
def isolated_data_files(tmp_path_factory):
    """Point the records DB and room bookings at temporary copies for the whole test run."""
    tmp = tmp_path_factory.mktemp("data")
    original_records, original_rooms = records_db.filepath, rooms.BOOKINGS_PATH
    records_db.filepath = tmp / "library_records.json"
    shutil.copy(original_records, records_db.filepath)
    rooms.BOOKINGS_PATH = tmp / "room_bookings.json"
    yield
    records_db.filepath, rooms.BOOKINGS_PATH = original_records, original_rooms
