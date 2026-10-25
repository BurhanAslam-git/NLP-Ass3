"""Optional live Ollama checks (skipped when Ollama is down)."""

import json
import uuid

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


def _ollama_up() -> bool:
    try:
        r = httpx.get("http://127.0.0.1:11434/api/tags", timeout=2.0)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


pytestmark = pytest.mark.skipif(not _ollama_up(), reason="Ollama not running")


def _chat(user_text: str) -> str:
    client = TestClient(app)
    session_id = str(uuid.uuid4())
    parts: list[str] = []
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_text(
            json.dumps({"type": "user_message", "session_id": session_id, "content": user_text})
        )
        while True:
            data = json.loads(ws.receive_text())
            if data["type"] == "token":
                parts.append(data.get("content", ""))
            elif data["type"] == "done":
                break
            elif data["type"] == "error":
                raise AssertionError(data.get("message", "error"))
    return "".join(parts)


def test_c1_speech_book_catalogue_row() -> None:
    reply = _chat("Do you have Speech and Language Processing?").lower()
    assert "speech and language processing" in reply
    # The LLM may phrase availability either way ("available" / "in stock").
    assert "available" in reply or "in stock" in reply
    assert "don't have" not in reply
    assert "do not have" not in reply
    assert "not in the catalogue" not in reply


def test_d1_joke_refusal() -> None:
    reply = _chat("tell me a joke plz").lower()
    assert "joke" not in reply[:60] or "library" in reply or "cannot" in reply or "can't" in reply
    assert "catalogue" in reply or "library" in reply or "assist" in reply


def test_c7_casual_hi_no_nuces_name() -> None:
    reply = _chat("hi there").lower()
    assert "i'm nuces" not in reply
    assert "i am nuces" not in reply


def test_e2_my_name_not_assistant_identity() -> None:
    reply = _chat("tell me my name plz").lower()
    assert "my name is alex" not in reply
    assert "i'm alex" not in reply
    assert "i am alex" not in reply
