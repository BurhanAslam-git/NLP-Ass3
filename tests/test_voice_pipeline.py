"""
Phase III: voice is NOT a separate system.

A typed turn and then a spoken turn (Piper-generated speech streamed as binary
frames, exactly like the browser) go to the same session_id over /ws/chat.
The spoken turn must see the typed turn's history.
Needs the local models (scripts/download_models.py) and, for the full turn, Ollama.
"""

import json
import uuid
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.app.config import settings
from backend.app.main import app, manager

MODELS = Path(settings.models_dir)
models_ready = (
    (MODELS / "whisper" / settings.asr_model / "model.bin").exists()
    and (MODELS / "piper" / f"{settings.tts_voice}.onnx").exists()
    and (MODELS / "vad" / "silero_vad.onnx").exists()
)


def _ollama_up() -> bool:
    try:
        return httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=2.0).status_code == 200
    except httpx.HTTPError:
        return False


pytestmark = pytest.mark.skipif(not models_ready, reason="voice models not downloaded")


def _collect_turn(ws) -> dict:
    """Read frames until the metrics frame that ends a turn."""
    frames: dict[str, list] = {}
    while True:
        frame = json.loads(ws.receive_text())
        frames.setdefault(frame["type"], []).append(frame)
        if frame["type"] == "error":
            raise AssertionError(frame["message"])
        if frame["type"] == "metrics":
            return frames


def _speak(ws, session_id: str, text: str, speak: bool = False) -> None:
    from backend.app.voice.tts import make_test_utterance

    pcm = make_test_utterance(text)
    ws.send_text(json.dumps({"type": "audio_start", "session_id": session_id,
                             "sample_rate": 16000, "format": "pcm16", "speak": speak}))
    for i in range(0, len(pcm), 1024):        # 32 ms binary frames, like the AudioWorklet
        ws.send_bytes(pcm[i:i + 1024])


def test_asr_transcribes_spoken_question():
    client = TestClient(app)
    with client.websocket_connect("/ws/chat") as ws:
        _speak(ws, "asr-only", "Do you have Clean Code?")
        while True:
            frame = json.loads(ws.receive_text())
            if frame["type"] == "transcript_final":
                assert "clean code" in frame["text"].lower()
                break
            assert frame["type"] in ("transcript_partial",), frame


@pytest.mark.skipif(not _ollama_up(), reason="Ollama not running")
def test_typed_then_voice_turn_share_history():
    session_id = f"voice-{uuid.uuid4()}"
    client = TestClient(app)
    with client.websocket_connect("/ws/chat") as ws:
        # Turn 1: typed
        ws.send_text(json.dumps({"type": "user_message", "session_id": session_id,
                                 "content": "Hello, my name is Ali and my card is P-78421."}))
        _collect_turn(ws)

        # Turn 2: spoken, same session, with spoken reply
        _speak(ws, session_id, "What is my name?", speak=True)
        frames = _collect_turn(ws)

    assert "name" in frames["transcript_final"][0]["text"].lower()
    reply = "".join(f["content"] for f in frames["token"]).lower()
    assert "ali" in reply                                   # memory from the typed turn
    assert frames["audio_chunk"][0]["seq"] == 0 and frames["tts_done"]
    metrics = frames["metrics"][0]
    assert metrics["asr_ms"] is not None and metrics["eos_to_first_audio_ms"] is not None

    # Both turns are in ONE server-side history.
    history = [m.content for m in manager._sessions[session_id].messages if m.role == "user"]
    assert history[0].startswith("Hello, my name is Ali") and "name" in history[1].lower()
