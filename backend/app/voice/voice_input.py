"""
Per-connection voice input: audio_start -> binary frames -> end of speech.

Flow for one spoken turn:
  1. start()      on {"type":"audio_start"}: new Utterance with a fresh VAD state.
  2. add_audio()  for every binary frame: VAD runs in a worker thread. About
                  every second of new speech we transcribe what we have so far
                  in the background and send transcript_partial.
  3. End of speech (600 ms silence, or audio_stop from the client):
                  transcribe the whole utterance ONCE, send transcript_final,
                  then call on_final(text, ...) which runs the normal chat turn
                  (same ConversationManager as typed text).
"""

import asyncio
import logging
import time

from backend.app.config import settings
from backend.app.models.schemas import ChatError, TranscriptFrame
from backend.app.voice.asr import asr
from backend.app.voice.utterance import Utterance
from backend.app.voice.vad import SileroVAD

logger = logging.getLogger(__name__)


class VoiceInput:
    def __init__(self, send, on_final) -> None:
        self._send = send            # async: send a frame to this client
        self._on_final = on_final    # async: on_final(session_id, text, speak, eos_time, asr_ms)
        self._utt: Utterance | None = None
        self._session_id = ""
        self._speak = True
        self._partial_task: asyncio.Task | None = None

    def start(self, session_id: str, speak: bool) -> None:
        self._session_id = session_id
        self._speak = speak
        self._utt = Utterance(SileroVAD(), settings.vad_silence_ms, settings.asr_max_utterance_s)

    async def add_audio(self, data: bytes) -> None:
        utt = self._utt
        if utt is None or utt.ended:
            return  # stray frames after end of speech are ignored
        ended = await asyncio.to_thread(utt.add, data)
        if ended:
            await self._finalise()
        elif utt.needs_partial(settings.asr_partial_interval_ms) and not self._partial_busy():
            utt.partial_at_bytes = len(utt.audio)
            self._partial_task = asyncio.create_task(self._partial(utt, bytes(utt.audio)))

    async def stop(self) -> None:
        """Client pressed Stop: treat it as end of speech."""
        if self._utt is not None and not self._utt.ended:
            self._utt.finish()
            await self._finalise()

    def _partial_busy(self) -> bool:
        # Only one partial at a time per user, to keep CPU cost reasonable.
        return self._partial_task is not None and not self._partial_task.done()

    async def _partial(self, utt: Utterance, audio: bytes) -> None:
        try:
            text = await asr.transcribe(audio)
            if text and not utt.ended:
                await self._send(TranscriptFrame(
                    type="transcript_partial", session_id=self._session_id, text=text))
        except Exception as exc:
            logger.warning("Partial transcript failed: %s", exc)

    async def _finalise(self) -> None:
        utt, self._utt = self._utt, None
        if utt is None:
            return
        if not utt.speech_started:
            await self._error("I didn't hear anything. Press Speak and try again.")
            return
        try:
            text = await asr.transcribe(bytes(utt.audio))
        except Exception as exc:
            logger.exception("ASR failed")
            await self._error(f"Speech recognition failed: {exc}")
            return
        asr_ms = (time.perf_counter() - utt.end_time) * 1000
        if not text:
            await self._error("Sorry, I couldn't make that out. Please try again or type your question.")
            return
        logger.info("ASR final (%.0f ms): %s", asr_ms, text)
        await self._send(TranscriptFrame(type="transcript_final", session_id=self._session_id, text=text))
        await self._on_final(self._session_id, text, self._speak, utt.end_time, asr_ms)

    async def _error(self, message: str) -> None:
        await self._send(ChatError(session_id=self._session_id or "unknown", message=message))
