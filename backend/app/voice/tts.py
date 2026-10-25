"""
Local text-to-speech with Piper (VITS voice, ONNX on CPU).

PiperTTS: loads the voice ONCE and turns one sentence into PCM16 audio.
TtsStream: used for one reply. LLM tokens go in with feed(); a background task
cuts them into sentences, synthesises each one in a worker thread AS SOON AS it
is complete, and sends audio_chunk frames in order (seq 0, 1, 2, ...). So the
first sentence is playing while the LLM is still writing the rest.
"""

import asyncio
import base64
import logging
import threading
import time
from pathlib import Path

from backend.app.config import settings
from backend.app.models.schemas import AudioChunkFrame, TtsDone
from backend.app.voice.sentences import SentenceSplitter, clean_for_speech

logger = logging.getLogger(__name__)


class PiperTTS:
    def __init__(self) -> None:
        self._voice = None
        self._load_lock = threading.Lock()

    def load(self) -> None:
        with self._load_lock:
            if self._voice is not None:
                return
            from piper import PiperVoice

            path = Path(settings.models_dir) / "piper" / f"{settings.tts_voice}.onnx"
            if not path.exists():
                raise RuntimeError(f"TTS voice not found: {path}. Run: python scripts/download_models.py")
            t0 = time.perf_counter()
            self._voice = PiperVoice.load(str(path))
            self._synthesize_sync("Ready.")  # warm-up
            logger.info("TTS ready: Piper %s (%.1f s)", settings.tts_voice, time.perf_counter() - t0)

    def _synthesize_sync(self, text: str) -> tuple[bytes, int]:
        """Return (PCM16 bytes, sample_rate) for one sentence."""
        pcm = bytearray()
        rate = self._voice.config.sample_rate
        for chunk in self._voice.synthesize(text):
            pcm += chunk.audio_int16_bytes
            rate = chunk.sample_rate
        return bytes(pcm), rate

    async def synthesize(self, text: str) -> tuple[bytes, int]:
        await asyncio.to_thread(self.load)
        return await asyncio.to_thread(self._synthesize_sync, text)


tts = PiperTTS()


def make_test_utterance(text: str, trailing_silence_s: float = 1.0) -> bytes:
    """
    Speak `text` with Piper and return 16 kHz mono PCM16 + trailing silence,
    i.e. exactly what the browser would stream. Used by tests and
    scripts/benchmark_voice.py to drive the real voice pipeline.
    """
    import numpy as np

    tts.load()
    pcm, rate = tts._synthesize_sync(text)
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if rate != 16000:  # simple linear resampling to 16 kHz
        positions = np.arange(0, len(audio), rate / 16000)
        audio = np.interp(positions, np.arange(len(audio)), audio)
    silence = np.zeros(int(16000 * trailing_silence_s))
    return np.concatenate([audio, silence]).astype(np.int16).tobytes()


class TtsStream:
    """Streams one reply as ordered audio chunks while tokens keep arriving."""

    def __init__(self, send, session_id: str) -> None:
        self._send = send                      # async function that sends a frame
        self._session_id = session_id
        self._splitter = SentenceSplitter(settings.tts_first_chunk_min_words)
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._seq = 0
        self.first_audio_at: float | None = None   # perf_counter() when chunk 0 was sent
        self.first_synth_ms: float | None = None   # time to synthesise the first chunk
        self._task = asyncio.create_task(self._worker())

    def feed(self, token: str) -> None:
        for sentence in self._splitter.feed(token):
            self._queue.put_nowait(sentence)

    async def finish(self) -> None:
        """No more tokens: speak the rest, wait for all audio, then send tts_done."""
        for sentence in self._splitter.flush():
            self._queue.put_nowait(sentence)
        self._queue.put_nowait(None)
        await self._task
        await self._send(TtsDone(session_id=self._session_id))

    def cancel(self) -> None:
        self._task.cancel()

    async def _worker(self) -> None:
        # One worker per reply => sentences are synthesised and sent strictly in order.
        while (sentence := await self._queue.get()) is not None:
            spoken = clean_for_speech(sentence)
            if not any(ch.isalnum() for ch in spoken):
                continue  # nothing speakable (e.g. only an ID or a symbol)
            try:
                t0 = time.perf_counter()
                pcm, rate = await tts.synthesize(spoken)
                synth_ms = (time.perf_counter() - t0) * 1000
            except Exception as exc:
                # TTS failure: the text reply still appears; we just skip the audio.
                logger.warning("TTS failed for %r: %s", spoken[:60], exc)
                continue
            await self._send(AudioChunkFrame(
                session_id=self._session_id,
                seq=self._seq,
                sample_rate=rate,
                data=base64.b64encode(pcm).decode("ascii"),
            ))
            if self._seq == 0:
                self.first_audio_at = time.perf_counter()
                self.first_synth_ms = synth_ms
            self._seq += 1
