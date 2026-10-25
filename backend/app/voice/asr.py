"""
Local speech-to-text with faster-whisper (CTranslate2 Whisper, int8 on CPU).

The model is loaded ONCE (at startup, or on first use) and shared by all
connections. Transcription runs in a worker thread via asyncio.to_thread so
the event loop keeps serving other users. num_workers=2 lets two threads
transcribe at the same time (two concurrent speakers).
"""

import asyncio
import logging
import re
import threading
import time
from pathlib import Path

import numpy as np

from backend.app.config import settings

logger = logging.getLogger(__name__)

# Domain words help Whisper spell library terms correctly.
_DOMAIN_PROMPT = (
    "NUCES library desk. Patron card P-78421. Is the book available? Renew, return, fine due on a date. "
    "Book study room 301, 302, 303, 304, 305 or 306."
)


class WhisperASR:
    def __init__(self) -> None:
        self._model = None
        self._load_lock = threading.Lock()

    def load(self) -> None:
        """Load + warm up the model (safe to call many times)."""
        with self._load_lock:
            if self._model is not None:
                return
            from faster_whisper import WhisperModel

            local_dir = Path(settings.models_dir) / "whisper" / settings.asr_model
            if not (local_dir / "model.bin").exists():
                raise RuntimeError(
                    f"ASR model not found in {local_dir}. Run: python scripts/download_models.py"
                )
            t0 = time.perf_counter()
            model = WhisperModel(
                str(local_dir),
                device="cpu",
                compute_type=settings.asr_compute_type,
                cpu_threads=settings.asr_cpu_threads,
                num_workers=2,
            )
            self._model = model
            self._transcribe_sync(np.zeros(16000, dtype=np.float32))  # warm-up: 1 s of silence
            logger.info("ASR ready: faster-whisper %s (%.1f s)", settings.asr_model, time.perf_counter() - t0)

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        segments, _info = self._model.transcribe(
            audio,
            language="en",
            beam_size=1,                       # greedy decoding: fastest on CPU
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,                  # we already did VAD ourselves
            initial_prompt=_DOMAIN_PROMPT,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        # Spoken card numbers come out as "P 78421", "P78421" or "P-78,421" -> "P-78421".
        return re.sub(r"\b[Pp][\s-]?(\d{2}),?(\d{3})\b", r"P-\1\2", text)

    async def transcribe(self, pcm16: bytes) -> str:
        """PCM16 16 kHz mono bytes -> text (runs in a worker thread)."""
        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        await asyncio.to_thread(self.load)
        return await asyncio.to_thread(self._transcribe_sync, audio)


asr = WhisperASR()
