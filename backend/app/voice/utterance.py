"""
Buffer one spoken utterance and detect the end of speech (endpointing).

The browser sends 16 kHz mono PCM16 audio in small binary frames. We cut it
into 512-sample (32 ms) frames, ask the VAD "is this frame speech?", and:

  - before speech starts: keep only a short pre-roll (so we don't feed
    seconds of silence to Whisper);
  - after speech starts: count consecutive silent milliseconds;
  - end of speech = silence >= VAD_SILENCE_MS (default 600 ms),
    or the utterance reached ASR_MAX_UTTERANCE_S.

`vad` is any object with `is_speech(frame_bytes) -> bool` (see vad.py).
"""

import time

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512                  # 32 ms, the frame size Silero VAD expects
FRAME_BYTES = FRAME_SAMPLES * 2      # PCM16 = 2 bytes per sample
FRAME_MS = 1000 * FRAME_SAMPLES / SAMPLE_RATE
PRE_ROLL_BYTES = 10 * FRAME_BYTES    # keep ~320 ms before speech starts
NO_SPEECH_TIMEOUT_MS = 8000          # give up if nobody speaks for 8 s


class Utterance:
    def __init__(self, vad, silence_ms: int, max_seconds: float) -> None:
        self.vad = vad
        self.silence_limit_ms = silence_ms
        self.max_bytes = int(max_seconds * SAMPLE_RATE) * 2
        self.audio = bytearray()       # PCM16 kept for Whisper
        self._leftover = b""           # bytes that don't fill a whole frame yet
        self.speech_started = False
        self.silence_ms = 0.0
        self.total_ms = 0.0
        self.ended = False
        self.end_time: float | None = None   # perf_counter() at end of speech
        self.partial_at_bytes = 0      # buffer size when the last partial ran

    def add(self, data: bytes) -> bool:
        """Add PCM16 bytes. Returns True once end of speech is detected."""
        if self.ended:
            return True
        data = self._leftover + data
        usable = len(data) - len(data) % FRAME_BYTES
        self._leftover = data[usable:]

        for offset in range(0, usable, FRAME_BYTES):
            frame = data[offset:offset + FRAME_BYTES]
            self.total_ms += FRAME_MS
            self.audio += frame
            if self.vad.is_speech(frame):
                self.speech_started = True
                self.silence_ms = 0.0
            elif self.speech_started:
                self.silence_ms += FRAME_MS
            else:
                del self.audio[:-PRE_ROLL_BYTES]  # still waiting for speech

            if (
                (self.speech_started and self.silence_ms >= self.silence_limit_ms)
                or len(self.audio) >= self.max_bytes
                or (not self.speech_started and self.total_ms >= NO_SPEECH_TIMEOUT_MS)
            ):
                self.finish()
                return True
        return False

    def finish(self) -> None:
        """Mark the end of speech (VAD silence, time limit, or user pressed Stop)."""
        if not self.ended:
            self.ended = True
            self.end_time = time.perf_counter()

    def needs_partial(self, interval_ms: int) -> bool:
        """True when enough new speech arrived since the last partial transcript."""
        new_bytes = len(self.audio) - self.partial_at_bytes
        return self.speech_started and not self.ended and new_bytes >= interval_ms * SAMPLE_RATE * 2 // 1000
