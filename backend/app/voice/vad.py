"""
Silero VAD (voice activity detection) run directly with onnxruntime.

For every 512-sample (32 ms) frame at 16 kHz the model returns the
probability that the frame contains speech. Silero is a small recurrent
network, so it carries a hidden `state` (and 64 samples of context) from one
frame to the next. The ONNX session is shared by everyone; each utterance gets
its own SileroVAD object, i.e. its own state.
"""

import threading
from pathlib import Path

import numpy as np

from backend.app.config import settings

_CONTEXT_SAMPLES = 64  # Silero prepends the last 64 samples of the previous frame
_session = None
_session_lock = threading.Lock()


def _get_session():
    """Load the ONNX model once (thread-safe)."""
    global _session
    with _session_lock:
        if _session is None:
            import onnxruntime as ort

            path = Path(settings.models_dir) / "vad" / "silero_vad.onnx"
            if not path.exists():
                raise RuntimeError(f"VAD model not found: {path}. Run: python scripts/download_models.py")
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1  # tiny model: one thread is plenty
            opts.inter_op_num_threads = 1
            _session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    return _session


class SileroVAD:
    def __init__(self, threshold: float | None = None) -> None:
        self.threshold = settings.vad_threshold if threshold is None else threshold
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT_SAMPLES), dtype=np.float32)
        self._in_speech = False

    def probability(self, frame: bytes) -> float:
        """Speech probability (0..1) for one 512-sample PCM16 frame."""
        x = np.frombuffer(frame, dtype=np.int16).astype(np.float32)[None, :] / 32768.0
        x = np.concatenate([self._context, x], axis=1)
        out, self._state = _get_session().run(
            None, {"input": x, "state": self._state, "sr": np.array(16000, dtype=np.int64)}
        )
        self._context = x[:, -_CONTEXT_SAMPLES:]
        return float(out[0][0])

    def is_speech(self, frame: bytes) -> bool:
        # Hysteresis (as recommended by Silero): start speech at `threshold`,
        # but only end it when the probability drops 0.15 below that.
        p = self.probability(frame)
        if self._in_speech:
            self._in_speech = p >= self.threshold - 0.15
        else:
            self._in_speech = p >= self.threshold
        return self._in_speech
