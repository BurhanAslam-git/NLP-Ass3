#!/usr/bin/env python3
"""
Two simultaneous users against the RUNNING app (start it with run.ps1 first):
  1. alone: one voice user, then one text user (baseline)
  2. voice + text at the same time
  3. voice + voice at the same time
Shows that neither user blocks the other (each session has its own lock; ASR,
TTS and embeddings run in worker threads), and how much latency grows under load.

Usage:  python scripts/concurrency_test.py
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_voice import text_turn, voice_turn  # noqa: E402

from backend.app.voice.tts import make_test_utterance  # noqa: E402

VOICE_Q = "Do you have Clean Code?"
VOICE_Q2 = "How many times can I renew a book?"
TEXT_Q = "Where is the Quiet Study Zone?"


def fmt(r: dict) -> str:
    keys = ["asr_ms", "llm_first_token_ms", "eos_to_first_audio_ms", "text_first_token_ms", "text_done_ms"]
    return ", ".join(f"{k}={r[k]:.0f}" for k in keys if r.get(k) is not None) + (f" ERROR {r['error']}" if r.get("error") else "")


async def timed(label: str, coro):
    t0 = time.perf_counter()
    result = await coro
    print(f"  {label:8s} finished at +{time.perf_counter() - t0:5.1f} s: {fmt(result)}")
    return result


async def main() -> None:
    voice_a, voice_b = make_test_utterance(VOICE_Q), make_test_utterance(VOICE_Q2)

    print("1) Baseline, one user at a time")
    await timed("voice", voice_turn(voice_a))
    await timed("text", text_turn(TEXT_Q))

    print("2) Voice + text at the same time")
    await asyncio.gather(timed("voice", voice_turn(voice_a)), timed("text", text_turn(TEXT_Q)))

    print("3) Voice + voice at the same time")
    await asyncio.gather(timed("voice A", voice_turn(voice_a)), timed("voice B", voice_turn(voice_b)))


if __name__ == "__main__":
    asyncio.run(main())
