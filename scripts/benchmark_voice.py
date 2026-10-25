#!/usr/bin/env python3
"""
Voice latency benchmark against the RUNNING app (start it with run.ps1 first).

For each test question we synthesise speech locally with Piper, stream it to
/ws/chat in real time (32 ms binary frames, like the browser's AudioWorklet),
and read back the server's per-turn `metrics` frame. Prints a markdown table of
mean / p50 / p95 per metric.

Usage:
    python scripts/benchmark_voice.py            # 3 runs per question
    python scripts/benchmark_voice.py -n 5
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import websockets  # noqa: E402  (installed with uvicorn[standard])

from backend.app.voice.tts import make_test_utterance  # noqa: E402

URL = "ws://127.0.0.1:8000/ws/chat"
QUESTIONS = {
    "catalogue": "Do you have Speech and Language Processing?",
    "fine": "My book was due on October eighteenth. How much is the fine if I return it today?",
    "study room": "Is study room 304 free tomorrow afternoon?",
    "policy (RAG)": "How many times can I renew a book?",
}
METRICS = ["asr_ms", "retrieval_ms", "tool_ms", "llm_first_token_ms", "tts_first_audio_ms", "eos_to_first_audio_ms"]


async def voice_turn(pcm: bytes, url: str = URL, session_id: str | None = None) -> dict:
    """Stream one spoken question in real time; return metrics + transcript + tools used."""
    session_id = session_id or f"bench-{uuid.uuid4()}"
    result: dict = {"tools": []}
    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "audio_start", "session_id": session_id,
                                  "sample_rate": 16000, "format": "pcm16", "speak": True}))
        for i in range(0, len(pcm), 1024):
            await ws.send(pcm[i:i + 1024])
            await asyncio.sleep(0.032)            # real time: 512 samples = 32 ms
        async for raw in ws:
            frame = json.loads(raw)
            kind = frame["type"]
            if kind == "transcript_final":
                result["transcript"] = frame["text"]
            elif kind == "tool_call":
                result["tools"].append(frame["name"])
            elif kind == "error":
                result["error"] = frame["message"]
                return result
            elif kind == "metrics":
                result.update({k: frame.get(k) for k in METRICS})
                return result
    return result


async def text_turn(text: str, url: str = URL) -> dict:
    """One typed question; returns time to first token and to done (ms)."""
    async with websockets.connect(url) as ws:
        t0 = time.perf_counter()
        await ws.send(json.dumps({"type": "user_message", "session_id": f"bench-{uuid.uuid4()}", "content": text}))
        first = None
        async for raw in ws:
            frame = json.loads(raw)
            if frame["type"] == "token" and first is None:
                first = (time.perf_counter() - t0) * 1000
            elif frame["type"] in ("done", "error"):
                done = (time.perf_counter() - t0) * 1000
                return {"text_first_token_ms": first, "text_done_ms": done, "error": frame.get("message")}
    return {}


def summary_table(rows: list[dict], metrics: list[str]) -> str:
    lines = ["| Metric | n | mean (ms) | p50 (ms) | p95 (ms) |", "|---|---|---|---|---|"]
    for m in metrics:
        values = sorted(r[m] for r in rows if r.get(m) is not None)
        if not values:
            lines.append(f"| {m} | 0 | — | — | — |")
            continue
        p95 = values[min(len(values) - 1, round(0.95 * (len(values) - 1)))]
        lines.append(f"| {m} | {len(values)} | {statistics.mean(values):.0f} | "
                     f"{statistics.median(values):.0f} | {p95:.0f} |")
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=3, help="runs per question")
    parser.add_argument("--url", default=URL)
    args = parser.parse_args()

    print("Synthesising test questions with Piper ...")
    audio = {name: make_test_utterance(q) for name, q in QUESTIONS.items()}

    rows = []
    for name, pcm in audio.items():
        for run in range(args.n):
            r = await voice_turn(pcm, args.url)
            r["question"] = name
            rows.append(r)
            print(f"{name:13s} run {run + 1}: transcript={r.get('transcript')!r} tools={r['tools']} "
                  f"eos->audio={r.get('eos_to_first_audio_ms')} {r.get('error', '')}")

    print("\n## All questions\n")
    print(summary_table(rows, METRICS))
    for name in QUESTIONS:
        print(f"\n## {name}\n")
        print(summary_table([r for r in rows if r["question"] == name], METRICS))


if __name__ == "__main__":
    asyncio.run(main())
