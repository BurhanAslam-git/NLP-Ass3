# Benchmarks (raw results)

Measured on 2026-10-25 on the development laptop: Intel Core i7-8550U (4 cores / 8 threads, 1.8 GHz),
7.9 GB RAM, Intel UHD 620 (no CUDA, everything on CPU), Windows 11, on mains power, "Balanced" power plan.
Models: qwen2.5:1.5b-instruct (Ollama), faster-whisper tiny.en int8, Silero VAD, Piper en_US-lessac-low,
all-MiniLM-L6-v2. The app was running (`run.ps1`) and warmed up before each benchmark.

## Voice pipeline — `python scripts/benchmark_voice.py -n 2`

Each question was synthesised with Piper and streamed to `/ws/chat` in real time (32 ms frames), new session each run.
`eos_to_first_audio_ms` starts when the VAD declares end of speech (600 ms after the last word).

```
Synthesising test questions with Piper ...
catalogue     run 1: transcript='Do you have speech and language processing?' tools=['check_patron_loans'] eos->audio=14960.104500001762 
catalogue     run 2: transcript='Do you have speech and language processing?' tools=['search_catalogue'] eos->audio=10475.999099959154 
fine          run 1: transcript='My book was due on October 18. How much is the fine if I return it today?' tools=[] eos->audio=11514.349400007632 
fine          run 2: transcript='My book was due on October 18. How much is the fine if I return it today?' tools=[] eos->audio=3617.5837999908254 
study room    run 1: transcript='Is study room 304 free tomorrow afternoon?' tools=[] eos->audio=8964.229900040664 
study room    run 2: transcript='Is study room 304 free tomorrow afternoon?' tools=['book_study_room'] eos->audio=15273.737300012726 
policy (RAG)  run 1: transcript='How many times can I renew a book?' tools=[] eos->audio=9023.78879999742 
policy (RAG)  run 2: transcript='How many times can I renew a book?' tools=[] eos->audio=4395.931100007147 

## All questions

| Metric | n | mean (ms) | p50 (ms) | p95 (ms) |
|---|---|---|---|---|
| asr_ms | 8 | 779 | 718 | 1029 |
| retrieval_ms | 8 | 17 | 16 | 42 |
| tool_ms | 3 | 1 | 1 | 2 |
| llm_first_token_ms | 8 | 6741 | 6553 | 12268 |
| tts_first_audio_ms | 8 | 501 | 485 | 863 |
| eos_to_first_audio_ms | 8 | 9778 | 9750 | 15274 |

## catalogue

| Metric | n | mean (ms) | p50 (ms) | p95 (ms) |
|---|---|---|---|---|
| asr_ms | 2 | 676 | 676 | 691 |
| retrieval_ms | 2 | 17 | 17 | 33 |
| tool_ms | 2 | 1 | 1 | 1 |
| llm_first_token_ms | 2 | 9150 | 9150 | 12268 |
| tts_first_audio_ms | 2 | 704 | 704 | 863 |
| eos_to_first_audio_ms | 2 | 12718 | 12718 | 14960 |

## fine

| Metric | n | mean (ms) | p50 (ms) | p95 (ms) |
|---|---|---|---|---|
| asr_ms | 2 | 982 | 982 | 1029 |
| retrieval_ms | 2 | 21 | 21 | 42 |
| tool_ms | 0 | � | � | � |
| llm_first_token_ms | 2 | 4952 | 4952 | 9231 |
| tts_first_audio_ms | 2 | 328 | 328 | 427 |
| eos_to_first_audio_ms | 2 | 7566 | 7566 | 11514 |

## study room

| Metric | n | mean (ms) | p50 (ms) | p95 (ms) |
|---|---|---|---|---|
| asr_ms | 2 | 709 | 709 | 746 |
| retrieval_ms | 2 | 16 | 16 | 31 |
| tool_ms | 1 | 2 | 2 | 2 |
| llm_first_token_ms | 2 | 9162 | 9162 | 11655 |
| tts_first_audio_ms | 2 | 452 | 452 | 553 |
| eos_to_first_audio_ms | 2 | 12119 | 12119 | 15274 |

## policy (RAG)

| Metric | n | mean (ms) | p50 (ms) | p95 (ms) |
|---|---|---|---|---|
| asr_ms | 2 | 749 | 749 | 822 |
| retrieval_ms | 2 | 16 | 16 | 31 |
| tool_ms | 0 | � | � | � |
| llm_first_token_ms | 2 | 3698 | 3698 | 6437 |
| tts_first_audio_ms | 2 | 520 | 520 | 627 |
| eos_to_first_audio_ms | 2 | 6710 | 6710 | 9024 |
```

## Two concurrent users — `python scripts/concurrency_test.py`

```
1) Baseline, one user at a time
  voice    finished at + 22.3 s: asr_ms=540, llm_first_token_ms=10768, eos_to_first_audio_ms=13525
  text     finished at + 27.5 s: text_first_token_ms=6193, text_done_ms=27497
2) Voice + text at the same time
  text     finished at + 10.2 s: text_first_token_ms=619, text_done_ms=10173
  voice    finished at + 34.0 s: asr_ms=1010, llm_first_token_ms=17648, eos_to_first_audio_ms=20208
3) Voice + voice at the same time
  voice A  finished at + 37.8 s: asr_ms=786, llm_first_token_ms=27061, eos_to_first_audio_ms=30822
  voice B  finished at + 60.3 s: asr_ms=1834, llm_first_token_ms=7643, eos_to_first_audio_ms=12590
```

## LLM comparison (same compact system prompt, warm model, direct Ollama calls)

| | llama3.2:3b | qwen2.5:1.5b-instruct |
|---|---|---|
| First token, no tools, prompt cached | 1.2–1.5 s | 0.8 s |
| Generation speed | 5–6 tok/s | 11–14 tok/s |
| Tool call ready (prompt cached) | 5.2 s | 2.9 s |
| Reading ~1,070 prompt tokens (not cached) | 29 s | 12 s |
| Correct tool decision (2 test questions) | 0/2 (called a tool for a policy question and the wrong tool for a catalogue question) | 2/2 |

With the original Assignment 2 system prompt (1,056 tokens) llama3.2:3b needed 44 s to read the prompt when not cached.

## Latency fixes applied (before → after)

| Change | Effect |
|---|---|
| Shorter system prompt (1,056 → 341 tokens) and shorter tool descriptions | fewer tokens to read every turn |
| llama3.2:3b → qwen2.5:1.5b-instruct | ~2.5× faster generation, correct tool choice |
| Warm-up at startup includes the tool schemas | fixed prompt prefix already cached for the first turn |
| Per-turn notes (date/time) moved from an extra system message into the user message | stopped the prompt cache being invalidated every minute: end-of-speech → first audio mean **21.1 s → 9.8 s** |
| RAG context per turn 2,000 → 800 characters | fewer new tokens per turn |
| `num_predict` 300 + `repeat_penalty` | no more runaway generations (one took > 6 minutes) |
