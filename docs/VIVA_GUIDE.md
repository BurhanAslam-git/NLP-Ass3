# Viva guide — NUCES Virtual Desk (Assignment 3)

Short explanations of how each part works, where the code is, and likely questions.
Everything runs locally on CPU: Ollama (LLM), faster-whisper (ASR), Silero (VAD), Piper (TTS), MiniLM + ChromaDB (RAG).

---

## 1. One turn, end to end

```
Browser mic ─AudioWorklet─▶ 16 kHz PCM16, 32 ms binary frames ─▶ /ws/chat
   VoiceInput: Silero VAD per frame → 600 ms silence = end of speech
   faster-whisper → transcript_final  ─┐
Typed text (user_message) ────────────┴▶ _run_turn() → ConversationManager.stream_reply()
   RAG retrieve (thread, 5 s timeout) → build prompt → chat_with_tools()
   Ollama streams tokens  ─▶ "token" frames (text bubble)
                          └▶ TtsStream: sentence splitter → Piper (thread) → audio_chunk seq 0,1,2…
   done → tts_done → metrics
```

Key point: **voice is not a separate system.** A spoken turn becomes text at `transcript_final` and then calls the exact same `_run_turn()` → `manager.stream_reply(session_id, text)` as a typed message. Same session history, same RAG, same tools, same LLM streaming. The only extra is that tokens are also fed to TTS.

Files: `backend/app/main.py` (`ws_chat`, `_run_turn`), `backend/app/voice/*`, `backend/app/conversation/tool_orchestrator.py`, `frontend/voice.js`, `frontend/audio-worklet.js`.

---

## 2. Streaming ASR + VAD (Phase I)

**Browser** (`frontend/audio-worklet.js`): an AudioWorklet runs on the audio thread. Mic audio arrives at the device rate (usually 48 kHz). We average every `48000/16000 = 3` samples into one (a simple low-pass + decimate), convert float [-1, 1] to Int16, and post 512-sample frames (32 ms). `voice.js` sends each frame as a **binary WebSocket frame** on the same `/ws/chat` socket, after an `audio_start` JSON message.

**Server** (`voice/utterance.py`, `voice/vad.py`, `voice/voice_input.py`):
- Frames are cut into exact 512-sample frames (leftover bytes carried over).
- **Silero VAD** (a small recurrent network, ONNX) gives a speech probability per frame. It keeps a hidden state between frames, so each utterance gets its own `SileroVAD` object. We use hysteresis: speech starts at p ≥ 0.5 and ends only when p < 0.35.
- Before speech starts, only ~320 ms of pre-roll is kept (Whisper doesn't waste time on silence).
- **End of speech** = 600 ms of continuous silence after speech (`VAD_SILENCE_MS`), or 20 s max, or the user presses Stop (`audio_stop`).
- **Partial transcripts**: every ~1 s of new speech we transcribe the audio so far in the background and send `transcript_partial`. Only one partial runs at a time per user, to bound CPU cost.
- **Final transcript**: ONE full transcription after end of speech → `transcript_final` → normal chat turn.

**Whisper**: faster-whisper (CTranslate2) `tiny.en`, int8, greedy decoding (`beam_size=1`), loaded once at startup and warmed up on 1 s of silence. `num_workers=2` so two users can transcribe in parallel. A short domain prompt ("NUCES library, OPAC, patron card P-78421…") helps it spell library words. Spoken card numbers ("P 78421") are normalised to `P-78421`.

**Why our own VAD instead of Whisper's `vad_filter`?** We need a decision *while audio is streaming* (frame by frame) to know when to stop listening. Whisper's filter only works on a finished recording.

---

## 3. Sentence-level TTS streaming (Phase II)

`voice/sentences.py` — `SentenceSplitter.feed(token)` buffers tokens and returns a chunk as soon as a sentence ends:
- `.` `!` `?` followed by whitespace, or a newline.
- Not inside numbers (`3.5`: next char isn't a space) or after abbreviations (`Dr.`, `e.g.`, `etc.`).
- The **first** chunk may also end at a comma once it has ≥ 6 words, so the first audio starts earlier.
- `clean_for_speech()` strips markdown, `[1]` citations and IDs like `BK-1004` before speaking. The text bubble keeps them.

`voice/tts.py` — `TtsStream` (one per reply): a single background task takes sentences from a queue, synthesises each with Piper **in a worker thread**, and sends `audio_chunk` frames with `seq = 0, 1, 2…`. One worker means strict order. Because it runs concurrently with the LLM stream, sentence 1 is being spoken while the LLM is still writing sentence 3.

**Browser playback** (`voice.js`): chunks are decoded (base64 → Int16 → Float32 AudioBuffer) and scheduled back to back on the Web Audio clock (`start(nextStartTime)`), so there are no gaps or overlaps. Chunks that arrive early wait in `pending` until their `seq` is next. "Stop audio" stops all scheduled sources and ignores the rest of that reply.

---

## 4. Tool calling (Phase IV)

**How the LLM picks a tool** (Ollama native tool calling):
1. `domain/tools.py` is the single registry: each tool has a name, description, JSON schema and an async handler.
2. `/api/chat` is called with `"tools": tool_specs()`. Ollama's chat template for llama3.2 puts those definitions into the prompt.
3. The model either writes text (streamed straight to the user) or outputs a tool call. Ollama parses it and returns `message.tool_calls = [{"function": {"name": "calculate_fine", "arguments": {...}}}]`.
4. `tool_orchestrator.run_tool()`: validate arguments against the schema (`validate_args`: required fields, types, enums; numbers sent as strings are converted) → run the handler with `asyncio.wait_for(timeout=3 s)`.
5. The result (or error) is appended as `{"role": "tool", "tool_name": ..., "content": json}` and the LLM is called again to explain it in words.
6. At most 2 tool rounds per turn. The last round is sent **without** tools, which forces a text answer, so the loop always ends.

There is no keyword regex any more (Assignment 2 used one): the LLM decides. Streaming with tools means a turn that needs no tool costs **one** LLM call, not two.

**Failures** never crash or hang the turn. Unknown tool, bad arguments, timeout or exception all become `{"success": false, "error": "..."}`. The LLM then explains it (and it's spoken in voice mode). The UI shows "Using search_catalogue…" from the `tool_call` event; `tool_result` has `ok` and `latency_ms`.

**Our three tools** (useful for a library):
- `search_catalogue` — live catalogue (status changes when books are issued/returned; RAG documents don't).
- `calculate_fine` — POL-002 exactly: PKR 20/day from the day after the due date, cap PKR 200/item, lost after 60 days, suspension above PKR 100. Uses `date.fromisoformat` (no eval).
- `book_study_room` — rooms 301–306 (FAQ-005), hours from POL-009, ≤ 3 h/day, ≤ 3 days ahead, no past times, no double-booking. Writes `data/room_bookings.json` atomically under an `asyncio.Lock`.

---

## 5. Timeouts and errors

| Failure | What happens |
|---|---|
| Tool too slow | `asyncio.wait_for` cancels it after `TOOL_TIMEOUT_SECONDS` (3 s); the LLM says the service is busy |
| Bad tool args / unknown tool | `validate_args` error returned to the LLM, which asks for the missing info |
| Retrieval slow/fails | 5 s timeout → answer without RAG (Assignment 2 behaviour) |
| Nothing heard / ASR fails | `error` frame: "I didn't hear anything…" (mic stops, chat continues) |
| TTS fails for a sentence | that sentence is skipped; the text still appears |
| Mic permission denied | browser shows a clear message how to allow the mic |
| Ollama down | `error` frame "Model error…", socket stays open |
| ASR/TTS model missing | logged at startup; text chat still works |

---

## 6. Concurrency

- One asyncio event loop serves everyone. **All CPU-heavy work runs in worker threads** (`asyncio.to_thread`): VAD, Whisper, Piper, MiniLM embeddings, Chroma queries. The loop only moves bytes.
- Per-session `asyncio.Lock` (Assignment 2): turns in one chat run in order; different chats run in parallel.
- One WebSocket can have several senders at once (tokens, TTS audio, partial transcripts), so `send()` uses a per-connection lock to keep frames whole.
- Shared models are loaded once: Whisper (`num_workers=2`), Piper and ONNX sessions are thread-safe for inference. The VAD *state* is per utterance.
- Limit: Ollama itself queues requests to the same model, so with 2 users the second LLM call may wait for the first. See README benchmarks.

---

## 7. Latency: what we measured and why

On our i7-8550U (no GPU): ASR ~0.8 s, retrieval ~17 ms, tools ~1 ms, TTS first sentence ~0.5 s, but LLM first token ~6.7 s (mean), so end of speech → first audio ≈ 9.8 s mean (best 3.6 s). The target of ~2 s is not met; the LLM is the bottleneck (see README §9, docs/BENCHMARKS.md).

**Prompt caching (good viva topic):** Ollama keeps the computed state (KV cache) of the last prompt. If a new request starts with the same tokens, it skips re-reading them. Our fixed start is system prompt + tool schemas (~1,100 tokens, ~12–30 s to read on this CPU!). We (1) warm up at startup *with the tools* so this start is cached, and (2) put everything that changes each turn (date/time, retrieved text) at the *end* of the conversation, in the user message. Before fix (2) the time note sat in a system message, which the chat template merges into the top block, so every new minute invalidated the cache: mean latency went 21.1 s → 9.8 s after the fix.

**Why qwen2.5:1.5b and not llama3.2:3b?** Measured: ~2.5× faster generation and it chose tools correctly (2/2 vs 0/2). Smaller models still fill arguments wrongly sometimes; schema validation + clear error messages keep that safe.

---

## 8. Likely viva questions

1. **Why does voice not need its own chat logic?** After `transcript_final` the text calls the same `_run_turn()` → `manager.stream_reply()` with the same `session_id` as typed messages.
2. **Why send audio as binary frames, not base64 JSON?** It's 33% smaller and has no encode/decode cost at ~31 frames/s. (Server→client audio is base64 JSON so it carries `seq` and `sample_rate` in one frame.)
3. **Why 16 kHz mono PCM16?** It is what Whisper and Silero expect; higher rates only add bytes.
4. **How is end of speech detected?** Silero VAD per 32 ms frame; 600 ms of continuous non-speech after speech has started.
5. **What if the user pauses mid-sentence?** A pause under 600 ms is ignored; longer pauses end the turn. `VAD_SILENCE_MS` trades responsiveness against cutting people off.
6. **How do partial transcripts work without wasting CPU?** Only after ~1 s of new speech, only one at a time per user, in a background task.
7. **How does TTS start before the LLM finishes?** Tokens go through a sentence splitter; each finished sentence is synthesised immediately in a thread and sent with a sequence number.
8. **How is playback kept in order and gap-free?** The client plays only the next expected `seq` and schedules each buffer at the end time of the previous one on the AudioContext clock.
9. **How does the LLM decide to call a tool?** Ollama's `tools` field. The model sees each tool's name, description and JSON schema and returns structured `tool_calls`; we don't parse text ourselves.
10. **What stops an infinite tool loop?** Max 2 tool rounds; the final call has no tools, so it must answer in text.
11. **What happens on a tool timeout?** `asyncio.wait_for` raises `TimeoutError`, which becomes `{"success": false, "error": ...}`. The LLM explains it, and the turn finishes normally (tested with a deliberately slow fake tool).
12. **How do you avoid double-booking a study room with two users?** All bookings go through one `asyncio.Lock`; the overlap check and the atomic write happen inside it.
13. **Why tiny.en and the "low" Piper voice?** CPU-only laptop: tiny.en (~0.7 s per question) is about 2× faster than base.en and half the download; the 16 kHz voice synthesises a sentence in ~0.2–0.3 s. Accuracy is lower (e.g. "304" heard as "300") but fine for short library questions.
14. **What does eos_to_first_audio_ms measure?** From the moment the VAD declares end of speech to the moment the first audio chunk is sent. The user also waits the 600 ms silence window before that.
15. **How do you know two users don't block each other?** `scripts/concurrency_test.py` runs voice+text and voice+voice at the same time and prints both users' latencies.
