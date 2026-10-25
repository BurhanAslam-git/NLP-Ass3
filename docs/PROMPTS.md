# Prompt log — Assignment 3 (generative AI use)

We used Claude Code (Claude Opus 5.5) inside VS Code to help build Assignment 3.
Every instruction we gave is recorded here with its date and outcome, so we can reproduce it in the viva.

---

## Prompt 5 — Team members, commit and push now (2026-10-25)

**Outcome:** Phases I, II and IV committed as separate commits and pushed to GitHub; then continued with the flaky test, Chrome check, benchmarks, README, docs.

```text
Team members for the README:
1. Burhan Aslam, 23I-3097
2. Saad Abdullah, 23I-3045

Because the deadline is today, commit what works RIGHT NOW (separate commits for
Phase I, II and IV as planned) and run "git push" after each commit, so the work
is saved on GitHub. Then continue with the remaining steps in order: flaky test,
Chrome check, benchmarks, README, PROMPTS.md, VIVA_GUIDE.md, then CRM bonus only
if time allows. Record this message in PROMPTS.md too.
```

## Prompt 4 — Status request (2026-10-25)

**Outcome:** Short status given (done / doing now / left).

```text
Give me a short status: what is done, what you're doing now, what's left.
```

## Prompt 3 — LLM choice (2026-10-25, answered in a multiple-choice question)

Claude measured llama3.2:3b on our i7-8550U: ~5 tokens/s generation, ~25–30 prompt tokens/s, and it called a (wrong) tool on almost every question when tools were offered. It asked how to proceed.

```text
Question: llama3.2:3b is ~5 tokens/s on this CPU and calls a (wrong) tool on almost every
question when tools are offered. How should I proceed with the LLM?
Answer chosen: "Pull qwen2.5:1.5b (Recommended)"
```

**Outcome:** qwen2.5:1.5b-instruct pulled and benchmarked: first token 0.8 s (vs 1.2–1.5 s), 11–14 tok/s (vs 5–6), correct tool choice 2/2 (vs 0/2). Made it the default model; llama3.2:3b still selectable with `OLLAMA_MODEL`.

---

## Prompt 2 — Decisions after Step 0 (2026-10-25)

**Outcome:** Stalled install was already dead (it failed with a connection reset). The pip cache was already empty. Install restarted on the faster network together with the voice packages. Plan approved; work started on Phase I.

```text
Decisions:
1. Network: I have switched to a faster connection. Kill the stalled pip install and
   restart it. Yes, run "pip cache purge" first to free disk space.
   Do NOT pull qwen2.5:1.5b-instruct yet; first measure latency with llama3.2:3b,
   then tell me if we need qwen.
2. Plan approved, including replacing the regex tool gate with LLM-decided tool
   calling and expanding the catalogue to 40 books. Also fix bugs 3, 7, 8 and 10
   since they affect Assignment 3.
3. We are short on time (due today). Prioritise in this order: Phase I, II, IV,
   then III, V, README. Commit after each phase. Do the CRM bonus only if
   everything else is done. Use tiny.en and a "low" Piper voice if downloads
   or latency are a problem.
```

(Bugs 3, 7, 8, 10 from the Step 0 report: `.env.example` points at a model that is not pulled; frontend closes the reply bubble after a 0.6 s token pause; `pytest-asyncio` missing from requirements; `/ws/chat` crashes on binary frames.)

---

## Prompt 1 — Initial brief (2026-10-25)

**Outcome:** Step 0 done. Hardware detected, `.venv` created, `setup.ps1` and `run.ps1` written, code reviewed, plan proposed. No Assignment 3 code written yet; waiting for approval.

```text
You are working in my group's NLP course repo (Windows 11, PowerShell, Python 3.13.2,
Ollama 0.40.6 with llama3.2:3b and qwen2.5:1.5b-instruct pulled).
This is Assignment 3 of a 3-part series. It builds DIRECTLY on our Assignment 2 code in this folder.
The full assignment spec is at docs/assignment3.pdf. READ IT COMPLETELY FIRST and treat it as the source of truth.
If anything in this prompt conflicts with the PDF, the PDF wins; tell me about the conflict.

## Step 0 — Before planning anything
1. Detect my hardware with PowerShell commands (CPU model, cores, RAM, GPU) and tell me what you found.
2. Create a Python virtual environment (.venv), install requirements.txt, run the RAG indexer
   (python -m backend.app.rag.indexer), start the Assignment 2 app, and run pytest, to confirm
   the existing project works on this machine. setup.sh/run.sh are bash scripts; create Windows
   equivalents (setup.ps1 / run.ps1). If any package doesn't support Python 3.13 on Windows, tell me
   and suggest the fix (e.g. installing Python 3.11 alongside).
3. Read the README and every file under backend/, frontend/, scripts/, tests/.
4. Then give me: (a) a short summary of how the current code works, (b) any failures from step 2,
   (c) bugs or README statements that are outdated (e.g. "Known limitations" still describes
   Assignment 1), (d) your step-by-step implementation plan with files to create/change and the
   final WebSocket protocol. Then WAIT for my approval before writing any Assignment 3 code.

## Context about our project (verify by reading the code — don't trust it blindly)
- Domain: "OPAC FAST NUCES" library assistant ("NUCES Virtual Desk"). We MUST keep this domain.
- Backend: FastAPI. backend/app/main.py has REST (/api/health, /api/library-info) and WebSocket /ws/chat.
- Conversation manager: backend/app/conversation/manager.py. Per-session history, per-session asyncio locks, tool orchestration.
- LLM: Ollama, local CPU, llama3.2:3b (alt qwen2.5:1.5b-instruct). backend/app/llm/ollama.py uses Ollama native tool calling.
- RAG: backend/app/rag/ (indexer.py, retriever.py). all-MiniLM-L6-v2 + persistent ChromaDB at data/chroma_db, k=5, LRU cache, 5 s timeout, 80 docs in data/documents.
- Existing tools: backend/app/domain/tools.py. issue_book, renew_book, return_book, check_patron_loans over data/library_records.json via backend/app/domain/records.py.
- Frontend: frontend/ (vanilla HTML/CSS/JS chat UI, multi-chat sidebar in localStorage, help page).
- Existing WebSocket contract (MUST NOT CHANGE):
  client→server: {"type":"user_message","session_id","content"}, {"type":"reset_session","session_id"}
  server→client: token, done, error, session_reset (as in backend/app/models/schemas.py).
- Tests in tests/ (pytest). Benchmark: scripts/benchmark.py.

## Hard constraints
1. EVERYTHING runs locally. No cloud ASR/TTS, no browser Web Speech API (no SpeechRecognition / speechSynthesis), no paid APIs, no internet-dependent tools.
2. The existing text WebSocket contract stays backward compatible. Text-only chat works exactly as before. Voice is ADDED alongside.
3. Voice is NOT a separate system. Spoken turns use the SAME ConversationManager, SAME session history (same session_id), SAME RAG, SAME tool orchestrator, SAME LLM streaming as typed turns. A user can type one turn and speak the next in the same chat with memory preserved.
4. Write the MINIMAL code needed. Reuse existing modules; don't rewrite working A2 code. We must explain every line in a viva, so keep code simple and readable with short comments.
5. Don't break existing tests. Run pytest after every phase.
6. All CPU-heavy work (ASR, TTS, embeddings) runs off the event loop (asyncio.to_thread or a worker pool) so two concurrent users never block each other.

## Model choices (use unless there's a concrete problem on my machine; justify any change)
- ASR: faster-whisper "base.en" (try "tiny.en" if too slow), int8 on CPU. Load ONCE at startup and warm up.
- End-of-speech: Silero VAD (or webrtcvad if Silero is a problem). Final transcript after ~600 ms silence (configurable).
- TTS: Piper (piper-tts), English voice such as en_US-lessac-medium. Load once at startup. Add a script that downloads voice files into models/ (gitignored).
- LLM: keep llama3.2:3b. Keep it warm (keep_alive + warm-up request at startup). If end-of-speech→first-audio can't reach ≤2 s, test qwen2.5:1.5b-instruct and report both.
Put all model names, thresholds and timeouts in config/.env with defaults, and document them.

## Phase I — Speech-to-Text
- Browser: getUserMedia + AudioWorklet, downsample to 16 kHz mono PCM16, send small chunks (~20–40 ms) over WebSocket.
- Additive protocol (refine if needed, document final version):
  client→server: {"type":"audio_start","session_id","sample_rate":16000,"format":"pcm16"}, binary frames = audio chunks, {"type":"audio_stop","session_id"}.
  server→client: {"type":"transcript_partial","session_id","text"}, {"type":"transcript_final","session_id","text"}.
  Decide: extra types on /ws/chat, or a separate /ws/voice sharing the same ConversationManager. Pick whichever keeps the old contract 100% unchanged and code simplest; explain in README.
- Backend: buffer audio per connection, VAD endpointing, partial transcripts periodically (~1 s of new audio, keep CPU cost reasonable), ONE final transcript at end of speech.
- Final transcript goes into the SAME code path as a typed user_message, shown as a user bubble in chat.
- UI: mic button with recording state, live partial transcript, clear error if mic permission denied.

## Phase II — Text-to-Speech
- As LLM tokens stream, split into sentences (handle . ! ? newlines, abbreviations/decimals like "Dr.", "e.g.", "3.5"). First chunk may be emitted at a comma once it has ≥ ~6 words.
- Synthesize each sentence with Piper in a worker thread as soon as complete. Do NOT wait for the full response.
- Stream audio in order: {"type":"audio_chunk","session_id","seq":n,...} (binary or base64, document it), then {"type":"tts_done","session_id"}.
- Browser plays chunks strictly in seq order with Web Audio API (gap-free). Text tokens keep appearing in chat simultaneously.
- Strip markdown/citation markers/IDs before speaking; the text bubble keeps them.
- Stop-playback button. Barge-in is nice-to-have, not required.
- "Speak replies" toggle in UI, on by default for voice turns.

## Phase III — Voice with streaming, memory and RAG
- Prove voice and text share one pipeline (same history, retrieval, tools).
- Automated test: a typed turn then a "voice" turn (WAV file through the voice path) in the same session_id; the second turn sees the first turn's history.

## Phase IV — Tools (orchestrator + three primary tools, useful for a LIBRARY)
Keep existing loan tools working. Implement and document these THREE as our primary Assignment 3 tools:
1. search_catalogue(query: str, author?: str, available_only?: bool, limit?: int)
   Searches the live catalogue in data/library_records.json (or a SQLite copy if simpler). Returns title, author, ID, location/shelf mark, status, due date if checked out.
2. calculate_fine(due_date: "YYYY-MM-DD", return_date?: "YYYY-MM-DD" (default today), item_type?: str)
   Computes overdue fines EXACTLY per our fines policy in data/documents (find the POL-* fines doc, use its real rates/caps). Safe date arithmetic, no eval. Include a breakdown.
3. book_study_room(action: "list_availability"|"book"|"cancel"|"my_bookings", card_number?: str, room_id?: str, date?: "YYYY-MM-DD", start_time?: "HH:MM", duration_minutes?: int, booking_id?: str)
   Calendar/scheduling over a local JSON file (data/room_bookings.json). Use real rooms/opening hours from FAC-* docs if present, else a small documented room list. Reject double-bookings, past times, out-of-hours and over-long bookings. Atomic writes with an asyncio.Lock.
For ALL tools (old and new):
- Clear name, description and JSON argument schema, defined in one registry.
- Orchestrator keeps Ollama native tool calling: LLM decides → validate args against schema → execute ASYNC with asyncio.wait_for timeout (default 3 s, configurable) → feed result or structured error back to LLM → stream final answer. Up to 2 tool rounds per turn.
- Failures (timeout, bad args, unknown tool, exception) give a sensible spoken/written message, never a hang or crash. Test each failure type, including a deliberately slow fake tool that times out.
- Additive WS events {"type":"tool_call","session_id","name","arguments"} and {"type":"tool_result","session_id","name","ok","latency_ms"}; UI shows "Using search_catalogue…".
- Tools work from SPOKEN requests and results are SPOKEN back (same TTS path).
- Explain in README and code comments exactly how the LLM picks a tool (Ollama "tools" parameter). We'll be asked in the viva.

## Phase V — Real-time behaviour and measurements
- Per voice turn, log and emit {"type":"metrics",...}: asr_ms (end of speech → final transcript), retrieval_ms, tool_ms, llm_first_token_ms, tts_first_audio_ms, eos_to_first_audio_ms. Show in a small collapsible UI panel.
- Targets: ideally ~1 s, max ~2 s end-of-speech → first audio; retrieval + tools ≤ ~2 s before generation. If missed, try in order: warm models, smaller whisper, shorter system prompt/num_ctx, earlier first TTS chunk, smaller LLM. Report before/after honestly. NEVER fake numbers.
- scripts/benchmark_voice.py: generate test WAVs locally with Piper (a catalogue question, a fines question, a room booking, a RAG policy question), run them through the real voice pipeline N times, print a markdown table of mean/p50/p95 for every metric.
- Concurrency test: 2 simultaneous clients (voice + text, then voice + voice). Show neither blocks the other; report latencies under load.

## BONUS — CRM tool (ONLY after everything above works and is tested)
patron_profile(action: "get"|"upsert"|"log_interaction", user_id: str, name?, email?, phone?, preferences?, note?)
- SQLite at data/crm.db (stdlib sqlite3 in a thread): name, contact details, preferences, past interactions keyed by user_id.
- Browser keeps a persistent user_id in localStorage and sends it at connection start (additive, don't break old contract).
- Returning session: retrieve profile so the assistant greets the user by name and recalls their last request.
- LLM calls upsert when user gives new info ("my name is Ali"). Interactions logged each turn. Must work in voice mode. Add tests.

## Working process
1. Phases in order: I → II → III → IV → V → README → Bonus. After EACH phase: run pytest, smoke check, git commit (e.g. "A3 Phase I: streaming ASR with VAD"), and give me a 5–10 line summary plus exact steps to test it myself in Chrome.
2. Pin new dependencies in requirements.txt. setup.ps1 installs everything and downloads ASR/TTS models. Model files go in models/ (gitignored).
3. If something can't work on my machine, stop and tell me with options. Never silently swap in a cloud service.
4. Keep docs/PROMPTS.md: record this prompt and every follow-up instruction I give, with date and outcome (we must reproduce prompts in the viva).
5. Create/update docs/VIVA_GUIDE.md: how ASR streaming + VAD works, sentence-level TTS streaming, tool calling detection/execution, timeouts/errors, concurrency, plus 15 likely viva questions with short answers.

## README.md (final deliverable — must cover ALL THREE assignments)
Rewrite README.md with EVERY item below; fix outdated A1/A2 statements. Before writing it, ASK me for both team members' names and roll numbers.
1. Overview + business use case (library), team members
2. Up-to-date Mermaid architecture diagram: Web UI, FastAPI, Conversation Manager, LLM, Retrieval Module, Voice Pipeline (ASR+VAD, TTS), Tool Orchestrator + each tool (+ CRM if done)
3. Model choices with reasons: LLM, ASR, VAD, TTS, embedding model
4. Document collection and retrieval setup (corpus, chunking, embeddings, vector store, top-k, thresholds, caching)
5. Conversation memory / context budgeting
6. Full WebSocket protocol: original text contract (unchanged) + all new voice/tool/metrics message types with JSON examples
7. For EACH tool: name, description, JSON schema, and a real example of the LLM invoking it (user utterance → tool call JSON → tool result → final answer), captured from an actual run
8. Failure handling (tool timeouts/errors, ASR/TTS failures, retrieval failure, mic denied)
9. Latency benchmarks table: ASR, retrieval, tools, LLM first token, TTS first audio, end-of-speech → first audio, plus concurrent-user results and the hardware measured on
10. Setup instructions from a clean Windows machine (Python, Ollama + model pull, ASR/TTS model download, indexing, run), plus Mac/Linux notes
11. How to run tests and benchmarks
12. Known limitations (honest)
13. Bonus CRM section (if done)
14. Use of generative AI: short statement + pointer to docs/PROMPTS.md
```
