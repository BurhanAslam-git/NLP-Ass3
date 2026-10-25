"""
ConversationManager — orchestrates all per-session logic.

Key changes for Assignment 2:
- Single OllamaEngine instance (P9 fix).
- SessionState.facts for in-session user identity recall (P4/P8 fix).
- Token-aware history trimming (P6 fix).
- RAG retrieval on every query (routing driven by similarity + confidence band).
- Varied refusal pool rotation (P3 fix).
- Auto-summarise at 95% context threshold (C1).
- Manual summarise via request_summarise WebSocket type (C1).
- context_usage frame emitted after each turn (C1).
- rag_sources frame emitted after RAG-grounded turns (Bonus).
- Patron hint injected for DEMO_PATRON card queries (prompt-based fast path).
- Structured logging throughout.
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime
from typing import AsyncIterator

from backend.app.config import settings
from backend.app.conversation.catalogue_hints import (
    build_patron_hint,
    is_casual_greeting,
    is_patron_query,
)
from backend.app.conversation.memory import (
    Message,
    SessionState,
    rough_token_count,
    session_tokens_used,
    trim_history_by_tokens,
)
from backend.app.conversation.orchestration import (
    build_inference_context,
    get_next_refusal,
    is_adversarial_or_off_scope,
)
from backend.app.domain.library import build_system_prompt
from backend.app.llm.ollama import OllamaEngine
from backend.app.models.schemas import (
    ContextUsageFrame,
    RagSourceItem,
    RagSourcesFrame,
    SummariseAck,
)
from backend.app.domain.records import records_db
from backend.app.conversation.tool_orchestrator import chat_with_tools
from backend.app.domain.tools import tool_specs
from backend.app.rag.retriever import LibraryRetriever
from backend.app.rag.schemas import RetrievalResult

logger = logging.getLogger(__name__)

# Regex to extract user name from self-introduction
_NAME_INTRO_RE = re.compile(
    r"(?:my name is|i am|i'm|call me)\s+([A-Za-z][A-Za-z\-']{1,30})",
    re.I,
)
# Regex to extract patron card number
_CARD_RE = re.compile(r"\bP-\d{4,6}\b", re.I)

# One-shot summarisation prompt
_SUMMARISE_SYSTEM = (
    "You are a conversation compressor. "
    "Summarise the following library assistant conversation in 3–4 sentences. "
    "Preserve all key facts: book titles mentioned, patron card numbers, names the user gave, "
    "decisions made (renewals confirmed, holds placed), and any important dates. "
    "Be concise and factual. Output ONLY the summary — no preamble."
)


class ConversationManager:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._lock = asyncio.Lock()
        self._session_turn_locks: dict[str, asyncio.Lock] = {}

        # Single shared LLM engine and retriever (P9 fix)
        self._llm = OllamaEngine()
        self._retriever = LibraryRetriever()

        logger.info("ConversationManager initialised.")

    # ------------------------------------------------------------------
    # Health check (exposed for /api/health endpoint)
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        return await self._llm.health_check()

    async def warm_up(self) -> None:
        """Startup: load the embedding model and preload the LLM into RAM."""
        try:
            await self._retriever.retrieve("library opening hours")
        except Exception as exc:
            logger.warning("Retriever warm-up failed: %s", exc)
        try:
            await self._llm.warm_up(
                [{"role": "system", "content": build_system_prompt()}, {"role": "user", "content": "hi"}],
                tools=tool_specs(),
            )
        except Exception as exc:
            logger.warning("LLM warm-up failed: %s", exc)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def _turn_lock(self, session_id: str) -> asyncio.Lock:
        async with self._lock:
            if session_id not in self._session_turn_locks:
                self._session_turn_locks[session_id] = asyncio.Lock()
            return self._session_turn_locks[session_id]

    async def reset_session(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)
            self._session_turn_locks.pop(session_id, None)
        logger.info("Session reset: %s", session_id)

    def _evict_stale_sessions(self) -> None:
        limit = settings.max_server_sessions
        if len(self._sessions) <= limit:
            return
        ranked = sorted(self._sessions.items(), key=lambda i: i[1].last_access)
        for sid, _ in ranked[: len(self._sessions) - limit]:
            self._sessions.pop(sid, None)
            self._session_turn_locks.pop(sid, None)
            logger.info("Session evicted (LRU): %s", sid)

    async def _get_or_create(self, session_id: str) -> SessionState:
        async with self._lock:
            if session_id not in self._sessions:
                self._evict_stale_sessions()
                self._sessions[session_id] = SessionState(
                    session_id=session_id,
                    messages=[Message(role="system", content=build_system_prompt())],
                )
                logger.info("New session created: %s", session_id)
            state = self._sessions[session_id]
            state.touch()
            return state

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------

    async def _append_and_trim(
        self, state: SessionState, role: str, content: str
    ) -> None:
        async with self._lock:
            state.messages.append(Message(role=role, content=content))  # type: ignore[arg-type]
            state.messages = trim_history_by_tokens(
                state.messages, settings.max_context_tokens
            )

    def _context_usage(self, state: SessionState) -> tuple[int, int]:
        """Return (tokens_used, tokens_max) for the current session."""
        used = session_tokens_used(state.messages)
        return used, settings.max_context_tokens

    # ------------------------------------------------------------------
    # User fact extraction
    # ------------------------------------------------------------------

    def _extract_facts(self, state: SessionState, user_text: str) -> None:
        """
        Extract user-provided name and card number from the message and store
        in state.facts for per-turn context injection.
        """
        m = _NAME_INTRO_RE.search(user_text)
        if m:
            name = m.group(1).strip().title()
            if name and name != state.facts.get("user_name"):
                state.facts["user_name"] = name
                logger.debug("Session %s: learned user name = %s", state.session_id, name)

        card_m = _CARD_RE.search(user_text)
        if card_m:
            card = card_m.group(0).upper()
            if card != state.facts.get("card_number"):
                state.facts["card_number"] = card
                logger.debug("Session %s: learned card = %s", state.session_id, card)

    # ------------------------------------------------------------------
    # Summarisation (C1)
    # ------------------------------------------------------------------

    async def summarise_session(self, session_id: str) -> SummariseAck:
        """
        Compress old dialogue turns into a single summary message.
        Always keeps: system prompt, the summary (if any), the 2 most recent turns.
        Returns SummariseAck with updated token counts.
        """
        state = await self._get_or_create(session_id)
        turn_lock = await self._turn_lock(session_id)

        async with turn_lock:
            async with self._lock:
                system_msgs = [m for m in state.messages if m.role == "system"]
                # Keep summary and user/assistant messages for compression
                dialogue = [m for m in state.messages if m.role != "system"]

            if len(dialogue) <= 4:
                # Too short to summarise — return current usage
                used, max_t = self._context_usage(state)
                return SummariseAck(
                    session_id=session_id,
                    tokens_used=used,
                    tokens_max=max_t,
                    message="Nothing to summarise yet — conversation is still short.",
                )

            # Turns to compress: everything except the 2 most recent
            to_compress = dialogue[:-2]
            recent = dialogue[-2:]

            # Build a plain-text transcript for the summariser
            transcript_lines = []
            for msg in to_compress:
                if msg.role == "summary":
                    transcript_lines.append(f"PREVIOUS SUMMARY:\n{msg.content}\n---")
                else:
                    role_label = "Patron" if msg.role == "user" else "Assistant"
                    transcript_lines.append(f"{role_label}: {msg.content}")
            transcript = "\n".join(transcript_lines)

            summarise_payload = [
                {"role": "system", "content": _SUMMARISE_SYSTEM},
                {"role": "user", "content": transcript},
            ]

            # Call LLM (non-streaming) for summary
            summary_parts: list[str] = []
            try:
                async for token in self._llm.stream_chat(summarise_payload):
                    summary_parts.append(token)
            except Exception as exc:
                logger.warning("Summarisation LLM call failed: %s", exc)
                used, max_t = self._context_usage(state)
                return SummariseAck(
                    session_id=session_id,
                    tokens_used=used,
                    tokens_max=max_t,
                    message="Summarisation failed — conversation unchanged.",
                )

            summary_text = "".join(summary_parts).strip()
            if not summary_text:
                used, max_t = self._context_usage(state)
                return SummariseAck(
                    session_id=session_id,
                    tokens_used=used,
                    tokens_max=max_t,
                    message="Summarisation produced empty output — conversation unchanged.",
                )

            # Rebuild messages: system + summary + recent turns
            async with self._lock:
                state.messages = (
                    system_msgs
                    + [Message(role="summary", content=f"[Summary of earlier conversation: {summary_text}]")]
                    + recent
                )

            used, max_t = self._context_usage(state)
            logger.info(
                "Session %s summarised. Tokens: %d → %d / %d",
                session_id,
                session_tokens_used(system_msgs + [Message(role="summary", content=summary_text)] + recent),
                used,
                max_t,
            )
            return SummariseAck(
                session_id=session_id,
                tokens_used=used,
                tokens_max=max_t,
            )

    # ------------------------------------------------------------------
    # Main streaming reply
    # ------------------------------------------------------------------

    async def stream_reply(
        self, session_id: str, user_text: str
    ) -> AsyncIterator[tuple[str, dict | None]]:
        """
        Yield (event_type, payload) tuples:
          ("token", {"content": "..."})       — streaming LLM token
          ("rag_sources", RagSourcesFrame)    — RAG citations (after done)
          ("context_usage", ContextUsageFrame) — token bar update (after done)
          ("timing", {...})                   — latency numbers for the metrics frame

        Caller is responsible for sending frames over WebSocket.
        """
        turn_lock = await self._turn_lock(session_id)
        async with turn_lock:
            state = await self._get_or_create(session_id)

            # --- Auto-summarise at threshold ---
            used, max_t = self._context_usage(state)
            if used / max(max_t, 1) >= settings.summarise_auto_threshold:
                logger.info(
                    "Session %s auto-summarising at %.0f%% context usage.",
                    session_id,
                    100 * used / max_t,
                )
                ack = await self.summarise_session(session_id)
                yield ("context_usage", ContextUsageFrame(
                    session_id=session_id,
                    tokens_used=ack.tokens_used,
                    tokens_max=ack.tokens_max,
                ))

            # Extract user facts (name, card) before appending to history
            self._extract_facts(state, user_text)

            await self._append_and_trim(state, "user", user_text)

            # --- Adversarial gate (only hard block — no LLM call) ---
            if is_adversarial_or_off_scope(user_text):
                refusal, new_idx = get_next_refusal(state.last_refusal_idx)
                state.last_refusal_idx = new_idx
                yield ("token", {"content": refusal})
                await self._append_and_trim(state, "assistant", refusal)
                used, max_t = self._context_usage(state)
                yield ("context_usage", ContextUsageFrame(
                    session_id=session_id,
                    tokens_used=used,
                    tokens_max=max_t,
                ))
                return

            # --- Pure greeting fast path (skip RAG, zero latency) ---
            t_retrieval = time.perf_counter()
            if is_casual_greeting(user_text):
                rag_result = RetrievalResult(chunks=[], confidence="none")
            else:
                # --- RAG retrieval (async, non-blocking) ---
                try:
                    rag_result = await self._retriever.retrieve(user_text)
                    logger.info(
                        "Session %s RAG: confidence=%s chunks=%d cached=%s latency=%.0f ms",
                        session_id,
                        rag_result.confidence,
                        len(rag_result.chunks),
                        rag_result.cached,
                        rag_result.latency_ms,
                    )
                except Exception as exc:
                    logger.warning("RAG retrieval error, proceeding without RAG: %s", exc)
                    rag_result = RetrievalResult(chunks=[], confidence="none")
            # Wall-clock retrieval time (includes the worker-thread hop), for the metrics frame.
            yield ("timing", {"retrieval_ms": round((time.perf_counter() - t_retrieval) * 1000, 1)})

            # --- Patron live account lookup from dynamic JSON database ---
            card_in_use = state.facts.get("card_number")
            if not card_in_use:
                card_m = _CARD_RE.search(user_text)
                if card_m:
                    card_in_use = card_m.group(0).upper()
                    state.facts["card_number"] = card_in_use
                elif is_patron_query(user_text):
                    card_in_use = "P-78421"

            live_loans = []
            if card_in_use:
                patron_data = await records_db.get_patron(card_in_use)
                if patron_data:
                    live_loans = patron_data.get("loans", [])

            # Today's date lets the LLM fill in dates for fines and room bookings.
            extra_notes: list[str] = [f"Today is {datetime.now():%A %Y-%m-%d}, time {datetime.now():%H:%M}."]
            patron_hint = build_patron_hint(user_text, live_loans)
            if patron_hint:
                extra_notes.append(patron_hint)

            # Dynamic account status check: If user asks about the status of a specific loan, inject live facts
            is_status_query = bool(
                re.search(
                    r"\b(status|due date|due on|renewals left|renewals used|how many renewals|can i renew|check on|holding)\b",
                    user_text,
                    re.I,
                )
            )
            if is_status_query and live_loans:
                for l in live_loans:
                    if l["item_id"].lower() in user_text.lower() or l["title"].lower() in user_text.lower():
                        u = l.get("renewals_used", 0)
                        rem = max(0, 2 - u)
                        extra_notes.append(
                            f"[Live Account Record: Patron {card_in_use} currently has '{l['title']}' ({l['item_id']}) "
                            f"checked out. Due date is {l['due_date']}. "
                            f"Renewals used: {u}/2 ({rem} renewal(s) remaining out of 2 max). "
                            f"CRITICAL: This book is currently ACTIVE ON LOAN to this patron, NOT on hold.]"
                        )
                        break

            # --- Build inference payload ---
            async with self._lock:
                history = [{"role": m.role, "content": m.content} for m in state.messages]

            payload = build_inference_context(
                messages=history,
                latest_user_text=user_text,
                retrieved_chunks=rag_result.chunks if rag_result.confidence != "none" else None,
                confidence=rag_result.confidence,
                session_facts=state.facts or None,
            )

            # Add per-turn notes (date, patron hint) to the last user
            # message. A3: they used to be an extra system message, but the chat
            # template merges system messages into the top of the prompt, so the
            # changing time invalidated Ollama's prompt cache (~15 s per turn).
            if extra_notes and payload and payload[-1]["role"] == "user":
                hint_content = "\n\n".join(extra_notes)
                # Put the note BEFORE the question: small models tend to echo text at the end.
                payload[-1] = {"role": "user", "content": f"({hint_content})\n\n{payload[-1]['content']}"}

            # --- LLM turn with tool calling (Assignment 3) ---
            # The LLM itself decides whether a tool is needed (Ollama native
            # tool calling); see conversation/tool_orchestrator.py. This replaces
            # the Assignment 2 keyword regex gate and regex fallback.
            full_parts: list[str] = []
            defaults = {"card_number": card_in_use} if card_in_use else None
            async for event_type, event_data in chat_with_tools(self._llm, payload, session_id, defaults):
                if event_type == "token":
                    full_parts.append(event_data["content"])
                yield (event_type, event_data)

            assistant_text = "".join(full_parts).strip()
            if assistant_text:
                await self._append_and_trim(state, "assistant", assistant_text)
                state.touch()

            # --- Emit RAG sources frame (Bonus: visible citations) ---
            if rag_result.chunks and rag_result.confidence != "none":
                sources = [
                    RagSourceItem(
                        doc_id=c.doc_id,
                        title=c.title,
                        similarity=round(c.similarity, 3),
                        category=c.category,
                    )
                    for c in rag_result.chunks
                ]
                yield ("rag_sources", RagSourcesFrame(session_id=session_id, sources=sources))

            # --- Emit context usage frame ---
            used, max_t = self._context_usage(state)
            yield ("context_usage", ContextUsageFrame(
                session_id=session_id,
                tokens_used=used,
                tokens_max=max_t,
            ))
