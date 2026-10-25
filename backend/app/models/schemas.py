"""WebSocket message schemas (Pydantic v2)."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Inbound (client → server)
# ---------------------------------------------------------------------------

class ChatInbound(BaseModel):
    type: Literal["user_message"] = "user_message"
    session_id: str = Field(..., min_length=1, max_length=128)
    content: str = Field(..., min_length=1, max_length=4000)

    @field_validator("content")
    @classmethod
    def normalize_content(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Message cannot be empty or whitespace only.")
        return cleaned


class ChatSessionReset(BaseModel):
    type: Literal["reset_session"] = "reset_session"
    session_id: str = Field(..., min_length=1, max_length=128)


class SummariseRequest(BaseModel):
    type: Literal["request_summarise"] = "request_summarise"
    session_id: str = Field(..., min_length=1, max_length=128)


# ---------------------------------------------------------------------------
# Outbound (server → client)
# ---------------------------------------------------------------------------

class ChatOutboundBase(BaseModel):
    session_id: str


class ChatToken(ChatOutboundBase):
    type: Literal["token"] = "token"
    content: str


class ChatDone(ChatOutboundBase):
    type: Literal["done"] = "done"


class ChatError(ChatOutboundBase):
    type: Literal["error"] = "error"
    message: str


class ChatRateLimit(ChatOutboundBase):
    type: Literal["rate_limit"] = "rate_limit"
    message: str = "Too many messages. Please wait a moment."


class ChatSessionResetAck(BaseModel):
    type: Literal["session_reset"] = "session_reset"
    session_id: str


class RagSourceItem(BaseModel):
    doc_id: str
    title: str
    similarity: float
    category: str = ""


class RagSourcesFrame(ChatOutboundBase):
    """Sent after 'done' when RAG was used — enables visible citations in UI."""
    type: Literal["rag_sources"] = "rag_sources"
    sources: list[RagSourceItem]


class ContextUsageFrame(ChatOutboundBase):
    """Sent after each assistant turn to drive the context window bar in the UI."""
    type: Literal["context_usage"] = "context_usage"
    tokens_used: int
    tokens_max: int


class SummariseAck(ChatOutboundBase):
    """Sent after summarisation completes."""
    type: Literal["summarise_done"] = "summarise_done"
    tokens_used: int
    tokens_max: int
    message: str = "Conversation compressed — older turns summarised to save context."


# ---------------------------------------------------------------------------
# Assignment 3 — voice and tool frames (additive; old frames are unchanged)
# ---------------------------------------------------------------------------

class AudioStart(BaseModel):
    """Client starts streaming mic audio; binary PCM16 frames follow."""
    type: Literal["audio_start"] = "audio_start"
    session_id: str = Field(..., min_length=1, max_length=128)
    sample_rate: Literal[16000] = 16000
    format: Literal["pcm16"] = "pcm16"
    speak: bool = True   # synthesise the reply as audio


class AudioStop(BaseModel):
    """Client stopped the mic (button). Server finalises what it has."""
    type: Literal["audio_stop"] = "audio_stop"
    session_id: str = Field(..., min_length=1, max_length=128)


class TranscriptFrame(ChatOutboundBase):
    type: Literal["transcript_partial", "transcript_final"]
    text: str


class AudioChunkFrame(ChatOutboundBase):
    """One synthesised sentence: base64 of raw little-endian PCM16 mono."""
    type: Literal["audio_chunk"] = "audio_chunk"
    seq: int
    sample_rate: int
    format: Literal["pcm16"] = "pcm16"
    data: str


class TtsDone(ChatOutboundBase):
    type: Literal["tts_done"] = "tts_done"


class ToolCallFrame(ChatOutboundBase):
    type: Literal["tool_call"] = "tool_call"
    name: str
    arguments: dict


class ToolResultFrame(ChatOutboundBase):
    type: Literal["tool_result"] = "tool_result"
    name: str
    ok: bool
    latency_ms: float


class MetricsFrame(ChatOutboundBase):
    """Per-turn latency breakdown in milliseconds (None = stage not used)."""
    type: Literal["metrics"] = "metrics"
    asr_ms: float | None = None
    retrieval_ms: float | None = None
    tool_ms: float | None = None
    llm_first_token_ms: float | None = None
    tts_first_audio_ms: float | None = None
    eos_to_first_audio_ms: float | None = None


# ---------------------------------------------------------------------------
# REST
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    model: str


class LibraryInfoResponse(BaseModel):
    library_name: str
    virtual_desk_availability: str
    assistant_name: str
    assistant_role: str
    tagline: str
    greeting: str
    sample_patron_card: str
    max_context_tokens: int
