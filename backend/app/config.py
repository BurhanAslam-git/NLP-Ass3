from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:1.5b-instruct"  # A3: ~2.5x faster than llama3.2:3b on our CPU, better tool choice
    request_timeout_seconds: float = 120.0
    ollama_keep_alive: str = "30m"   # keep the model loaded in RAM between turns
    ollama_num_ctx: int = 4096       # context window sent to Ollama

    # Session management
    max_server_sessions: int = 128

    # Context window (token-aware trimming)
    max_context_tokens: int = 3072   # rough-token budget for session history

    # RAG
    rag_top_k: int = 5
    rag_similarity_threshold: float = 0.30
    rag_max_chars: int = 1200        # max chars of RAG context per turn (A3: was 2000; less to read = faster)
    rag_chroma_dir: str = "data/chroma_db"
    rag_documents_dir: str = "data/documents"
    rag_collection: str = "library_v1"
    rag_cache_size: int = 256        # LRU cache entries for retrieval results
    embedding_model: str = "models/all-MiniLM-L6-v2"  # local copy; falls back to the HF name

    def embedding_model_path(self) -> str:
        """Local folder if downloaded (fully offline), else the Hugging Face model id."""
        from pathlib import Path
        return self.embedding_model if Path(self.embedding_model).is_dir() else "all-MiniLM-L6-v2"

    # Summariser
    summarise_auto_threshold: float = 0.95  # trigger auto-summarise at 95% context

    # Voice — ASR (Assignment 3)
    models_dir: str = "models"            # local model files (gitignored)
    asr_model: str = "tiny.en"            # faster-whisper size: tiny.en (fast) / base.en (more accurate)
    asr_compute_type: str = "int8"        # int8 quantisation on CPU
    asr_cpu_threads: int = 4
    vad_threshold: float = 0.5            # Silero speech probability threshold
    vad_silence_ms: int = 600             # this much silence = end of speech
    asr_partial_interval_ms: int = 1000   # new audio needed before a partial transcript
    asr_max_utterance_s: float = 20.0     # force end of speech after this long

    # Voice — TTS
    tts_voice: str = "en_US-lessac-low"     # Piper voice (16 kHz "low" = faster than "medium")
    tts_first_chunk_min_words: int = 6      # first chunk may end at a comma after this many words

    # Tools
    tool_timeout_seconds: float = 3.0
    tool_max_rounds: int = 2

    # Server
    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()
