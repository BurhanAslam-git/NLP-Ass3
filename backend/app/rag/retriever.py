"""
Async RAG retriever with:
- all-MiniLM-L6-v2 sentence-transformer embedding (loaded once at startup).
- ChromaDB persistent vector store.
- LRU-style in-memory cache (max RAG_CACHE_SIZE entries).
- 5-band confidence scoring (high / medium_strong / medium / low / none).
- asyncio.to_thread wrapping so retrieval never blocks the event loop.

Usage:
    retriever = LibraryRetriever()          # call once at startup
    result = await retriever.retrieve(query)
"""

import asyncio
import logging
import time
from collections import OrderedDict
from pathlib import Path

from backend.app.config import settings
from backend.app.rag.schemas import RetrievalResult, RetrievedChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Confidence band logic
# ---------------------------------------------------------------------------

def _confidence_band(chunks: list[RetrievedChunk]) -> str:
    """
    Classify retrieval confidence using a 5-band scheme.

    Band        Similarity          Strategy
    ----        ----------          --------
    high        >= 0.60             Full injection; cite source.
    medium_str  0.40–0.60, gap≥0.10 Strong single match; inject top only.
    medium      0.40–0.60, clustered Partial match; inject with caveat.
    low         0.30–0.40           Background context only; heavy caveat.
    none        < 0.30              Skip injection; graceful fallback.
    """
    if not chunks:
        return "none"

    top = chunks[0].similarity

    if top >= 0.60:
        return "high"

    if top >= 0.40:
        second = chunks[1].similarity if len(chunks) > 1 else 0.0
        gap = top - second
        return "medium_strong" if gap >= 0.10 else "medium"

    if top >= settings.rag_similarity_threshold:
        return "low"

    return "none"


# ---------------------------------------------------------------------------
# LibraryRetriever
# ---------------------------------------------------------------------------

class LibraryRetriever:
    """
    Singleton-style retriever. Instantiate once in ConversationManager.

    The embedding model and ChromaDB client are loaded lazily on first use
    to avoid slowing down server startup.
    """

    def __init__(self) -> None:
        self._model = None          # sentence_transformers.SentenceTransformer
        self._chroma_client = None  # chromadb.PersistentClient
        self._collection = None     # chromadb.Collection
        self._cache: OrderedDict[str, RetrievalResult] = OrderedDict()
        self._ready = False

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_ready(self) -> None:
        """Load models and open ChromaDB on first use (called inside to_thread)."""
        if self._ready:
            return

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            import chromadb  # type: ignore

            logger.info("Loading embedding model all-MiniLM-L6-v2 ...")
            self._model = SentenceTransformer(settings.embedding_model_path())

            chroma_path = str(Path(settings.rag_chroma_dir).resolve())
            logger.info("Opening ChromaDB at %s ...", chroma_path)
            self._chroma_client = chromadb.PersistentClient(path=chroma_path)
            self._collection = self._chroma_client.get_or_create_collection(
                name=settings.rag_collection,
                metadata={"hnsw:space": "cosine"},
            )
            self._ready = True
            logger.info(
                "RAG retriever ready. Collection '%s' has %d chunks.",
                settings.rag_collection,
                self._collection.count(),
            )
        except Exception as exc:
            logger.error("Failed to initialise RAG retriever: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_get(self, key: str) -> RetrievalResult | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _cache_set(self, key: str, result: RetrievalResult) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= settings.rag_cache_size:
                self._cache.popitem(last=False)  # evict oldest
            self._cache[key] = result

    # ------------------------------------------------------------------
    # Core retrieval (synchronous — wrapped in to_thread by retrieve)
    # ------------------------------------------------------------------

    def _retrieve_sync(self, query: str) -> RetrievalResult:
        """Run embedding + Chroma query synchronously (called in a thread)."""
        self._ensure_ready()

        t0 = time.perf_counter()
        k = settings.rag_top_k

        # Embed query
        embedding = self._model.encode([query], normalize_embeddings=True).tolist()[0]

        # Query ChromaDB
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info("RAG retrieval latency: %.1f ms (query: %.60s)", latency_ms, query)

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        # ChromaDB cosine distance → similarity: similarity = 1 - distance
        chunks: list[RetrievedChunk] = []
        for doc, meta, dist in zip(docs, metas, distances):
            similarity = max(0.0, 1.0 - float(dist))
            chunks.append(
                RetrievedChunk(
                    content=doc,
                    doc_id=meta.get("doc_id", "unknown"),
                    category=meta.get("category", "general"),
                    similarity=round(similarity, 4),
                    source_file=meta.get("source_file", ""),
                    title=meta.get("title", meta.get("doc_id", "Unknown")),
                )
            )

        # Sort by similarity descending (Chroma usually returns sorted, but be safe)
        chunks.sort(key=lambda c: c.similarity, reverse=True)

        # Apply context window guard: truncate lowest-similarity chunks if total
        # chars exceed MAX_RAG_CHARS.
        total_chars = 0
        kept: list[RetrievedChunk] = []
        for chunk in chunks:
            if total_chars + len(chunk.content) > settings.rag_max_chars:
                if not kept:  # always keep the best chunk, cut to the budget
                    chunk.content = chunk.content[: settings.rag_max_chars]
                    kept.append(chunk)
                break
            kept.append(chunk)
            total_chars += len(chunk.content)

        # Confidence uses ALL retrieved chunks (top-1 vs top-2 gap), not just
        # the ones that fit the character budget.
        confidence = _confidence_band(chunks)
        return RetrievalResult(chunks=kept, confidence=confidence, latency_ms=latency_ms)

    # ------------------------------------------------------------------
    # Public async interface
    # ------------------------------------------------------------------

    async def retrieve(self, query: str) -> RetrievalResult:
        """
        Retrieve the most relevant chunks for `query`.

        - Checks the in-memory cache first (O(1), zero latency).
        - Falls back to embedding + Chroma query in a thread pool.
        - Falls back gracefully on any error (returns empty RetrievalResult).
        """
        cache_key = query.strip().lower()

        # Cache hit
        cached = self._cache_get(cache_key)
        if cached is not None:
            logger.debug("RAG cache hit for query: %.60s", query)
            result = RetrievalResult(
                chunks=cached.chunks,
                confidence=cached.confidence,
                cached=True,
                latency_ms=0.0,
            )
            return result

        # Ensure model and collection are ready before querying
        if not self._ready:
            try:
                await asyncio.to_thread(self._ensure_ready)
            except Exception as exc:
                logger.error("Failed to ensure RAG retriever readiness: %s", exc)
                return RetrievalResult(chunks=[], confidence="none", latency_ms=0.0)

        # Run retrieval in a thread (non-blocking) with timeout guard
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._retrieve_sync, query),
                timeout=5.0,
            )
        except asyncio.TimeoutError:
            logger.warning("RAG retrieval timed out (>5.0s) — falling back to no-RAG")
            return RetrievalResult(chunks=[], confidence="none", latency_ms=5000.0)
        except Exception as exc:
            logger.warning("RAG retrieval failed — falling back to no-RAG: %s", exc)
            return RetrievalResult(chunks=[], confidence="none", latency_ms=0.0)

        self._cache_set(cache_key, result)
        return result

    def cache_clear(self) -> None:
        """Clear the retrieval cache (e.g. after index update)."""
        self._cache.clear()
