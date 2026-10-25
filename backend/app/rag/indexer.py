"""
Offline indexing pipeline for the NUCES library document corpus.

Run from the project root:
    python -m backend.app.rag.indexer

Features:
- Walks data/documents/ for .md and .txt files.
- Parses YAML-style header block (--- ... ---) for metadata.
- Chunks body text at paragraph boundaries with token-aware size limit and overlap.
- Embeds chunks with all-MiniLM-L6-v2 (sentence-transformers).
- Upserts into ChromaDB collection (re-runnable; no duplicates).
- Prints summary: docs processed, chunks stored, elapsed time.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config (pulled from settings when run as module; standalone defaults shown)
# ---------------------------------------------------------------------------

try:
    from backend.app.config import settings as _cfg
    CHROMA_DIR = str(Path(_cfg.rag_chroma_dir).resolve())
    DOCUMENTS_DIR = str(Path(_cfg.rag_documents_dir).resolve())
    COLLECTION_NAME = _cfg.rag_collection
except ImportError:
    CHROMA_DIR = "data/chroma_db"
    DOCUMENTS_DIR = "data/documents"
    COLLECTION_NAME = "library_v1"

CHUNK_MAX_CHARS = 1_600   # ≈ 400 tokens at 4 chars/token
CHUNK_OVERLAP_CHARS = 200  # ≈ 50 tokens overlap between consecutive chunks
EMBED_BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Document parsing
# ---------------------------------------------------------------------------

_HEADER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)
_KV_RE = re.compile(r"^(\w+)\s*:\s*(.+)$", re.M)


def _parse_document(path: Path) -> tuple[dict[str, str], str]:
    """Return (metadata_dict, body_text) for a document file."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    meta: dict[str, str] = {"source_file": str(path.name)}

    m = _HEADER_RE.match(raw)
    if m:
        header_block = m.group(1)
        body = raw[m.end():]
        for kv in _KV_RE.finditer(header_block):
            meta[kv.group(1).strip().lower()] = kv.group(2).strip()
    else:
        body = raw

    # Derive title from metadata or filename
    if "title" not in meta:
        meta["title"] = path.stem.replace("_", " ").replace("-", " ").title()

    if "doc_id" not in meta:
        meta["doc_id"] = path.stem.upper()

    if "category" not in meta:
        meta["category"] = "general"

    return meta, body.strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _split_into_sentences(text: str) -> list[str]:
    """Naive sentence splitter on '. ', '! ', '? '."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _chunk_body(body: str, doc_id: str) -> list[tuple[int, str]]:
    """
    Split body into (chunk_index, chunk_text) pairs.

    Strategy:
    1. Split on paragraph boundaries (double newline).
    2. If a paragraph exceeds CHUNK_MAX_CHARS, sub-split at sentence boundaries.
    3. Apply CHUNK_OVERLAP_CHARS overlap between consecutive chunks.
    """
    paragraphs: list[str] = [p.strip() for p in re.split(r"\n{2,}", body) if p.strip()]

    # Flatten into sentence-level units
    units: list[str] = []
    for para in paragraphs:
        if len(para) <= CHUNK_MAX_CHARS:
            units.append(para)
        else:
            sentences = _split_into_sentences(para)
            units.extend(sentences)

    # Slide a window over units to create overlapping chunks
    chunks: list[tuple[int, str]] = []
    current: list[str] = []
    current_len = 0
    chunk_idx = 0

    for unit in units:
        unit_len = len(unit)

        if current_len + unit_len > CHUNK_MAX_CHARS and current:
            # Emit current chunk
            chunks.append((chunk_idx, " ".join(current)))
            chunk_idx += 1

            # Keep overlap: walk back from end until we have ≈ CHUNK_OVERLAP_CHARS
            overlap: list[str] = []
            overlap_len = 0
            for part in reversed(current):
                if overlap_len + len(part) > CHUNK_OVERLAP_CHARS:
                    break
                overlap.insert(0, part)
                overlap_len += len(part)
            current = overlap
            current_len = overlap_len

        current.append(unit)
        current_len += unit_len

    if current:
        chunks.append((chunk_idx, " ".join(current)))

    return chunks


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_indexer() -> None:
    t_start = time.perf_counter()

    # --- imports (heavy; only when this module runs) ---
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        import chromadb  # type: ignore
    except ImportError as exc:
        logger.error(
            "Required packages missing. Run: pip install sentence-transformers chromadb\nError: %s",
            exc,
        )
        return

    logger.info("Loading embedding model all-MiniLM-L6-v2 ...")
    from backend.app.config import settings
    model = SentenceTransformer(settings.embedding_model_path())

    logger.info("Opening ChromaDB at %s ...", CHROMA_DIR)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    docs_dir = Path(DOCUMENTS_DIR)
    if not docs_dir.is_dir():
        logger.error("Documents directory not found: %s", docs_dir)
        return

    doc_files = sorted(docs_dir.glob("**/*.md")) + sorted(docs_dir.glob("**/*.txt"))
    logger.info("Found %d document files in %s", len(doc_files), docs_dir)

    total_chunks = 0
    processed_docs = 0

    # Process in batches
    pending_ids: list[str] = []
    pending_docs: list[str] = []
    pending_metas: list[dict] = []
    pending_texts_for_embed: list[str] = []

    def _flush_batch() -> None:
        nonlocal total_chunks
        if not pending_texts_for_embed:
            return
        embeddings = model.encode(
            pending_texts_for_embed,
            batch_size=EMBED_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        collection.upsert(
            ids=pending_ids,
            embeddings=embeddings,
            documents=pending_docs,
            metadatas=pending_metas,
        )
        total_chunks += len(pending_ids)
        pending_ids.clear()
        pending_docs.clear()
        pending_metas.clear()
        pending_texts_for_embed.clear()

    for doc_path in doc_files:
        try:
            meta, body = _parse_document(doc_path)
        except Exception as exc:
            logger.warning("Skipping %s — parse error: %s", doc_path.name, exc)
            continue

        if not body:
            logger.warning("Skipping %s — empty body.", doc_path.name)
            continue

        chunks = _chunk_body(body, meta["doc_id"])
        for chunk_idx, chunk_text in chunks:
            # Stable ID: hash of doc_id + chunk_index (upsert is idempotent)
            chunk_id = hashlib.md5(
                f"{meta['doc_id']}::{chunk_idx}".encode()
            ).hexdigest()

            pending_ids.append(chunk_id)
            pending_docs.append(chunk_text)
            pending_metas.append({
                "doc_id": meta["doc_id"],
                "title": meta.get("title", meta["doc_id"]),
                "category": meta.get("category", "general"),
                "source_file": meta["source_file"],
                "chunk_index": chunk_idx,
            })
            pending_texts_for_embed.append(chunk_text)

            if len(pending_ids) >= EMBED_BATCH_SIZE:
                _flush_batch()

        processed_docs += 1
        logger.info("  Indexed: %s (%d chunks)", doc_path.name, len(chunks))

    _flush_batch()  # remaining items

    elapsed = time.perf_counter() - t_start
    final_count = collection.count()
    logger.info(
        "\n✅ Indexing complete.\n"
        "   Documents processed : %d\n"
        "   Chunks this run     : %d\n"
        "   Total in collection : %d\n"
        "   Elapsed             : %.1f s",
        processed_docs,
        total_chunks,
        final_count,
        elapsed,
    )


if __name__ == "__main__":
    run_indexer()
