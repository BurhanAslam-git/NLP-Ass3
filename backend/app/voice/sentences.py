"""
Split streaming LLM tokens into speakable chunks for TTS.

Tokens arrive a few characters at a time. We buffer them and cut a chunk as soon
as a sentence ends, so TTS can start speaking long before the reply is finished.

Rules:
- A sentence ends at '.', '!' or '?' followed by whitespace, or at a newline.
- '.' does NOT end a sentence inside numbers ("3.5") or after abbreviations
  ("Dr.", "e.g.").
- The FIRST chunk may also end at a comma once it has enough words, so the
  first audio comes out sooner.
"""

import re

_ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "e.g", "i.e", "etc", "vs", "no",
    "st", "jr", "sr", "approx", "dept", "fig", "vol", "ed", "min", "max",
}


class SentenceSplitter:
    def __init__(self, first_chunk_min_words: int = 6) -> None:
        self.buffer = ""
        self.first_chunk_min_words = first_chunk_min_words
        self.chunks_emitted = 0

    def feed(self, text: str) -> list[str]:
        """Add new tokens; return any chunks that are now complete."""
        self.buffer += text
        chunks = []
        while (cut := self._find_cut()) is not None:
            chunk = self.buffer[:cut].strip()
            self.buffer = self.buffer[cut:]
            if chunk:
                chunks.append(chunk)
                self.chunks_emitted += 1
        return chunks

    def flush(self) -> list[str]:
        """Return whatever is left at the end of the reply."""
        chunk = self.buffer.strip()
        self.buffer = ""
        return [chunk] if chunk else []

    def _find_cut(self) -> int | None:
        """Index just after the first sentence end in the buffer, or None."""
        for i, ch in enumerate(self.buffer):
            if ch == "\n":
                return i + 1
            if ch in ".!?":
                if i + 1 >= len(self.buffer):
                    return None  # can't tell yet ("3." might become "3.5")
                if not self.buffer[i + 1].isspace():
                    continue  # "3.5", "e.g" mid-word, "BK-1001.md"
                if ch == "." and self._is_abbreviation(i):
                    continue
                return i + 1
            if ch == "," and self.chunks_emitted == 0:
                if len(self.buffer[:i].split()) >= self.first_chunk_min_words:
                    return i + 1
        return None

    def _is_abbreviation(self, dot_index: int) -> bool:
        match = re.search(r"(\S+)$", self.buffer[:dot_index])
        if not match:
            return False
        # Note: single letters are NOT treated as initials, because library
        # replies often end with "Stacks A." / "Stacks B.".
        return match.group(1).lstrip("(\"'").lower() in _ABBREVIATIONS


# ---------------------------------------------------------------------------
# Text clean-up before speaking (the chat bubble keeps the original text)
# ---------------------------------------------------------------------------

_DOC_ID = r"(?:BK|POL|FAQ|FAC|RL|STAFF)-\d+"


def clean_for_speech(text: str) -> str:
    """Remove markdown, citations and document IDs so TTS reads naturally."""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)          # [label](url) -> label
    text = re.sub(rf"\(\s*{_DOC_ID}\s*\)|\[\s*{_DOC_ID}\s*\]", "", text)  # (BK-1004)
    text = re.sub(r"\[\d+\]", "", text)                             # [1] citations
    text = re.sub(rf"\b{_DOC_ID}\b", "", text)                      # bare IDs
    text = re.sub(r"[*_`#>|]+", "", text)                           # markdown symbols
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)              # bullet markers
    text = re.sub(r"\s+([,.!?])", r"\1", text)                       # space before punctuation
    return re.sub(r"\s{2,}", " ", text).strip()
