"""Unit tests for the voice helpers (no models needed)."""

from backend.app.voice.sentences import SentenceSplitter, clean_for_speech
from backend.app.voice.utterance import FRAME_BYTES, Utterance


def _split(text: str, step: int = 3) -> list[str]:
    """Feed text a few characters at a time, like LLM tokens."""
    splitter = SentenceSplitter(first_chunk_min_words=6)
    chunks: list[str] = []
    for i in range(0, len(text), step):
        chunks += splitter.feed(text[i:i + step])
    return chunks + splitter.flush()


def test_splits_on_sentence_end_not_abbreviation_or_decimal():
    chunks = _split("Dr. Smith said e.g. 3.5 days. The fine is PKR 20! Is it in Stacks B? Yes.")
    assert chunks == [
        "Dr. Smith said e.g. 3.5 days.",
        "The fine is PKR 20!",
        "Is it in Stacks B?",
        "Yes.",
    ]


def test_first_chunk_may_end_at_comma_after_six_words():
    chunks = _split("Yes, we have Speech and Language Processing by Jurafsky, in Stacks B. Later, more text here.")
    assert chunks[0] == "Yes, we have Speech and Language Processing by Jurafsky,"
    assert chunks[1] == "in Stacks B."
    # The comma rule only applies to the first chunk.
    assert chunks[2] == "Later, more text here."


def test_newline_ends_a_chunk():
    assert _split("Line one\nLine two") == ["Line one", "Line two"]


def test_clean_for_speech_strips_markdown_and_ids():
    text = "**Clean Code** (BK-1004) is in [Stacks A](http://x) [1].\n- Due BK-1003 soon"
    assert clean_for_speech(text) == "Clean Code is in Stacks A.\nDue soon"


class FakeVAD:
    """Treats any non-zero frame as speech."""

    def is_speech(self, frame: bytes) -> bool:
        return any(frame)


def test_end_of_speech_after_silence():
    utt = Utterance(FakeVAD(), silence_ms=600, max_seconds=20)
    silence, speech = bytes(FRAME_BYTES), b"\x10\x00" * (FRAME_BYTES // 2)
    assert not utt.add(silence * 5)          # waiting for speech
    assert not utt.add(speech * 20)          # speaking
    assert not utt.add(silence * 10)         # 320 ms silence: not yet
    assert utt.add(silence * 10)             # > 600 ms silence: end
    assert utt.ended and utt.end_time is not None


def test_pre_roll_is_trimmed_before_speech():
    utt = Utterance(FakeVAD(), silence_ms=600, max_seconds=20)
    utt.add(bytes(FRAME_BYTES) * 100)        # 3.2 s of silence
    assert len(utt.audio) == 10 * FRAME_BYTES


def test_partial_frames_are_carried_over():
    utt = Utterance(FakeVAD(), silence_ms=600, max_seconds=20)
    half = b"\x10\x00" * (FRAME_BYTES // 4)  # half a frame
    utt.add(half)
    assert not utt.speech_started
    utt.add(half)
    assert utt.speech_started
