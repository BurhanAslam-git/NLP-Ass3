#!/usr/bin/env python3
"""Download the local ASR (faster-whisper) and TTS (Piper) models into models/.

Usage (from the project root):
    python scripts/download_models.py                 # models named in .env / config
    python scripts/download_models.py --asr tiny.en --voice en_US-lessac-low

Run once; after that everything works offline.
"""

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.config import settings  # noqa: E402

PIPER_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
SILERO_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"


def download_whisper(size: str, models_dir: Path) -> None:
    from faster_whisper import download_model

    out = models_dir / "whisper" / size
    if (out / "model.bin").exists():
        print(f"Whisper {size}: already present")
        return
    print(f"Downloading faster-whisper {size} ...")
    download_model(size, output_dir=str(out))
    print(f"Whisper {size}: saved to {out}")


def download_piper_voice(voice: str, models_dir: Path) -> None:
    # Voice names look like en_US-lessac-medium -> en/en_US/lessac/medium/
    lang, speaker, quality = voice.split("-")
    folder = f"{lang.split('_')[0]}/{lang}/{speaker}/{quality}"
    out_dir = models_dir / "piper"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in (f"{voice}.onnx", f"{voice}.onnx.json"):
        target = out_dir / name
        if target.exists():
            print(f"Piper {name}: already present")
            continue
        print(f"Downloading Piper {name} ...")
        tmp = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(f"{PIPER_BASE}/{folder}/{name}", tmp)
        tmp.replace(target)  # only keep complete files
    print(f"Piper voice {voice}: saved to {out_dir}")


def download_vad(models_dir: Path) -> None:
    target = models_dir / "vad" / "silero_vad.onnx"
    if target.exists():
        print("Silero VAD: already present")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading Silero VAD ...")
    tmp = target.with_suffix(".part")
    urllib.request.urlretrieve(SILERO_URL, tmp)
    tmp.replace(target)


def download_embeddings(models_dir: Path) -> None:
    """all-MiniLM-L6-v2 for RAG, saved locally so retrieval works offline."""
    base = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main"
    out = models_dir / "all-MiniLM-L6-v2"
    files = ["modules.json", "config_sentence_transformers.json", "sentence_bert_config.json", "config.json",
             "tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json",
             "1_Pooling/config.json", "model.safetensors"]
    for name in files:
        target = out / name
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MiniLM {name} ...")
        tmp = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(f"{base}/{name}", tmp)
        tmp.replace(target)
    print(f"Embedding model: ready in {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr", default=settings.asr_model)
    parser.add_argument("--voice", default=settings.tts_voice)
    args = parser.parse_args()

    models_dir = ROOT / settings.models_dir
    download_embeddings(models_dir)
    download_vad(models_dir)
    download_whisper(args.asr, models_dir)
    download_piper_voice(args.voice, models_dir)
    print("All models ready.")


if __name__ == "__main__":
    main()
