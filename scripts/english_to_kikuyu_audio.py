#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kikuyu_tts.chunking import chunk_text, normalize_text
from scripts.synthesize_mms_vits_checkpoint import (
    load_checkpoint_for_synthesis,
    pick_device,
    synthesize_text,
)


DEFAULT_TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"
DEFAULT_SRC_LANG = "eng_Latn"
DEFAULT_TGT_LANG = "kik_Latn"
DEFAULT_TTS_REPO_ID = "kihahu/mms-tts-kik-waxal-v1"
DEFAULT_TTS_CHECKPOINT = "mms_vits_finetune/vits/logs/mms_kik_waxal_single_speaker/G_77100.pth"
DEFAULT_TTS_CONFIG = "mms_vits_finetune/vits/logs/mms_kik_waxal_single_speaker/config.json"
DEFAULT_TTS_VOCAB = "mms_vits_finetune/vits/logs/mms_kik_waxal_single_speaker/vocab.txt"


@dataclass(frozen=True)
class PipelineChunk:
    index: int
    source_text: str
    kikuyu_text: str
    audio_path: str
    sample_rate: int
    samples: int
    duration_sec: float


def choose_translation_device(raw: str) -> str:
    if raw != "auto":
        return raw
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def read_input_text(path: str, text: str) -> str:
    if path:
        return Path(path).read_text(encoding="utf-8")
    return text


def translate_chunks_nllb(
    *,
    chunks: list[str],
    model_name: str,
    src_lang: str,
    tgt_lang: str,
    device: str,
    max_new_tokens: int,
    num_beams: int,
) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    if forced_bos_token_id is None or forced_bos_token_id == tokenizer.unk_token_id:
        raise ValueError(f"Target language code {tgt_lang!r} is not supported by {model_name!r}.")

    translations: list[str] = []
    for source_text in chunks:
        encoded = tokenizer(source_text, return_tensors="pt", truncation=True).to(device)
        with torch.inference_mode():
            translated = model.generate(
                **encoded,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
        decoded = tokenizer.batch_decode(translated, skip_special_tokens=True)[0].strip()
        translations.append(decoded)
    return translations


def translate_chunks(
    *,
    chunks: list[str],
    backend: str,
    model_name: str,
    src_lang: str,
    tgt_lang: str,
    device: str,
    max_new_tokens: int,
    num_beams: int,
) -> list[str]:
    if backend == "identity":
        return chunks
    if backend == "nllb":
        return translate_chunks_nllb(
            chunks=chunks,
            model_name=model_name,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            device=device,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
    raise ValueError(f"Unknown translation backend: {backend}")


def concatenate_audio(parts: list[np.ndarray], silence_samples: int) -> np.ndarray:
    if not parts:
        return np.zeros(0, dtype=np.float32)
    silence = np.zeros(max(0, silence_samples), dtype=np.float32)
    out: list[np.ndarray] = []
    for idx, part in enumerate(parts):
        if idx:
            out.append(silence)
        out.append(np.asarray(part, dtype=np.float32))
    return np.concatenate(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate English text to Kikuyu and synthesize one Waxal G_77100 Kikuyu audio file."
    )
    parser.add_argument("--input", default="", help="UTF-8 English input file. Use --text for inline input.")
    parser.add_argument("--text", default="", help="Inline English input text.")
    parser.add_argument("--output-wav", default="artifacts/english_to_kikuyu_audio/output.wav")
    parser.add_argument("--translated-output", default="artifacts/english_to_kikuyu_audio/kikuyu.txt")
    parser.add_argument("--manifest-json", default="artifacts/english_to_kikuyu_audio/manifest.json")
    parser.add_argument("--chunk-wav-dir", default="artifacts/english_to_kikuyu_audio/chunks")
    parser.add_argument("--source-chunk-size", type=int, default=700)
    parser.add_argument("--tts-chunk-size", type=int, default=320)
    parser.add_argument("--max-chunks", type=int, default=0, help="Optional cap for quick smoke tests.")
    parser.add_argument("--silence-ms", type=int, default=350)

    parser.add_argument("--translation-backend", choices=("nllb", "identity"), default="nllb")
    parser.add_argument("--translation-model", default=DEFAULT_TRANSLATION_MODEL)
    parser.add_argument("--src-lang", default=DEFAULT_SRC_LANG)
    parser.add_argument("--tgt-lang", default=DEFAULT_TGT_LANG)
    parser.add_argument("--translation-device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--num-beams", type=int, default=4)

    parser.add_argument("--tts-repo-id", default=DEFAULT_TTS_REPO_ID)
    parser.add_argument("--tts-checkpoint", default=DEFAULT_TTS_CHECKPOINT)
    parser.add_argument("--tts-config", default=DEFAULT_TTS_CONFIG)
    parser.add_argument("--tts-vocab", default=DEFAULT_TTS_VOCAB)
    parser.add_argument("--vits-repo", default="artifacts/vits_inference")
    parser.add_argument("--tts-device", default="cpu", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--noise-scale", type=float, default=0.667)
    parser.add_argument("--noise-scale-w", type=float, default=0.8)
    parser.add_argument("--length-scale", type=float, default=1.0)
    args = parser.parse_args()

    source_text = normalize_text(read_input_text(args.input, args.text))
    if not source_text:
        raise ValueError("Provide English text with --input or --text.")

    source_chunks = [chunk.text for chunk in chunk_text(source_text, max_chars=args.source_chunk_size)]
    if args.max_chunks:
        source_chunks = source_chunks[: args.max_chunks]
    if not source_chunks:
        raise ValueError("Input produced no source chunks.")

    translation_device = choose_translation_device(args.translation_device)
    kikuyu_chunks = translate_chunks(
        chunks=source_chunks,
        backend=args.translation_backend,
        model_name=args.translation_model,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        device=translation_device,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )

    translated_path = Path(args.translated_output)
    translated_path.parent.mkdir(parents=True, exist_ok=True)
    translated_path.write_text("\n\n".join(kikuyu_chunks), encoding="utf-8")

    tts_model = load_checkpoint_for_synthesis(
        repo_id=args.tts_repo_id,
        checkpoint=args.tts_checkpoint,
        config=args.tts_config,
        vocab=args.tts_vocab,
        vits_repo=Path(args.vits_repo),
        device=pick_device(args.tts_device),
    )

    chunk_wav_dir = Path(args.chunk_wav_dir)
    chunk_wav_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[PipelineChunk] = []
    audio_parts: list[np.ndarray] = []
    sample_rate = int(tts_model.hps.data.sampling_rate)

    audio_index = 0
    for source_index, kikuyu_text in enumerate(kikuyu_chunks):
        tts_chunks = [chunk.text for chunk in chunk_text(kikuyu_text, max_chars=args.tts_chunk_size)]
        for tts_text in tts_chunks:
            audio = synthesize_text(
                tts_model,
                tts_text,
                noise_scale=args.noise_scale,
                noise_scale_w=args.noise_scale_w,
                length_scale=args.length_scale,
            )
            chunk_path = chunk_wav_dir / f"chunk_{audio_index:04d}.wav"
            sf.write(chunk_path, audio, sample_rate, subtype="PCM_16")
            manifest.append(
                PipelineChunk(
                    index=audio_index,
                    source_text=source_chunks[source_index],
                    kikuyu_text=tts_text,
                    audio_path=str(chunk_path),
                    sample_rate=sample_rate,
                    samples=int(len(audio)),
                    duration_sec=round(float(len(audio) / sample_rate), 4),
                )
            )
            audio_parts.append(audio)
            audio_index += 1

    full_audio = concatenate_audio(audio_parts, int(sample_rate * args.silence_ms / 1000.0))
    output_wav = Path(args.output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_wav, full_audio, sample_rate, subtype="PCM_16")

    manifest_path = Path(args.manifest_json)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_input": args.input or "<inline>",
        "translation_backend": args.translation_backend,
        "translation_model": args.translation_model if args.translation_backend == "nllb" else "",
        "src_lang": args.src_lang,
        "tgt_lang": args.tgt_lang,
        "tts_repo_id": args.tts_repo_id,
        "tts_checkpoint": args.tts_checkpoint,
        "output_wav": str(output_wav),
        "translated_output": str(translated_path),
        "manifest_json": str(manifest_path),
        "sample_rate": sample_rate,
        "duration_sec": round(float(len(full_audio) / sample_rate), 4),
        "chunks": [asdict(row) for row in manifest],
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({k: payload[k] for k in ("output_wav", "translated_output", "manifest_json", "duration_sec") if k in payload}, indent=2))


if __name__ == "__main__":
    main()
