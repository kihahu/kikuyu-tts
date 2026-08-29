#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForCTC, AutoProcessor, AutoTokenizer, VitsModel

try:
    from scripts.normalize_kikuyu import normalize_kikuyu_text
except ModuleNotFoundError:
    from normalize_kikuyu import normalize_kikuyu_text


DEFAULT_TTS_MODELS = [
    "facebook/mms-tts-kik",
    "gateremark/kikuyu-tts-v1",
    "BrianMwangi/African-Kikuyu-TTS",
]
DEFAULT_ASR_MODEL = "kihahu/mms-asr-kik-waxal-ctc"


@dataclass(frozen=True)
class AudioMetrics:
    duration_sec: float
    rms: float
    peak_abs: float
    clipping_rate: float
    silence_ratio: float


def normalize_text(text: str) -> str:
    return normalize_kikuyu_text(text)


def read_prompts(path: Path) -> list[str]:
    prompts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            prompts.append(line)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def slug_model_id(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "__", model_id.strip())


def calc_audio_metrics(samples: np.ndarray, sample_rate: int) -> AudioMetrics:
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    duration_sec = float(samples.size / sample_rate) if sample_rate > 0 else 0.0
    if samples.size == 0:
        return AudioMetrics(duration_sec, 0.0, 0.0, 0.0, 1.0)
    abs_samples = np.abs(samples)
    rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
    peak_abs = float(np.max(abs_samples))
    clipping_rate = float(np.mean(abs_samples >= 0.99))
    silence_ratio = float(np.mean(abs_samples < 1e-4))
    return AudioMetrics(duration_sec, rms, peak_abs, clipping_rate, silence_ratio)


def edit_distance(left: list[str] | str, right: list[str] | str) -> int:
    left_items = list(left)
    right_items = list(right)
    prev = list(range(len(right_items) + 1))
    for i, left_item in enumerate(left_items, start=1):
        cur = [i]
        for j, right_item in enumerate(right_items, start=1):
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + (0 if left_item == right_item else 1),
                )
            )
        prev = cur
    return prev[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    if not ref_words:
        return 0.0 if not hypothesis.split() else 1.0
    return edit_distance(ref_words, hypothesis.split()) / len(ref_words)


def char_error_rate(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(reference, hypothesis) / len(reference)


def synthesize_model(model_id: str, prompts: list[str], output_dir: Path, device: str) -> list[dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id).to(device)
    model.eval()
    sample_rate = int(model.config.sampling_rate)

    rows: list[dict[str, Any]] = []
    model_dir = output_dir / slug_model_id(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)

    for idx, prompt in enumerate(prompts):
        audio_path = model_dir / f"sample_{idx:03d}.wav"
        row: dict[str, Any] = {
            "model": model_id,
            "prompt_id": f"prompt_{idx:03d}",
            "prompt": prompt,
            "audio_path": str(audio_path),
            "success": "0",
            "error": "",
        }
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                output = model(**inputs)
            waveform = output.waveform[0].detach().cpu().numpy()
            sf.write(audio_path, waveform, sample_rate, subtype="PCM_16")
            metrics = calc_audio_metrics(waveform, sample_rate)
            row.update(
                {
                    "success": "1",
                    "sample_rate": sample_rate,
                    "duration_sec": round(metrics.duration_sec, 4),
                    "rms": round(metrics.rms, 6),
                    "peak_abs": round(metrics.peak_abs, 6),
                    "clipping_rate": round(metrics.clipping_rate, 6),
                    "silence_ratio": round(metrics.silence_ratio, 6),
                }
            )
        except Exception as exc:  # noqa: BLE001 - report all model failures in the eval table.
            row["error"] = repr(exc)
        rows.append(row)
    return rows


def load_asr(model_id: str, device: str) -> tuple[Any, Any]:
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCTC.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


def transcribe_audio(audio_path: Path, processor: Any, model: Any, device: str) -> str:
    samples, sample_rate = sf.read(audio_path, dtype="float32")
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    target_rate = int(getattr(processor.feature_extractor, "sampling_rate", 16000))
    if sample_rate != target_rate:
        raise ValueError(f"{audio_path} has sample rate {sample_rate}; expected {target_rate}")
    inputs = processor(samples, sampling_rate=sample_rate, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0]


def add_asr_roundtrip(rows: list[dict[str, Any]], asr_model_id: str, device: str) -> None:
    processor, model = load_asr(asr_model_id, device)
    for row in rows:
        if row.get("success") != "1":
            continue
        try:
            hyp = transcribe_audio(Path(row["audio_path"]), processor, model, device)
            ref_norm = normalize_text(row["prompt"])
            hyp_norm = normalize_text(hyp)
            row["asr_model"] = asr_model_id
            row["asr_transcript"] = hyp
            row["roundtrip_wer"] = round(float(word_error_rate(ref_norm, hyp_norm)), 6)
            row["roundtrip_cer"] = round(float(char_error_rate(ref_norm, hyp_norm)), 6)
        except Exception as exc:  # noqa: BLE001 - ASR is optional eval telemetry.
            row["asr_error"] = repr(exc)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["model"]), []).append(row)

    summary_rows = []
    for model_id, model_rows in grouped.items():
        successes = [row for row in model_rows if row.get("success") == "1"]
        clipping = [float(row.get("clipping_rate", 0.0) or 0.0) for row in successes]
        silence = [float(row.get("silence_ratio", 0.0) or 0.0) for row in successes]
        cers = [float(row["roundtrip_cer"]) for row in successes if row.get("roundtrip_cer") not in (None, "")]
        summary_rows.append(
            {
                "model": model_id,
                "samples": len(model_rows),
                "successful_samples": len(successes),
                "synthesis_success_rate": round(len(successes) / len(model_rows), 6) if model_rows else 0.0,
                "avg_clipping_rate": round(float(np.mean(clipping)), 6) if clipping else "",
                "avg_silence_ratio": round(float(np.mean(silence)), 6) if silence else "",
                "avg_roundtrip_cer": round(float(np.mean(cers)), 6) if cers else "",
            }
        )
    write_csv(path, summary_rows)


def write_manual_score_sheet(path: Path, rows: list[dict[str, Any]]) -> None:
    sheet_rows = [
        {
            "model": row["model"],
            "prompt_id": row["prompt_id"],
            "prompt": row["prompt"],
            "audio_path": row["audio_path"],
            "naturalness_1_to_5": "",
            "pronunciation_1_to_5": "",
            "notes": "",
        }
        for row in rows
        if row.get("success") == "1"
    ]
    write_csv(path, sheet_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and evaluate Kikuyu TTS model samples.")
    parser.add_argument("--prompts", default="data/eval/kikuyu_prompts.txt")
    parser.add_argument("--models", nargs="+", default=DEFAULT_TTS_MODELS)
    parser.add_argument("--output-dir", default="artifacts/tts_eval")
    parser.add_argument("--details-csv", default="artifacts/tts_eval/details.csv")
    parser.add_argument("--summary-csv", default="artifacts/tts_eval/summary.csv")
    parser.add_argument("--manual-csv", default="artifacts/tts_eval/manual_scores.csv")
    parser.add_argument("--json-output", default="artifacts/tts_eval/details.json")
    parser.add_argument("--asr-model", default=DEFAULT_ASR_MODEL)
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    prompts = read_prompts(Path(args.prompts))
    output_dir = Path(args.output_dir)
    rows: list[dict[str, Any]] = []
    for model_id in args.models:
        rows.extend(synthesize_model(model_id, prompts, output_dir, device))

    if not args.skip_asr:
        add_asr_roundtrip(rows, args.asr_model, device)

    write_csv(Path(args.details_csv), rows)
    write_summary(Path(args.summary_csv), rows)
    write_manual_score_sheet(Path(args.manual_csv), rows)
    Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_output).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.summary_csv}")


if __name__ == "__main__":
    main()
