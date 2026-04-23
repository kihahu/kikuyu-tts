#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer, VitsModel

VALID_SELECTION_STRATEGIES = {"primary_only", "fallback_on_failure", "benchmark_all"}
VALID_VOICE_MODES = {"single_speaker_average"}
DEFAULT_CANDIDATES = [
    "facebook/mms-tts-kik",
    "facebook/mms-tts-swh",
    "facebook/mms-tts-kin",
    "facebook/mms-tts-lug",
]


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                prompts.append(text)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def manifest_charset(rows: list[dict[str, Any]]) -> set[str]:
    chars: set[str] = set()
    for row in rows:
        chars.update(str(row.get("text") or ""))
    return chars


def dominant_speaker(rows: list[dict[str, Any]]) -> tuple[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        speaker = str(row.get("speaker") or "unknown_speaker")
        counts[speaker] = counts.get(speaker, 0) + 1
    speaker, n_rows = max(counts.items(), key=lambda item: (item[1], item[0]))
    return speaker, n_rows


def tokenizer_coverage(tokenizer: AutoTokenizer, texts: list[str]) -> tuple[int, int]:
    total_chars = 0
    unknown_chars = 0
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for text in texts:
        total_chars += len(text)
        if unk_id is None:
            continue
        encoded = tokenizer(text, add_special_tokens=False)
        unknown_chars += sum(1 for item in encoded["input_ids"] if item == unk_id)
    return total_chars, unknown_chars


def benchmark_candidate(model_id: str, prompts: list[str]) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id)
    model.eval()

    success_count = 0
    clip_count = 0
    silent_count = 0
    durations: list[float] = []

    with torch.inference_mode():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model(**inputs)
            waveform = outputs.waveform.squeeze().detach().cpu().numpy().astype(np.float32)
            if waveform.size > 0 and np.isfinite(waveform).all():
                success_count += 1
            if waveform.size == 0 or float(np.max(np.abs(waveform))) < 0.01:
                silent_count += 1
            if waveform.size > 0 and float(np.max(np.abs(waveform))) >= 0.99:
                clip_count += 1
            durations.append(float(waveform.size) / float(model.config.sampling_rate))

    total = len(prompts)
    success_rate = success_count / total
    clipping_rate = clip_count / total
    silent_rate = silent_count / total
    mean_duration_sec = sum(durations) / len(durations)
    passes_quality_gate = success_rate == 1.0 and silent_rate <= 0.2 and clipping_rate <= 0.5 and mean_duration_sec >= 0.15

    return {
        "model_id": model_id,
        "sampling_rate": int(model.config.sampling_rate),
        "success_rate": round(success_rate, 4),
        "clipping_rate": round(clipping_rate, 4),
        "silent_rate": round(silent_rate, 4),
        "mean_duration_sec": round(mean_duration_sec, 4),
        "passes_quality_gate": passes_quality_gate,
    }


def choose_candidate(strategy: str, reports: list[dict[str, Any]]) -> dict[str, Any]:
    if strategy == "primary_only":
        return reports[0]
    if strategy == "fallback_on_failure":
        for report in reports:
            if report["passes_quality_gate"]:
                return report
        return reports[0]

    ranked = sorted(
        reports,
        key=lambda item: (
            item["passes_quality_gate"],
            item["success_rate"],
            1.0 - item["clipping_rate"],
            1.0 - item["silent_rate"],
            item["mean_duration_sec"],
        ),
        reverse=True,
    )
    return ranked[0]


def ensure_config(config: dict[str, Any]) -> None:
    strategy = config["training"]["selection_strategy"]
    if strategy not in VALID_SELECTION_STRATEGIES:
        raise ValueError(f"Unsupported selection_strategy={strategy!r}")
    voice_mode = config["training"]["target_voice_mode"]
    if voice_mode not in VALID_VOICE_MODES:
        raise ValueError(f"Unsupported target_voice_mode={voice_mode!r}")
    candidates = config["model"].get("base_model_candidates") or []
    if not candidates:
        raise ValueError("model.base_model_candidates must contain at least one candidate.")


def write_candidate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank_hint",
        "model_id",
        "sampling_rate",
        "success_rate",
        "clipping_rate",
        "silent_rate",
        "mean_duration_sec",
        "passes_quality_gate",
        "manifest_total_chars",
        "manifest_unknown_chars",
        "manifest_unknown_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the MMS-first Kikuyu TTS workflow: validate candidates, benchmark prompts, and export the selected base."
    )
    parser.add_argument("--config", default="configs/finetune_mms_tts_kik.yaml")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    repo_root = config_path.parent.parent.resolve()
    config = read_yaml(config_path)
    config["model"].setdefault("base_model_candidates", DEFAULT_CANDIDATES)
    config["training"].setdefault("selection_strategy", "primary_only")
    config["training"].setdefault("target_voice_mode", "single_speaker_average")
    ensure_config(config)

    train_manifest = (repo_root / config["data"]["train_manifest"]).resolve()
    dev_manifest = (repo_root / config["data"]["dev_manifest"]).resolve()
    test_manifest = (repo_root / config["data"]["test_manifest"]).resolve()
    prompts_file = (repo_root / config["evaluation"]["fixed_prompt_file"]).resolve()
    export_hf_dir = (repo_root / config["outputs"]["export_hf_dir"]).resolve()
    eval_report_csv = (repo_root / config["outputs"]["eval_report_csv"]).resolve()
    selection_report = (repo_root / config["outputs"]["selection_report_json"]).resolve()

    train_rows = read_jsonl(train_manifest)
    dev_rows = read_jsonl(dev_manifest)
    test_rows = read_jsonl(test_manifest)
    prompts = load_prompts(prompts_file)
    all_rows = train_rows + dev_rows + test_rows
    if not all_rows:
        raise ValueError("No rows found across train/dev/test manifests.")

    canonical_speaker, canonical_row_count = dominant_speaker(all_rows)
    charset = sorted(manifest_charset(all_rows))
    texts_for_coverage = [str(row.get("text") or "") for row in all_rows[:512]]

    reports: list[dict[str, Any]] = []
    for idx, model_id in enumerate(config["model"]["base_model_candidates"]):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        total_chars, unknown_chars = tokenizer_coverage(tokenizer, texts_for_coverage)
        benchmark = benchmark_candidate(model_id, prompts)
        benchmark["manifest_total_chars"] = total_chars
        benchmark["manifest_unknown_chars"] = unknown_chars
        benchmark["manifest_unknown_rate"] = round((unknown_chars / total_chars), 6) if total_chars else 0.0
        benchmark["rank_hint"] = idx + 1
        reports.append(benchmark)

    selected = choose_candidate(config["training"]["selection_strategy"], reports)
    config["model"]["base_model"] = selected["model_id"]

    tokenizer = AutoTokenizer.from_pretrained(selected["model_id"])
    model = VitsModel.from_pretrained(selected["model_id"])
    export_hf_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(export_hf_dir)
    model.save_pretrained(export_hf_dir)

    report = {
        "config": str(config_path),
        "resolved_base_model": selected["model_id"],
        "selection_strategy": config["training"]["selection_strategy"],
        "target_voice_mode": config["training"]["target_voice_mode"],
        "canonical_single_speaker": {
            "speaker": canonical_speaker,
            "rows": canonical_row_count,
        },
        "manifests": {
            "train_manifest": str(train_manifest),
            "dev_manifest": str(dev_manifest),
            "test_manifest": str(test_manifest),
        },
        "character_inventory_size": len(charset),
        "character_inventory_preview": "".join(charset[:80]),
        "candidate_reports": reports,
        "selected_report": selected,
        "training_backend_status": {
            "trainable_in_transformers_vits": False,
            "detail": "transformers.VitsModel currently supports inference/export only and raises NotImplementedError for training labels.",
            "recommended_next_step": "Use the exported selected base for inference and benchmarking, or keep Coqui VITS as the only in-repo trainable path until a trainable MMS backend is added.",
        },
    }

    selection_report.parent.mkdir(parents=True, exist_ok=True)
    with selection_report.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    write_candidate_csv(eval_report_csv, reports)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
