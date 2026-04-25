#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import evaluate
import numpy as np
import torch
import yaml
from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset
from transformers import (
    AutoProcessor,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    Wav2Vec2ForCTC,
)
from transformers.data.data_collator import DataCollatorMixin


MULTISPACE_RE = re.compile(r"\s+")
PUNCT_REPLACEMENTS = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
}


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_kikuyu_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    for src, dst in PUNCT_REPLACEMENTS.items():
        text = text.replace(src, dst)
    text = text.lower().strip()
    return MULTISPACE_RE.sub(" ", text)


def resolve_repo_path(repo_root: Path, value: str) -> Path:
    return (repo_root / value).resolve()


def _reports_to_mlflow(report_to: str | list[str] | None) -> bool:
    if report_to is None or report_to == "none":
        return False
    if isinstance(report_to, str):
        parts = [p.strip() for p in report_to.split(",")]
    else:
        parts = [str(p).strip() for p in report_to]
    return any(p == "mlflow" for p in parts if p)


@dataclass
class DataCollatorCTCWithPadding(DataCollatorMixin):
    processor: Any
    padding: str | bool = "longest"
    return_tensors: str = "pt"

    def torch_call(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        # as_target_processor() was removed from Wav2Vec2Processor; pad labels with the tokenizer.
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


class SaveProcessorCallback(TrainerCallback):
    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        self.processor.save_pretrained(checkpoint_dir)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.processor.save_pretrained(args.output_dir)
        return control


def load_training_dataset(config: dict[str, Any]) -> DatasetDict:
    dataset_cfg = config["dataset"]
    raw = load_dataset(dataset_cfg["name"], dataset_cfg["config"])

    train_splits = dataset_cfg.get("train_splits", ["train", "validation"])
    train_parts = [raw[split] for split in train_splits]
    eval_split = dataset_cfg.get("eval_split", "test")

    if not train_parts:
        raise ValueError("dataset.train_splits must contain at least one split")
    if eval_split not in raw:
        raise ValueError(f"Missing eval split '{eval_split}' in dataset")

    dataset = DatasetDict(
        {
            "train": concatenate_datasets(train_parts) if len(train_parts) > 1 else train_parts[0],
            "eval": raw[eval_split],
        }
    )
    return dataset.select_columns(["audio", dataset_cfg.get("text_column", "text")])


def prepare_datasets(
    dataset: DatasetDict,
    processor: Any,
    text_column: str,
    max_audio_seconds: float,
    min_audio_seconds: float,
    num_proc: int,
) -> DatasetDict:
    sampling_rate = processor.feature_extractor.sampling_rate
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

    vocab = processor.tokenizer.get_vocab()

    def prepare_example(example: dict[str, Any]) -> dict[str, Any]:
        audio = example["audio"]
        text = normalize_kikuyu_text(str(example[text_column]))
        example["input_values"] = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
        ).input_values[0]
        example["input_length"] = len(audio["array"]) / audio["sampling_rate"]
        example["labels"] = processor(text=text).input_ids
        example["target_text"] = text
        example["oov_char_count"] = sum(1 for char in text if char not in vocab)
        return example

    prepared = dataset.map(
        prepare_example,
        remove_columns=dataset["train"].column_names,
        num_proc=num_proc,
    )
    prepared["train"] = prepared["train"].filter(
        lambda length: min_audio_seconds <= length <= max_audio_seconds,
        input_columns=["input_length"],
        num_proc=num_proc,
    )
    prepared["eval"] = prepared["eval"].filter(
        lambda length: min_audio_seconds <= length <= max_audio_seconds,
        input_columns=["input_length"],
        num_proc=num_proc,
    )
    return prepared


def build_compute_metrics(processor: Any):
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = np.argmax(pred.predictions, axis=-1)
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(label_ids, group_tokens=False)

        pred_str = [normalize_kikuyu_text(text) for text in pred_str]
        label_str = [normalize_kikuyu_text(text) for text in label_str]

        return {
            "wer": wer_metric.compute(predictions=pred_str, references=label_str),
            "cer": cer_metric.compute(predictions=pred_str, references=label_str),
        }

    return compute_metrics


def summarize_dataset(dataset: DatasetDict) -> dict[str, Any]:
    train_duration = sum(dataset["train"]["input_length"]) if len(dataset["train"]) else 0.0
    eval_duration = sum(dataset["eval"]["input_length"]) if len(dataset["eval"]) else 0.0
    train_oov = int(sum(dataset["train"]["oov_char_count"])) if len(dataset["train"]) else 0
    eval_oov = int(sum(dataset["eval"]["oov_char_count"])) if len(dataset["eval"]) else 0
    return {
        "train_rows": len(dataset["train"]),
        "eval_rows": len(dataset["eval"]),
        "train_hours": round(train_duration / 3600.0, 3),
        "eval_hours": round(eval_duration / 3600.0, 3),
        "train_oov_char_count": train_oov,
        "eval_oov_char_count": eval_oov,
        "train_unique_chars": sorted(set("".join(dataset["train"]["target_text"]))) if len(dataset["train"]) else [],
        "eval_unique_chars": sorted(set("".join(dataset["eval"]["target_text"]))) if len(dataset["eval"]) else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune MMS ASR for Kikuyu using paired audio-text data."
    )
    parser.add_argument("--config", default="configs/train_mms_asr_kik.yaml")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    repo_root = config_path.parent.parent.resolve()
    config = read_yaml(config_path)

    model_cfg = config["model"]
    training_cfg = config["training"]
    runtime_cfg = config.get("runtime", {})
    outputs_cfg = config["outputs"]

    processor = AutoProcessor.from_pretrained(
        model_cfg["name"],
        target_lang=model_cfg["target_lang"],
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        model_cfg["name"],
        target_lang=model_cfg["target_lang"],
        ignore_mismatched_sizes=True,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    model.freeze_feature_encoder()
    model.config.ctc_zero_infinity = True
    model.config.use_cache = False

    dataset = load_training_dataset(config)
    prepared = prepare_datasets(
        dataset=dataset,
        processor=processor,
        text_column=config["dataset"].get("text_column", "text"),
        max_audio_seconds=float(config["dataset"].get("max_audio_seconds", 30.0)),
        min_audio_seconds=float(config["dataset"].get("min_audio_seconds", 0.5)),
        num_proc=int(runtime_cfg.get("num_proc", 1)),
    )

    output_dir = resolve_repo_path(repo_root, outputs_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_dataset(prepared)
    summary["model_name"] = model_cfg["name"]
    summary["target_lang"] = model_cfg["target_lang"]
    summary["dataset_name"] = config["dataset"]["name"]
    summary["dataset_config"] = config["dataset"]["config"]
    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    push_to_hub = bool(outputs_cfg.get("push_to_hub", False))
    hub_model_id = outputs_cfg.get("hub_model_id")
    hub_strategy = outputs_cfg.get("hub_strategy", "end")
    hub_private_repo = outputs_cfg.get("hub_private_repo")
    if push_to_hub and not hub_model_id:
        raise ValueError("outputs.push_to_hub is true but outputs.hub_model_id is missing (e.g. kihahu/mms-asr-kik-finetuned).")

    report_to = training_cfg.get("report_to", "none")
    run_name = training_cfg.get("run_name")
    if (run_name is None or run_name == "") and _reports_to_mlflow(report_to):
        run_name = f"mms-asr-kik-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(training_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
        learning_rate=float(training_cfg["learning_rate"]),
        warmup_steps=int(training_cfg["warmup_steps"]),
        max_steps=int(training_cfg["max_steps"]),
        eval_strategy=training_cfg.get("eval_strategy", "steps"),
        save_strategy=training_cfg.get("save_strategy", "steps"),
        eval_steps=int(training_cfg["eval_steps"]),
        save_steps=int(training_cfg["save_steps"]),
        logging_steps=int(training_cfg["logging_steps"]),
        save_total_limit=int(training_cfg.get("save_total_limit", 2)),
        load_best_model_at_end=bool(training_cfg.get("load_best_model_at_end", True)),
        metric_for_best_model=training_cfg.get("metric_for_best_model", "wer"),
        greater_is_better=bool(training_cfg.get("greater_is_better", False)),
        fp16=bool(training_cfg.get("fp16", torch.cuda.is_available())),
        gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", True)),
        remove_unused_columns=False,
        report_to=report_to,
        run_name=run_name,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        hub_strategy=hub_strategy,
        hub_private_repo=hub_private_repo,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=prepared["train"],
        eval_dataset=prepared["eval"],
        processing_class=processor,
        data_collator=DataCollatorCTCWithPadding(processor=processor),
        compute_metrics=build_compute_metrics(processor),
        callbacks=[SaveProcessorCallback(processor)],
    )

    trainer.train(resume_from_checkpoint=training_cfg.get("resume_from_checkpoint"))
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
