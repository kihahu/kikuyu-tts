#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import evaluate
import numpy as np
import torch
import yaml
from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset
from huggingface_hub import HfApi, hf_hub_download
from transformers import (
    AutoProcessor,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    Wav2Vec2ForCTC,
)
from transformers.data.data_collator import DataCollatorMixin

from normalize_kikuyu import normalize_kikuyu_text

DEFAULT_HUB_REQUIRED_FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "vocab.json",
    "trainer_state.json",
    "eval_results.json",
    "dataset_summary.json",
]
FINAL_UPLOAD_IGNORE_PATTERNS = ["checkpoint-*", "checkpoint-*/*"]


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def _require_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "outputs.push_to_hub is true, but HF_TOKEN is not set. "
            "Pass a write-capable token to HF Jobs with `--secrets HF_TOKEN`."
        )
    return token


def _commit_oid(commit_info: Any) -> str | None:
    return getattr(commit_info, "oid", None) or getattr(commit_info, "commit_oid", None)


def _matching_required_files(repo_files: list[str], required: list[str]) -> tuple[list[str], list[str]]:
    present: list[str] = []
    missing: list[str] = []
    for pattern in required:
        matches = [path for path in repo_files if path == pattern or fnmatch(path, pattern)]
        if matches:
            present.append(pattern)
        else:
            missing.append(pattern)
    return present, missing


def preflight_hub_repo(
    api: HfApi,
    repo_id: str,
    private: bool | None,
    token: str,
) -> None:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    sentinel_path = f".publish_preflight/{timestamp}.json"
    payload = {
        "repo_id": repo_id,
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": "kikuyu-tts MMS ASR publish preflight",
    }

    try:
        api.whoami(token=token)
        api.create_repo(
            repo_id=repo_id,
            private=private,
            repo_type="model",
            exist_ok=True,
            token=token,
        )
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as f:
            json.dump(payload, f, indent=2)
            temp_path = f.name
        try:
            api.upload_file(
                path_or_fileobj=temp_path,
                path_in_repo=sentinel_path,
                repo_id=repo_id,
                repo_type="model",
                token=token,
                commit_message="publish preflight",
            )
        finally:
            Path(temp_path).unlink(missing_ok=True)

        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=sentinel_path,
            repo_type="model",
            token=token,
        )
        with Path(downloaded).open("r", encoding="utf-8") as f:
            downloaded_payload = json.load(f)
        if downloaded_payload.get("repo_id") != repo_id:
            raise RuntimeError(f"Preflight sentinel readback mismatch for {repo_id}.")
    except Exception as exc:
        raise RuntimeError(
            "Hugging Face Hub preflight failed before training. "
            f"Verify HF_TOKEN can create/write the model repo '{repo_id}'. "
            f"Original error: {exc}"
        ) from exc


def upload_folder_to_hub(
    api: HfApi,
    folder_path: Path,
    repo_id: str,
    token: str,
    commit_message: str,
    path_in_repo: str | None = None,
    ignore_patterns: list[str] | None = None,
) -> Any:
    try:
        return api.upload_folder(
            folder_path=folder_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message=commit_message,
            ignore_patterns=ignore_patterns,
        )
    except Exception as exc:
        target = f"{repo_id}/{path_in_repo}" if path_in_repo else repo_id
        raise RuntimeError(f"Failed to upload {folder_path} to Hub target {target}: {exc}") from exc


def verify_hub_model(
    repo_id: str,
    token: str,
    required_files: list[str],
    target_lang: str,
) -> dict[str, Any]:
    api = HfApi()
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="model", token=token)
    present, missing = _matching_required_files(repo_files, required_files)
    if missing:
        raise RuntimeError(
            f"Hub repo {repo_id} is missing required files after publish: {', '.join(missing)}"
        )

    for filename in required_files:
        if "*" not in filename:
            hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model", token=token)

    AutoProcessor.from_pretrained(repo_id, target_lang=target_lang, token=token)
    Wav2Vec2ForCTC.from_pretrained(repo_id, token=token)
    return {
        "repo_id": repo_id,
        "required_files_present": present,
        "file_count": len(repo_files),
    }


def write_publish_report(output_dir: Path, report: dict[str, Any]) -> Path:
    path = output_dir / "publish_report.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return path


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


class HubCheckpointUploadCallback(TrainerCallback):
    def __init__(
        self,
        api: HfApi,
        repo_id: str,
        token: str,
        processor: Any,
        upload_every_n_saves: int,
    ) -> None:
        self.api = api
        self.repo_id = repo_id
        self.token = token
        self.processor = processor
        self.upload_every_n_saves = max(1, upload_every_n_saves)
        self.save_count = 0

    def on_save(self, args, state, control, **kwargs):
        self.save_count += 1
        if self.save_count % self.upload_every_n_saves != 0:
            return control
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint_dir.is_dir():
            raise RuntimeError(f"Expected checkpoint directory is missing: {checkpoint_dir}")
        self.processor.save_pretrained(checkpoint_dir)
        upload_folder_to_hub(
            api=self.api,
            folder_path=checkpoint_dir,
            path_in_repo=checkpoint_dir.name,
            repo_id=self.repo_id,
            token=self.token,
            commit_message=f"checkpoint step {state.global_step}",
        )
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

    push_to_hub = bool(outputs_cfg.get("push_to_hub", False))
    hub_model_id = outputs_cfg.get("hub_model_id")
    hub_strategy = outputs_cfg.get("hub_strategy", "end")
    hub_private_repo = outputs_cfg.get("hub_private_repo")
    hub_verify_after_push = bool(outputs_cfg.get("hub_verify_after_push", push_to_hub))
    hub_upload_checkpoints = bool(outputs_cfg.get("hub_upload_checkpoints", push_to_hub))
    hub_upload_every_n_saves = int(outputs_cfg.get("hub_upload_every_n_saves", 1))
    hub_required_files = list(outputs_cfg.get("hub_required_files") or DEFAULT_HUB_REQUIRED_FILES)
    hub_token = _require_hf_token() if push_to_hub else ""
    hub_api = HfApi() if push_to_hub else None
    if push_to_hub and not hub_model_id:
        raise ValueError(
            "outputs.push_to_hub is true but outputs.hub_model_id is missing "
            "(e.g. kihahu/mms-asr-kik-finetuned)."
        )
    if push_to_hub:
        assert hub_api is not None
        preflight_hub_repo(
            api=hub_api,
            repo_id=hub_model_id,
            private=hub_private_repo,
            token=hub_token,
        )

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
    if push_to_hub and hub_upload_checkpoints:
        assert hub_api is not None
        trainer.add_callback(
            HubCheckpointUploadCallback(
                api=hub_api,
                repo_id=hub_model_id,
                token=hub_token,
                processor=processor,
                upload_every_n_saves=hub_upload_every_n_saves,
            )
        )

    trainer.train(resume_from_checkpoint=training_cfg.get("resume_from_checkpoint"))
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    trainer.save_state()

    if push_to_hub:
        assert hub_api is not None
        final_commit = None
        report: dict[str, Any] = {
            "status": "started",
            "repo_id": hub_model_id,
            "target_lang": model_cfg["target_lang"],
            "output_dir": str(output_dir),
            "global_step": int(trainer.state.global_step),
            "metrics": metrics,
            "required_files": hub_required_files,
            "created_at": datetime.now(UTC).isoformat(),
        }
        try:
            final_commit = upload_folder_to_hub(
                api=hub_api,
                folder_path=output_dir,
                repo_id=hub_model_id,
                token=hub_token,
                commit_message=f"final model step {trainer.state.global_step}",
                ignore_patterns=FINAL_UPLOAD_IGNORE_PATTERNS,
            )
            report["final_commit_oid"] = _commit_oid(final_commit)
            if hub_verify_after_push:
                report["verification"] = verify_hub_model(
                    repo_id=hub_model_id,
                    token=hub_token,
                    required_files=hub_required_files,
                    target_lang=model_cfg["target_lang"],
                )
            report["status"] = "success"
            write_publish_report(output_dir, report)
            hub_api.upload_file(
                path_or_fileobj=output_dir / "publish_report.json",
                path_in_repo="publish_report.json",
                repo_id=hub_model_id,
                repo_type="model",
                token=hub_token,
                commit_message=f"publish report step {trainer.state.global_step}",
            )
        except Exception as exc:
            report["status"] = "failed"
            report["error"] = str(exc)
            write_publish_report(output_dir, report)
            try:
                hub_api.upload_file(
                    path_or_fileobj=output_dir / "publish_report.json",
                    path_in_repo="publish_report.json",
                    repo_id=hub_model_id,
                    repo_type="model",
                    token=hub_token,
                    commit_message=f"failed publish report step {trainer.state.global_step}",
                )
            except Exception:
                pass
            raise


if __name__ == "__main__":
    main()
