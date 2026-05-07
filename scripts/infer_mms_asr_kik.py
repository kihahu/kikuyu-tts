#!/usr/bin/env python3
"""Run ASR inference on one audio file using a fine-tuned MMS Wav2Vec2 CTC checkpoint."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub.errors import RepositoryNotFoundError
from transformers import AutoProcessor, Wav2Vec2ForCTC

# Matches configs/train_mms_asr_kik.yaml outputs.hub_model_id (multisource fine-tune).
_DEFAULT_HUB_MODEL_ID = "kihahu/mms-asr-kik-finetuned"
# Training uses this checkpoint; Hub fine-tune repos sometimes omit processor files.
_DEFAULT_PROCESSOR_ID = "facebook/mms-1b-all"


def build_hub_auth(cli_token: str | None = None) -> dict[str, str]:
    """HF Hub auth: ``--token`` overrides ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` (trimmed)."""
    if cli_token is not None and str(cli_token).strip():
        return {"token": str(cli_token).strip()}
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        raw = os.environ.get(key)
        if raw:
            token = raw.strip()
            if token:
                return {"token": token}
    return {}


def _exception_blob(exc: BaseException) -> str:
    return " ".join(str(part).lower() for part in _walk_exception_causes(exc))


def _is_hub_access_failure(exc: BaseException) -> bool:
    blob = _exception_blob(exc)
    for part in _walk_exception_causes(exc):
        if isinstance(part, RepositoryNotFoundError):
            return True
    return any(
        snippet in blob
        for snippet in (
            "401",
            "403",
            "404",
            "invalid token",
            "unauthorized",
            "could not authenticate",
            "repository not found",
            "not a valid model identifier",
        )
    )


def _exit_on_hub_access_failure(exc: OSError, hub_auth: dict[str, str]) -> None:
    if not _is_hub_access_failure(exc):
        return
    if hub_auth:
        raise SystemExit(
            "Hugging Face could not read the requested model repo using this token. For private repos, the Hub "
            "may return 404 instead of 401/403. Verify that the repo exists, the token has read access, and the "
            "repo contains the expected files such as config.json. Underlying error:\n"
            f"  {exc}"
        ) from exc
    raise SystemExit(
        "Hugging Face could not read the requested model repo and no token was passed to the script. Private "
        "repos often surface as 404, not 401/403. Set `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` in the process "
        "environment, or pass `--token` explicitly. Underlying error:\n"
        f"  {exc}"
    ) from exc


def _is_local_model_path(model_id: str) -> bool:
    p = Path(model_id)
    return p.is_dir() or p.is_absolute()


def resolve_pretrained_source(raw: str) -> str:
    """Normalize local paths to absolute so HF Hub does not treat ./... as a repo id."""
    p = Path(raw).expanduser()
    if p.is_dir() or p.is_absolute() or raw.startswith((".", "~")):
        return str(p.resolve())
    return raw


def _walk_exception_causes(exc: BaseException):
    cur: BaseException | None = exc
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        yield cur
        seen.add(id(cur))
        cur = cur.__cause__


def _processor_files_missing(exc: BaseException) -> bool:
    """True when the checkpoint is readable but missing processor-specific files."""
    blob = _exception_blob(exc)
    return (
        "unrecognized processing class" in blob
        or "can't instantiate a processor" in blob
        or "preprocessor_config.json" in blob
        or "processor_config.json" in blob
        or "tokenizer_config.json" in blob
        or "special_tokens_map.json" in blob
        or ("processor_config" in blob and "not found" in blob)
        or ("does not appear to have a file named" in blob and "config.json" not in blob)
        or ("could not find a file named" in blob and "config.json" not in blob)
    )


def load_processor(
    model_id: str,
    target_lang: str,
    processor_id: str | None,
    revision: str | None,
    hub_auth: dict[str, str],
) -> AutoProcessor:
    """Load Wav2Vec2Processor; optional explicit repo, else model_id with MMS base fallback."""
    hub_kw = {**hub_auth}
    if revision is not None:
        hub_kw["revision"] = revision

    if processor_id:
        return AutoProcessor.from_pretrained(processor_id, target_lang=target_lang, **hub_kw)
    try:
        return AutoProcessor.from_pretrained(model_id, target_lang=target_lang, **hub_kw)
    except (OSError, ValueError) as exc:
        if isinstance(exc, OSError):
            _exit_on_hub_access_failure(exc, hub_auth)
        if not _processor_files_missing(exc):
            raise
    # Base checkpoint is public; do not pass fine-tune `revision` here.
    return AutoProcessor.from_pretrained(_DEFAULT_PROCESSOR_ID, target_lang=target_lang, **hub_auth)


def load_audio_mono_16k(path: Path) -> np.ndarray:
    data, sr = sf.read(str(path), always_2d=False)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = np.asarray(data, dtype=np.float32)
    if sr != 16_000:
        duration = len(data) / float(sr)
        n_new = max(1, int(duration * 16_000))
        t_old = np.linspace(0.0, duration, num=len(data), endpoint=False)
        t_new = np.linspace(0.0, duration, num=n_new, endpoint=False)
        data = np.interp(t_new, t_old, data).astype(np.float32)
    return data


def main() -> None:
    p = argparse.ArgumentParser(
        description="Transcribe one WAV/FLAC/etc. with a saved MMS ASR checkpoint.",
        epilog=(
            f"Default --model-dir is {_DEFAULT_HUB_MODEL_ID} (private Hub repo: export HF_TOKEN or "
            "HUGGING_FACE_HUB_TOKEN with read access; whitespace is stripped)."
        ),
    )
    p.add_argument(
        "--model-dir",
        type=str,
        default=_DEFAULT_HUB_MODEL_ID,
        help=f"HF model id or local path with config + weights (default: {_DEFAULT_HUB_MODEL_ID}).",
    )
    p.add_argument(
        "--revision",
        default=None,
        help="Hub git revision (branch, tag, or commit). Default: latest on the repo default branch.",
    )
    p.add_argument(
        "--processor-id",
        default=None,
        help=f"HF id for AutoProcessor (default: use --model-dir, or {_DEFAULT_PROCESSOR_ID} if that repo has no processor).",
    )
    p.add_argument(
        "--token",
        default=None,
        help="Hugging Face access token (overrides HF_TOKEN / HUGGING_FACE_HUB_TOKEN; use if your IDE does not inherit shell exports).",
    )
    p.add_argument("--audio", type=Path, required=True, help="Path to audio file (resampled to 16 kHz mono internally).")
    p.add_argument(
        "--target-lang",
        default="kik",
        help="MMS language id (must match training; default kik).",
    )
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto).")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model_id = resolve_pretrained_source(args.model_dir)
    # revision only applies to Hub ids; local dirs must not pass revision into from_pretrained.
    hub_revision = None if _is_local_model_path(model_id) else args.revision
    hub_auth = build_hub_auth(args.token)
    rev_kw = {**hub_auth}
    if hub_revision is not None:
        rev_kw["revision"] = hub_revision
    processor = load_processor(model_id, args.target_lang, args.processor_id, hub_revision, hub_auth)
    # Base MMS checkpoints (e.g. facebook/mms-1b-all) keep per-language weights in adapter.<lang>.bin.
    # Trainer fine-tunes and saves a merged checkpoint without that sidecar; loading with target_lang
    # then fails with OSError missing adapter.*.bin — load the full weights without target_lang.
    try:
        model = Wav2Vec2ForCTC.from_pretrained(model_id, target_lang=args.target_lang, **rev_kw)
    except OSError as exc:
        if "adapter." in str(exc) and "does not appear to have a file named" in str(exc):
            model = Wav2Vec2ForCTC.from_pretrained(model_id, **rev_kw)
        else:
            _exit_on_hub_access_failure(exc, hub_auth)
            raise
    model.eval().to(device)

    wav = load_audio_mono_16k(args.audio)
    inputs = processor(wav, sampling_rate=16_000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        logits = model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    text = processor.batch_decode(pred_ids)[0]
    print(text)


if __name__ == "__main__":
    main()
