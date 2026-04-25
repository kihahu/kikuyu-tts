# Fine-Tune MMS ASR For Kikuyu

This repo now includes a direct MMS ASR path for Kikuyu using [`facebook/mms-1b-all`](https://huggingface.co/facebook/mms-1b-all) with the Kikuyu language code **`kik`** (ISO 639-3 key in the MMS `vocab.json`; there is no `ki` entry in that checkpoint).

## Why This Path

- MMS ASR with the `kik` adapter is a strong open-weight fit for Kikuyu ASR compared to Whisper on small paired data.
- The current public `WaxalNLP` release still exposes `kik_tts`, not `kik_asr`, but the `kik_tts` split is still usable for paired `audio` + `text` ASR fine-tuning.
- The main tradeoff is data quality and diversity: `kik_tts` is small and narrow compared to a real multi-speaker ASR corpus.

## Files Added

- training script: `scripts/train_mms_asr_kik.py`
- HF Jobs bootstrap (FFmpeg + install + train): `scripts/hf_jobs_train_mms_asr_kik.sh`
- default config: `configs/train_mms_asr_kik.yaml`
- output dir: `artifacts/mms_asr_kik`

## Run It

```bash
python scripts/train_mms_asr_kik.py \
  --config configs/train_mms_asr_kik.yaml
```

The default config:

- loads `facebook/mms-1b-all`
- sets `target_lang: kik`
- trains on `google/WaxalNLP`, config `kik_tts`
- combines `train + validation` for training
- evaluates on `test`
- filters clips outside `0.5s` to `30s`

## What It Writes

- `artifacts/mms_asr_kik/dataset_summary.json`
- Trainer checkpoints under `artifacts/mms_asr_kik/checkpoint-*`
- final processor and model files under `artifacts/mms_asr_kik/`
- eval metrics via the Hugging Face Trainer save hooks

## Hugging Face Jobs (GPU)

`datasets[audio]` decodes columns through **torchcodec**, which needs **FFmpeg** on the machine image. Bare `python:3.12` does not ship it, so install FFmpeg before training. Use `scripts/hf_jobs_train_mms_asr_kik.sh` after cloning the repo (script runs `apt-get` when available, then `pip install -e .` and training).

Example (adjust `--flavor`, branch name, and `--detach` as you like):

```bash
hf jobs run --detach --flavor a10g-large --timeout 8h --secrets HF_TOKEN \
  python:3.12 bash -c \
  'git clone -b initial-import https://github.com/kihahu/kikuyu-tts.git /workspace/kikuyu-tts && bash /workspace/kikuyu-tts/scripts/hf_jobs_train_mms_asr_kik.sh /workspace/kikuyu-tts'
```

Use a `-b` ref that already contains `scripts/hf_jobs_train_mms_asr_kik.sh` (for example your feature branch) until that file exists on `initial-import`.

## Notes

- The script uses `Wav2Vec2ForCTC` because MMS ASR is CTC-based.
- It freezes the feature encoder first, which is a safer default for a small Kikuyu dataset.
- `wer` is the main model-selection metric, and `cer` is also logged.
- MMS is licensed `CC-BY-NC-4.0`, so this path is not suitable for commercial use without resolving the licensing constraint.

## Practical Next Step

This is the right bootstrap run. If the results are promising, the next quality jump will come from adding real multi-speaker Kikuyu ASR audio, not from swapping model families again.
