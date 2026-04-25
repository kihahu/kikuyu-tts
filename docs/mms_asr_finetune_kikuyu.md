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

## After a Hugging Face Job: get the checkpoint locally

The **Jobs web page** shows status and **logs**; it does **not** list the container’s filesystem as downloadable “outputs.” The CLI has **no** `hf jobs download`. During training, files lived under **`/workspace/kikuyu-tts/artifacts/mms_asr_kik/`** on the worker; when the job ends, that disk is gone unless you copied it elsewhere.

**Recommended:** turn on Hub upload in `configs/train_mms_asr_kik.yaml` (defaults in-repo use `kihahu/mms-asr-kik-finetuned`; change `hub_model_id` if you want a different repo name):

```yaml
outputs:
  output_dir: artifacts/mms_asr_kik
  push_to_hub: true
  hub_model_id: YOUR_USERNAME/mms-asr-kik-finetuned
  hub_private_repo: true
  hub_strategy: end
```

The `Trainer` calls **`create_repo(..., exist_ok=True)`** when `push_to_hub` is enabled, so you do **not** need to create the model repo manually first.

Use **`--secrets HF_TOKEN`** on `hf jobs run` with a token that can **create/write** that model repo. After the run, `hf download YOUR_USERNAME/mms-asr-kik-finetuned` locally.

Other options:

1. **HF bucket volume** — `-v hf://buckets/org/bucket:/mnt` and `cp -a artifacts/mms_asr_kik /mnt/…`, then `hf sync` to your laptop.
2. **One-off upload job** — only helps if the worker still exists (not after completion).

## Smoke-test inference (local)

With a directory that contains `config.json`, `model.safetensors` (or `pytorch_model.bin`), and tokenizer / preprocessor files:

```bash
python scripts/infer_mms_asr_kik.py \
  --model-dir /path/to/artifacts/mms_asr_kik \
  --audio /path/to/kikuyu_clip.wav
```

Optional: `--target-lang kik` (default) and `--device cuda` or `cpu`.

## MLflow experiment tracking

Training defaults to `report_to: mlflow` in `configs/train_mms_asr_kik.yaml`. The Hugging Face `Trainer` uses the built-in **`MLflowCallback`** (`transformers` integration).

**Local UI:** after a run, `mlflow ui` (defaults to tracking store under `./mlruns` unless you override the URI).

**Common environment variables** (see also [HF `MLflowCallback` docs](https://huggingface.co/docs/transformers/main/en/main_classes/callback#transformers.integrations.MLflowCallback)):

| Variable | Purpose |
|----------|---------|
| `MLFLOW_TRACKING_URI` | e.g. `http://127.0.0.1:5000` or `sqlite:///mlflow.db` |
| `MLFLOW_EXPERIMENT_NAME` | Experiment name (created if missing) |
| `MLFLOW_TAGS` | JSON object string, e.g. `'{"task":"mms-asr-kik"}'` |
| `HF_MLFLOW_LOG_ARTIFACTS` | `true` / `1` to log checkpoint folders as artifacts (can be slow) |
| `DISABLE_MLFLOW_INTEGRATION` | `TRUE` to turn MLflow off without editing YAML |

To **disable** tracking for a one-off run, set `training.report_to: none` in the YAML or export `DISABLE_MLFLOW_INTEGRATION=TRUE`.

## Hugging Face Jobs (GPU)

`datasets[audio]` decodes columns through **torchcodec**, which needs **FFmpeg** on the machine image. Bare `python:3.12` does not ship it, so install FFmpeg before training. Use `scripts/hf_jobs_train_mms_asr_kik.sh` after cloning the repo (script runs `apt-get` when available, then `pip install -e .`, uninstalls Apple-only **MLX** wheels on Linux so `transformers` does not crash on `import mlx.core`, then runs training).

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
