# Colab Workflow: Kikuyu VITS From Scratch (WaxalNLP `kik_tts`)

This runbook executes the full plan in Colab without editing plan files.

## 1) Setup runtime and dependencies

```bash
python --version
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv
python3.11 -m venv /content/tts311
source /content/tts311/bin/activate
python -m pip install -U pip
python -m pip install datasets[audio] soundfile librosa pyyaml huggingface_hub
python -m pip install coqpit trainer TTS==0.22.0
```

Use the Python 3.11 environment for all subsequent commands:

```bash
source /content/tts311/bin/activate
python --version  # should be 3.11.x
```

If using Drive for checkpoint safety:

```python
from google.colab import drive
drive.mount("/content/drive")
```

## 2) Clone repo and prepare data

```bash
cd /content
git clone https://github.com/<your-user>/kikuyu-tts.git
cd kikuyu-tts
```

```bash
source /content/tts311/bin/activate
python scripts/prepare_waxal_kik_tts.py \
  --dataset-name google/WaxalNLP \
  --dataset-config kik_tts \
  --split train \
  --output-dir data/waxal_kik_tts \
  --target-sample-rate 16000 \
  --min-duration-sec 0.6 \
  --max-duration-sec 25.0 \
  --min-rms 0.0035 \
  --seed 42 \
  --dev-ratio 0.10 \
  --test-ratio 0.05
```

## 3) Build tokenizer/vocab from normalized manifests

```bash
source /content/tts311/bin/activate
python scripts/build_kikuyu_vocab.py \
  --train-manifest data/waxal_kik_tts/manifests/train.jsonl \
  --dev-manifest data/waxal_kik_tts/manifests/dev.jsonl \
  --out-dir artifacts/tokenizer_kikuyu_char
```

## 4) Launch training with resume-safe checkpointing

Clone Coqui trainer once per runtime:

```bash
cd /content
git clone https://github.com/coqui-ai/TTS.git
cd /content/kikuyu-tts
```

Start fresh:

```bash
source /content/tts311/bin/activate
python scripts/colab_train_vits_scratch.py \
  --config configs/train_kikuyu_vits_scratch_colab.yaml \
  --trainer-repo /content/TTS
```

Resume from Drive checkpoint:

```bash
source /content/tts311/bin/activate
python scripts/colab_train_vits_scratch.py \
  --config configs/train_kikuyu_vits_scratch_colab.yaml \
  --trainer-repo /content/TTS \
  --resume
```

Optional: push checkpoints to HF Hub:

```bash
source /content/tts311/bin/activate
huggingface-cli login
python scripts/colab_train_vits_scratch.py \
  --config configs/train_kikuyu_vits_scratch_colab.yaml \
  --trainer-repo /content/TTS \
  --resume \
  --push-hf
```

## 5) Evaluate checkpoints and select best

Prepare a CSV (one row per checkpoint) with:

- `checkpoint`
- `synthesis_success_rate`
- `clipping_rate`
- `mos_lite`
- `wer_proxy`

Then run:

```bash
source /content/tts311/bin/activate
python scripts/evaluate_and_select.py \
  --metrics-csv artifacts/checkpoint_metrics.csv \
  --out-json artifacts/best_checkpoint_selection.json \
  --out-csv artifacts/tts_eval_summary.csv
```

## 6) Package best checkpoint for local integration

```bash
source /content/tts311/bin/activate
python scripts/prepare_local_integration.py \
  --best-checkpoint-dir artifacts/colab_runs/kikuyu_vits_scratch/checkpoint_best \
  --tokenizer-dir artifacts/tokenizer_kikuyu_char \
  --eval-summary-csv artifacts/tts_eval_summary.csv \
  --out-dir artifacts/local_integration/kikuyu_vits_best \
  --model-id kikuyu-vits-scratch-waxal
```

This creates:

- local model bundle
- tokenizer files
- evaluation summary copy
- `integration_metadata.json` for the next local `kikuyu-tts` wiring step

