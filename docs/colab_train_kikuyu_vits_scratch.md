# Colab Workflow: Kikuyu VITS From Scratch (WaxalNLP `kik_tts`)

This runbook matches `notebooks/train_kikuyu_vits_scratch.ipynb` on the default branch.

## 1) Setup runtime and dependencies (Coqui from PyPI)

The notebook uses the Colab image Python and `coqui-tts` from PyPI (no separate 3.11 venv), then runs repo scripts from a clone of this repository.

```bash
pip install -U pip setuptools wheel
pip install "datasets[audio]" soundfile librosa pyyaml huggingface_hub
pip install coqui-tts
```

**Alternative (older Colab / strict patch pins):** use a Python 3.11 venv, then `pip install coqpit trainer TTS==0.22.0` as in earlier revisions.

If using Drive for checkpoint safety:

```python
from google.colab import drive
drive.mount("/content/drive")
```

## 2) Clone repo and prepare data

```bash
cd /content
[ -d kikuyu-tts ] || git clone https://github.com/kihahu/kikuyu-tts.git
cd kikuyu-tts
git pull
```

The prepare step also writes `data/waxal_kik_tts/manifests/train_coqui.txt` and `dev_coqui.txt` for the Coqui `coqui` dataset formatter (see `scripts/colab_train_vits_scratch.py`).

```bash
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
python scripts/build_kikuyu_vocab.py \
  --train-manifest data/waxal_kik_tts/manifests/train.jsonl \
  --dev-manifest data/waxal_kik_tts/manifests/dev.jsonl \
  --out-dir artifacts/tokenizer_kikuyu_char \
  --wikipedia-orthography \
  --extra-chars 'ñÑ'
```

- **`--wikipedia-orthography`**: add all letters from the [Kikuyu language](https://en.wikipedia.org/wiki/Kikuyu_language) *Alphabet* section (fetched from the live article when online; otherwise the same line is read from a frozen string in `scripts/kikuyu_orthography_wikipedia.py`). Use **`--wikipedia-offline`** in air‑gapped environments to skip network I/O.
- **`--extra-chars`**: any extra symbols you still need (e.g. loanword letters not in the wiki line).

`scripts/colab_train_vits_scratch.py` reads `tokenizer.extra_characters` in the YAML and the built `vocab.json` to fill Coqui’s `characters` block.

## 4) Launch training with resume-safe checkpointing

`scripts/colab_train_vits_scratch.py` writes `artifacts/colab_runs/kikuyu_vits_scratch/coqui_vits_config.yaml` and runs training. It uses the Coqui TTS **git** checkout only if `TTS/bin/train_tts.py` exists under `--trainer-repo`; otherwise it uses `python -m TTS.bin.train_tts` (e.g. after `pip install coqui-tts`).

**Trainer repo** defaults to the kikuyu-tts tree (next to `configs/`). To use a separate clone of [coqui-ai/TTS](https://github.com/coqui-ai/TTS) instead, pass e.g. `--trainer-repo /content/TTS`.

Start fresh:

```bash
python scripts/colab_train_vits_scratch.py \
  --config configs/train_kikuyu_vits_scratch_colab.yaml \
  --trainer-repo /content/kikuyu-tts
```

Resume from Drive checkpoint:

```bash
python scripts/colab_train_vits_scratch.py \
  --config configs/train_kikuyu_vits_scratch_colab.yaml \
  --trainer-repo /content/kikuyu-tts \
  --resume
```

Optional: push checkpoints to HF Hub:

```bash
huggingface-cli login
python scripts/colab_train_vits_scratch.py \
  --config configs/train_kikuyu_vits_scratch_colab.yaml \
  --trainer-repo /content/kikuyu-tts \
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
python scripts/evaluate_and_select.py \
  --metrics-csv artifacts/checkpoint_metrics.csv \
  --out-json artifacts/best_checkpoint_selection.json \
  --out-csv artifacts/tts_eval_summary.csv
```

## 6) Package best checkpoint for local integration

```bash
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

