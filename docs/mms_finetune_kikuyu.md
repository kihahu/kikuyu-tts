# MMS Kikuyu TTS Workflow (Waxal `kik_tts`)

This is the preferred repo workflow for Kikuyu-capable open-weight TTS models. The repo now has two MMS paths:

- real full-checkpoint continuation with the official VITS training stack
- Hugging Face-format baseline selection/evaluation for inference-only checkpoints

## Real Fine-Tuning Path

Prepare the public Waxal Kikuyu TTS data first:

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

This writes:

- `data/waxal_kik_tts/filelists/train_single_speaker.txt`
- `data/waxal_kik_tts/filelists/dev_single_speaker.txt`
- `data/waxal_kik_tts/speaker_map_single_speaker.json`
- fairseq-style `.tsv/.txt/.uid/.spk/.lang` bundles for analysis

Bootstrap the MMS/VITS continuation run:

```bash
python scripts/bootstrap_mms_kikuyu_tts_finetune.py \
  --config configs/train_mms_tts_kik_waxal.yaml \
  --download-checkpoint
```

That downloads the full `facebook/mms-tts-kik` checkpoint archive, patches its config for Waxal, and writes:

- `artifacts/mms_tts_kik_waxal_finetune/generated_config/mms_kik_waxal_single_speaker.json`
- `artifacts/mms_tts_kik_waxal_finetune/launch_finetune.sh`
- `artifacts/mms_tts_kik_waxal_finetune/bootstrap_summary.json`

On HF Jobs, run the wrapper from a cloned repo with `HF_TOKEN` passed as a secret:

```bash
hf jobs run --detach --flavor a10g-large --timeout 8h --secrets HF_TOKEN \
  python:3.10 \
  'git clone https://github.com/kihahu/kikuyu-tts.git /workspace/kikuyu-tts && bash /workspace/kikuyu-tts/scripts/hf_jobs_train_mms_tts_kik_waxal.sh /workspace/kikuyu-tts'
```

For the first paid validation run, cap training with environment overrides:

```bash
hf jobs run --detach --flavor a10g-large --timeout 1h --secrets HF_TOKEN \
  --env KIK_TTS_EPOCHS=1 --env KIK_TTS_EVAL_INTERVAL=10 --env KIK_TTS_LOG_INTERVAL=1 \
  python:3.10 \
  'git clone https://github.com/kihahu/kikuyu-tts.git /workspace/kikuyu-tts && bash /workspace/kikuyu-tts/scripts/hf_jobs_train_mms_tts_kik_waxal.sh /workspace/kikuyu-tts'
```

The wrapper uploads in-progress and final artifacts to `kihahu/mms-tts-kik-waxal-v1` by default. Change `hub.repo_id` in `configs/train_mms_tts_kik_waxal.yaml` before launching if you want a different destination.

## Baseline Selection Path

`scripts/prepare_mms_tts_finetune.py` does four things:

1. reads the Waxal single-speaker manifests from `configs/finetune_mms_tts_kik.yaml`
2. validates that each MMS candidate can be loaded through `transformers`
3. benchmarks the candidates on fixed Kikuyu prompts
4. exports the selected base model in Hugging Face format for downstream inference

## Run It

Prepare the dataset first if you have not already:

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

Then run the MMS workflow:

```bash
python scripts/prepare_mms_tts_finetune.py \
  --config configs/finetune_mms_tts_kik.yaml
```

Outputs:

- `artifacts/mms_tts_selected_base/`
- `artifacts/mms_tts_candidate_report.csv`
- `artifacts/mms_tts_selection_report.json`

## Which Base Should I Pick?

| Base | Status | Fit For Kikuyu | License | Notes |
| --- | --- | --- | --- | --- |
| `facebook/mms-tts-kik` | Preferred | Best | CC-BY-NC-4.0 | Default base and first candidate. |
| `facebook/mms-tts-swh` | Fallback | Medium | CC-BY-NC-4.0 | Nearby regional language fallback. |
| `facebook/mms-tts-kin` | Fallback | Medium | CC-BY-NC-4.0 | Useful A/B warm-start candidate. |
| `facebook/mms-tts-lug` | Fallback | Medium | CC-BY-NC-4.0 | Last MMS fallback in the shortlist. |
| `coqui/XTTS-v2` | Not default | Low | CPML | Multilingual and cloning-oriented, but Kikuyu is not an officially supported language. |
| Scratch Coqui VITS | Secondary | Variable | Depends on training setup | No base weights, but highest training risk and longest path to quality. |

## Selection Strategies

- `primary_only`: use `facebook/mms-tts-kik`, benchmark it, and export it.
- `fallback_on_failure`: move down the shortlist only if the earlier candidate fails the quality gate.
- `benchmark_all`: benchmark all shortlist candidates and export the strongest one.

## Important Limitation

Hugging Face `transformers` currently exposes MMS VITS checkpoints for inference and export, but not for training. If you pass labels into `VitsModel`, it raises `NotImplementedError`.

That means:

- use `scripts/eval_tts.py` and `scripts/synthesize_mms_tts_kik.py` for Hugging Face-format inference checkpoints
- use `scripts/bootstrap_mms_kikuyu_tts_finetune.py` plus the generated VITS launcher for real gradient-based MMS continuation
- scratch Coqui VITS remains the fallback if the MMS continuation stack is unstable

## Evaluation

Generate baseline or candidate samples:

```bash
python scripts/eval_tts.py \
  --prompts data/eval/kikuyu_prompts.txt \
  --models facebook/mms-tts-kik gateremark/kikuyu-tts-v1 BrianMwangi/African-Kikuyu-TTS \
  --output-dir artifacts/tts_eval
```

Outputs:

- `artifacts/tts_eval/details.csv`
- `artifacts/tts_eval/summary.csv`
- `artifacts/tts_eval/manual_scores.csv`
- one WAV directory per model

By default the evaluator also transcribes generated audio with `kihahu/mms-asr-kik-waxal-ctc` and reports round-trip CER/WER. Use `--skip-asr` when you only want synthesis/audio checks.
