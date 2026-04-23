# MMS-First Kikuyu Workflow (Waxal `kik_tts`)

This is the preferred repo workflow for Kikuyu-capable open-weight TTS models.

## What It Does

`scripts/prepare_mms_tts_finetune.py` does four things:

1. reads the Waxal single-speaker manifests from `configs/finetune_mms_tts_kik.yaml`
2. validates that each MMS candidate can be loaded through `transformers`
3. benchmarks the candidates on fixed Kikuyu prompts
4. exports the selected base model in Hugging Face format for downstream inference

## Run It

Prepare the dataset first:

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

- this repo can now select, benchmark, and export the best MMS base for Waxal
- the repo still cannot fine-tune MMS weights directly with `transformers`
- scratch Coqui VITS remains the only in-repo trainable backend until a trainable MMS backend is added
