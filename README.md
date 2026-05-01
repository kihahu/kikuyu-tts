# kikuyu-tts

English-to-Kikuyu translation and speech synthesis pipeline with an MMS-first TTS workflow.

## Current TTS Paths

- Preferred: MMS base-model selection and export via `configs/finetune_mms_tts_kik.yaml` and `scripts/prepare_mms_tts_finetune.py`
- Secondary: scratch Coqui VITS training via `scripts/colab_train_vits_scratch.py`

## Current ASR Path

- Preferred: MMS ASR fine-tuning via `configs/train_mms_asr_kik.yaml` and `scripts/train_mms_asr_kik.py`

Run it with:

```bash
python scripts/train_mms_asr_kik.py \
  --config configs/train_mms_asr_kik.yaml
```

This trains `facebook/mms-1b-all` with the Kikuyu MMS head (`kik`) on the paired `audio` + `text` data from `google/WaxalNLP`, config `kik_tts`. Metrics and hyperparameters are logged to **MLflow** by default (`report_to: mlflow`); see `docs/mms_asr_finetune_kikuyu.md`.

## Recommended Workflow

1. Prepare Waxal manifests with `scripts/prepare_waxal_kik_tts.py`. This now writes both speaker-disjoint manifests and canonical single-speaker manifests for the dominant Waxal speaker.
2. Run the MMS workflow script to validate, benchmark, rank, and export a selected open-weight base:

```bash
python scripts/prepare_mms_tts_finetune.py \
  --config configs/finetune_mms_tts_kik.yaml
```

3. Use the exported Hugging Face-format checkpoint with the existing MMS inference path in `src/kikuyu_tts/mlx_fork.py`.

## MMS Fine-Tune Prep

If you want to fine-tune the actual Kikuyu MMS checkpoint on `Anv-ke/kikuyu`, start with:

```bash
python scripts/prepare_anv_kikuyu_mms_tts.py \
  --dataset-name Anv-ke/kikuyu \
  --output-dir data/anv_kikuyu_mms_tts \
  --target-sample-rate 16000 \
  --min-duration-sec 1.0 \
  --max-duration-sec 15.0
```

That produces fairseq-style manifests and normalized 16 kHz audio for the full MMS checkpoint route. See `docs/anv_kikuyu_mms_finetune_prep.md`.

To bootstrap the real MMS continuation run after prep:

```bash
python scripts/bootstrap_anv_kikuyu_mms_finetune.py \
  --prepared-dir data/anv_kikuyu_mms_tts \
  --output-dir artifacts/anv_kikuyu_mms_finetune \
  --download-checkpoint
```

## Base Model Ranking

Default shortlist:

1. `facebook/mms-tts-kik`
2. `facebook/mms-tts-swh`
3. `facebook/mms-tts-kin`
4. `facebook/mms-tts-lug`

## Important Constraint

The current Hugging Face `transformers` `VitsModel` supports inference and export, but not gradient-based training. The MMS workflow implemented here prepares and selects the best base model for Waxal and exports it in Hugging Face format; the only in-repo trainable path today remains the Coqui scratch-VITS workflow.
