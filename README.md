# kikuyu-tts

English-to-Kikuyu translation and speech synthesis pipeline with an MMS-first TTS workflow.

## Current TTS Paths

- Preferred: MMS base-model selection and export via `configs/finetune_mms_tts_kik.yaml` and `scripts/prepare_mms_tts_finetune.py`
- Secondary: scratch Coqui VITS training via `scripts/colab_train_vits_scratch.py`

## Recommended Workflow

1. Prepare Waxal manifests with `scripts/prepare_waxal_kik_tts.py`. This now writes both speaker-disjoint manifests and canonical single-speaker manifests for the dominant Waxal speaker.
2. Run the MMS workflow script to validate, benchmark, rank, and export a selected open-weight base:

```bash
python scripts/prepare_mms_tts_finetune.py \
  --config configs/finetune_mms_tts_kik.yaml
```

3. Use the exported Hugging Face-format checkpoint with the existing MMS inference path in `src/kikuyu_tts/mlx_fork.py`.

## Base Model Ranking

Default shortlist:

1. `facebook/mms-tts-kik`
2. `facebook/mms-tts-swh`
3. `facebook/mms-tts-kin`
4. `facebook/mms-tts-lug`

## Important Constraint

The current Hugging Face `transformers` `VitsModel` supports inference and export, but not gradient-based training. The MMS workflow implemented here prepares and selects the best base model for Waxal and exports it in Hugging Face format; the only in-repo trainable path today remains the Coqui scratch-VITS workflow.
