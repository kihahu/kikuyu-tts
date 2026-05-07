# kikuyu-tts

English-to-Kikuyu translation and speech synthesis pipeline with an MMS-first TTS workflow.

## Current TTS Paths

- Preferred: MMS full-checkpoint continuation on Waxal via `configs/train_mms_tts_kik_waxal.yaml` and `scripts/bootstrap_mms_kikuyu_tts_finetune.py`
- Baseline: MMS base-model selection and export via `configs/finetune_mms_tts_kik.yaml` and `scripts/prepare_mms_tts_finetune.py`
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

### English To Kikuyu Audio

The first end-to-end app surface is a script. It translates English text to Kikuyu with NLLB, then synthesizes Kikuyu audio with the current best Waxal MMS/VITS checkpoint, `G_77100`.

Smoke test the Waxal audio path with known Kikuyu text:

```bash
python scripts/english_to_kikuyu_audio.py \
  --translation-backend identity \
  --text "Ni wega gukwona umuthi." \
  --output-wav artifacts/english_to_kikuyu_audio/smoke_identity.wav
```

Run English text through translation and TTS:

```bash
python scripts/english_to_kikuyu_audio.py \
  --input path/to/shakespeare_chapter.txt \
  --output-wav artifacts/english_to_kikuyu_audio/shakespeare_chapter.wav \
  --translated-output artifacts/english_to_kikuyu_audio/shakespeare_chapter.kik.txt \
  --manifest-json artifacts/english_to_kikuyu_audio/shakespeare_chapter.manifest.json \
  --translation-device cpu \
  --tts-device cpu
```

The compatibility wrapper `scripts/pipeline.py` calls the same implementation:

```bash
python scripts/pipeline.py \
  --input path/to/shakespeare_chapter.txt \
  --output-wav artifacts/english_to_kikuyu_audio/shakespeare_chapter.wav
```

Default TTS checkpoint:

```text
kihahu/mms-tts-kik-waxal-v1
mms_vits_finetune/vits/logs/mms_kik_waxal_single_speaker/G_77100.pth
```

1. Prepare Waxal manifests with `scripts/prepare_waxal_kik_tts.py`. This writes JSONL, fairseq-style bundles, and VITS filelists for both all-speaker and dominant single-speaker training.
2. For a real MMS fine-tune, bootstrap the full-checkpoint continuation run:

```bash
python scripts/bootstrap_mms_kikuyu_tts_finetune.py \
  --config configs/train_mms_tts_kik_waxal.yaml \
  --download-checkpoint
```

3. Run the generated launcher on a GPU machine, or launch the HF Jobs wrapper from a cloned repo:

```bash
bash scripts/hf_jobs_train_mms_tts_kik_waxal.sh /workspace/kikuyu-tts
```

4. Generate baseline samples and score candidates:

```bash
python scripts/eval_tts.py \
  --prompts data/eval/kikuyu_prompts.txt \
  --output-dir artifacts/tts_eval
```

5. Synthesize one sentence from any Hugging Face-format MMS/VITS model:

```bash
python scripts/synthesize_mms_tts_kik.py \
  --model facebook/mms-tts-kik \
  --text "Nĩ wega tũkĩe na ũgima na ũmwe." \
  --output artifacts/tts_eval/single.wav
```

To compare existing open Kikuyu-capable TTS checkpoints before training, run the baseline workflow:

```bash
python scripts/prepare_mms_tts_finetune.py \
  --config configs/finetune_mms_tts_kik.yaml
```

Then use the exported Hugging Face-format checkpoint with the existing MMS inference path in `src/kikuyu_tts/mlx_fork.py`.

## MMS Fine-Tune Prep

If you want to fine-tune the actual Kikuyu MMS checkpoint on `Anv-ke/kikuyu`, start with:

```bash
python scripts/prepare_anv_kikuyu_mms_tts.py \
  --dataset-name Anv-ke/kikuyu \
  --output-dir data/anv_kikuyu_mms_tts \
  --target-sample-rate 16000 \
  --min-duration-sec 1.0 \
  --max-duration-sec 15.0 \
  --orthography preserve
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

The current Hugging Face `transformers` `VitsModel` supports inference and export, but not gradient-based training. Real MMS fine-tuning uses the full MMS/VITS checkpoint route bootstrapped by `scripts/bootstrap_mms_kikuyu_tts_finetune.py`.
