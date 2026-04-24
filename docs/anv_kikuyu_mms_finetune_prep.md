# Prepare `Anv-ke/kikuyu` For MMS Kikuyu Fine-Tuning

This workflow prepares the Hugging Face dataset [`Anv-ke/kikuyu`](https://huggingface.co/datasets/Anv-ke/kikuyu) for MMS Kikuyu TTS fine-tuning against the full `facebook/mms-tts-kik` checkpoint.

## What The Script Produces

`scripts/prepare_anv_kikuyu_mms_tts.py` downloads and filters the dataset, normalizes text, resamples audio to 16 kHz, and writes:

- `data/anv_kikuyu_mms_tts/clips/...`
- `data/anv_kikuyu_mms_tts/manifests/{train,dev,dev_test}.jsonl`
- `data/anv_kikuyu_mms_tts/manifests/{train,dev,dev_test}.tsv`
- `data/anv_kikuyu_mms_tts/manifests/{train,dev,dev_test}.txt`
- `data/anv_kikuyu_mms_tts/manifests/{train,dev,dev_test}.uid`
- `data/anv_kikuyu_mms_tts/manifests/{train,dev,dev_test}.spk`
- `data/anv_kikuyu_mms_tts/manifests/{train,dev,dev_test}.lang`
- `data/anv_kikuyu_mms_tts/filelists/{train,dev,dev_test}.txt`

The `.tsv/.txt/.uid/.spk/.lang` bundle is the fairseq-style manifest set. The filelists are a practical fallback for VITS-style training wrappers.

## Default Recipe

The default script configuration is intentionally conservative:

- dataset: `Anv-ke/kikuyu`
- splits: `train`, `dev`, `dev_test`
- text: scripted only, from `actualSentence`
- sample rate: `16000`
- duration filter: `1s` to `15s`
- speakers: all speakers retained

That is the right first prep for a real MMS continuation run. It keeps the official split boundaries and avoids mixing in unscripted transcripts until the scripted path is stable.

## Run It

```bash
python scripts/prepare_anv_kikuyu_mms_tts.py \
  --dataset-name Anv-ke/kikuyu \
  --output-dir data/anv_kikuyu_mms_tts \
  --target-sample-rate 16000 \
  --min-duration-sec 1.0 \
  --max-duration-sec 15.0
```

Optional flags:

- `--include-unscripted`
- `--speaker-mode dominant_only`
- `--max-rows-per-split 5000`

## Next Step: Full MMS Checkpoint

For actual fine-tuning, do not use only the Hugging Face inference checkpoint. Use the full MMS Kikuyu checkpoint instead:

- base model: [`facebook/mms-tts-kik`](https://huggingface.co/facebook/mms-tts-kik)
- full checkpoint archive: `https://dl.fbaipublicfiles.com/mms/tts/full_model/kik.tar.gz`

The MMS README is explicit that full-model archives include the discriminator and optimizer state needed for training continuation:

- [MMS README](https://github.com/facebookresearch/fairseq/blob/main/examples/mms/README.md)

## Bootstrap The Fine-Tune Run

After dataset prep, stage the full MMS checkpoint and generate the VITS training launcher:

```bash
python scripts/bootstrap_anv_kikuyu_mms_finetune.py \
  --prepared-dir data/anv_kikuyu_mms_tts \
  --output-dir artifacts/anv_kikuyu_mms_finetune \
  --download-checkpoint
```

This writes:

- `artifacts/anv_kikuyu_mms_finetune/base_checkpoint/kik/...`
- `artifacts/anv_kikuyu_mms_finetune/generated_config/mms_kik_anv_ms.json`
- `artifacts/anv_kikuyu_mms_finetune/launch_finetune.sh`

Then run:

```bash
bash artifacts/anv_kikuyu_mms_finetune/launch_finetune.sh
```

The launcher clones the official VITS repo if needed, builds monotonic alignment, stages the MMS full checkpoint into the VITS `logs/<run_name>` directory, and starts `train_ms.py`.

## Notes

- `dev_test` is kept because it exists in the ANV public dataset and is useful as a held-out evaluation split before touching the locked `test` set.
- If the first full MMS run is unstable, keep the same prep but rerun with `--speaker-mode dominant_only` to reduce speaker variance before debugging optimizer or checkpoint issues.
