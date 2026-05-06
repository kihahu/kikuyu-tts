# Prepare `Anv-ke/kikuyu` For MMS Kikuyu Fine-Tuning

This workflow prepares the Hugging Face dataset [`Anv-ke/kikuyu`](https://huggingface.co/datasets/Anv-ke/kikuyu) for MMS Kikuyu TTS fine-tuning against the full `facebook/mms-tts-kik` checkpoint.

## What The Script Produces

`scripts/prepare_anv_kikuyu_mms_tts.py` downloads and filters the dataset, normalizes text, resamples audio to 16 kHz, and writes:

- `data/anv_kikuyu_mms_tts/clips/...`
- `data/anv_kikuyu_mms_tts/manifests/{train,dev,test}.jsonl`
- `data/anv_kikuyu_mms_tts/manifests/{train,dev,test}.tsv`
- `data/anv_kikuyu_mms_tts/manifests/{train,dev,test}.txt`
- `data/anv_kikuyu_mms_tts/manifests/{train,dev,test}.uid`
- `data/anv_kikuyu_mms_tts/manifests/{train,dev,test}.spk`
- `data/anv_kikuyu_mms_tts/manifests/{train,dev,test}.lang`
- `data/anv_kikuyu_mms_tts/filelists/{train,dev,test}.txt`

The `.tsv/.txt/.uid/.spk/.lang` bundle is the fairseq-style manifest set. The filelists are a practical fallback for VITS-style training wrappers.

## Default Recipe

The default script configuration is intentionally conservative:

- dataset: `Anv-ke/kikuyu`
- source splits: `train`, `validation`, `test`
- output splits: `train`, `dev`, `test`
- text: scripted rows from `actualSentence` on older exports or `transcription` on the current export
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

## Cheap Row Probe

Before spending GPU time, prove that the gated ANV dataset can yield at least one usable audio/text row:

```bash
python scripts/prepare_anv_kikuyu_mms_tts.py \
  --dataset-name Anv-ke/kikuyu \
  --probe-one-row \
  --probe-split train \
  --probe-max-rows 200
```

This streams only the requested split, keeps `Audio(decode=False)`, prints the first row with usable text and audio payload metadata, and exits before writing clips or manifests. To also prove `ffmpeg` can decode that row without preparing the whole dataset, add `--probe-decode-audio`.

For Hugging Face Jobs, run the same gate through the training wrapper:

```bash
hf jobs run --detach --flavor cpu-basic --timeout 30m --secrets HF_TOKEN \
  --env ANV_SMOKE_ONLY=1 \
  --env ANV_PROBE_SPLIT=train \
  --env ANV_PROBE_MAX_ROWS=200 \
  python:3.10 \
  'git clone https://github.com/kihahu/kikuyu-tts.git /workspace/kikuyu-tts && bash /workspace/kikuyu-tts/scripts/hf_jobs_train_mms_tts_kik_anv.sh /workspace/kikuyu-tts'
```

Only after this succeeds should you run a decode smoke with `ANV_SMOKE_DECODE_AUDIO=1`, then a bounded prep-only job, and only then a GPU training job.

Bounded prep-only job:

```bash
hf jobs run --detach --flavor cpu-basic --timeout 30m --secrets HF_TOKEN \
  --env ANV_PREP_ONLY=1 \
  --env ANV_STREAMING=1 \
  --env ANV_STREAM_RETRIES=5 \
  --env ANV_MAX_ROWS_PER_SPLIT=1 \
  python:3.10 \
  'git clone https://github.com/kihahu/kikuyu-tts.git /workspace/kikuyu-tts && bash /workspace/kikuyu-tts/scripts/hf_jobs_train_mms_tts_kik_anv.sh /workspace/kikuyu-tts'
```

Bootstrap-only checkpoint/resume gate:

```bash
hf jobs run --detach --flavor cpu-upgrade --timeout 2h --secrets HF_TOKEN \
  --env ANV_BOOTSTRAP_ONLY=1 \
  --env ANV_STREAMING=1 \
  --env ANV_STREAM_RETRIES=8 \
  --env ANV_MAX_ROWS_PER_SPLIT=1 \
  python:3.10 \
  'git clone https://github.com/kihahu/kikuyu-tts.git /workspace/kikuyu-tts && bash /workspace/kikuyu-tts/scripts/hf_jobs_train_mms_tts_kik_anv.sh /workspace/kikuyu-tts'
```

For a training-start smoke when the token cannot create or write the configured Hub repo, set `ANV_DISABLE_HUB_UPLOAD=1`. That skips repo creation and artifact sync but still runs local prep, bootstrap, and VITS training.

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

## Latest ANV Continuation Checkpoint

The first persisted ANV continuation run used 5,000 accepted rows per split in `single_speaker` mode, resumed from `kihahu/mms-tts-kik-waxal-v1` step `77100`, and uploaded artifacts to `kihahu/mms-tts-kik-waxal-anv-v1`.

Verified latest generator checkpoint:

```text
mms_vits_finetune/vits/logs/mms_kik_waxal_anv_single_speaker/G_4381800.pth
```

Matching discriminator checkpoint:

```text
mms_vits_finetune/vits/logs/mms_kik_waxal_anv_single_speaker/D_4381800.pth
```

To continue from this ANV checkpoint instead of restarting from the Waxal checkpoint, pass:

```bash
--env KIK_TTS_RESUME_REPO_ID=kihahu/mms-tts-kik-waxal-anv-v1 \
--env KIK_TTS_RESUME_RUN_NAME=mms_kik_waxal_anv_single_speaker \
--env KIK_TTS_RESUME_STEP=4381800
```

The checkpoint was smoke-tested locally with:

```bash
python scripts/synthesize_mms_vits_checkpoint.py \
  --repo-id kihahu/mms-tts-kik-waxal-anv-v1 \
  --checkpoint mms_vits_finetune/vits/logs/mms_kik_waxal_anv_single_speaker/G_4381800.pth \
  --config mms_vits_finetune/vits/logs/mms_kik_waxal_anv_single_speaker/config.json \
  --vocab mms_vits_finetune/vits/logs/mms_kik_waxal_anv_single_speaker/vocab.txt \
  --text 'Ni wega gukwona umuthi.' \
  --output artifacts/tts_eval/anv_g4381800_smoke.wav \
  --device cpu
```

## Notes

- The current ANV dataset splits are `train`, `validation`, and `test`; the prep maps them to `train`, `dev`, and `test` for VITS/fairseq compatibility.
- If the first full MMS run is unstable, keep the same prep but rerun with `--speaker-mode dominant_only` to reduce speaker variance before debugging optimizer or checkpoint issues.
