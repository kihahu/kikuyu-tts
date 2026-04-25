# Fine-Tune MMS ASR For Kikuyu

This repo now includes a direct MMS ASR path for Kikuyu using [`facebook/mms-1b-all`](https://huggingface.co/facebook/mms-1b-all) with the Kikuyu language adapter `ki`.

## Why This Path

- MMS ASR explicitly supports `ki`, so this is a cleaner fit than Whisper for Kikuyu ASR.
- The current public `WaxalNLP` release still exposes `kik_tts`, not `kik_asr`, but the `kik_tts` split is still usable for paired `audio` + `text` ASR fine-tuning.
- The main tradeoff is data quality and diversity: `kik_tts` is small and narrow compared to a real multi-speaker ASR corpus.

## Files Added

- training script: `scripts/train_mms_asr_kik.py`
- default config: `configs/train_mms_asr_kik.yaml`
- output dir: `artifacts/mms_asr_kik`

## Run It

```bash
python scripts/train_mms_asr_kik.py \
  --config configs/train_mms_asr_kik.yaml
```

The default config:

- loads `facebook/mms-1b-all`
- sets `target_lang: ki`
- trains on `google/WaxalNLP`, config `kik_tts`
- combines `train + validation` for training
- evaluates on `test`
- filters clips outside `0.5s` to `30s`

## What It Writes

- `artifacts/mms_asr_kik/dataset_summary.json`
- Trainer checkpoints under `artifacts/mms_asr_kik/checkpoint-*`
- final processor and model files under `artifacts/mms_asr_kik/`
- eval metrics via the Hugging Face Trainer save hooks

## Notes

- The script uses `Wav2Vec2ForCTC` because MMS ASR is CTC-based.
- It freezes the feature encoder first, which is a safer default for a small Kikuyu dataset.
- `wer` is the main model-selection metric, and `cer` is also logged.
- MMS is licensed `CC-BY-NC-4.0`, so this path is not suitable for commercial use without resolving the licensing constraint.

## Practical Next Step

This is the right bootstrap run. If the results are promising, the next quality jump will come from adding real multi-speaker Kikuyu ASR audio, not from swapping model families again.
