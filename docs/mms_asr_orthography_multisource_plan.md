# Plan: MMS Kikuyu ASR — orthography alignment and multi-source training

This document turns the orthography / `<unk>` discussion into an execution plan. It applies to fine-tuning `facebook/mms-1b-all` with `target_lang: kik` using `scripts/train_mms_asr_kik.py` and `configs/train_mms_asr_kik.yaml`.

## Problem statement

- The MMS `kik` CTC head uses a **fixed tokenizer vocabulary** (on the order of a few dozen graphemes). The model can only emit symbols that exist in that vocab; anything else tends to decode as `<unk>` or wrong splits.
- Training today uses **Waxal**-style paired data (`google/WaxalNLP`, `kik_tts`). Audio and transcripts from other sources (e.g. GRN) differ in **recording domain** and often in **spelling conventions**.
- **Training does not invent new output letters.** To “understand” another orthography, you either **normalize labels into the MMS alphabet** or undertake **vocab expansion + head resize** (out of scope here; prefer normalization).

## Guiding principle

**One canonical grapheme inventory for labels:** every supervised transcript character must appear in `AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="kik").tokenizer.get_vocab()` after normalization. Target **zero** out-of-vocabulary characters in training labels.

## Phase 1 — Inventory and normalization spec

1. **Dump the allowed character set** (once):

   ```python
   from transformers import AutoProcessor
   p = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="kik")
   allowed = sorted(p.tokenizer.get_vocab().keys())
   ```

2. **Collect “source” orthography samples** from all corpora you care about (Waxal, GRN, hand transcripts): a few hundred lines or a frequency table of Unicode code points.

3. **Define an explicit mapping** (document in code as data, not ad hoc edits):

   - Use **NFKC** normalization and lowercase for labels (matches current `normalize_kikuyu_text` in `scripts/train_mms_asr_kik.py`).
   - Map unsupported letters to supported digraphs/trigraphs (example pattern only—verify against Waxal): e.g. precomposed **ŋ** → **`ng`** if `ŋ` is not in vocab.
   - Unify apostrophe / glottal variants to **one** character that exists in vocab (`'` vs `ʼ` etc.).
   - Strip or replace punctuation and symbols **not** in vocab (periods, commas, digits in narration if they are not in your target alphabet).
   - Collapse whitespace; apply the same punctuation replacements already in `PUNCT_REPLACEMENTS` where relevant.

4. **Implement** the mapping in one shared normalization source of truth (for example `scripts/normalize_kikuyu.py`, imported by training and data-prep scripts). Keep behavior **identical** on Waxal unless you intentionally fix bugs.

5. **Retire duplicated normalizers** so rules do not drift between pipelines:
   - `scripts/train_mms_asr_kik.py`
   - `scripts/prepare_waxal_kik_tts.py`
   - `scripts/prepare_anv_kikuyu_mms_tts.py`
   - `scripts/build_manifests.py` (already imports `normalize_kikuyu_text`; keep it on the shared module path)

6. **Validate** on a held-out slice of each source:

   - For every line, assert `all(c in vocab for c in text)` after normalization using tokenizer keys from `processor.tokenizer.get_vocab()`. If you want a stricter linguistic inventory, maintain and validate against a separate explicit `allowed_graphemes` set.
   - The training script already computes `oov_char_count` per example in `prepare_datasets`; use `dataset_summary.json` and per-row stats to catch regressions.

## Phase 2 — New paired data (e.g. GRN)

1. **Audio:** segment long files into clips within the trainer’s duration bounds (default **0.5–30 s** in `configs/train_mms_asr_kik.yaml`). Resample to **16 kHz mono** if you build files offline; the trainer also casts via `datasets.Audio` to the processor rate.

2. **Text:** obtain or create time-aligned transcripts; apply **the same** normalization as Phase 1 so labels are strictly in-vocab.

3. **Format:** produce a `datasets`-compatible table: columns at least `audio`, `text` (or your chosen `text_column` name). Options:

   - Push a **Hugging Face Dataset** repo and load by name; or
   - Store **JSONL/Parquet** locally and load with `load_dataset("json", data_files=...)`.

4. **Quality gate:** discard or fix clips with empty text, wrong language, or persistent OOV after normalization.

## Phase 3 — Combine Waxal + new data in training

**Current code** loads a single Hub dataset via `load_training_dataset` in `scripts/train_mms_asr_kik.py` (`dataset.name`, `dataset.config`). Pick one:

- **Option A (no code change):** publish one **merged** HF dataset (Waxal train/val splits concatenated with your new rows under the same schema) and point `configs/train_mms_asr_kik.yaml` `dataset.name` / `config` at it.

- **Option B (small code change):** extend `load_training_dataset` to accept a list (e.g. `dataset.sources: [{name, config, splits}, ...]`) and `concatenate_datasets` internally, with optional per-source row caps or oversampling.

**Mixing strategy:**

- If the new corpus is small, **oversample** it or use a higher sampling weight so the optimizer sees it often enough (for example with `datasets.interleave_datasets(..., probabilities=[...])`, or by duplicating rows in the smaller source before merge).
- Keep Waxal in the mix to reduce **catastrophic forgetting** on the original domain.

## Phase 4 — Continued fine-tuning from your checkpoint

1. **Start from** `kihahu/mms-asr-kik-finetuned` (or a local `artifacts/mms_asr_kik` export), not from bare `facebook/mms-1b-all`, unless you have a reason to retrain from scratch.

2. **Hyperparameters:** use a **lower learning rate** and **enough steps** for the new data volume; enable `resume_from_checkpoint` in YAML when resuming a partial run.

3. **Eval:** hold out a **GRN-only** eval split and a **Waxal-only** eval split so you can see domain tradeoffs (WER/CER via existing `compute_metrics`). Make this concrete with one of:
   - Two eval runs (separate YAMLs or CLI overrides, each pointing at a different eval split/source).
   - One combined eval dataset with a `source` column plus a small code change to report per-source metrics.

4. **Hub:** use a **new** `hub_model_id` (or revision) so you can compare checkpoints without overwriting production.

## Phase 5 — Inference and evaluation alignment

1. **Inference:** keep using `scripts/infer_mms_asr_kik.py` for one-off transcription output.
2. **Evaluation alignment:** normalize **reference** text with the same normalizer wherever WER/CER is computed (training eval loop or a dedicated offline eval script), so scores are comparable across sources.

3. **Display:** if you need GRN-style spelling in a UI, maintain a **display map** (inverse transforms) only where it is well-defined; do not feed unnormalized text as CTC labels.

## Phase 6 — Verification checklist

- [ ] Allowed vocab enumerated and checked into review (or script output archived).
- [ ] Normalization table reviewed by a Kikuyu reader for your target tradition(s).
- [ ] Post-normalization OOV rate **~0%** on train and eval.
- [ ] Eval WER on Waxal **not** severely regressed vs baseline.
- [ ] Eval WER on new source **improved** vs baseline zero-shot on that source.
- [ ] Normalization mapping has table-driven tests (input → expected output) and passes.
- [ ] Waxal regression check run (character histogram/OOV count does not regress unexpectedly after mapping updates).
- [ ] Dataset provenance recorded (dataset revision/hash, split boundaries, and speaker-overlap policy).
- [ ] Licensing/consent requirements verified for external corpora (e.g. GRN) before publication or Hub push.
- [ ] Spot-listen failed / high-loss utterances for alignment or label errors.

## Out of scope (explicit)

- **Adding new Unicode letters to MMS** without tokenizer + CTC head resize and careful initialization.
- Replacing CTC with a full seq2seq model (different stack).

## File references in this repo

| Piece | Location |
|--------|----------|
| Label normalization | `normalize_kikuyu_text` in `scripts/train_mms_asr_kik.py` |
| OOV accounting | `prepare_datasets` (`oov_char_count`) in `scripts/train_mms_asr_kik.py` |
| Default train config | `configs/train_mms_asr_kik.yaml` |
| ASR inference | `scripts/infer_mms_asr_kik.py` |
| ASR training notes | `docs/mms_asr_finetune_kikuyu.md` |

## Summary

**Normalize all transcripts into the MMS `kik` character set**, add paired in-domain audio with those labels, **mix** with Waxal, and **continue fine-tuning** from your existing checkpoint with monitored eval on both domains. That is the supported path to “understand” GRN-style audio without fighting the fixed CTC alphabet.
