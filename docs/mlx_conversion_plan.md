# Convert Kikuyu ASR and TTS to MLX

## Summary

Convert the working Kikuyu speech stack to MLX so it can run efficiently on Apple Silicon unified memory. Target models are:

- ASR: `kihahu/mms-asr-kik-waxal-ctc`
- TTS: `kihahu/mms-tts-kik-waxal-v1`, checkpoint `mms_vits_finetune/vits/logs/mms_kik_waxal_single_speaker/G_77100.pth`

ASR is the lower-risk first conversion because local `mlx-audio` already has an MMS/Wav2Vec2 CTC backend. TTS is larger because the current `mlx-audio` MMS-TTS backend is only a scaffold, so real VITS inference must be implemented before the fine-tuned checkpoint can produce proper audio in MLX.

## Key Changes

- Add MLX conversion outputs under local model directories:
  - `models/mlx/mms-asr-kik-waxal-ctc-fp16/`
  - `models/mlx/mms-asr-kik-waxal-ctc-4bit/`
  - `models/mlx/mms-tts-kik-waxal-g77100-fp16/`
- Add ASR conversion and inference scripts:
  - Convert the HF Transformers MMS ASR safetensors/config/tokenizer into the local `mlx-audio` STT format.
  - Preserve processor/tokenizer files so CTC decoding matches the PyTorch model.
  - Add an MLX ASR smoke inference path for `audio -> Kikuyu transcript`.
- Implement the missing real MMS/VITS graph in local `mlx-audio` before converting TTS:
  - Text encoder
  - Stochastic duration predictor inference path
  - Prior flow
  - Decoder/HiFi-GAN generator
  - Correct Conv1d/ConvTranspose1d weight layout conversion
  - Raw `G_77100.pth` key mapping into MLX parameter names
- Add TTS conversion and inference scripts:
  - Input: Hub repo, checkpoint path, config, vocab.
  - Output: MLX weights/config/vocab directory.
  - Inference: `text -> wav` using the MLX backend.
- Update the existing YouTube pipeline after both backends pass parity:
  - Add `--asr-backend torch|mlx`
  - Add `--tts-backend torch|mlx`
  - Keep PyTorch as fallback until MLX parity is proven.

## Validation Plan

- ASR parity:
  - Run the same known Kikuyu audio through PyTorch ASR and MLX ASR.
  - Compare transcript text and CER/WER.
  - Confirm long audio chunking still works.
- TTS parity:
  - Generate the fixed prompt set with PyTorch `G_77100.pth` and MLX `G_77100`.
  - Check duration, RMS, silence, clipping, and waveform validity.
  - Run generated audio back through ASR for rough round-trip CER.
  - Do manual listening before accepting the MLX TTS output.
- Pipeline smoke:
  - Run YouTube/audio input through MLX ASR, feed transcript into MLX TTS, write final WAV.
  - Confirm outputs are valid 16 kHz mono WAV files.
- Performance:
  - Record load time, peak memory, synthesis/transcription time, and real-time factor for PyTorch vs MLX on the same Mac.

## Assumptions

- Start with FP16 MLX conversion. Only produce 4-bit/8-bit artifacts after FP16 parity passes.
- ASR conversion is implemented first because existing MLX support is close.
- TTS requires actual MLX VITS implementation work; the current local MMS-TTS MLX code is not enough for usable speech.
- The accepted TTS checkpoint is `G_77100.pth`, since it sounded better than earlier checkpoints.
- Artifacts stay local first; upload to Hugging Face only after quality and parity checks pass.
