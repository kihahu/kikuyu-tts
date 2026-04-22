# MLX MMS-TTS Port Plan (facebook/mms-tts-kik)

## Goal

Port `facebook/mms-tts-kik` (Transformers VITS) to a true MLX-compatible TTS backend so it can be quantized via `mlx_audio.convert` and used with `mlx_audio.tts.generate`.

## Why this is needed

- `facebook/mms-tts-kik` is `VitsModel` (`model_type: vits`) and not currently supported by upstream `mlx-audio` TTS modules.
- Existing conversion fails due to architecture mismatch and unresolved parameter mapping.
- Current working path uses PyTorch VITS fallback, not MLX-native quantized inference.

## File-by-file implementation checklist

1. Create model module:
   - `mlx_audio/tts/models/mms_tts/__init__.py`
   - `mlx_audio/tts/models/mms_tts/config.py`
   - `mlx_audio/tts/models/mms_tts/layers.py`
   - `mlx_audio/tts/models/mms_tts/mms_tts.py`

2. Register detection hints and loader routing:
   - Add `DETECTION_HINTS` so `VitsModel` + MMS path resolves to `mms_tts`.
   - Update `mlx_audio/tts/utils.py` model dispatch for `vits` / `mms-tts`.

3. Implement `ModelConfig` for VITS fields:
   - `hidden_size`, `ffn_dim`, `num_hidden_layers`, `num_attention_heads`
   - `flow_size`, `duration_predictor_*`, `prior_encoder_*`, `posterior_encoder_*`
   - `upsample_*`, `resblock_*`, `sampling_rate`, `speaking_rate`, `noise_scale*`

4. Implement MLX model components:
   - Text encoder (transformer blocks)
   - Duration predictor (including flow path)
   - Prior/posterior encoders
   - HiFi-GAN-like decoder and upsampler

5. Implement `sanitize(weights)`:
   - Map Hugging Face VITS parameter names to MLX model tree.
   - Handle shape/layout conversions.
   - Report matched/missing/unexpected keys clearly.

6. Quantization readiness:
   - Ensure quantizable modules expose compatible interfaces.
   - Add/adjust `model_quant_predicate`.
   - Validate 4-bit and 8-bit modes.

7. Add tests:
   - Detection test
   - Load/convert test
   - Generation smoke test
   - Quantized generation test

8. Document usage:
   - Conversion command
   - Quantization command
   - Inference command
   - Known limitations

## Starter skeletons

### `mlx_audio/tts/models/mms_tts/__init__.py`

```python
from .mms_tts import Model, ModelConfig, DETECTION_HINTS

__all__ = ["Model", "ModelConfig", "DETECTION_HINTS"]
```

### `mlx_audio/tts/models/mms_tts/config.py`

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_type: str = "vits"
    vocab_size: int = 37
    hidden_size: int = 192
    ffn_dim: int = 768
    num_hidden_layers: int = 6
    num_attention_heads: int = 2
    flow_size: int = 192
    spectrogram_bins: int = 513
    sampling_rate: int = 16000
    duration_predictor_filter_channels: int = 256
    duration_predictor_kernel_size: int = 3
    duration_predictor_num_flows: int = 4
    duration_predictor_flow_bins: int = 10
    prior_encoder_num_flows: int = 4
    prior_encoder_num_wavenet_layers: int = 4
    posterior_encoder_num_wavenet_layers: int = 16
    upsample_initial_channel: int = 512
    upsample_rates: tuple[int, ...] = (8, 8, 2, 2)
    upsample_kernel_sizes: tuple[int, ...] = (16, 16, 4, 4)
    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    noise_scale: float = 0.667
    noise_scale_duration: float = 0.8
    speaking_rate: float = 1.0

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        allowed = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**allowed)
```

### `mlx_audio/tts/models/mms_tts/mms_tts.py` (scaffold)

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig


DETECTION_HINTS = {
    "architectures": ["VitsModel"],
    "path_patterns": ["mms-tts", "vits"],
    "config_keys": [
        "flow_size",
        "upsample_rates",
        "resblock_kernel_sizes",
        "duration_predictor_num_flows",
        "posterior_encoder_num_wavenet_layers",
    ],
}


@dataclass
class GenerationResult:
    audio: mx.array
    sample_rate: int

    def save(self, path: str) -> None:
        from mlx_audio.utils import save_audio
        save_audio(self.audio, path, self.sample_rate)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.text_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.text_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.audio_head = nn.Linear(config.hidden_size, 256)

    @property
    def sampling_rate(self) -> int:
        return self.config.sampling_rate

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        return weights

    def model_quant_predicate(self, path: str, module: nn.Module) -> bool:
        return isinstance(module, nn.Linear)

    def _forward_tokens(self, input_ids: mx.array) -> mx.array:
        x = self.text_embed(input_ids)
        x = self.text_proj(x)
        return x

    def _decode_stub(self, hidden: mx.array) -> mx.array:
        x = self.audio_head(hidden)
        x = mx.mean(x, axis=-1)
        return mx.repeat(x, repeats=200, axis=1)

    def generate(self, text: str, tokenizer=None, **kwargs) -> Iterable[GenerationResult]:
        if tokenizer is None:
            raise ValueError("Tokenizer is required for MMS-TTS generation.")
        tokens = tokenizer(text, return_tensors="np")
        input_ids = mx.array(tokens["input_ids"])
        hidden = self._forward_tokens(input_ids)
        wav = self._decode_stub(hidden)[0]
        yield GenerationResult(audio=wav, sample_rate=self.sampling_rate)
```

## Validation sequence

1. Convert fp16:
   - `python -m mlx_audio.convert --hf-path facebook/mms-tts-kik --mlx-path ./mms-kik-mlx-fp16 --dtype float16`

2. Generate:
   - `mlx_audio.tts.generate --model ./mms-kik-mlx-fp16 --text "..." --output_path ./out`

3. Quantize:
   - `python -m mlx_audio.convert --hf-path facebook/mms-tts-kik --mlx-path ./mms-kik-mlx-4bit --quantize --q-bits 4`

4. Re-test quality, latency, memory on Apple Silicon.
