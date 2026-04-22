from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import numpy as np
import soundfile as sf
import torch
from transformers import AutoTokenizer, VitsModel

from mlx_audio.tts.utils import load_model as mlx_load_model


@dataclass
class GenerationResult:
    audio: np.ndarray
    sample_rate: int

    def save(self, path: str) -> None:
        sf.write(path, self.audio, self.sample_rate)


class MmsTtsModel:
    """Fallback TTS backend for MMS checkpoints.

    This keeps project-level compatibility with mlx-audio generate semantics while
    upstream mlx-audio does not yet expose an MMS-TTS backend.
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = VitsModel.from_pretrained(model_id).to(self.device)
        self.sampling_rate = self.model.config.sampling_rate

    def generate(self, text: str, **_: object) -> Iterable[GenerationResult]:
        with torch.inference_mode():
            inputs = self.tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            output = self.model(**inputs)
            audio = output.waveform.squeeze().detach().cpu().numpy().astype(np.float32)
        yield GenerationResult(audio=audio, sample_rate=self.sampling_rate)


def load_model(model_name: str):
    lowered = model_name.lower()
    if "mms-tts" in lowered:
        return MmsTtsModel(model_name)
    return mlx_load_model(model_name)


def quantize_mx_array(arr: mx.array, bits: int = 4) -> mx.array:
    """Light helper for future MLX-native MMS backend."""
    if bits <= 0:
        raise ValueError("bits must be positive")
    levels = (1 << bits) - 1
    clipped = mx.clip(arr, -1.0, 1.0)
    scaled = (clipped + 1.0) * (levels / 2.0)
    rounded = mx.round(scaled)
    return (rounded / (levels / 2.0)) - 1.0
