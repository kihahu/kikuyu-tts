#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download


@dataclass
class VitsCheckpoint:
    commons: object
    hps: object
    net_g: torch.nn.Module
    text_to_sequence: object
    device: str


def run(command: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(command, cwd=cwd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}")


def ensure_vits_dependencies() -> None:
    missing = []
    for module, package in (
        ("Cython", "Cython"),
        ("numpy", "numpy"),
        ("unidecode", "Unidecode"),
        ("matplotlib", "matplotlib"),
        ("librosa", "librosa"),
        ("scipy", "scipy"),
    ):
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        package_list = " ".join(missing)
        raise RuntimeError(
            "Missing local dependencies required by raw MMS/VITS inference: "
            f"{', '.join(missing)}.\n"
            "Install them in this Python environment with:\n"
            f"  {sys.executable} -m pip install {package_list}"
        )


def ensure_vits_repo(path: Path) -> None:
    if not path.exists():
        run(["git", "clone", "https://github.com/jaywalnut310/vits.git", str(path)])


def patch_vits_repo(path: Path, vocab_path: Path) -> None:
    if (path / ".git").exists():
        run(
            [
                "git",
                "checkout",
                "--",
                "text/symbols.py",
                "text/__init__.py",
                "text/cleaners.py",
                "monotonic_align/setup.py",
            ],
            cwd=path,
        )

    symbols_path = path / "text" / "symbols.py"
    symbols_path.write_text(
        "\n".join(
            [
                "import os",
                f"vocab_file = os.environ.get('MMS_VOCAB_FILE', {str(vocab_path)!r})",
                "with open(vocab_file, encoding='utf-8') as f:",
                "    symbols = [line.rstrip('\\n') for line in f]",
                "SPACE_ID = symbols.index(' ') if ' ' in symbols else 0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    text_init_path = path / "text" / "__init__.py"
    text_init = text_init_path.read_text(encoding="utf-8")
    text_init = text_init.replace(
        "symbol_id = _symbol_to_id[symbol]\n    sequence += [symbol_id]",
        "symbol_id = _symbol_to_id.get(symbol)\n    if symbol_id is not None:\n      sequence += [symbol_id]",
    )
    text_init = text_init.replace(
        "sequence = [_symbol_to_id[symbol] for symbol in cleaned_text]",
        "sequence = [_symbol_to_id[symbol] for symbol in cleaned_text if symbol in _symbol_to_id]",
    )
    text_init_path.write_text(text_init, encoding="utf-8")

    cleaners_path = path / "text" / "cleaners.py"
    cleaners = cleaners_path.read_text(encoding="utf-8")
    cleaners = cleaners.replace(
        "from phonemizer import phonemize",
        "\ntry:\n  from phonemizer import phonemize\nexcept ImportError:\n  phonemize = None",
    )
    cleaners = cleaners.replace(
        "  phonemes = phonemize(text, language='en-us', backend='espeak', strip=True)",
        "  if phonemize is None:\n    raise RuntimeError('phonemizer is required for english_cleaners')\n  phonemes = phonemize(text, language='en-us', backend='espeak', strip=True)",
    )
    cleaners = cleaners.replace(
        "  phonemes = phonemize(text, language='en-us', backend='espeak', strip=True, preserve_punctuation=True, with_stress=True)",
        "  if phonemize is None:\n    raise RuntimeError('phonemizer is required for english_cleaners2')\n  phonemes = phonemize(text, language='en-us', backend='espeak', strip=True, preserve_punctuation=True, with_stress=True)",
    )
    cleaners_path.write_text(cleaners, encoding="utf-8")

    nested_align = path / "monotonic_align" / "monotonic_align"
    nested_align.mkdir(parents=True, exist_ok=True)
    (nested_align / "__init__.py").write_text("", encoding="utf-8")
    setup_path = path / "monotonic_align" / "setup.py"
    setup_path.write_text(
        "\n".join(
            [
                "from distutils.core import setup",
                "from Cython.Build import cythonize",
                "import numpy",
                "",
                "setup(",
                "  name='monotonic_align',",
                "  ext_modules=cythonize('core.pyx'),",
                "  include_dirs=[numpy.get_include()]",
                ")",
                "",
            ]
        ),
        encoding="utf-8",
    )
    run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=path / "monotonic_align")


def load_vits_modules(path: Path):
    sys.path.insert(0, str(path))
    import commons  # type: ignore
    import utils  # type: ignore
    from models import SynthesizerTrn  # type: ignore
    from text import text_to_sequence  # type: ignore

    return commons, utils, SynthesizerTrn, text_to_sequence


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def pick_device(raw: str) -> str:
    if raw == "auto":
        return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return raw


def load_checkpoint_for_synthesis(
    *,
    repo_id: str,
    checkpoint: str,
    config: str,
    vocab: str,
    vits_repo: Path,
    device: str,
) -> VitsCheckpoint:
    ensure_vits_dependencies()

    checkpoint_path = Path(hf_hub_download(repo_id, checkpoint, repo_type="model"))
    config_path = Path(hf_hub_download(repo_id, config, repo_type="model"))
    vocab_path = Path(hf_hub_download(repo_id, vocab, repo_type="model"))

    vits_repo = vits_repo.resolve()
    ensure_vits_repo(vits_repo)
    patch_vits_repo(vits_repo, vocab_path)
    commons, utils, SynthesizerTrn, text_to_sequence = load_vits_modules(vits_repo)

    hps = utils.get_hparams_from_file(str(config_path))
    net_g = SynthesizerTrn(
        len(__import__("text.symbols", fromlist=["symbols"]).symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)
    net_g.eval()
    utils.load_checkpoint(str(checkpoint_path), net_g, None)
    return VitsCheckpoint(
        commons=commons,
        hps=hps,
        net_g=net_g,
        text_to_sequence=text_to_sequence,
        device=device,
    )


def synthesize_text(
    model: VitsCheckpoint,
    text: str,
    *,
    noise_scale: float = 0.667,
    noise_scale_w: float = 0.8,
    length_scale: float = 1.0,
) -> np.ndarray:
    text = normalize_text(text)
    if not text:
        return np.zeros(0, dtype=np.float32)

    sequence = model.text_to_sequence(text, model.hps.data.text_cleaners)
    if model.hps.data.add_blank:
        sequence = model.commons.intersperse(sequence, 0)
    x = torch.LongTensor(sequence).unsqueeze(0).to(model.device)
    x_lengths = torch.LongTensor([x.size(1)]).to(model.device)

    with torch.inference_mode():
        audio = model.net_g.infer(
            x,
            x_lengths,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
        )[0][0, 0].detach().cpu().numpy()
    return np.asarray(audio, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize Kikuyu audio from a raw MMS/VITS checkpoint.")
    parser.add_argument("--repo-id", default="kihahu/mms-tts-kik-waxal-v1")
    parser.add_argument(
        "--checkpoint",
        default="mms_vits_finetune/vits/logs/mms_kik_waxal_single_speaker/G_73340.pth",
    )
    parser.add_argument(
        "--config",
        default="mms_vits_finetune/vits/logs/mms_kik_waxal_single_speaker/config.json",
    )
    parser.add_argument(
        "--vocab",
        default="mms_vits_finetune/vits/logs/mms_kik_waxal_single_speaker/vocab.txt",
    )
    parser.add_argument("--text", default="")
    parser.add_argument("--input", default="", help="UTF-8 text file to synthesize.")
    parser.add_argument("--output", default="artifacts/tts_eval/mms_vits_waxal.wav")
    parser.add_argument("--vits-repo", default="artifacts/vits_inference")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--noise-scale", type=float, default=0.667)
    parser.add_argument("--noise-scale-w", type=float, default=0.8)
    parser.add_argument("--length-scale", type=float, default=1.0)
    args = parser.parse_args()

    text = args.text
    if args.input:
        text = Path(args.input).read_text(encoding="utf-8")
    text = normalize_text(text)
    if not text:
        raise ValueError("Provide text with --text or --input.")

    model = load_checkpoint_for_synthesis(
        repo_id=args.repo_id,
        checkpoint=args.checkpoint,
        config=args.config,
        vocab=args.vocab,
        vits_repo=Path(args.vits_repo),
        device=pick_device(args.device),
    )
    audio = synthesize_text(
        model,
        text,
        noise_scale=args.noise_scale,
        noise_scale_w=args.noise_scale_w,
        length_scale=args.length_scale,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, int(model.hps.data.sampling_rate), subtype="PCM_16")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
