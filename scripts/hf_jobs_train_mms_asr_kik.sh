#!/usr/bin/env bash
# Bootstrap for Hugging Face Jobs (Debian/Ubuntu images): FFmpeg is required so
# datasets[audio] can decode via torchcodec. Then editable install + MMS ASR train.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

if command -v apt-get >/dev/null 2>&1; then
  apt-get update
  apt-get install -y --no-install-recommends ffmpeg
fi

ROOT="${1:-}"
if [[ -z "${ROOT}" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

cd "${ROOT}"
pip install -U pip
pip install -e .

# mlx-* is pulled in via pyproject for Apple Silicon workflows; on Linux x86_64 HF Jobs the
# wheels do not ship libmlx.so, and transformers' dtype helpers try `import mlx.core` and crash.
if [[ "$(uname -s)" == Linux ]]; then
  pip uninstall -y mlx mlx-audio mlx-lm miniaudio 2>/dev/null || true
fi

exec python scripts/train_mms_asr_kik.py --config configs/train_mms_asr_kik.yaml
