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
exec python scripts/train_mms_asr_kik.py --config configs/train_mms_asr_kik.yaml
