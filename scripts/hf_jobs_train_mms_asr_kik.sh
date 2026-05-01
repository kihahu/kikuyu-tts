#!/usr/bin/env bash
# Bootstrap for Hugging Face Jobs (Debian/Ubuntu images): FFmpeg is required so
# datasets[audio] can decode via torchcodec. Then editable install + MMS ASR train.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required. Pass a write-capable token with: hf jobs run ... --secrets HF_TOKEN" >&2
  exit 2
fi

if command -v apt-get >/dev/null 2>&1; then
  apt-get update
  apt-get install -y --no-install-recommends ffmpeg
fi

ROOT="${1:-}"
if [[ -z "${ROOT}" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

cd "${ROOT}"
echo "Repo root: ${ROOT}"
if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Git branch: $(git rev-parse --abbrev-ref HEAD)"
  echo "Git commit: $(git rev-parse HEAD)"
fi
echo "Config path: configs/train_mms_asr_kik.yaml"
awk '
  /^outputs:/ { in_outputs = 1; next }
  /^[^[:space:]]/ { in_outputs = 0 }
  in_outputs && /hub_model_id|output_dir|push_to_hub|hub_upload_checkpoints|hub_verify_after_push/ { print }
' configs/train_mms_asr_kik.yaml

pip install -U pip
pip install -e .

# mlx-* is pulled in via pyproject for Apple Silicon workflows; on Linux x86_64 HF Jobs the
# wheels do not ship libmlx.so, and transformers' dtype helpers try `import mlx.core` and crash.
if [[ "$(uname -s)" == Linux ]]; then
  pip uninstall -y mlx mlx-audio mlx-lm miniaudio 2>/dev/null || true
fi

exec python scripts/train_mms_asr_kik.py --config configs/train_mms_asr_kik.yaml
