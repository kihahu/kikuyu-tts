#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-/workspace/kikuyu-tts}"
CONFIG_PATH="${2:-configs/train_mms_tts_kik_waxal.yaml}"

cd "$REPO_DIR"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is required. Pass it with: hf jobs run ... --secrets HF_TOKEN" >&2
  exit 1
fi

if command -v apt-get >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg git build-essential espeak-ng
fi

python -m pip install --upgrade pip
python -m pip install "torch>=2.3,<2.6" "huggingface_hub>=0.24.0" "datasets[audio]>=3.0.0" soundfile pyyaml

python scripts/prepare_waxal_kik_tts.py \
  --dataset-name google/WaxalNLP \
  --dataset-config kik_tts \
  --split train \
  --output-dir data/waxal_kik_tts \
  --target-sample-rate 16000 \
  --min-duration-sec 0.6 \
  --max-duration-sec 25.0 \
  --min-rms 0.0035 \
  --seed 42 \
  --dev-ratio 0.10 \
  --test-ratio 0.05

BOOTSTRAP_ARGS=()
if [ -n "${KIK_TTS_EPOCHS:-}" ]; then
  BOOTSTRAP_ARGS+=(--epochs "$KIK_TTS_EPOCHS")
fi
if [ -n "${KIK_TTS_BATCH_SIZE:-}" ]; then
  BOOTSTRAP_ARGS+=(--batch-size "$KIK_TTS_BATCH_SIZE")
fi
if [ -n "${KIK_TTS_EVAL_INTERVAL:-}" ]; then
  BOOTSTRAP_ARGS+=(--eval-interval "$KIK_TTS_EVAL_INTERVAL")
fi
if [ -n "${KIK_TTS_LOG_INTERVAL:-}" ]; then
  BOOTSTRAP_ARGS+=(--log-interval "$KIK_TTS_LOG_INTERVAL")
fi

python scripts/bootstrap_mms_kikuyu_tts_finetune.py \
  --config "$CONFIG_PATH" \
  --download-checkpoint \
  "${BOOTSTRAP_ARGS[@]}"

readarray -t HUB_CONFIG < <(CONFIG_PATH="$CONFIG_PATH" python - <<'PY'
import os
import yaml
with open(os.environ["CONFIG_PATH"], "r", encoding="utf-8") as f:
    hub = (yaml.safe_load(f) or {}).get("hub", {})
print(hub.get("repo_id", ""))
for pattern in hub.get("upload_patterns", []):
    print(pattern)
PY
)
HUB_REPO_ID="${HUB_CONFIG[0]:-}"
UPLOAD_PATTERNS=("${HUB_CONFIG[@]:1}")
UPLOAD_PATTERNS_JSON="$(printf '%s\n' "${UPLOAD_PATTERNS[@]}" | python -c 'import json,sys; print(json.dumps([line.strip() for line in sys.stdin if line.strip()]))')"

if [ -n "$HUB_REPO_ID" ]; then
  python - <<PY
from huggingface_hub import HfApi
api = HfApi()
api.create_repo(repo_id="$HUB_REPO_ID", repo_type="model", exist_ok=True)
PY
fi

SYNC_PID=""
if [ -n "$HUB_REPO_ID" ]; then
  (
    while true; do
      sleep "${HF_UPLOAD_INTERVAL_SEC:-900}"
      python - <<PY || true
import json
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    repo_id="$HUB_REPO_ID",
    repo_type="model",
    folder_path="artifacts/mms_tts_kik_waxal_finetune",
    path_in_repo="mms_vits_finetune",
    allow_patterns=json.loads('''$UPLOAD_PATTERNS_JSON'''),
    commit_message="Sync in-progress MMS Kikuyu TTS Waxal artifacts",
)
PY
    done
  ) &
  SYNC_PID="$!"
  trap 'if [ -n "$SYNC_PID" ]; then kill "$SYNC_PID" 2>/dev/null || true; fi' EXIT
fi

bash artifacts/mms_tts_kik_waxal_finetune/launch_finetune.sh

if [ -n "$HUB_REPO_ID" ]; then
  if [ -n "$SYNC_PID" ]; then
    kill "$SYNC_PID" 2>/dev/null || true
    wait "$SYNC_PID" 2>/dev/null || true
    SYNC_PID=""
  fi
  python - <<PY
import json
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    repo_id="$HUB_REPO_ID",
    repo_type="model",
    folder_path="artifacts/mms_tts_kik_waxal_finetune",
    path_in_repo="mms_vits_finetune",
    allow_patterns=json.loads('''$UPLOAD_PATTERNS_JSON'''),
    commit_message="Upload MMS Kikuyu TTS Waxal fine-tune artifacts",
)
PY
fi
