#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-/workspace/kikuyu-tts}"
CONFIG_PATH="${2:-configs/train_mms_tts_kik_anv.yaml}"

cd "$REPO_DIR"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is required. Pass it with: hf jobs run ... --secrets HF_TOKEN" >&2
  exit 1
fi

ANV_PREP_ONLY="${ANV_PREP_ONLY:-0}"
ANV_LIGHTWEIGHT_PREP=0
if [ "${ANV_SMOKE_ONLY:-0}" = "1" ] || [ "$ANV_PREP_ONLY" = "1" ]; then
  ANV_LIGHTWEIGHT_PREP=1
fi

if command -v apt-get >/dev/null 2>&1; then
  apt-get update
  if [ "$ANV_LIGHTWEIGHT_PREP" = "1" ]; then
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg
  else
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg git build-essential espeak-ng
  fi
fi

python -m pip install --upgrade pip

if [ "${ANV_SMOKE_ONLY:-0}" = "1" ]; then
  python -m pip install "huggingface_hub>=0.24.0" "datasets>=3.0.0,<4.0.0" numpy soundfile
  PROBE_ARGS=()
  if [ "${ANV_SMOKE_DECODE_AUDIO:-0}" = "1" ]; then
    PROBE_ARGS+=(--probe-decode-audio)
  fi
  if [ "${ANV_INCLUDE_UNSCRIPTED:-0}" = "1" ]; then
    PROBE_ARGS+=(--include-unscripted)
  fi
  python scripts/prepare_anv_kikuyu_mms_tts.py \
    --dataset-name Anv-ke/kikuyu \
    --target-sample-rate 16000 \
    --probe-one-row \
    --probe-split "${ANV_PROBE_SPLIT:-train}" \
    --probe-max-rows "${ANV_PROBE_MAX_ROWS:-200}" \
    "${PROBE_ARGS[@]}"
  exit 0
fi

if [ "$ANV_LIGHTWEIGHT_PREP" = "1" ]; then
  python -m pip install "huggingface_hub>=0.24.0" "datasets>=3.0.0,<4.0.0" numpy soundfile
else
  python -m pip install "torch>=2.3,<2.6" "huggingface_hub>=0.24.0" "datasets[audio]>=3.0.0,<4.0.0" soundfile pyyaml
fi

python scripts/prepare_anv_kikuyu_mms_tts.py \
  --dataset-name Anv-ke/kikuyu \
  --output-dir data/anv_kikuyu_mms_tts \
  --target-sample-rate 16000 \
  --min-duration-sec "${ANV_MIN_DURATION_SEC:-1.0}" \
  --max-duration-sec "${ANV_MAX_DURATION_SEC:-15.0}" \
  --speaker-mode "${ANV_SPEAKER_MODE:-single_speaker}" \
  --max-rows-per-split "${ANV_MAX_ROWS_PER_SPLIT:-0}" \
  --stream-retries "${ANV_STREAM_RETRIES:-3}" \
  ${ANV_STREAMING:+--streaming}

if [ "$ANV_PREP_ONLY" = "1" ]; then
  exit 0
fi

BOOTSTRAP_ARGS=()
if [ -n "${KIK_TTS_EPOCHS:-}" ]; then
  BOOTSTRAP_ARGS+=(--epochs "$KIK_TTS_EPOCHS")
fi
if [ -n "${KIK_TTS_BATCH_SIZE:-}" ]; then
  BOOTSTRAP_ARGS+=(--batch-size "$KIK_TTS_BATCH_SIZE")
fi
if [ -n "${KIK_TTS_LEARNING_RATE:-}" ]; then
  BOOTSTRAP_ARGS+=(--learning-rate "$KIK_TTS_LEARNING_RATE")
fi
if [ -n "${KIK_TTS_EVAL_INTERVAL:-}" ]; then
  BOOTSTRAP_ARGS+=(--eval-interval "$KIK_TTS_EVAL_INTERVAL")
fi
if [ -n "${KIK_TTS_LOG_INTERVAL:-}" ]; then
  BOOTSTRAP_ARGS+=(--log-interval "$KIK_TTS_LOG_INTERVAL")
fi
if [ "${KIK_TTS_RESET_OPTIMIZER:-1}" = "1" ]; then
  BOOTSTRAP_ARGS+=(--reset-optimizer)
else
  BOOTSTRAP_ARGS+=(--keep-optimizer)
fi
if [ -n "${KIK_TTS_RESUME_REPO_ID:-}" ]; then
  BOOTSTRAP_ARGS+=(--resume-repo-id "$KIK_TTS_RESUME_REPO_ID")
fi
if [ -n "${KIK_TTS_RESUME_STEP:-}" ]; then
  BOOTSTRAP_ARGS+=(--resume-step "$KIK_TTS_RESUME_STEP")
fi
if [ -n "${KIK_TTS_RESUME_RUN_NAME:-}" ]; then
  BOOTSTRAP_ARGS+=(--resume-run-name "$KIK_TTS_RESUME_RUN_NAME")
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
    folder_path="artifacts/mms_tts_kik_anv_finetune",
    path_in_repo="mms_vits_finetune",
    allow_patterns=json.loads('''$UPLOAD_PATTERNS_JSON'''),
    commit_message="Sync in-progress MMS Kikuyu TTS ANV artifacts",
)
PY
    done
  ) &
  SYNC_PID="$!"
  trap 'if [ -n "$SYNC_PID" ]; then kill "$SYNC_PID" 2>/dev/null || true; fi' EXIT
fi

bash artifacts/mms_tts_kik_anv_finetune/launch_finetune.sh

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
    folder_path="artifacts/mms_tts_kik_anv_finetune",
    path_in_repo="mms_vits_finetune",
    allow_patterns=json.loads('''$UPLOAD_PATTERNS_JSON'''),
    commit_message="Upload MMS Kikuyu TTS ANV fine-tune artifacts",
)
PY
fi
