#!/usr/bin/env bash
set -uo pipefail

cd /home/aaquib/pi_ws/openpi-LBMfailure

LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/pi0_autorestart.log"
mkdir -p "${LOG_DIR}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG_FILE}"
}

# If this script itself dies for any reason, log it
trap 'log "WRAPPER EXIT: status=$? (outer bash died before printing train exit code)"' EXIT

# Properly init conda so we do not hit exit 127
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate openpi

max_restarts=20
restarts=0

while true; do
  log "========== Starting pi0 training (restart=${restarts}) =========="

  start_time=$(date +%s)

  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    uv run scripts/train.py "$@" 2>&1 | tee -a "${LOG_FILE}"
  exit_code=${PIPESTATUS[0]}

  end_time=$(date +%s)
  runtime=$((end_time - start_time))

  log "Training exited with code ${exit_code} after ${runtime}s"

  if [ "${exit_code}" -eq 0 ]; then
    log "Training finished successfully. Not restarting."
    break
  fi

  if [ "${runtime}" -lt 60 ]; then
    log "Crashed in under 60s. Not auto-restarting to avoid crash loops."
    break
  fi

  restarts=$((restarts + 1))
  if [ "${restarts}" -ge "${max_restarts}" ]; then
    log "Hit max_restarts=${max_restarts}. Stopping."
    break
  fi

  log "Training crashed, sleeping 10s before restart (attempt ${restarts}/${max_restarts})..."
  sleep 10
done

log "=== Autorestart loop finished ==="
