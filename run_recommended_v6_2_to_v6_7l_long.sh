#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/train-data-1-hdd/guancheng/miniconda3/envs/kintwin/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROFILES=(
  recommended_v6_2
  recommended_v6_3
  recommended_v6_4
  recommended_v6_5
  recommended_v6_6
  recommended_v6_7
  recommended_v6_7l_long
)

cd "${REPO_ROOT}"

for profile in "${PROFILES[@]}"; do
  echo "============================================================"
  echo "Starting profile: ${profile}"
  echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "============================================================"

  "${PYTHON_BIN}" kintwin/run_train_from_json.py --profile "${profile}"

  echo "============================================================"
  echo "Finished profile: ${profile}"
  echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "============================================================"
done

echo "All profiles finished."
