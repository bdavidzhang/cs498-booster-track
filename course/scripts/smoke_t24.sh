#!/usr/bin/env bash
# CPU smoke test for a training task. Run from course/ on the login node.
#   bash scripts/smoke_t24.sh [Task-Id]   (defaults to the PPO-T24 task)
set -uo pipefail

TASK=${1:-Course-HW2-DoubleCartpole-Balance-PPO-T24}
LOG=smoke_${TASK}.log

envs=$(uv run list-envs 2>&1)
grep -qF -- "$TASK" <<<"$envs" || { echo "FAIL: $TASK not registered"; exit 1; }

WANDB_MODE=offline uv run train "$TASK" \
  --gpu-ids None --env.scene.num-envs 8 --agent.max-iterations 3 \
  2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}

if [[ $status -ne 0 ]] || grep -q Traceback "$LOG"; then
  echo "FAIL: training crashed (see $LOG)"; exit 1
fi
if grep -qiE '\bnan\b|\binf\b' "$LOG"; then
  echo "WARN: nan/inf in output (see $LOG)"; exit 1
fi
echo "PASS: $TASK ran 3 iterations on CPU"
