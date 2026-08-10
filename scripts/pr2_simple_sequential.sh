#!/bin/bash
# Sequential driver for SIMPLE matrix v6: no mtp, c ∈ {1,8,32}.
# Runs 4 qwen36 jobs in order:
#   1. qwen36_tp2_triton  (will mostly SKIP — already has c=1/8/32 mtp0 OKs)
#   2. qwen36_tp2_exllama (0 rows → full 9 cells)
#   3. qwen36_tp4_triton  (4 rows → ~5 cells to do)
#   4. qwen36_tp4_exllama (0 rows → full 9 cells)
# Auto-detects SKIP via CSV pre-existing OK/TIMEOUT rows.

MATRIX_DIR="/home/chenco_adm/Projects/infrastructure/gfx1030_optimized/benchmark_results/pr2_gfx1030_matrix"
LOG_DIR="$MATRIX_DIR/logs"
mkdir -p "$LOG_DIR"

JOBS=(
  "qwen36 2 triton_w4a16"
  "qwen36 2 exllama"
  "qwen36 4 triton_w4a16"
  "qwen36 4 exllama"
)

for job in "${JOBS[@]}"; do
  read -r model tp kernel <<< "$job"
  job_log="$LOG_DIR/seq_simple_${model}_tp${tp}_${kernel}.log"
  echo "[$(date +%H:%M:%S)] SIMPLE JOB ${model}_tp${tp}_${kernel} starting"
  bash "$MATRIX_DIR/pr2_simple_runner.sh" "$model" "$tp" "$kernel" > "$job_log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] SIMPLE JOB ${model}_tp${tp}_${kernel} done rc=$rc"
done

echo "[$(date +%H:%M:%S)] ALL SIMPLE JOBS DONE"