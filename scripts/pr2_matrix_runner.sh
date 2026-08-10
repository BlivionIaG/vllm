#!/bin/bash
# PR #2 matrix runner v5.
# - Increased timeout (50 min) for Qwen3.6 heavy cells
# - GPU-scoped cleanup only
# - No set -u
# - Hardcoded venv path

MATRIX_DIR="/home/chenco_adm/Projects/infrastructure/gfx1030_optimized/benchmark_results/pr2_gfx1030_matrix"
CSV="$MATRIX_DIR/results.csv"
LOG_DIR="$MATRIX_DIR/cell_logs"
mkdir -p "$LOG_DIR"

MODEL_ID="${1:-qwen36}"
TP="${2:-4}"
KERNEL="${3:-triton_w4a16}"

VENV="/home/chenco_adm/Apps/vllm/venv-7.2.0"
source "$VENV/bin/activate"

if [ ! -s "$CSV" ]; then
  echo "timestamp,model,tp,kernel,mtp,workload_in,workload_out,concurrency,num_prompts,out_tok_s,total_tok_s,ttft_p50_ms,status,cell_log" > "$CSV"
fi

case "$MODEL_ID" in
  qwen36) MODEL_PATH="/home/chenco_adm/.cache/huggingface/hub/models--wizardeur--Qwen3.6-27B-GPTQ-W4A16-G32/snapshots/591e1cc676ea8c2399370e1a1be93faed87471e5" ;;
  qwen25) MODEL_PATH="/home/chenco_adm/.cache/huggingface/hub/qwen25-7b-gptq-int4" ;;
  qwen3_4b) MODEL_PATH="/home/chenco_adm/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/manual" ;;
  *) echo "unknown model: $MODEL_ID"; exit 1 ;;
esac

case "$KERNEL" in
  triton_w4a16) unset VLLM_DISABLED_KERNELS; KERNEL_NAME="Triton W4A16" ;;
  exllama) export VLLM_DISABLED_KERNELS="RDNA2W4A16LinearKernel,RDNA3W4A16LinearKernel,RDNAHybridW4A16LinearKernel,TritonW4A16LinearKernel"; KERNEL_NAME="Exllama" ;;
  default) unset VLLM_DISABLED_KERNELS; KERNEL_NAME="Default" ;;
  *) echo "unknown kernel: $KERNEL"; exit 1 ;;
esac

export PATH=/opt/rocm-7.2.0/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm-7.2.0/lib:${LD_LIBRARY_PATH:-}
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ROCM_USE_AITER=0
export VLLM_ROCM_USE_AITER_MOE=0
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export VLLM_RDNA_FORCE_FP16=1
export TORCH_BLAS_PREFER_HIPBLASLT=0
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
export GPU_MAX_HW_QUEUES=2
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HIP_FORCE_DEV_KERNARG=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export SETUPTOOLS_SCM_PRETEND_VERSION=0.20.1.dev99
export VLLM_BATCH_INVARIANT=0

case "$MODEL_ID" in
  qwen36) DTYPE="float16"; QUANT_FLAG="--quantization compressed-tensors" ;;
  qwen25) DTYPE="float16"; QUANT_FLAG="--quantization gptq" ;;
  qwen3_4b) DTYPE="float16"; QUANT_FLAG="" ;;
esac

WORKLOADS=("1024 256" "4096 512" "16384 1024")
MTPS=(0 1)

# Per-model timeout: qwen36 needs 50 min (heavy Triton compile + 27B prefill), others 25 min
case "$MODEL_ID" in
  qwen36) CELL_TIMEOUT=3000 ;;
  *) CELL_TIMEOUT=1500 ;;
esac

get_num_prompts() {
  if [ "$1" -le 8 ]; then echo 32
  elif [ "$1" -le 16 ]; then echo 64
  else echo 128
  fi
}

case "$TP" in
  1) GPU_LIST="4" ;;
  2) GPU_LIST="0,1" ;;
  4) GPU_LIST="0,1,2,3" ;;
esac

GMU=0.72

get_gpu_used_mib() {
  local g=$1
  local bytes=$(/opt/rocm/bin/rocm-smi --showmeminfo vram -d $g 2>/dev/null | grep "Used Memory" | grep -oE '[0-9]+' | tail -1)
  if [ -z "$bytes" ]; then echo 0; else echo $((bytes / 1024 / 1024)); fi
}

wait_drained() {
  local max_wait=120
  local waited=0
  while [ $waited -lt $max_wait ]; do
    local drained=1
    IFS=',' read -ra GA <<< "$GPU_LIST"
    for g in "${GA[@]}"; do
      local mib=$(get_gpu_used_mib "$g")
      if [ "$mib" -gt 500 ]; then drained=0; break; fi
    done
    if [ $drained -eq 1 ]; then return 0; fi
    sleep 5
    waited=$((waited + 5))
  done
  return 1
}

gpu_cleanup() {
  IFS=',' read -ra GA <<< "$GPU_LIST"
  local all_pids=""
  for g in "${GA[@]}"; do
    local pids=$(/opt/rocm/bin/rocm-smi --showpidgpus -d $g 2>/dev/null | grep -oE 'PID [0-9]+' | awk '{print $2}' | sort -u)
    all_pids="$all_pids $pids"
  done
  for pid in $all_pids; do
    [ -z "$pid" ] && continue
    kill -9 "$pid" 2>/dev/null
  done
  sleep 8
}

run_cell() {
  local mtp=$1 workload_in=$2 workload_out=$3 conc=$4
  local num_prompts=$(get_num_prompts "$conc")
  local cell_name="${MODEL_ID}_tp${TP}_${KERNEL_NAME// /_}_mtp${mtp}_in${workload_in}_out${workload_out}_c${conc}"
  local cell_log="$LOG_DIR/${cell_name}.log"
  local max_num_seqs=$conc
  [ "$conc" -gt 8 ] && max_num_seqs=8
  local max_model_len=$((workload_in + workload_out + 64))

  local mtp_args=""
  if [ "$mtp" -gt 0 ]; then
    mtp_args="--speculative-config {\"method\":\"qwen3_next_mtp\",\"num_speculative_tokens\":$mtp}"
  fi

  gpu_cleanup
  if ! wait_drained; then
    echo "[$(date +%H:%M:%S)] SKIP $cell_name (not drained)"
    echo "$(date +%Y-%m-%dT%H:%M:%S),$MODEL_ID,$TP,$KERNEL_NAME,$mtp,$workload_in,$workload_out,$conc,$num_prompts,,,,,SKIP,$cell_name" >> "$CSV"
    return 0
  fi

  echo "[$(date +%H:%M:%S)] START $cell_name (timeout=${CELL_TIMEOUT}s)"
  (
    trap 'echo "[runner-v5] TIMEOUT after ${CELL_TIMEOUT}s, killing bench"; exit 124' TERM
    HIP_VISIBLE_DEVICES="$GPU_LIST" \
    timeout $CELL_TIMEOUT \
    "$VENV/bin/vllm" bench throughput \
      --model "$MODEL_PATH" \
      --input-len $workload_in \
      --output-len $workload_out \
      --num-prompts $num_prompts \
      --max-num-seqs $max_num_seqs \
      --gpu-memory-utilization $GMU \
      --dtype $DTYPE \
      --tensor-parallel-size $TP \
      --max-model-len $max_model_len \
      $QUANT_FLAG \
      --trust-remote-code \
      --language-model-only \
      --skip-mm-profiling \
      --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE", "compile_ranges_endpoints": []}' \
      $mtp_args
  ) > "$cell_log" 2>&1
  local bench_rc=$?

  local out_tok_s="" total_tok_s="" ttft_p50=""
  if [ -s "$cell_log" ]; then
    out_tok_s=$(grep -E '^Throughput:' "$cell_log" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | tail -1)
    total_tok_s=$(grep -E '^Throughput:' "$cell_log" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | sed -n '2p')
    ttft_p50=$(grep -E 'TTFT p50:' "$cell_log" | head -1 | grep -oE '[0-9]+\.[0-9]+')
  fi

  gpu_cleanup

  local status="OK"
  if [ $bench_rc -ne 0 ] || [ -z "$out_tok_s" ]; then status="FAIL"; fi
  if [ $bench_rc -eq 124 ]; then status="TIMEOUT"; fi
  echo "$(date +%Y-%m-%dT%H:%M:%S),$MODEL_ID,$TP,$KERNEL_NAME,$mtp,$workload_in,$workload_out,$conc,$num_prompts,$out_tok_s,$total_tok_s,$ttft_p50,$status,$cell_name" >> "$CSV"
  echo "[$(date +%H:%M:%S)] DONE $cell_name out=${out_tok_s:-N/A} status=$status rc=$bench_rc"

  local total_lines=$(wc -l < "$CSV")
  if [ $((total_lines % 50)) -eq 0 ] && [ $total_lines -gt 1 ]; then
    local backup="/home/chenco_adm/Projects/infrastructure/gfx1030_matrix/backups/results_${total_lines}.csv"
    cp "$CSV" "$backup"
    echo "[$(date +%H:%M:%S)] BACKUP $backup"
  fi
}

gpu_cleanup
for mtp in "${MTPS[@]}"; do
  for wl in "${WORKLOADS[@]}"; do
    in_len=$(echo $wl | awk '{print $1}')
    out_len=$(echo $wl | awk '{print $2}')
    for conc in 1 2 4 8 16 32; do
      run_cell $mtp $in_len $out_len $conc
    done
  done
done

echo "[$(date +%H:%M:%S)] ALL DONE for $MODEL_ID TP=$TP kernel=$KERNEL"
