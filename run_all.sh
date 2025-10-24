#!/usr/bin/env bash
set -euo pipefail

DEVICE="cuda:0"   # GPU
TASKS="aime24,ifeval" # benchmarks to run
OUTDIR="results"
BATCH="auto:8"  # Try up to batch 8 (fallback to lower if needed)
MAX_BATCH="8"
DTYPE="float16"   # Use bfloat16 if you prefer: DTYPE="bfloat16"
MAX_GEN_TOKS="512"  # Max tokens to generate per example
LIMIT="--limit 100"  # Run only 100 examples per task  

# Evaluate 3 models
MODELS=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
)

echo "Device: $DEVICE"
echo "Tasks:  $TASKS"
echo "Out:    $OUTDIR"
echo "Batch:  $BATCH (max ${MAX_BATCH})"
echo "Max gen tokens: $MAX_GEN_TOKS"
echo

for MODEL in "${MODELS[@]}"; do
  SAFE_NAME="$(echo "$MODEL" | tr '/' '_' )"
  MODEL_OUT="$OUTDIR/$SAFE_NAME"
  mkdir -p "$MODEL_OUT"

  # If results already exist, skip to avoid re-running
  if ls "$MODEL_OUT"/results_*.json >/dev/null 2>&1; then
    echo "✓ Found existing results for $MODEL, skipping."
    echo
    continue
  fi

  echo "=== Evaluating: $MODEL ==="
  lm-eval --model hf \
    --model_args "pretrained=${MODEL},dtype=${DTYPE}" \
    --tasks "$TASKS" \
    --device "$DEVICE" \
    --batch_size "$BATCH" \
    --max_batch_size "$MAX_BATCH" \
    --gen_kwargs "max_gen_toks=${MAX_GEN_TOKS}" \
    --apply_chat_template \
    --output_path "$MODEL_OUT" \
    --log_samples \
    $LIMIT

  echo
done

echo "✅ Evaluation complete! Results saved in: $OUTDIR"