#!/usr/bin/env bash
# Serial real LoRA training. GPU work starts ONLY if EXECUTE=1.
set -euo pipefail
: "${MODEL:?Set MODEL to a compatible pinned local checkpoint}"
: "${BASE_ID:?Set BASE_ID to the immutable operator-verified checkpoint ID}"
RUN_ROOT="${RUN_ROOT:-runs/spear-real-lora-pilot}"
REVISION="${REVISION:-main}"
MODEL_CLASS="${MODEL_CLASS:-causal}"
QUANTIZATION="${QUANTIZATION:-4bit}"
TARGETS="${TARGETS:-schema}"
SOURCES="${SOURCES:-blind docs teacher probe}"
SEED="${SEED:-42}"
STEPS="${STEPS:-80}"

if [[ ! -f "${RUN_ROOT}/manifest.json" ]]; then
  python -m experiments.real_lora.prepare --seed "$SEED" \
    --train-groups "${TRAIN_GROUPS:-48}" \
    --validation-groups "${VAL_GROUPS:-12}" \
    --test-groups "${TEST_GROUPS:-24}" \
    --out "${RUN_ROOT}/manifest.json"
fi
python -m experiments.real_lora.run plan --manifest "${RUN_ROOT}/manifest.json" --out "$RUN_ROOT"

if [[ "${EXECUTE:-0}" != "1" ]]; then
  echo "Dry run only: set EXECUTE=1 to train real LoRA adapters."
  exit 0
fi
for target in $TARGETS; do
  for source in $SOURCES; do
    adapter="${RUN_ROOT}/adapters/${target}/${source}"
    out="${RUN_ROOT}/eval/${target}/${source}"
    python -m experiments.real_lora.run train \
      --manifest "${RUN_ROOT}/manifest.json" \
      --model "$MODEL" --revision "$REVISION" --base-id "$BASE_ID" \
      --model-class "$MODEL_CLASS" --quantization "$QUANTIZATION" \
      --target "$target" --source "$source" --seed "$SEED" --steps "$STEPS" \
      --out "$adapter"
    python -m experiments.real_lora.run evaluate \
      --manifest "${RUN_ROOT}/manifest.json" \
      --model "$MODEL" --revision "$REVISION" --base-id "$BASE_ID" \
      --model-class "$MODEL_CLASS" --quantization "$QUANTIZATION" \
      --target "$target" --adapter "$adapter" --out "$out"
  done
done
python -m experiments.real_lora.run report --root "$RUN_ROOT" \
  --out "${RUN_ROOT}/counterfactual_report.json"
echo "Final report: ${RUN_ROOT}/counterfactual_report.json"
