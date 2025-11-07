#!/bin/bash

# SimpleTIR launcher that mirrors the configuration style from the upstream
# SimpleTIR README. Set the environment variables below before invoking, e.g.
#
# MODEL_PATH=/path/to/Qwen3.5-7B \
# DATA_PATH=/path/to/datasets \
# CHECKPOINT_PATH=/path/to/output \
# LOG_PATH=/path/to/logs \
# NNODES=1 \
# GPUS_PER_NODE=8 \
# MODE=train \
# bash scripts/run-simpletir-qwen3-4B.sh
#
# MODE=train  – full RL training
# MODE=eval   – evaluation only (val_only=True)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${ROOT_DIR}/scripts/models/qwen3-4B.sh"
export PYTHONBUFFERED=16

MODE=${MODE:-train}
MODEL_PATH=${MODEL_PATH:-"/root/Qwen3-4B"}
DATA_PATH=${DATA_PATH:-"/root/SimpleTIR/datasets"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/root/Qwen3-4B_simpletir"}
LOG_PATH=${LOG_PATH:-"${CHECKPOINT_PATH}/logs"}
NNODES=${NNODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
RESUME=${RESUME:-False}
CONFIG_NAME=${CONFIG_NAME:-simpletir_trainer}
MODEL_NAME=${MODEL_NAME:-"Qwen3-4B"}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-16000}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8000}
MAX_TURNS=${MAX_TURNS:-5}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}
VAL_SAMPLE_SIZE=${VAL_SAMPLE_SIZE:-50}
N_VAL=${N_VAL:-16}

MAX_PROMPT_LENGTH_EVAL=${MAX_PROMPT_LENGTH_EVAL:-36000}
MAX_RESPONSE_LENGTH_EVAL=${MAX_RESPONSE_LENGTH_EVAL:-12000}
N_VAL_EVAL=${N_VAL_EVAL:-32}
VAL_INTERVAL=${VAL_INTERVAL:-$VAL_SAMPLE_SIZE}
NUM_ROLLOUT=${NUM_ROLLOUT:-1000}

TRAIN_DATASETS=${TRAIN_DATASETS:-"simplelr_math_35/train deepscaler/train"}
VALID_DATASETS=${VALID_DATASETS:-"deepscaler/aime"}

mkdir -p "${CHECKPOINT_PATH}" "${LOG_PATH}"

# helper: convert dataset spec "deepscaler/train" -> "${DATA_PATH}/deepscaler/train.parquet"
dataset_to_path() {
  local spec="$1"
  echo "${DATA_PATH}/${spec}.parquet"
}

first_dataset_path() {
  local spec_list="$1"
  for spec in ${spec_list}; do
    dataset_to_path "${spec}"
    return
  done
}

first_dataset_name() {
  local spec_list="$1"
  for spec in ${spec_list}; do
    echo "${spec%%/*}"
    return
  done
}

if [[ "${MODE}" == "eval" ]]; then
  NUM_ROLLOUT=0
  MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH_EVAL}
  MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH_EVAL}
  N_VAL=${N_VAL_EVAL}
  VAL_INTERVAL=1
fi

LOAD_PATH="${CHECKPOINT_PATH}"
if [[ "${RESUME}" != "False" && "${RESUME}" != "True" ]]; then
  LOAD_PATH="${RESUME}"
fi

PROMPT_DATA_PATH=$(first_dataset_path "${TRAIN_DATASETS}")
VALID_NAME=$(first_dataset_name "${VALID_DATASETS}")
VALID_PATH=$(first_dataset_path "${VALID_DATASETS}")

if (( $(wc -w <<<"${TRAIN_DATASETS}") > 1 )); then
  echo "WARNING: multiple training datasets detected; using ${PROMPT_DATA_PATH} (first entry)." >&2
fi
if (( $(wc -w <<<"${VALID_DATASETS}") > 1 )); then
  echo "WARNING: multiple validation datasets detected; using ${VALID_PATH} (first entry)." >&2
fi

if [[ ! -f "${PROMPT_DATA_PATH}" ]]; then
  echo "Prompt data not found: ${PROMPT_DATA_PATH}"
  exit 1
fi
if [[ ! -f "${VALID_PATH}" ]]; then
  echo "Validation data not found: ${VALID_PATH}"
  exit 1
fi

COMMON_ARGS=(
  --hf-checkpoint "${MODEL_PATH}"
  --load "${LOAD_PATH}"
  --save "${CHECKPOINT_PATH}"
  --save-interval 50
  --num-rollout "${NUM_ROLLOUT}"
  --model-name "${MODEL_NAME}"
  --actor-num-nodes "${NNODES}"
  --actor-num-gpus-per-node "${GPUS_PER_NODE}"
  --colocate
  --prompt-data "${PROMPT_DATA_PATH}"
  --apply-chat-template
  --metadata-key extra_info
  --custom-generate-function-path examples.simpletir.generate.custom_generate
  --custom-rm-path examples.simpletir.reward.async_reward
  --simpletir-train-data "${PROMPT_DATA_PATH}"
  --simpletir-max-turns "${MAX_TURNS}"
  --simpletir-mask-void-turns
  --global-batch-size "${TRAIN_BATCH_SIZE}"
  --rollout-max-prompt-len "${MAX_PROMPT_LENGTH}"
  --rollout-max-response-len "${MAX_RESPONSE_LENGTH}"
  --rollout-batch-size 32
  --n-samples-per-prompt 8
  --rollout-temperature 1.0
  --balance-data
  --tensor-model-parallel-size 2
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu 9216
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef 0.0
  --kl-loss-type low_var_kl
  --entropy-coef 0.0
  --eps-clip 0.2
  --eps-clip-high 0.28
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --rollout-num-gpus-per-engine 2
  --sglang-mem-fraction-static 0.7
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

EVAL_ARGS=(
  --eval-prompt-data "${VALID_NAME}" "${VALID_PATH}"
  --n-samples-per-eval-prompt "${N_VAL}"
  --eval-max-response-len "${MAX_RESPONSE_LENGTH}"
  --eval-interval "${VAL_INTERVAL}"
)

log_suffix=$(date +"%Y%m%d-%H%M%S")_${MODE}
LOG_FILE="${LOG_PATH}/simpletir_${log_suffix}.log"

{
  echo "MODE=${MODE}"
  echo "MODEL_PATH=${MODEL_PATH}"
  echo "DATA_PATH=${DATA_PATH}"
  echo "CHECKPOINT_PATH=${CHECKPOINT_PATH}"
  echo "PROMPT_DATA=${PROMPT_DATA_PATH}"
  echo "VALID_DATA=${VALID_PATH}"
  echo "LOAD_PATH=${LOAD_PATH}"
  echo "Running..."
} | tee "${LOG_FILE}"

python3 train.py \
  "${MODEL_ARGS[@]}" \
  "${COMMON_ARGS[@]}" \
  "${EVAL_ARGS[@]}" | tee -a "${LOG_FILE}"
