#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
export TOKENIZERS_PARALLELISM=false

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen3-8B.sh"

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-8B
   #--hf-checkpoint /root/Qwen3-8B-FP8
   --ref-load /root/Qwen3-8B_torch_dist
   --load /root/Qwen3-8B_slime/
   --save /root/Qwen3-8B_slime/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data /root/coding_dataset/train.jsonl
   --input-key prompt
   --label-key input_output
   --apply-chat-template
   --rollout-shuffle
   --rm-type coding
   --num-epoch 1
   --rollout-batch-size 128
   --n-samples-per-prompt 16
   --rollout-max-response-len 27648
   --rollout-temperature 1.0

   --global-batch-size 2048
   --balance-data
   # --partial-rollout
   # --over-sampling-batch-size 64
)

EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data LiveCodeBench /root/coding_dataset/lcb_v5_2410_2502.json
   # --eval-prompt-data Codeforces /root/coding_dataset/codeforces.jsonl \
   #                    LiveCodeBench /root/coding_dataset/lcb_v5_2410_2502.jsonl \
   #                    CodeContests /root/coding_dataset/code_contest_all.jsonl
   --eval-input-key prompt
   --eval-label-key input_output
   --n-samples-per-eval-prompt 2
   --eval-max-response-len 32768
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 2e-5
   --lr-decay-style constant
   --weight-decay 0.05
   --adam-beta1 0.9
   --adam-beta2 0.95
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project Qwen3-8B-Coding
   --wandb-group test
   --wandb-key ${WANDB_API_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.4
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-"12345"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} 