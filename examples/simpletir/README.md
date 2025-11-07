## SimpleTIR Example (preview)

This directory contains the components required to experiment with the
SimpleTIR tasks inside `slime`. The integration currently covers:

- Loading the official SimpleTIR parquet datasets (train / eval).
- Multi-turn rollouts that execute Python code blocks inside a sandbox and
  feed observations back to the model.
- Verifier rewards: math answers are checked with `math_verify`; code tasks
  execute the model output inside a sandbox and compare stdout against the
  official tests (see `examples/simpletir/reward_score/`).
- Hook points (`custom_generate_function_path`, `custom_rm_path`) so that
  rollouts plug into the standard `slime` training entrypoint.
- Optional sandbox helpers for executing Python snippets inside a Firejail-
  style environment (see `sandbox/README.md`).

### Data preparation

1. Download the official datasets (e.g. clone `ltzheng/SimpleTIR` or copy the
   `datasets/` directory from a release).
2. Convert the parquet files into the JSONL format expected by `slime`:

   ```bash
   python examples/simpletir/prepare_data.py \
     --source /path/to/SimpleTIR/datasets \
     --output /root/datasets/simpletir
   ```

   The command produces files such as
   `/root/datasets/simpletir/deepscaler_train.jsonl` and preserves all
   metadata required by the generator / reward hooks.

### Usage

1. Point `slime` to the SimpleTIR datasets (parquet files) and enable the
   custom rollout / reward hooks, e.g.

   ```bash
   python train.py \
     --rollout-function-path slime.rollout.sglang_rollout.generate_rollout \
     --custom-generate-function-path examples.simpletir.generate.custom_generate \
     --custom-rm-path examples.simpletir.reward.async_reward \
     --prompt-data /path/to/train.parquet \
     --eval-prompt-data simpletir_eval /path/to/eval.parquet \
     --metadata-key extra_info \
     ...
   ```

2. If you wish to execute model-generated code, run a sandbox service and
   set `SANDBOX_ENDPOINT=http://host:port/faas/sandbox/`. The helpers in
   `examples/simpletir/sandbox/` provide a reference FastAPI implementation.

3. (Optional) Use `examples/simpletir/run_simpletir.sh` as a starting point
   for launching Ray + `train.py`. The script mirrors the stock
   `scripts/run-qwen3-4B.sh`; edit the hard-coded paths inside the file if you
   need to point at different checkpoints or datasets.

### Status

The current implementation focuses on rule-based rewards and Python-tool
execution. More advanced reward models (e.g. verifier ensembles) and the
original megatron-specific trainer from the SimpleTIR release are not yet
ported. Contributions are welcome.
