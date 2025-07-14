# Example: Coding

[中文版](./README_zh.md)

This is a simple example of supporting coding task and local evaluation in slime.

## Environment Setup

After pulling the `zhuzilin/slime:latest` image, initialize the image environment as follows:

```bash
cd /root/
git clone https://github.com/THUDM/slime.git
pip install -e .
```

Download the dataset：

```bash
huggingface-cli download --repo-type dataset inclusionAI/AReaL-boba-2-RL-Code \
  --local-dir /root/coding_data
```

Initialize the Qwen3-4B model:

```bash
# hf checkpoint
huggingface-cli download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B

# mcore checkpoint
cd /root/slime
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    --hf-checkpoint /root/Qwen3-4B \
    --save /root/Qwen3-4B_torch_dist
```

## Running the Script

```bash
cd slime/
bash examples/coding/run-qwen3-4B.sh
```
