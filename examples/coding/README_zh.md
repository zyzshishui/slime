# 示例：Coding

[English](./README.md)

这里是一个在 slime 中接入 coding 任务及本地代码评测的简单样例。

## 配置环境

拉取 `zhuzilin/slime:latest` 镜像后，用如下方式初始化镜像环境：

```bash
cd /root/
git clone https://github.com/THUDM/slime.git
pip install -e .
```

下载并处理数据：

```bash
huggingface-cli download --repo-type dataset inclusionAI/AReaL-boba-2-RL-Code --local-dir /root/coding_data

python examples/coding/convert_datasets.py
```
或直接下载处理好的数据：
```bash
huggingface-cli download --repo-type dataset zyzshishui0627/AReaL-boba-2-RL-Code-openai-format --local-dir /root/coding_dataset
```

初始化 Qwen3-4B 模型：

```bash
# hf checkpoint
huggingface-cli download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B

# mcore checkpoint
cd /root/slime
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    --hf-checkpoint /root/Qwen3-4B \
    --save /root/Qwen3-4B_torch_dist
```

## 运行脚本

```bash
cd slime/
bash examples/coding/run-qwen3-4B.sh
```
