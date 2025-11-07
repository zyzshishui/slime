from __future__ import annotations

import copy
import os
import re
from typing import Any

from slime.rollout.sglang_rollout import GenerateState, generate
from slime.utils.types import Sample

from .dataset import get_dataset, get_dataset_from_path
from .reward import compute_rule_reward
from .text_utils import truncate_content

try:
    if os.getenv("SANDBOX_ENDPOINT"):
        from .sandbox.local_sandbox import parallel_sandbox
    else:
        from .sandbox.internal_sandbox import parallel_sandbox
except Exception:  # noqa: BLE001 - optional dependency
    parallel_sandbox = None


MAX_OBS_CHARS = 512
CODE_PATTERN = re.compile(r"```(?:py|python)?\n(.*?)```", re.DOTALL)


def _get_tokenizer(args):
    tokenizer = getattr(args, "_simpletir_tokenizer", None)
    if tokenizer is None:
        state = GenerateState(args)
        tokenizer = state.tokenizer
        setattr(args, "_simpletir_tokenizer", tokenizer)
    return tokenizer


def _render_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    except Exception:  # noqa: BLE001
        rendered = []
        for msg in messages:
            role = msg.get("role", "user").capitalize()
            rendered.append(f"{role}: {msg.get('content', '')}")
        rendered.append("Assistant:")
        return "\n".join(rendered)


async def _run_code(code: str):
    if parallel_sandbox is None:
        return {"success": False, "stdout": "", "stderr": "Sandbox not configured."}

    try:
        success_list, stdout_list, stderr_list = await parallel_sandbox([code])
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "stdout": "", "stderr": f"Sandbox failure: {exc}"}

    stdout = stdout_list[0] if stdout_list else ""
    stderr = stderr_list[0] if stderr_list else ""
    success = bool(success_list and success_list[0])
    return {
        "success": success,
        "stdout": truncate_content(stdout, MAX_OBS_CHARS) if stdout else "",
        "stderr": truncate_content(stderr, MAX_OBS_CHARS) if stderr else "",
    }


async def custom_generate(args, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    """Generate multi-turn SimpleTIR rollouts with optional sandbox execution."""
    tokenizer = _get_tokenizer(args)
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    if metadata is None:
        metadata = {}

    dataset_path = metadata.get("simpletir_dataset_path")
    dataset_split_hint = metadata.get("simpletir_split") or metadata.get("split") or "train"
    if dataset_path:
        dataset = get_dataset_from_path(dataset_path, split=str(dataset_split_hint))
    else:
        dataset = get_dataset(args, split="train")

    record = None
    if "index" in metadata:
        try:
            record = dataset.get_record(extra_index=int(metadata["index"]))
        except KeyError:
            record = None
    if record is None:
        record = dataset.get_record(row_id=sample.index % len(dataset))

    record_index = metadata.get("index")
    if record_index is None:
        record_index = record.extra_info.get("index")

    metadata.update(
        {
            "ability": record.ability,
            "reward_style": record.reward_style,
            "ground_truth": record.ground_truth,
            "data_source": record.data_source,
            "simpletir_dataset": str(dataset.path),
            "simpletir_dataset_path": str(dataset.path),
            "simpletir_split": dataset_split_hint or "train",
            "simpletir_record_index": record_index,
        }
    )
    sample.metadata = metadata

    if record.ground_truth:
        sample.label = record.ground_truth
        if record.ability == "math":
            sample.metadata.setdefault("rm_type", "math")

    messages = copy.deepcopy(record.prompt) or [{"role": "user", "content": ""}]
    max_turns = getattr(args, "simpletir_max_turns", 4)
    mask_void_turns = getattr(args, "simpletir_mask_void_turns", True)

    turns_taken = 0
    void_turns = 0
    code_turns = 0
    successful_code = 0
    sandbox_logs: list[dict[str, Any]] = []

    for turn in range(max_turns):
        turns_taken = turn + 1
        prompt_text = _render_prompt(tokenizer, messages)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        sample.tokens = prompt_ids
        sample.response = ""
        sample.response_length = 0
        sample.loss_mask = None
        sample.rollout_log_probs = None
        sample.prompt = prompt_text
        sample.status = Sample.Status.PENDING

        sample = await generate(args, sample, sampling_params)
        response_text = sample.response or ""
        messages.append({"role": "assistant", "content": response_text})

        boxed_answer = "\\boxed{" in response_text
        code_blocks = CODE_PATTERN.findall(response_text)

        if code_blocks:
            code_turns += 1
            execution = await _run_code(code_blocks[-1])
            successful_code += int(execution["success"])
            sandbox_logs.append({"turn": turn, **execution})

            if execution["stderr"]:
                observation = f"\nCode execution result: {truncate_content(execution['stderr'], MAX_OBS_CHARS)}\n"
            elif execution["stdout"]:
                observation = f"\nCode execution result: {truncate_content(execution['stdout'], MAX_OBS_CHARS)}\n"
            else:
                observation = "\nCode execution result: \n"

            messages.append({"role": "user", "content": observation})
            if boxed_answer:
                break
            continue

        if not response_text.strip():
            void_turns += 1
            if mask_void_turns:
                break

        if boxed_answer:
            break

        # No tool use and no final answer – exit to avoid empty loops.
        break

    sample.prompt = messages
    metadata.update(
        {
            "turns_taken": turns_taken,
            "void_turns": void_turns,
            "code_turns": code_turns,
            "successful_code": successful_code,
        }
    )
    if sandbox_logs:
        metadata["sandbox_logs"] = sandbox_logs

    if record.ground_truth:
        reward = compute_rule_reward(record, sample)
        if reward is not None:
            sample.reward = reward

    if sample.loss_mask is None and sample.response_length:
        sample.loss_mask = [1] * sample.response_length

    return sample
