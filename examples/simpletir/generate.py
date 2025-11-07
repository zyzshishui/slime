from __future__ import annotations

import copy
from typing import Any

from slime.rollout.sglang_rollout import GenerateState, generate
from slime.utils.types import Sample

from .dataset import get_dataset, get_dataset_from_path
from .reward import compute_rule_reward

SIMPLETIR_CONFIG = {
    "max_turns": 5,
    "mask_void_turns": True,
}


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


async def custom_generate(args, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    """Generate multi-turn SimpleTIR rollouts."""
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

    max_prompt_tokens = getattr(args, "rollout_max_prompt_len", None)
    dataset.ensure_prompt_limit(tokenizer, max_prompt_tokens)

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
    max_turns = SIMPLETIR_CONFIG["max_turns"]
    mask_void_turns = SIMPLETIR_CONFIG["mask_void_turns"]

    base_sampling_params = dict(sampling_params or {})
    response_budget = base_sampling_params.get("max_new_tokens")
    if response_budget is None:
        response_budget = getattr(args, "rollout_max_response_len", 0)
    response_budget = int(response_budget or 0)
    initial_response_budget = response_budget

    turns_taken = 0
    void_turns = 0

    for turn in range(max_turns):
        turns_taken = turn + 1
        if response_budget <= 0:
            sample.status = Sample.Status.TRUNCATED
            break

        prompt_text = _render_prompt(tokenizer, messages)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        current_sampling_params = dict(base_sampling_params)
        current_sampling_params["max_new_tokens"] = max(response_budget, 0)

        sample.tokens = prompt_ids
        sample.response = ""
        sample.response_length = 0
        sample.loss_mask = None
        sample.rollout_log_probs = None
        sample.prompt = prompt_text
        sample.status = Sample.Status.PENDING

        sample = await generate(args, sample, current_sampling_params)
        response_text = sample.response or ""
        messages.append({"role": "assistant", "content": response_text})

        generated_tokens = sample.response_length or max(len(sample.tokens) - len(prompt_ids), 0)
        response_budget -= max(generated_tokens, 0)

        boxed_answer = "\\boxed{" in response_text

        if not response_text.strip():
            void_turns += 1
            if mask_void_turns:
                break

        if boxed_answer:
            break

        if response_budget <= 0:
            sample.status = Sample.Status.TRUNCATED
            break

        # No tool use and no final answer – exit to avoid empty loops.
        break

    sample.prompt = messages
    metadata.update(
        {
            "turns_taken": turns_taken,
            "void_turns": void_turns,
            "response_budget_remaining": max(response_budget, 0),
            "response_budget_initial": initial_response_budget,
        }
    )
    if record.ground_truth:
        reward = compute_rule_reward(record, sample)
        if reward is not None:
            sample.reward = reward

    if sample.loss_mask is None and sample.response_length:
        sample.loss_mask = [1] * sample.response_length

    return sample
