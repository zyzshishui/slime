import asyncio
import copy
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Counter, DefaultDict
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from slime.utils.async_utils import run
from slime.utils.data import JsonlDataset
from slime.utils.http_utils import get, post
from slime.utils.misc import load_function
from slime.utils.types import Sample

from .rm_hub import async_rm, batched_async_rm

__all__ = ["generate_rollout"]


@dataclass
class GenerateState:
    remaining_batch_size: int = 0  # the number of samples that are done/running/pending
    pendings: set = None  # the set of pending tasks
    num_stats: DefaultDict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    rollout_start_time: float = None
    completion_tokens_list: list = field(default_factory=list)

    def bump(self, group: str, value: str, n: int):
        self.num_stats[group][value] += n

    def stats_to_string(self) -> str:
        flat = {f"{g}_{k}_samples": v for g, ctr in self.num_stats.items() for k, v in ctr.items()}
        return json.dumps(flat, ensure_ascii=False)


def update_num_stats(state: GenerateState, group: str, value: Any, n: int = 1):
    vstr = value.name.lower() if isinstance(value, Enum) else str(value)
    state.bump(group, vstr, n)


TOKENIZER = None
SEMAPHORE = None


async def generate(args, sample: Sample, sampling_params) -> Sample:
    global TOKENIZER, SEMAPHORE
    if TOKENIZER is None:
        TOKENIZER = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)

    if SEMAPHORE is None:
        SEMAPHORE = asyncio.Semaphore(
            args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
        )

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    assert (
        sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED
    ), f"Sample status is {sample.status}"
    is_partial_sample = sample.status == Sample.Status.ABORTED
    # Handle partial rollout samples: continue generation from existing response
    if is_partial_sample:
        # Continue generation from partial response
        input_text = sample.prompt + sample.response
    else:
        # Regular generation from prompt
        input_text = sample.prompt

    payload = {
        "text": input_text,
        "sampling_params": sampling_params,
    }

    max_retries = 60
    retry_count = 0
    while retry_count < max_retries:
        try:
            async with SEMAPHORE:
                output = await post(url, payload, use_http2=args.use_http2)
        except Exception as e:
            retry_count += 1
            print(f"Error: {e}, retrying... (attempt {retry_count}/{max_retries})")
            if retry_count >= max_retries:
                print(f"Max retries ({max_retries}) reached, failing...")
                raise e
            await asyncio.sleep(1)
            continue
        break

    prompt_tokens_ids = TOKENIZER(sample.prompt, add_special_tokens=False)["input_ids"]
    sample.completion_tokens = output["meta_info"]["completion_tokens"]

    if is_partial_sample:
        # For partial rollout: combine existing response with new generation
        sample.response = sample.response + output["text"]
    else:
        # Regular generation
        sample.response = output["text"]

    response_token_ids = TOKENIZER(sample.response, add_special_tokens=False)["input_ids"]
    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    match output["meta_info"]["finish_reason"]["type"]:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED

    return sample


async def generate_and_rm(args, sample: Sample, sampling_params: dict, evaluation=False) -> Sample:
    # For samples with existing response, check if they're complete
    if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
        assert sample.response is not None and sample.reward is not None
        return sample

    # generate
    if args.custom_generate_function_path is not None:
        custom_generate_func = load_function(args.custom_generate_function_path)
        sample = await custom_generate_func(args, sample, sampling_params)
    else:
        sample = await generate(args, sample, sampling_params)

    if sample.status == Sample.Status.ABORTED:
        return sample

    # for the rm that need the whole group, we will not do the rm here
    if args.group_rm:
        return sample

    reward = await async_rm(args, sample)
    if not evaluation and args.reward_key:
        reward = reward[args.reward_key]
    elif evaluation and args.eval_reward_key:
        reward = reward[args.eval_reward_key]
    sample.reward = reward

    return sample


async def generate_rollout_async(args, rollout_id: int, data_buffer) -> list[Sample]:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_buffer: the data buffer to store the generated samples

    Returns:
        list[Sample]: a list of samples generated by the rollout, the length of the list is exactly the same as the `rollout_batch_size`
    """
    assert args.rollout_global_dataset

    sampling_params = dict(
        temperature=args.rollout_temperature,
        top_p=args.rollout_top_p,
        top_k=args.rollout_top_k,
        max_new_tokens=args.rollout_max_response_len,
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

    # sampling_batch_size refers to the number of samples to get at a time.
    # if the number of valid samples obtained is insufficient to support rollout, start the next round of sampling.
    # redundant samples: aborted samples and excess completed and truncated samples.
    # for non-partial rollout with dynamic filter, on-policy is required and all redundant samples are dropped, so the sampling_batch_size should not be too large.
    # for partial rollout, redundant samples are stored in the buffer and will be used in the next round of sampling.

    if args.sampling_batch_size is not None:
        sampling_batch_size = args.sampling_batch_size
    else:
        sampling_batch_size = args.rollout_batch_size

    state = GenerateState(
        remaining_batch_size=0,
        pendings=set(),
    )

    dynamic_filter = None
    if args.dynamic_sampling_filter_path is not None:
        # dynamic filter is used to filter out samples that is not suitable for training
        dynamic_filter = load_function(args.dynamic_sampling_filter_path)
    over_sampling_filter = None
    if args.over_sampling_filter_path is not None:
        assert args.over_sampling_filter_input_size is not None
        # over sampling filter ensures over_sampling_filter_input_size samples are rollout. And pick rollout_batch_size samples from them.
        over_sampling_filter = load_function(args.over_sampling_filter_path)

    def submit_generate_tasks(samples: list[Sample]):
        for sample in samples:
            if sample.status == Sample.Status.PENDING:
                assert len(sample.metadata) == 0, f"Sample {sample} has metadata {sample.metadata}"
            state.pendings.add(
                asyncio.create_task(
                    generate_and_rm(
                        args,
                        sample,
                        sampling_params=sampling_params,
                        evaluation=False,
                    )
                )
            )
        state.remaining_batch_size += len(samples) // args.n_samples_per_prompt

    data_group = {}
    data = []

    async def abort_rollout():
        print(f"DEBUG: Sending abort. Current data: {len(data)}, pending tasks: {len(state.pendings)}", flush=True)
        try:
            response = await get(
                f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers", use_http2=args.use_http2
            )
            print(f"DEBUG: List workers: {response}", flush=True)
        except Exception as e:
            print(f"Error: {e}, Failed to get list_workers", flush=True)
            return []

        for url in response["urls"]:
            # abort all the requests
            # NOTE: Using empty string as rid to abort ALL requests by startswith() match
            print(f"Abort request for {url}", flush=True)
            await post(f"{url}/abort_request", {"rid": ""}, use_http2=False)

    have_aborted = False

    # target_data_size is the total number of valid samples to get
    # if over_sampling_filter_input_size is set, we will use it as the target data size, otherwise, we will use the rollout_batch_size
    target_data_size = (
        args.over_sampling_filter_input_size if over_sampling_filter is not None else args.rollout_batch_size
    ) * args.n_samples_per_prompt
    # rollout_info is used for sending info to the buffer
    rollout_info = {
        "rollout_id": rollout_id,
    }

    state.rollout_start_time = time.time()
    pbar = tqdm(total=target_data_size, desc="Rollout generation")
    while len(data) < target_data_size:
        while state.remaining_batch_size < target_data_size // args.n_samples_per_prompt:
            # get samples from the buffer and submit the generation requests.
            samples = await data_buffer.get_samples(sampling_batch_size * args.n_samples_per_prompt, rollout_info)
            submit_generate_tasks(samples)
            for sample in samples:
                update_num_stats(state, "input", sample.status)

        # wait for the generation to finish
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        # Always finish all done tasks. This will make the code of partial rollout cleaner.
        # The assumption here is that group_rm is not too slow.
        for task in done:
            sample = task.result()

            if sample.completion_tokens is not None:
                state.completion_tokens_list.append(sample.completion_tokens)

            # add sample to its group
            group_index = sample.index // args.n_samples_per_prompt
            if group_index not in data_group:
                data_group[group_index] = []
            data_group[group_index].append(sample)

            assert (
                sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED
            ), f"Sample {sample.index} has status {sample.status}, but should be completed or truncated. Rollout {rollout_id}, Sample prompt: {sample.prompt}, Sample response: {sample.response}"
            update_num_stats(state, "output", sample.status)

            if not len(data_group[group_index]) == args.n_samples_per_prompt:
                # wait for the data_group for this prompt finishing
                continue

            # For some rm, we need to do the rm for all samples in the group at the same time
            if args.group_rm:
                # TODO: this will stuck the asyncio loop.
                rewards = await batched_async_rm(args, data_group[group_index])
                for i, sample in enumerate(data_group[group_index]):
                    if args.reward_key:
                        sample.reward = rewards[i][args.reward_key]
                    else:
                        sample.reward = rewards[i]

            if dynamic_filter is not None and not dynamic_filter(args, data_group[group_index]):
                # Delete the invalid samples, don't use them in partial rollout.
                del data_group[group_index]
                state.remaining_batch_size -= 1
                update_num_stats(state, "excluded", "dynamic_sampling_filter", args.n_samples_per_prompt)
                continue

            # add the samples to the data
            if len(data) < target_data_size:
                data.extend(data_group[group_index])
                del data_group[group_index]
                pbar.update(args.n_samples_per_prompt)

            # When having enough samples, try abort the rollout and continue
            if len(data) >= target_data_size and not have_aborted:
                await abort_rollout()
                have_aborted = True

    pbar.close()

    print(f"[DEBUG] Rollout {rollout_id}: Got {len(data)} samples", flush=True)

    # there are still some unfinished requests, abort them
    if state.pendings:
        if not have_aborted:
            await abort_rollout()
            have_aborted = True

    if args.partial_rollout:
        # put unfinished samples to data group
        while state.pendings:
            done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                sample = task.result()
                update_num_stats(state, "output", sample.status)
                state.completion_tokens_list.append(sample.completion_tokens)

                group_index = sample.index // args.n_samples_per_prompt
                if group_index not in data_group:
                    data_group[group_index] = []
                data_group[group_index].append(sample)

        # try cache unfinished samples and excess valid samples back to buffer
        for group_index, samples in data_group.items():
            assert (
                len(samples) == args.n_samples_per_prompt
            ), f"Got {len(samples)} samples, expected {args.n_samples_per_prompt}"
            cached_samples = []
            for sample in samples:
                update_num_stats(state, "cached", sample.status)
                cached_samples.append(sample)

            if len(cached_samples) > 0:
                # Add cached samples back to buffer for next iteration
                await data_buffer.add_samples(cached_samples, rollout_info)

    if over_sampling_filter is not None:
        update_num_stats(
            state, "excluded", "over_sampling_filter", len(data) - args.rollout_batch_size * args.n_samples_per_prompt
        )
        data = over_sampling_filter(args, data)[: args.rollout_batch_size * args.n_samples_per_prompt]
    else:
        data.sort(key=lambda sample: sample.index)

    rollout_time = time.time() - state.rollout_start_time

    completion_tokens_stats = {}
    completion_tokens_array = np.array(state.completion_tokens_list)
    completion_tokens_stats = {
        'total_completion_tokens': np.sum(completion_tokens_array).item(),
        'completion_tokens_mean': np.mean(completion_tokens_array).item(),
        'completion_tokens_std': np.std(completion_tokens_array).item(),
        'completion_tokens_count': len(completion_tokens_array),
    }
    
    if len(data) > 0:
        data[0].metadata.update({
            'rollout_time': rollout_time,
            'completion_tokens_stats': completion_tokens_stats,
        })

    print(f"[DEBUG] Rollout {rollout_id}: {state.stats_to_string()}", flush=True)
    if completion_tokens_stats:
        print(f"[DEBUG] Rollout {rollout_id}: Completion tokens stats: {completion_tokens_stats}", flush=True)
    assert (
        len(data) == args.rollout_batch_size * args.n_samples_per_prompt
    ), f"Got {len(data)} samples, expected {args.rollout_batch_size * args.n_samples_per_prompt}"
    return data


EVAL_PROMPT_DATASET = {}


async def eval_rollout(args, rollout_id):
    assert not args.group_rm, "Group RM is not supported for eval rollout"
    results = {}
    for i in range(0, len(args.eval_prompt_data), 2):
        name, path = args.eval_prompt_data[i : i + 2]
        results.update(await eval_rollout_single_dataset(args, rollout_id, name, path))
    return results


async def eval_rollout_single_dataset(args, rollout_id, name, path):
    """An example to implement the eval_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        name: str, the name of the dataset
        path: str, the path of the dataset
    """
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    global EVAL_PROMPT_DATASET

    if name not in EVAL_PROMPT_DATASET:
        tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        EVAL_PROMPT_DATASET[name] = JsonlDataset(
            path,
            tokenizer=tokenizer,
            max_length=args.rollout_max_prompt_len,
            prompt_key=args.input_key if args.eval_input_key is None else args.eval_input_key,
            label_key=args.label_key if args.eval_label_key is None else args.eval_label_key,
            metadata_key=args.metadata_key,
            tool_key=args.tool_key if args.eval_tool_key is None else args.eval_tool_key,
            apply_chat_template=args.apply_chat_template,
        )
    dataset = EVAL_PROMPT_DATASET[name]

    sampling_params = dict(
        temperature=args.rollout_temperature if args.eval_temperature is None else args.eval_temperature,
        top_p=args.rollout_top_p if args.eval_top_p is None else args.eval_top_p,
        top_k=args.rollout_top_k if args.eval_top_k is None else args.eval_top_k,
        max_new_tokens=(
            args.rollout_max_response_len if args.eval_max_response_len is None else args.eval_max_response_len
        ),
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

    tasks = []
    # do multiple samples for eval prompts
    sample_index = 0
    for i, prompt_sample in enumerate(dataset.samples):
        for j in range(args.n_samples_per_eval_prompt):
            # use the same prompt for multiple samples
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample_index += 1
            tasks.append(
                generate_and_rm(
                    args,
                    sample,
                    sampling_params=sampling_params,
                    evaluation=True,
                )
            )

    data = []
    do_print = True
    pbar = tqdm(total=len(tasks), desc="Rollout generation", disable=not do_print)
    for coro in asyncio.as_completed(tasks):
        sample = await coro
        if do_print:
            print([sample.prompt + sample.response], sample.reward)
            do_print = False
        data.append(sample)
        pbar.update(1)
    pbar.close()

    data.sort(key=lambda sample: sample.index)

    return {
        name: {
            "rewards": [sample.reward for sample in data],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
        }
    }


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_buffer: the data buffer to store the generated samples
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        list[Sample]: a list of samples generated by the rollout
    """
    assert args.rollout_global_dataset
    if evaluation:
        return run(eval_rollout(args, rollout_id))
    return run(generate_rollout_async(args, rollout_id, data_buffer))
