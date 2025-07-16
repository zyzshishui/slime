from typing import Union

import torch
from megatron.core import mpu

from slime.utils.misc import load_function
from slime.utils.ppo_utils import (
    compute_approx_kl,
    compute_entropy_from_logits,
    compute_log_probs,
    compute_policy_loss,
    get_grpo_returns,
    get_reinforce_plus_plus_returns,
    get_reinforce_plus_plus_baseline_advantages,
)
from slime.utils.distributed_utils import distributed_masked_whiten

from .cp_utils import get_logits_and_tokens_offset_with_cp, get_sum_of_sample_mean
from .data import get_local_storage, set_local_storage


def calculate_log_probs_and_entropy(logits, tokens, with_entropy: bool = False):
    logits = logits.contiguous()
    # TODO: not sure why we need to clone the logits here.
    # Without the clone, the backward will trigger inplace edit error.
    # It seems that the function with tp will modify the logits inplace.
    if logits.size(0) != 0:
        log_prob = compute_log_probs(logits.clone(), tokens, mpu.get_tensor_model_parallel_group())
    else:
        log_prob = logits.new_zeros((0,))

    if with_entropy:
        if logits.size(0) != 0:
            entropy = compute_entropy_from_logits(logits.clone(), mpu.get_tensor_model_parallel_group())
        else:
            entropy = logits.new_zeros((0,))
    else:
        entropy = None
    return log_prob, entropy


def get_log_probs_and_entropy(
    logits: torch.Tensor,
    *,
    args,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
) -> dict[str, list[torch.Tensor]]:
    assert logits.size(0) == 1, f"{logits.shape}"
    assert logits.dtype == torch.float32, f"{logits.dtype}"

    logits = logits.squeeze(0)
    logits.div_(args.rollout_temperature)

    cp_size = mpu.get_context_parallel_world_size()

    log_probs_list = []
    if with_entropy:
        entropy_list = []
    end = 0
    for tokens, total_length, response_length in zip(unconcat_tokens, total_lengths, response_lengths):
        if cp_size == 1:
            end += total_length
            start = end - response_length
            logits_chunk = logits[start - 1 : end - 1]
            tokens_chunk = tokens[-response_length:]

            log_prob, entropy = calculate_log_probs_and_entropy(logits_chunk, tokens_chunk, with_entropy=with_entropy)
        else:
            # TODO: this is super ugly... do better abstraction.
            chunk_size, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length
            )

            logits_0, logits_1 = logits[end : end + chunk_size], logits[end + chunk_size : end + 2 * chunk_size]

            logits_0 = logits_0[logits_offset[0][0] - chunks_offset[0][0] : logits_offset[0][1] - chunks_offset[0][0]]
            tokens_0 = tokens[tokens_offset[0][0] : tokens_offset[0][1]]

            logits_1 = logits_1[logits_offset[1][0] - chunks_offset[1][0] : logits_offset[1][1] - chunks_offset[1][0]]
            tokens_1 = tokens[tokens_offset[1][0] : tokens_offset[1][1]]

            assert logits_0.size(0) == tokens_0.size(0), f"{logits_0.size(0)} vs {tokens_0.size(0)}"
            assert logits_1.size(0) == tokens_1.size(0), f"{logits_1.size(0)} vs {tokens_1.size(0)}"

            log_prob_0, entropy_0 = calculate_log_probs_and_entropy(
                logits_0,
                tokens_0,
                with_entropy=with_entropy,
            )
            log_prob_1, entropy_1 = calculate_log_probs_and_entropy(
                logits_1,
                tokens_1,
                with_entropy=with_entropy,
            )
            log_prob = torch.cat([log_prob_0, log_prob_1], dim=0)
            if with_entropy:
                entropy = torch.cat([entropy_0, entropy_1], dim=0)

            end += 2 * chunk_size

        log_probs_list.append(log_prob.squeeze(-1))
        if with_entropy:
            entropy_list.append(entropy)

    res = {
        "log_probs": log_probs_list,
    }
    if with_entropy:
        res["entropy"] = entropy_list
    return res


def compute_advantages_and_returns(args):
    log_probs: list[torch.Tensor] = get_local_storage("log_probs")
    ref_log_probs: list[torch.Tensor] = get_local_storage("ref_log_probs")
    rewards: list[float] = get_local_storage("rewards")
    values: Union[None, list[torch.Tensor]] = get_local_storage("values")
    response_lengths: list[int] = get_local_storage("response_lengths")
    loss_masks: list[torch.Tensor] = get_local_storage("loss_masks")
    total_lengths: list[int] = get_local_storage("total_lengths")

    if log_probs is None:
        return

    if args.kl_coef == 0:
        # when kl_coef is 0, we won't compute ref_log_prob
        kl = [
            torch.zeros_like(
                log_probs[i],
                dtype=torch.float32,
                device=log_probs[i].device,
            )
            for i in range(len(log_probs))
        ]
    else:
        kl = [
            compute_approx_kl(
                log_probs[i],
                ref_log_probs[i],
                kl_loss_type=args.kl_loss_type,
            )
            for i in range(len(log_probs))
        ]
    rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)

    if args.advantage_estimator == "grpo":
        returns = get_grpo_returns(rewards, kl)
        # TODO: is the copy necessary?
        advantages = [r for r in returns]

    elif args.advantage_estimator == "reinforce_plus_plus":
        returns = get_reinforce_plus_plus_returns(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            response_lengths=response_lengths,
            total_lengths=total_lengths,
            kl_coef=args.kl_coef,
            gamma=args.gamma,
        )
        advantages = [r for r in returns]

    elif args.advantage_estimator == "reinforce_plus_plus_baseline":
        advantages = get_reinforce_plus_plus_baseline_advantages(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            kl_coef=args.kl_coef,
        )
        returns = advantages

    else:
        raise NotImplementedError(f"advantage_estimator {args.advantage_estimator} is not supported. ")

    # TODO: OpenRLHF always does advantages normalization but veRL doesn't seem to do it.
    if args.normalize_advantages:
        all_advs = torch.cat(advantages)
        cp_size = mpu.get_context_parallel_world_size()
        if cp_size == 1:
            all_masks = torch.cat(loss_masks)
        else:
            mask_chunks = []
            for i in range(len(loss_masks)):
                full_mask = loss_masks[i]
                total_len, response_len, prompt_len = total_lengths[i], response_lengths[i], total_lengths[i] - response_lengths[i]

                _, _, _, my_offsets = get_logits_and_tokens_offset_with_cp(total_len, response_len)
                
                s_start, e_start = my_offsets[0][0] - prompt_len, my_offsets[0][1] - prompt_len
                s_end, e_end = my_offsets[1][0] - prompt_len, my_offsets[1][1] - prompt_len
                
                mask_chunk = torch.cat([full_mask[s_start:e_start], full_mask[s_end:e_end]])
                mask_chunks.append(mask_chunk)
            all_masks = torch.cat(mask_chunks)

        assert all_advs.size() == all_masks.size(), \
            f"Shape mismatch before whitening: advantages {all_advs.size()}, masks {all_masks.size()}"

        whitened_advs_flat = distributed_masked_whiten(all_advs, all_masks, shift_mean=True)
        chunk_lengths = [chunk.size(0) for chunk in advantages]
        advantages = list(torch.split(whitened_advs_flat, chunk_lengths))

    set_local_storage("advantages", advantages)
    set_local_storage("returns", returns)


def policy_loss_function(args, batch, logits, sum_of_sample_mean):
    advantages = torch.cat(batch["advantages"], dim=0)
    old_log_probs = torch.cat(batch["log_probs"], dim=0)

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=True,
    )

    log_probs = log_probs_and_entropy["log_probs"]
    entropy = log_probs_and_entropy["entropy"]

    log_probs = torch.cat(log_probs, dim=0)
    entropy = torch.cat(entropy, dim=0)

    entropy_loss = sum_of_sample_mean(entropy)

    pg_loss, pg_clipfrac, ppo_kl = compute_policy_loss(
        log_probs, old_log_probs, advantages, args.eps_clip, args.eps_clip_high
    )
    pg_loss = sum_of_sample_mean(pg_loss)
    pg_clipfrac = sum_of_sample_mean(pg_clipfrac)
    ppo_kl = sum_of_sample_mean(ppo_kl)

    loss = pg_loss - args.entropy_coef * entropy_loss

    if args.use_kl_loss:
        ref_log_probs = batch["ref_log_probs"]
        ref_log_probs = torch.cat(ref_log_probs, dim=0)
        kl = compute_approx_kl(
            log_probs,
            ref_log_probs,
            kl_loss_type=args.kl_loss_type,
        )
        kl_loss = sum_of_sample_mean(kl)

        loss = loss + args.kl_loss_coef * kl_loss
    else:
        kl_loss = torch.tensor(0.0, device=log_probs.device)

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    return (
        loss,
        {
            "loss": loss.clone().detach(),
            "pg_loss": pg_loss.clone().detach(),
            "entropy_loss": entropy_loss.clone().detach(),
            "pg_clipfrac": pg_clipfrac.clone().detach(),
            "ppo_kl": ppo_kl.clone().detach(),
            "kl_loss": kl_loss.clone().detach(),
        },
    )


def sft_loss_function(args, batch, logits, sum_of_sample_mean):
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
    )

    log_probs = log_probs_and_entropy["log_probs"]
    log_probs = torch.cat(log_probs, dim=0)
    loss = -sum_of_sample_mean(log_probs)

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    return (
        loss,
        {
            "loss": loss.clone().detach(),
        },
    )


def loss_function(args, batch, num_microbatches, logits):
    num_tokens = sum(batch["response_lengths"])
    num_samples = len(batch["response_lengths"])

    sum_of_sample_mean = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.calculate_per_token_loss,
    )

    loss_function_kwargs = {
        "args": args,
        "batch": batch,
        "logits": logits,
        "sum_of_sample_mean": sum_of_sample_mean,
    }

    match args.loss_type:
        case "policy_loss":
            loss, log = policy_loss_function(**loss_function_kwargs)
        case "sft_loss":
            loss, log = sft_loss_function(**loss_function_kwargs)
        case "custom_loss":
            custom_loss_function = load_function(args.custom_loss_function_path)
            loss, log = custom_loss_function(**loss_function_kwargs)
        case _:
            raise ValueError(f"Unknown loss type: {args.loss_type}")

    # Here we need to divide by cp_size because to cancel the multiply in Megatron.
    loss = (
        loss * num_microbatches / args.global_batch_size * mpu.get_data_parallel_world_size(with_context_parallel=True)
    )

    return (
        loss,
        num_tokens if args.calculate_per_token_loss else 1,
        {
            "keys": list(log.keys()),
            "values": torch.tensor(
                [
                    num_samples if not args.calculate_per_token_loss else num_tokens,
                ]
                + list(log.values()),
                device=logits.device,
            ),
        },
    )
