# Adapt from https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/models/utils.py
# and https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/trainer/ppo_utils/experience_maker.py
from typing import Optional, List, Tuple

import torch
import torch.distributed as dist

from megatron.core import mpu
from slime.utils.distributed_utils import distributed_masked_whiten
from slime.backends.megatron_utils.cp_utils import get_logits_and_tokens_offset_with_cp

@torch.compile(dynamic=True)
def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    kl_loss_type: str,
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        action_mask: Mask for actions.
    """

    log_ratio = log_probs.float() - log_probs_base.float()

    if kl_loss_type == "kl":
        return log_ratio
    elif kl_loss_type == "k2":
        log_ratio = log_probs.float() - log_probs_base.float()
        log_ratio = log_ratio**2 / 2.0
        return log_ratio
    elif kl_loss_type == "k3":
        log_ratio = -log_ratio
        log_ratio = log_ratio.exp() - 1 - log_ratio
        return log_ratio
    elif kl_loss_type == "low_var_kl":
        # The non negative kl approximation in
        # http://joschu.net/blog/kl-approx.html
        # Besides non negative, it is also unbiased and have lower variance.
        log_ratio = -log_ratio
        log_ratio = log_ratio.exp() - 1 - log_ratio
        return torch.clamp(log_ratio, min=-10, max=10)
    else:
        raise ValueError(f"Unknown kl_loss_type: {kl_loss_type}")


@torch.compile(dynamic=True)
def compute_policy_loss(
    log_probs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
    eps_clip_c: Optional[float] = None,
):
    approx_kl = old_logprobs - log_probs
    ratio = (-approx_kl).exp()
    pg_losses1 = -ratio * advantages
    pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    clipfrac = torch.gt(pg_losses2, pg_losses1).float()

    if eps_clip_c is not None:
        assert (
            eps_clip_c > 1.0
        ), f"The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0, but get the value: {eps_clip_c}."
        pg_losses3 = -eps_clip_c * advantages
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    else:
        pg_losses = clip_pg_losses1

    return pg_losses, clipfrac, approx_kl


def compute_log_probs(logits: torch.Tensor, tokens: torch.Tensor, process_group: Optional[dist.ProcessGroup]):
    from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy

    # convert to [seq_len, batch_size, vocab_size] as expected by fused_vocab_parallel_cross_entropy
    logits = logits.unsqueeze(1)
    tokens = tokens.unsqueeze(1)
    return -fused_vocab_parallel_cross_entropy(logits, tokens, process_group)


# from https://github.com/volcengine/verl/blob/0bdf7f469854815177e73dcfe9e420836c952e6e/verl/utils/megatron/tensor_parallel.py#L99
class _VocabParallelEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor, process_group: dist.ProcessGroup) -> torch.Tensor:

        @torch.compile(dynamic=True)
        def mul_reduce(a, b):
            return (a * b).sum(dim=-1, keepdim=True)

        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=process_group)
        normalized_vocab_parallel_logits = vocab_parallel_logits - logits_max
        normalized_exp_logits = normalized_vocab_parallel_logits.exp_()
        normalized_sum_exp_logits = normalized_exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(normalized_sum_exp_logits, group=process_group)
        softmax_logits = normalized_exp_logits.div_(normalized_sum_exp_logits)
        sum_softmax_times_logits = mul_reduce(softmax_logits, vocab_parallel_logits)
        dist.all_reduce(sum_softmax_times_logits, group=process_group)
        entropy = logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits
        ctx.save_for_backward(vocab_parallel_logits, softmax_logits, sum_softmax_times_logits)
        return entropy.squeeze(dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = ctx.saved_tensors
        # reuse softmax_logits as grad
        vocab_parallel_logits.sub_(sum_softmax_times_logits)
        softmax_logits.mul_(vocab_parallel_logits)
        softmax_logits.mul_(grad_output.unsqueeze(dim=-1))
        # recover vocab_parallel_logits
        vocab_parallel_logits.add_(sum_softmax_times_logits)
        softmax_logits.mul_(-1)
        return softmax_logits, None


def compute_entropy_from_logits(logits: torch.Tensor, process_group) -> torch.Tensor:
    return _VocabParallelEntropy.apply(logits, process_group)


def get_grpo_returns(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
):
    returns = []
    for i in range(len(rewards)):
        returns.append(torch.ones_like(kl[i]) * rewards[i])
    return returns


def get_reinforce_plus_plus_returns(
    rewards: torch.Tensor,
    kl: List[torch.Tensor],
    loss_masks: List[torch.Tensor],
    response_lengths: List[int],
    total_lengths: List[int],
    kl_coef: float,
    gamma: float,
) -> List[torch.Tensor]:
    """
    Calculates discounted returns for REINFORCE++ (https://arxiv.org/pdf/2501.03262)

    Args:
        rewards (Tensor): A tensor of scalar rewards for each sequence.
        kl (List[Tensor]): List of per-token KL divergence tensors for sequence chunks.
        loss_masks (List[Tensor]): List of response-only loss masks for each full sequence.
        response_lengths (List[int]): The full length of each response sequence.
        total_lengths (List[int]): The full length of each sequence (prompt + response).
        kl_coef (float): Coefficient for the KL penalty.
        gamma (float): The discount factor.

    Returns:
        List[torch.Tensor]: A list of return (G_t) tensors for the
                            local sequence chunks owned by the current GPU rank.
    """
    cp_size = mpu.get_context_parallel_world_size()

    # Step 1: Reconstruct full sequences and calculate full returns.
    batch_returns = []
    batch_offsets = [] if cp_size > 1 else None

    for i in range(len(rewards)):
        if cp_size == 1:
            full_kl_response = kl[i]
        else:
            my_kl_chunk = kl[i]
            total_len, response_len, prompt_len = total_lengths[i], response_lengths[i], total_lengths[i] - response_lengths[i]
            device, dtype = my_kl_chunk.device, my_kl_chunk.dtype

            _, _, _, my_offsets = get_logits_and_tokens_offset_with_cp(total_len, response_len)
            batch_offsets.append(my_offsets)

            # Gather all chunks and their layout information from all ranks in the CP group.
            object_to_gather = {"kl_chunk": my_kl_chunk.cpu(), "offsets": my_offsets}
            gathered_objects = [None] * cp_size
            dist.all_gather_object(gathered_objects, object_to_gather, group=mpu.get_context_parallel_group())
            
            full_kl_response = torch.zeros(response_len, device=device, dtype=dtype)
            for obj in gathered_objects:
                kl_chunk, global_offsets = obj['kl_chunk'], obj['offsets']
                s_start, e_start = global_offsets[0][0] - prompt_len, global_offsets[0][1] - prompt_len
                s_end, e_end = global_offsets[1][0] - prompt_len, global_offsets[1][1] - prompt_len
                kl_chunk_start, kl_chunk_end = torch.split(kl_chunk.to(device), [e_start - s_start, e_end - s_end])

                if kl_chunk_start.numel() > 0:
                    full_kl_response[s_start:e_start] = kl_chunk_start
                if kl_chunk_end.numel() > 0:
                    full_kl_response[s_end:e_end] = kl_chunk_end

        token_level_rewards = -kl_coef * full_kl_response
        full_mask = loss_masks[i]
        assert full_mask.sum().item() > 0, f"Sequence at index {i} is fully masked."
        last_idx = full_mask.nonzero(as_tuple=True)[0][-1]
        token_level_rewards[last_idx] += rewards[i]
        
        returns_for_seq = torch.zeros_like(token_level_rewards)
        running_return = 0.0
        for t in reversed(range(token_level_rewards.size(0))):
            # G_t = r_t + gamma * G_{t+1}
            running_return = token_level_rewards[t] + gamma * running_return
            returns_for_seq[t] = running_return
        
        batch_returns.append(returns_for_seq)

    # Step 2: Extract the final, local chunks from the full returns.
    final_returns_chunks = []
    for i in range(len(rewards)):
        full_returns = batch_returns[i]
        if cp_size == 1:
            returns_chunk = full_returns
        else:
            my_offsets = batch_offsets[i]
            prompt_len = total_lengths[i] - response_lengths[i]
            s_start, e_start = my_offsets[0][0] - prompt_len, my_offsets[0][1] - prompt_len
            s_end, e_end = my_offsets[1][0] - prompt_len, my_offsets[1][1] - prompt_len
            returns_chunk = torch.cat([full_returns[s_start:e_start], full_returns[s_end:e_end]])
        
        final_returns_chunks.append(returns_chunk)
        
    return final_returns_chunks

def get_reinforce_plus_plus_baseline_advantages(
    rewards: torch.Tensor,
    kl: List[torch.Tensor],
    loss_masks: List[torch.Tensor],
    kl_coef: float,
) -> List[torch.Tensor]:
    """
    Calculates the unwhitened advantages for the REINFORCE++-baseline algorithm.
    Broadcasting the scalar (reward - group_baseline) to each token.

    Args:
        rewards (Tensor): A tensor of scalar rewards, where the group-wise
                                baseline has already been subtracted.
        kl (list[Tensor]): A list of per-token KL divergence tensors. Used to
                                 get the shape for broadcasting.
        loss_masks (list[Tensor]): A list of per-token loss masks.
        kl_coef (float): Coefficient for the KL penalty.

    Returns:
        list[Tensor]: A list of tensors containing the unwhitened advantages.
    """
    # Broadcast to get unwhitened advantages
    unwhitened_advantages = [
        torch.ones_like(kl_tensor) * reward_val - kl_coef * kl_tensor
        for kl_tensor, reward_val in zip(kl, rewards)
    ]
    
    return unwhitened_advantages
