import torch
from einops import rearrange
from typing import Callable, Literal
from cs336_alignment.utils.utils import masked_normalize

def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],  # rollout_batch_size = n_prompts_per_rollout_batch * group_size
    repeated_ground_truths: list[str], # rollout_batch_size = n_prompts_per_rollout_batch * group_size
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    rewards = [
        reward_fn(rollout_response, repeated_ground_truth)['reward']
        for rollout_response, repeated_ground_truth in zip(rollout_responses, repeated_ground_truths)
    ] # rollout_batch_size
    rewards = torch.tensor(rewards, dtype=torch.float32)  # rollout_batch_size
    n_prompts_per_rollout_batch = rewards.shape[0] // group_size
    rewards = rearrange(rewards, '(b g) -> b g', g=group_size)  # n_prompts_per_rollout_batch x group_size
    group_mean = torch.mean(rewards, dim=1, keepdim=True)  # n_prompts_per_rollout_batch x 1
    if normalize_by_std:
        group_std = torch.std(rewards, dim=1, keepdim=True) # n_prompts_per_rollout_batch x 1
        advantages = (rewards - group_mean) / (group_std + advantage_eps)  # n_prompts_per_rollout_batch x group_size
    else:
        advantages = rewards - group_mean  # n_prompts_per_rollout_batch x group_size
    advantages = rearrange(advantages, 'b g -> (b g)')  # rollout_batch_size
    
    rewards = rearrange(rewards, 'b g -> (b g)')  # rollout_batch_size
    return advantages, rewards, {'mean_reward': group_mean.mean().item(), 'mean_adv': advantages.mean().item(), 'max_adv': advantages.max().item(), 'min_adv': advantages.min().item()}

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,  # (batch_size, 1)
    policy_log_probs: torch.Tensor, # (batch_size, seq_len)
) -> torch.Tensor:
    return - raw_rewards_or_advantages * policy_log_probs # (batch_size, seq_len)

def compute_grpo_clip_loss(
    advantages: torch.Tensor, # (batch_size, 1)
    policy_log_probs: torch.Tensor, # (batch_size, seq_len)
    old_log_probs: torch.Tensor,  # (batch_size, seq_len)
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1): 
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length): 
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss 
                (used to compute clip fraction).
    """
    importance_ratios = torch.exp(policy_log_probs - old_log_probs)  # (batch_size, seq_len)
    unclipped_loss = advantages * importance_ratios  # (batch_size, seq_len)
    clipped_loss = advantages * torch.clamp(importance_ratios, 1.0 - cliprange, 1.0 + cliprange)  # (batch_size, seq_len)
    is_clipped = clipped_loss < unclipped_loss
    loss = - torch.min(unclipped_loss, clipped_loss)  # (batch_size, seq_len)
    return loss, {'is_clipped': is_clipped}

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor, # (batch_size, seq_len)
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,  # (batch_size, 1)
    advantages: torch.Tensor | None = None,  # (batch_size, 1)
    old_log_probs: torch.Tensor | None = None, # (batch_size, seq_len)
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.
    """
    if loss_type == "no_baseline":
        assert raw_rewards is not None
        loss = compute_naive_policy_gradient_loss(
            raw_rewards,
            policy_log_probs,
        )
        return loss, {}
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None
        loss = compute_naive_policy_gradient_loss(
            advantages,
            policy_log_probs,
        )
        return loss, {}
    elif loss_type == "grpo_clip":
        assert advantages is not None
        assert old_log_probs is not None
        assert cliprange is not None
        return compute_grpo_clip_loss(
            advantages,
            policy_log_probs,
            old_log_probs,
            cliprange,
        )
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    masked_tensor = tensor * mask
    summed = masked_tensor.sum(dim=dim)
    count = mask.sum(dim=dim)
    return summed / count

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,  # (batch_size, seq_len)
    response_mask: torch.Tensor,  # (batch_size, seq_len)
    gradient_accumulation_steps: int, 
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,  # (batch_size, 1)
    advantages: torch.Tensor | None = None,  # (batch_size, 1)
    old_log_probs: torch.Tensor | None = None,  # (batch_size, seq_len)
    cliprange: float | None = None,
    constant_normalize_factor: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length): 
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], 
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio. 
            Needed for loss_type="grpo_clip".
        constant_normalize_factor: int | None, provided if we want to sum over 
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: 
            the policy gradient loss and its metadata.
    """
    per_token_loss, loss_metadata = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange,
    )
    if constant_normalize_factor is None:
        loss = masked_mean(
            per_token_loss,
            response_mask,
            dim=-1,
        ).mean()
        scaled_loss = loss / gradient_accumulation_steps
    else:
        loss = masked_normalize(
            per_token_loss,
            response_mask,
            constant_normalize_factor,
            dim=-1,
        )
        scaled_loss = loss / gradient_accumulation_steps
    scaled_loss.backward()
    loss_metadata.update({
        "loss": loss.item(),
        "scaled_loss": scaled_loss.item(),
    })
    return scaled_loss, loss_metadata



    