from transformers import AutoTokenizer
import torch
from transformers import PreTrainedModel
from typing import Callable

def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    assert len(prompt_strs) == len(output_strs), "The number of prompts must match the number of outputs."
    batch_size = len(prompt_strs)
    prompt_token_ids = tokenizer(
        prompt_strs,
    ).input_ids
    output_token_ids = tokenizer(
        output_strs,
    ).input_ids
    # pad_token_id = tokenizer.pad_token_id
    max_length = max(len(prompt_ids + output_ids) for prompt_ids, output_ids in zip(prompt_token_ids, output_token_ids))
    input_ids = torch.empty((batch_size, max_length-1), dtype=torch.int).fill_(tokenizer.pad_token_id)
    labels = torch.empty((batch_size, max_length-1), dtype=torch.int).fill_(tokenizer.pad_token_id)
    response_mask = torch.zeros((batch_size, max_length-1), dtype=torch.bool)
    for i in range(batch_size):
        prompt_ids = prompt_token_ids[i]
        output_ids = output_token_ids[i]
        input_len = len(prompt_ids)
        output_len = len(output_ids)
        if input_len + output_len - 1 < max_length - 1:
            input_ids[i, :input_len+output_len] = torch.tensor(prompt_ids+output_ids, dtype=torch.int)  # for passing the test
        else:
            input_ids[i, :input_len+output_len-1] = torch.tensor(prompt_ids+output_ids[:-1], dtype=torch.int)
        labels[i, :input_len+output_len-1] = torch.tensor(prompt_ids[1:]+output_ids, dtype=torch.int)
        response_mask[i, input_len-1:input_len+output_len-1] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }

def compute_entropy(logits) -> torch.Tensor:
    """
    Compute the entropy of the probability distribution defined by the logits.

    Args:
        logits: A tensor of shape (batch_size, seq_len, vocab_size) representing the logits.

    Returns:
        A tensor of shape (batch_size, seq_len) representing the entropy for each position.
    """
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,  # (batch_size, seq_len)
    labels: torch.Tensor,   # (batch_size, seq_len)
    return_token_entropy: bool = False
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits   # (batch_size, seq_len, vocab_size)
    log_probs = torch.log_softmax(logits, dim=-1) # (batch_size, seq_len, vocab_size)
    response_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=labels.unsqueeze(-1).long()  # (batch_size, seq_len, 1)
    )
    response_log_probs = response_log_probs.squeeze(-1)  # (batch_size, seq_len)
    if return_token_entropy:
        token_entropy = compute_entropy(logits)  # (batch_size, seq_len)
        return {
            "log_probs": response_log_probs,
            "token_entropy": token_entropy
        }
    else:
        return {
            "log_probs": response_log_probs
        }

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Sum over a dimension and normalize by a constant, considering only those elements where mask == 1.

    Args:
        tensor: The input tensor to be summed and normalized.
        mask: A boolean tensor of the same shape as `tensor`, where True indicates the elements to consider.
        normalize_constant: The constant by which to normalize the summed values.
        dim: The dimension or dimensions to sum over. If None, sum over all dimensions.
    """
    masked_tensor = tensor * mask
    summed = masked_tensor.sum(dim=dim)
    normalized = summed / normalize_constant
    return normalized
    
def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,  # (batch_size, seq_len)
    response_mask: torch.Tensor,  # (batch_size, seq_len)
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
    per_token_loss: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if per_token_loss:
        response_lengths = response_mask.sum(dim=-1)
        loss = - masked_normalize(policy_log_probs, response_mask, response_lengths, dim=-1).mean()
    else:
        loss = - masked_normalize(policy_log_probs, response_mask, normalize_constant, dim=-1).mean()
    scaled_loss = loss / gradient_accumulation_steps
    scaled_loss.backward()
    meta_data = {
        "loss": loss.item(),
        "scaled_loss": scaled_loss.item()
    }
    return scaled_loss, meta_data

def log_generations():
    pass
    


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("/data1/chenweicong/models/Qwen/Qwen2.5-Math-1.5B")
    prompt_strs = ['Hello, world!', 'This is a test.', 'This is another test.']
    output_strs = ['Hello, world!', 'This is a test.', 'This is another test.']
    tokenized_outputs = tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
    print("Tokenized Outputs:", tokenized_outputs)