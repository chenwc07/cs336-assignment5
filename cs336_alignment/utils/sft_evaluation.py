import json
import torch
from vllm import LLM, SamplingParams
from typing import Callable, List, Dict
from transformers import PreTrainedModel, AutoTokenizer
from tqdm import tqdm
from cs336_alignment.utils.utils import get_response_log_probs, tokenize_prompt_and_output

def load_evaluation_data(
    data_path: str,
    max_samples: int = 64,
    prompt_template_path: str = None
):
    if prompt_template_path is not None:
        with open(prompt_template_path, 'r') as f:
            prompt_template = f.read()
    else:
        prompt_template = "{question}"

    prompts = []
    answers = []
    with open(data_path, "r") as f:
        for line in tqdm(f):
            data = json.loads(line)
            prompts.append(prompt_template.format(question=data["problem"]))
            answers.append(data["solution"])
            if len(prompts) >= max_samples:
                break
    return prompts, answers


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], Dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams,
    answers: List[str]
):
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics
    """
    outputs = vllm_model.generate(prompts, sampling_params=eval_sampling_params)
    results = []
    format_reward = 0.0
    answer_reward = 0.0
    total_reward = 0.0
    for output, answer in zip(outputs, answers):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        rewards = reward_fn(generated_text, answer)
        results.append({
            "prompt": prompt,
            "generated_text": generated_text,
            "answer": answer,
            "rewards": rewards
        })

        format_reward += rewards.get("format_reward", 0.0)
        answer_reward += rewards.get("answer_reward", 0.0)
        total_reward += rewards.get("reward", 0.0)

    num_samples = len(prompts)
    avg_format_reward = format_reward / num_samples
    avg_answer_reward = answer_reward / num_samples
    avg_total_reward = total_reward / num_samples
    return {
        "results": results,
        "avg_format_reward": avg_format_reward,
        "avg_answer_reward": avg_answer_reward,
        "avg_total_reward": avg_total_reward
    }

@torch.no_grad()
def evaluate_model(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    evaluation_vllm_results: dict,
    cfg=None
):
    # cfg
    device = cfg.policy.device
    test_batch_size = cfg.eval.batch_size

    # get data
    vllm_results = evaluation_vllm_results["results"]
    prompts = [item["prompt"] for item in vllm_results]
    vllm_response = [item["generated_text"] for item in vllm_results]
    answers = [item["answer"] for item in vllm_results]
    rewards = [item["rewards"]["reward"] for item in vllm_results]

    # metrics
    response_token_entropy_accum = 0.0
    response_token_log_prob_accum = 0.0
    response_token_count = 0
    correct_response_token_count = 0
    incorrect_response_token_count = 0
    num_correct = 0
    num_incorrect = 0

    # evaluate
    for i in range(0, len(prompts), test_batch_size):
        batch_prompts = prompts[i: i + test_batch_size]
        batch_responses = vllm_response[i: i + test_batch_size]
        batch_rewards = rewards[i: i + test_batch_size]

        tokenized_batch = tokenize_prompt_and_output(
            batch_prompts,
            batch_responses,
            tokenizer
        )

        input_ids = tokenized_batch["input_ids"].to(device)
        labels = tokenized_batch["labels"].to(device)
        response_mask = tokenized_batch["response_mask"].to(device)

        log_probs_dict = get_response_log_probs(
            model,
            input_ids,
            labels,
            return_token_entropy=True
        )

        log_probs = log_probs_dict["log_probs"]  # (batch_size, seq_len)
        token_entropy = log_probs_dict["token_entropy"]  # (batch_size, seq_len)

        # 回答长度
        response_token_count += response_mask.sum().item()

        # 累计回答部分的token entropy 和log_probs
        response_token_entropy_accum += (token_entropy * response_mask).sum().item()
        response_token_log_prob_accum += (log_probs * response_mask).sum().item()

        # 记录正确/错误回答的长度，累计回答部分的序列log概率
        for j, reward in enumerate(batch_rewards):
            num_response_tokens = response_mask[j].sum().item()
            # response_seq_log_prob_accum += (log_probs[j] * response_mask[j]).sum().item() / num_response_tokens
            if reward > 0:
                correct_response_token_count += num_response_tokens
                num_correct += 1
            else:
                incorrect_response_token_count += num_response_tokens
                num_incorrect += 1

    avg_response_token_entropy = response_token_entropy_accum / response_token_count
    avg_response_token_log_prob = response_token_log_prob_accum / response_token_count
    avg_correct_response_length = correct_response_token_count / num_correct if num_correct > 0 else 0.0
    avg_incorrect_response_length = incorrect_response_token_count / num_incorrect if num_incorrect > 0 else 0.0

    model.train()
    return {
        "avg_response_token_entropy": avg_response_token_entropy,
        "avg_response_token_log_prob": avg_response_token_log_prob,
        "avg_correct_response_length": avg_correct_response_length,
        "avg_incorrect_response_length": avg_incorrect_response_length,
        "avg_format_reward": evaluation_vllm_results["avg_format_reward"],
        "avg_answer_reward": evaluation_vllm_results["avg_answer_reward"],
        "avg_total_reward": evaluation_vllm_results["avg_total_reward"],
    }