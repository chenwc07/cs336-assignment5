import json
import random

from typing import Callable, Dict
from vllm import LLM, SamplingParams
from cs336_alignment.utils.grpo_utils import compute_group_normalized_rewards

def grpo_rollout(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], Dict[str, float]],
    grpo_step: int,
    cfg=None
):
    grpo_sampling_params = SamplingParams(
        temperature=cfg.rollout.sampling_temperature, 
        top_p=1.0, 
        max_tokens=cfg.rollout.sampling_max_tokens, 
        min_tokens=cfg.rollout.sampling_min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        n=cfg.rollout.group_size,
        seed=42
    )

    # load training data
    prompt_template_path=cfg.data.prompt_template_path
    if prompt_template_path is not None:
        with open(prompt_template_path, 'r') as f:
            prompt_template = f.read()
    else:
        prompt_template = "{question}"

    train_data = []
    with open(cfg.train.data_path, "r") as f:
        for line in f:
            train_data.append(json.loads(line))

    assert cfg.rollout.rollout_batch_size % cfg.rollout.group_size == 0, (
    "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = cfg.rollout.rollout_batch_size // cfg.rollout.group_size
    sampled_data = random.sample(train_data, n_prompts_per_rollout_batch)

    # generate rollouts
    prompts = [prompt_template.format(question=data["problem"]) for data in sampled_data]
    outputs = vllm_model.generate(prompts, sampling_params=grpo_sampling_params)

    grpo_samples = []
    for output, data in zip(outputs, sampled_data):
        prompt = output.prompt
        output_texts = [out.text for out in output.outputs] # list of generated texts per prompt
        for gen_text in output_texts:
            # rewards = reward_fn(gen_text, data["solution"])
            grpo_samples.append({
                "problem": prompt,
                "rollout": gen_text,
                "solution": data["solution"],
            })
    
    with open(cfg.rollout.rollout_path + f"grpo_samples_{grpo_step}.jsonl", "w") as f:
        for sample in grpo_samples:
            f.write(json.dumps(sample) + "\n")


    # compute advantates
    repeated_ground_truths = [sample["solution"] for sample in grpo_samples]
    rollout_responses = [sample["rollout"] for sample in grpo_samples]
    advantages, rewards, advantage_stats = compute_group_normalized_rewards(
        reward_fn,
        rollout_responses,
        repeated_ground_truths,
        cfg.rollout.group_size,
        cfg.rollout.advantage_eps,
        cfg.rollout.use_std_normalization
    )

    for sample, adv, rew in zip(grpo_samples, advantages, rewards):
        sample["advantage"] = adv.item()
        sample["reward"] = rew.item()

    return {
        "rollouts": grpo_samples,
        "mean_reward": advantage_stats['mean_reward'],
        "mean_adv": advantage_stats['mean_adv'],
        "max_adv": advantage_stats['max_adv'],
        "min_adv": advantage_stats['min_adv'],
    }


class GRPOTrainDataset():
    def __init__(self, samples):
        self.ptr = 0
        self._load_data(samples)

    def _load_data(self, samples):
        self.samples = []
        for data in samples:
            prompt = data["problem"]
            answer = data["rollout"]
            reward = data["reward"]
            advantage = data["advantage"]
            self.samples.append((prompt, answer, reward, advantage))
        random.shuffle(self.samples)

    def get_batch(self, batch_size):
        batch_prompts = []
        batch_answers = []
        batch_rewards = []
        batch_advantages = []
        for _ in range(batch_size):
            if self.ptr >= len(self.samples):
                self.reset_pointer()
            prompt, answer, reward, advantage = self.samples[self.ptr]
            batch_prompts.append(prompt)
            batch_answers.append(answer)
            batch_rewards.append(reward)
            batch_advantages.append(advantage)
            self.ptr += 1
        return batch_prompts, batch_answers, batch_rewards, batch_advantages

    def get_length(self):
        return len(self.samples)
    
    def reset_pointer(self):
        self.ptr = 0
        # shuffle the data here if needed
        random.shuffle(self.samples)