import json
import random
from typing import Callable, Dict

from vllm import LLM, SamplingParams


def generate_ei_samples(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], Dict[str, float]],
    ei_step: int,
    cfg=None
):
    train_data_path=cfg.train.data_path
    n_samples_from_trainset=cfg.train.n_samples_from_trainset
    G_per_sample=cfg.train.G_per_sample
    prompt_template_path=cfg.data.prompt_template_path

    # sampling parameters for EI generation
    ei_sampling_params = SamplingParams(
        temperature=1.0, 
        top_p=1.0, 
        max_tokens=1024, 
        min_tokens=4,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        n=G_per_sample,
        seed=42
    )

    # load training data
    if prompt_template_path is not None:
        with open(prompt_template_path, 'r') as f:
            prompt_template = f.read()
    else:
        prompt_template = "{question}"

    train_data = []
    with open(train_data_path, "r") as f:
        for line in f:
            train_data.append(json.loads(line))
    
    sampled_data = random.sample(train_data, n_samples_from_trainset)
    prompts = [prompt_template.format(question=data["problem"]) for data in sampled_data]
    outputs = vllm_model.generate(prompts, sampling_params=ei_sampling_params)

    ei_samples = []
    for output, data in zip(outputs, sampled_data):
        prompt = output.prompt
        output_texts = [out.text for out in output.outputs] # list of generated texts per prompt
        for gen_text in output_texts:
            rewards = reward_fn(gen_text, data["solution"])
            if rewards.get("reward", 0.0) > 0.0:
                ei_samples.append({
                    "problem": prompt,
                    "rollout": gen_text,
                    "solution": data["solution"],
                })
    
    with open(cfg.train.ei_rollout_path + f"ei_samples_{ei_step}.jsonl", "w") as f:
        for sample in ei_samples:
            f.write(json.dumps(sample) + "\n")

    return {
        'path': cfg.train.ei_rollout_path + f"ei_samples_{ei_step}.jsonl",
        'rollout_acc': len(ei_samples) / (n_samples_from_trainset * G_per_sample),
        'n_ei_samples': len(ei_samples)
    }

class EITrainDataset():
    def __init__(self, data_path, prompt_template_path=None, max_samples=None):
        self.data_path = data_path
        self.prompt_template_path = prompt_template_path
        self.max_samples = max_samples
        self.ptr = 0
        self._load_data()

    def _load_data(self):
        if self.prompt_template_path is not None:
            with open(self.prompt_template_path, 'r') as f:
                self.prompt_template = f.read()
        else:
            self.prompt_template = "{question}"

        self.samples = []
        with open(self.data_path, "r") as f:
            for line in f:
                data = json.loads(line)
                prompt = self.prompt_template.format(question=data["problem"])
                answer = data["rollout"]
                self.samples.append((prompt, answer))
                if self.max_samples is not None and len(self.samples) >= self.max_samples:
                    break
        random.shuffle(self.samples)

    def get_batch(self, batch_size):
        batch_prompts = []
        batch_answers = []
        for _ in range(batch_size):
            if self.ptr >= len(self.samples):
                self.reset_pointer()
            prompt, answer = self.samples[self.ptr]
            batch_prompts.append(prompt)
            batch_answers.append(answer)
            self.ptr += 1
        return batch_prompts, batch_answers

    def get_length(self):
        return len(self.samples)
    
    def reset_pointer(self):
        self.ptr = 0
        # shuffle the data here if needed
        random.shuffle(self.samples)