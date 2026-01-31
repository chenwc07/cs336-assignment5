import json
from vllm import LLM, SamplingParams
from typing import Callable, List, Dict
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], Dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams,
    answers: List[str]
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    outputs = vllm_model.generate(prompts, sampling_params=eval_sampling_params)
    results = []
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
    
    # Serialize results to disk
    with open("data/hendrycks_math/evaluation_results.jsonl", "a") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    sampling_params = SamplingParams(
        temperature=1.0, 
        top_p=1.0, 
        max_tokens=1024, 
        stop=["</answer>"],
        include_stop_str_in_output=True
    )
    
    llm = LLM(
        model="/data1/chenweicong/models/Qwen/Qwen2.5-Math-1.5B",
        tensor_parallel_size=4,
        )

    with open('cs336_alignment/prompts/r1_zero.prompt', 'r') as f:
        prompt_template = f.read()


    with open("data/hendrycks_math/validation.jsonl", "r") as f:
        prompts = []
        answers = []
        for line in f:
            data = json.loads(line)
            prompts.append(prompt_template.format(question=data["problem"]))
            answers.append(data["solution"])
            if len(prompts) >= 64:
                evaluate_vllm(
                    vllm_model=llm,
                    reward_fn=r1_zero_reward_fn,
                    prompts=prompts,
                    eval_sampling_params=sampling_params,
                    answers=answers
                )
                prompts = []
                answers = []
        if prompts:
            evaluate_vllm(
                vllm_model=llm,
                reward_fn=r1_zero_reward_fn,
                prompts=prompts,
                eval_sampling_params=sampling_params,
                answers=answers
            )