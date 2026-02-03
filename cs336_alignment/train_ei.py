import os
from dotenv import load_dotenv
load_dotenv()
import hydra
from omegaconf import DictConfig
import torch
import wandb
from tqdm import tqdm

from cs336_alignment.utils.sft_evaluation import evaluate_vllm, evaluate_model, load_evaluation_data
from cs336_alignment.utils.helper_fns import init_vllm, load_policy_into_vllm_instance
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.utils.utils import tokenize_prompt_and_output, get_response_log_probs, sft_microbatch_train_step
from cs336_alignment.utils.ei_dataloader import generate_ei_samples, EITrainDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import SamplingParams

@hydra.main(config_path="../conf", config_name="ei", version_base=None)
def main(cfg: DictConfig):
    # ========== Initialize VLLM Model and Policy Model ==========
    sampling_params = SamplingParams(
        temperature=1.0, 
        top_p=1.0, 
        max_tokens=1024, 
        stop=["</answer>"],
        include_stop_str_in_output=True
    )
    vllm_model = init_vllm(
        model_id=cfg.vllm.model_id,
        device=cfg.vllm.device,
        seed=cfg.vllm.seed,
    )
    policy_model = AutoModelForCausalLM.from_pretrained(
        cfg.policy.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation=cfg.policy.attn_implementation,
        device_map=cfg.policy.device
    )
    
    # Enable gradient checkpointing to save memory
    policy_model.gradient_checkpointing_enable()
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.policy.model_id)

    # ========== Load Evaluation Data ==========
    eval_prompts, eval_answers = load_evaluation_data(
        data_path=cfg.eval.data_path,
        max_samples=cfg.eval.max_eval_samples,
        prompt_template_path=cfg.data.prompt_template_path
    )

    # ========== Load Optimizer ==========
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay
    )

    # ========== Initialize WandB Logger ==========
    wandb.login(key=os.getenv("WANDB_KEY"))
    logger = wandb.init(
        entity="cwc7", 
        project="cs336-alignment-ei", 
        name=cfg.train.run_name
    )
    wandb.define_metric("train_step") # the x‑axis for training
    wandb.define_metric("ei_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="ei_step")
    wandb.define_metric("ei/*", step_metric="ei_step")

    # ========== Evaluate before training ==========
    vllm_results = evaluate_vllm(
        vllm_model=vllm_model,
        reward_fn=r1_zero_reward_fn,
        prompts=eval_prompts,
        eval_sampling_params=sampling_params,
        answers=eval_answers
    )

    model_results = evaluate_model(
        model=policy_model,
        tokenizer=tokenizer,
        evaluation_vllm_results=vllm_results,
        cfg=cfg
    )
    logger.log({
        "ei_step": 0,
        "eval/token_entropy": model_results["avg_response_token_entropy"],
        "eval/token_log_prob": model_results["avg_response_token_log_prob"],
        "eval/avg_correct_response_length": model_results["avg_correct_response_length"],
        "eval/avg_incorrect_response_length": model_results["avg_incorrect_response_length"],
        "eval/avg_format_reward": model_results["avg_format_reward"],
        "eval/avg_answer_reward": model_results["avg_answer_reward"],
        "eval/avg_total_reward": model_results["avg_total_reward"],
    })

    torch.cuda.empty_cache()
    print("VLLM Evaluation Results:")
    for key, value in model_results.items():
        print(f"{key}: {value}")

    # ========== EI Training Loop ==========
    total_step = 0
    for ei_step in range(1, cfg.train.n_ei_steps + 1):
        # ========== Generate EI Samples ==========
        ei_generation_stats = generate_ei_samples(
            vllm_model=vllm_model,
            reward_fn=r1_zero_reward_fn,
            ei_step=ei_step,
            cfg=cfg
        )

        logger.log({
            "ei_step": ei_step,
            "ei/rollout_acc": ei_generation_stats["rollout_acc"],
            "ei/n_ei_samples": ei_generation_stats["n_ei_samples"]
        })

        # ========== Load EI Training Dataset ==========
        train_dataset = EITrainDataset(data_path=ei_generation_stats["path"])
        n_epochs = cfg.train.epoch_per_ei_step
        local_total_steps = (len(train_dataset.samples) // (cfg.train.micro_batch_size * cfg.train.gradient_accumulation_steps)) * n_epochs

        # ========== Training Loop ==========
        for step in tqdm(range(total_step, total_step + local_total_steps), desc="Training"):
            loss_accum = 0.0
            entropy_accum = 0.0
            total_response_tokens = 0
            for microbatch_idx in range(cfg.train.gradient_accumulation_steps):
                # ========== Get Microbatch Data ==========
                batch_prompts, batch_answers = train_dataset.get_batch(cfg.train.micro_batch_size)

                # ========== Microbatch Train Step ==========
                tokenized_batch = tokenize_prompt_and_output(
                    batch_prompts,
                    batch_answers,
                    tokenizer,
                )
                input_ids = tokenized_batch["input_ids"].to(policy_model.device)
                labels = tokenized_batch["labels"].to(policy_model.device)
                response_mask = tokenized_batch["response_mask"].to(policy_model.device)

                # forward pass
                log_probs_dict = get_response_log_probs(
                    policy_model,
                    input_ids,
                    labels,
                    return_token_entropy=True
                )
                policy_log_probs = log_probs_dict["log_probs"]  # (batch_size, seq_len)
                token_entropy = log_probs_dict["token_entropy"]  # (batch_size, seq_len)

                # backward
                scaled_loss, loss_dict = sft_microbatch_train_step(
                    policy_log_probs,
                    response_mask,
                    cfg.train.gradient_accumulation_steps,
                    normalize_constant=cfg.train.normalize_constant,
                    per_token_loss=cfg.train.per_token_loss
                )

                # accumulate metrics
                loss_accum += loss_dict["loss"]
                total_response_tokens += response_mask.sum().item()
                entropy_accum += (token_entropy * response_mask).sum().item()
                
                # Clear cache after each microbatch to prevent memory buildup
                del input_ids, labels, response_mask, log_probs_dict, policy_log_probs, token_entropy, scaled_loss, loss_dict, tokenized_batch
                if microbatch_idx % 4 == 0:  # Clear cache every 4 microbatches
                    torch.cuda.empty_cache()

            # ========== Optimizer Step ==========
            normalized_grad_norm = torch.nn.utils.clip_grad_norm_(
                policy_model.parameters(),
                cfg.optimizer.max_grad_norm
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)  # set_to_none=True saves memory

            # ========== Logging ==========
            avg_token_entropy = entropy_accum / total_response_tokens
            avg_loss = loss_accum / cfg.train.gradient_accumulation_steps
            tqdm.write(f"Step {step}: Train loss={avg_loss:.4f}, Token Entropy={avg_token_entropy:.4f}")
            logger.log({
                "train_step": step,
                "train/loss": avg_loss,
                "train/token_entropy": avg_token_entropy,
                "train/grad_norm": normalized_grad_norm.item()
            })
        total_step += local_total_steps

        # ========== Evaluation ==========
        load_policy_into_vllm_instance(policy_model, vllm_model)

        vllm_results = evaluate_vllm(
            vllm_model=vllm_model,
            reward_fn=r1_zero_reward_fn,
            prompts=eval_prompts,
            eval_sampling_params=sampling_params,
            answers=eval_answers
        )

        model_results = evaluate_model(
            model=policy_model,
            tokenizer=tokenizer,
            evaluation_vllm_results=vllm_results,
            cfg=cfg
        )
            
        logger.log({
            "ei_step": ei_step,
            "eval/token_entropy": model_results["avg_response_token_entropy"],
            "eval/token_log_prob": model_results["avg_response_token_log_prob"],
            "eval/avg_correct_response_length": model_results["avg_correct_response_length"],
            "eval/avg_incorrect_response_length": model_results["avg_incorrect_response_length"],
            "eval/avg_format_reward": model_results["avg_format_reward"],
            "eval/avg_answer_reward": model_results["avg_answer_reward"],
            "eval/avg_total_reward": model_results["avg_total_reward"],
        })
        torch.cuda.empty_cache()

        print("VLLM Evaluation Results:")
        for key, value in model_results.items():
            print(f"{key}: {value}")

        # ========== Checkpointing ==========
        policy_model.save_pretrained(cfg.checkpoint.save_dir+f"/step_{step}")
        tokenizer.save_pretrained(cfg.checkpoint.save_dir+f"/step_{step}")

if __name__ == "__main__":
    main()