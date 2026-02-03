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
from cs336_alignment.utils.utils import tokenize_prompt_and_output, get_response_log_probs
from cs336_alignment.utils.grpo_utils import grpo_microbatch_train_step
from cs336_alignment.utils.grpo_dataloader import grpo_rollout, GRPOTrainDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import SamplingParams

@hydra.main(config_path="../conf", config_name="grpo", version_base=None)
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
    # policy_model.gradient_checkpointing_enable()
    
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
        weight_decay=cfg.optimizer.weight_decay,
        betas=cfg.optimizer.betas
    )

    # ========== Initialize WandB Logger ==========
    wandb.login(key=os.getenv("WANDB_KEY"))
    logger = wandb.init(
        entity="cwc7", 
        project="cs336-alignment-grpo", 
        name=cfg.train.run_name
    )
    wandb.define_metric("train_step") # the x‑axis for training
    wandb.define_metric("grpo_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="train_step")
    wandb.define_metric("rollouts/*", step_metric="grpo_step")

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
        "train_step": 0,
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

    # ========== GRPO Training Loop ==========
    total_step = 0
    for grpo_step in range(1, cfg.train.n_grpo_steps + 1):
        # ========== Rollouts ==========
        load_policy_into_vllm_instance(policy_model, vllm_model)
        grpo_rollout_state_dict = grpo_rollout(
            vllm_model=vllm_model,
            reward_fn=r1_zero_reward_fn,
            grpo_step=grpo_step,
            cfg=cfg
        )

        logger.log({
            "grpo_step": grpo_step,
            "rollouts/mean_reward": grpo_rollout_state_dict["mean_reward"],
            "rollouts/mean_adv": grpo_rollout_state_dict['mean_adv'],
            "rollouts/max_adv": grpo_rollout_state_dict['max_adv'],
            "rollouts/min_adv": grpo_rollout_state_dict['min_adv'],
        })

        # ========== Load GRPO Training Dataset ==========
        train_dataset = GRPOTrainDataset(samples=grpo_rollout_state_dict["rollouts"])
        n_epochs = cfg.train.epoch_per_grpo_step
        micro_train_batch_size = cfg.train.train_batch_size // cfg.train.gradient_accumulation_steps
        local_total_steps = (train_dataset.get_length() * n_epochs) // cfg.train.train_batch_size

        # ==========compute old log probs for grpo_clip loss ==========
        train_dataset.get_data_log_prob(
            policy_model=policy_model,
            tokenizer=tokenizer,
            cfg=cfg
        )

        # ========== Training Loop ==========
        for step in tqdm(range(total_step, total_step + local_total_steps), desc="Training"):
            loss_accum = 0.0
            entropy_accum = 0.0
            total_response_tokens = 0
            for microbatch_idx in range(cfg.train.gradient_accumulation_steps):
                # ========== Get Microbatch Data ==========
                batch_prompts, batch_answers, batch_rewards, batch_advantages, batch_old_log_probs = train_dataset.get_batch(micro_train_batch_size)

                # ========== Microbatch Train Step ==========
                tokenized_batch = tokenize_prompt_and_output(
                    batch_prompts,
                    batch_answers,
                    tokenizer,
                )
                input_ids = tokenized_batch["input_ids"].to(policy_model.device)
                labels = tokenized_batch["labels"].to(policy_model.device)
                response_mask = tokenized_batch["response_mask"].to(policy_model.device)
                batch_advantages = torch.tensor(batch_advantages).unsqueeze(-1).to(policy_model.device)
                if cfg.train.loss_type == "no_baseline":
                    batch_rewards = torch.tensor(batch_rewards).unsqueeze(-1).to(policy_model.device)
                else:
                    batch_rewards = None
                if cfg.train.loss_type != "grpo_clip":
                    old_log_probs = None
                    cliprange = None
                else:
                    # old_log_probs and cliprange for grpo_clip loss
                    old_log_probs = torch.empty_like(input_ids, dtype=torch.float32).fill_(0)
                    for i in range(len(batch_old_log_probs)):
                        min_length = min(len(batch_old_log_probs[i]), old_log_probs.size(1))
                        old_log_probs[i, :min_length] = torch.tensor(batch_old_log_probs[i][:min_length], dtype=torch.float32)
                    old_log_probs = old_log_probs.to(policy_model.device)
                    cliprange = cfg.train.grpo_cliprange

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
                scaled_loss, loss_dict = grpo_microbatch_train_step(
                    policy_log_probs,
                    response_mask,
                    cfg.train.gradient_accumulation_steps,
                    cfg.train.loss_type,
                    raw_rewards=batch_rewards,
                    advantages=batch_advantages,
                    old_log_probs=old_log_probs,
                    cliprange=cliprange,
                )

                # accumulate metrics
                loss_accum += loss_dict["loss"]
                total_response_tokens += response_mask.sum().item()
                entropy_accum += (token_entropy * response_mask).sum().item()
                
                # Clear cache after each microbatch to prevent memory buildup
                # del input_ids, labels, response_mask, log_probs_dict, policy_log_probs, token_entropy, scaled_loss, loss_dict, tokenized_batch
                # if microbatch_idx % 4 == 0:  # Clear cache every 4 microbatches
                #     torch.cuda.empty_cache()

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

            # ========== Evaluate ==========
            if step > 0 and step % cfg.eval.eval_interval_steps == 0:
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
                    "train_step": step,
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

        total_step += local_total_steps

                

        # ========== Checkpointing ==========
        if grpo_step % cfg.checkpoint.save_grpo_interval == 0:
            policy_model.save_pretrained(cfg.checkpoint.save_dir+f"/step_{step}")
            tokenizer.save_pretrained(cfg.checkpoint.save_dir+f"/step_{step}")

if __name__ == "__main__":
    main()