uv run cs336_alignment/train_grpo.py \
    checkpoint.save_dir=./checkpoints/grpo_qwen2.5_math/grpo_off_policy_grpo200_b256_ep4 \
    train.run_name=grpo_off_policy_grpo200_b256_ep4 \
    train.n_grpo_steps=200 \
    train.epoch_per_grpo_step=4 \
    train.train_batch_size=256 \
    train.gradient_accumulation_steps=128 \