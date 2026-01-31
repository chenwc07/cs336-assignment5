uv run cs336_alignment/train_sft.py \
    train.per_token_loss=true \
    checkpoint.save_dir=./checkpoints/sft_qwen2.5_math/sft_math_pertokenloss \
    train.run_name=sft_math_pertokenloss \