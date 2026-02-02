uv run cs336_alignment/train_grpo.py \
    policy.model_id=/home/chenweicong/projects/assignment5-alignment/checkpoints/sft_qwen2.5_math/sft_math_pertokenloss/step_90 \
    vllm.model_id=/home/chenweicong/projects/assignment5-alignment/checkpoints/sft_qwen2.5_math/sft_math_pertokenloss/step_90 \
    train.run_name=sft_grpo_math \