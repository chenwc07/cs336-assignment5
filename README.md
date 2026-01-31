# CS336 Spring 2025 Assignment 5: Alignment

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment5_alignment.pdf](./cs336_spring2025_assignment5_alignment.pdf)

We include a supplemental (and completely optional) assignment on safety alignment, instruction tuning, and RLHF at [cs336_spring2025_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2025_assignment5_supplement_safety_rlhf.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

As in previous assignments, we use `uv` to manage dependencies.

1. Install all packages except `flash-attn`, then all packages (`flash-attn` is weird)
```
uv sync --no-install-package flash-attn
uv sync
```

2. Run unit tests:

``` sh
uv run pytest
```

3. sft_data
- The sft data is not provided in the assignment.
- I download the math-12k dataset from [here](https://huggingface.co/datasets/EleutherAI/hendrycks_math).
- Firstly I generate my own sft dataset using deepseek api (deepseek-reasoner), putting the reasoning content in ```<think> <\think>``` and the content in ```<answer> <\answer>```
- But the sequence length is too long (maximum length 40k) to train on my GPU, so I used deepseek-v3.2 to summarize the cots and the final answer into a concise version (maximun lenth 2k). The code that I make my sft data is [here](./cs336_alignment/make_sft_data.py).

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

