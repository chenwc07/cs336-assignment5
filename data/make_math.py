import pandas as pd
import json
import glob
import os

if __name__ == "__main__":
    # for train in glob.glob("hendrycks_math/*/train*"):
    #     df = pd.read_parquet(train)
    #     path = os.path.dirname(train)
    #     print(f"Saving train.jsonl to {path}")
    #     df.to_json(
    #         os.path.join(path, "train.jsonl"),
    #         orient="records",
    #         lines=True,
    #     )

    # for test in glob.glob("hendrycks_math/*/test*"):
    #     df = pd.read_parquet(test)
    #     path = os.path.dirname(test)
    #     print(f"Saving test.jsonl to {path}")
    #     df.to_json(
    #         os.path.join(path, "test.jsonl"),
    #         orient="records",
    #         lines=True,
    #     )

    # combine all train.jsonl files into one
    all_dfs = []
    for train in glob.glob("hendrycks_math/*/train.jsonl"):
        df = pd.read_json(train, lines=True)
        all_dfs.append(df)
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Saving combined train.jsonl to hendrycks_math/train.jsonl")
    combined_df.to_json(
        "hendrycks_math/train.jsonl",
        orient="records",
        lines=True,
    )

    # combine all test.jsonl files into one
    all_dfs = []
    for test in glob.glob("hendrycks_math/*/test.jsonl"):
        df = pd.read_json(test, lines=True)
        all_dfs.append(df)
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Saving combined test.jsonl to hendrycks_math/validation.jsonl")
    combined_df.to_json(
        "hendrycks_math/validation.jsonl",
        orient="records",
        lines=True,
    )