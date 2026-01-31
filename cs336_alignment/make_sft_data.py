
import os
import json
from openai import AsyncOpenAI
from dotenv import load_dotenv
import tqdm
import asyncio
load_dotenv()


aclient = AsyncOpenAI(api_key=os.environ.get('API_KEY'), base_url=os.environ.get('BASE_URL'))

# async def process_one(query, gt_answer, semaphore=None):
#     if semaphore is not None:
#         async with semaphore:
#             return await _process_one(query, gt_answer)
#     else:
#         return await _process_one(query, gt_answer)

# async def _process_one(query, gt_answer):
#     try:
#         response = await aclient.chat.completions.create(
#             model="deepseek-reasoner",
#             messages=[
#                 {"role": "system", "content": "answer the question with the final answer in latex boxed"},
#                 {"role": "user", "content": query},
#             ],
#             stream=False,
#         )
#         ds_answer = response.choices[0].message.content
#         think_trace = response.choices[0].message.reasoning_content

#         with open('data/hendrycks_math/sft.jsonl', 'a') as f:
#             f.write(json.dumps({
#                 'problem': query,
#                 'solution': gt_answer,
#                 'ds_answer': f"{think_trace} </think> <answer> {ds_answer} </answer>"
#             }) + '\n')
#     except Exception as e:
#         print(f"Error processing query: {query}\nError: {e}")
#         pass

async def process_one(query, gt_answer, ds_answer, semaphore=None):
    if semaphore is not None:
        async with semaphore:
            return await make_sft_short(query, gt_answer, ds_answer)
    else:
        return await make_sft_short(query, gt_answer, ds_answer)

async def make_sft_short(query, gt_answer, ds_answer):
    think_trace = ds_answer.split('</think> <answer>')[0].strip()
    answer = ds_answer.split('</think> <answer>')[1].replace('</answer>', '').strip()

    user_prompt = f"""Read the following math problem and the reasoning steps and the answer. Your task is to:
    1. Summarize the reasoning steps into a concise version, keeping all important formula, steps and logic.
    2. Summarize the answer into a concise version, keeping the final answer unchanged in latex boxed format.
    output the result in json format:
    {{
        "concise_think_trace": "...",
        "concise_answer": "..."
    }}
    Problem: {query}
    Reasoning Steps: {think_trace} 
    Answer: {answer}
    """
    try:
        response = await aclient.chat.completions.create(
            model="DeepSeek-V3.2",
            messages=[
                {"role": "system", "content": "You are an expert at summarizing math reasoning steps and answers."},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
        )
        concise_output = response.choices[0].message.content
        concise_output = concise_output.strip()
        if concise_output.startswith("```json"):
            concise_output = concise_output[7:-3].strip()
        elif concise_output.startswith("```"):
            concise_output = concise_output[3:-3].strip()
        concise_json = json.loads(concise_output)

        concise_think_trace = concise_json["concise_think_trace"]
        concise_answer = concise_json["concise_answer"]

        with open('data/hendrycks_math/sft_short_v2.jsonl', 'a') as f:
            f.write(json.dumps({
                'problem': query,
                'solution': gt_answer,
                'ds_answer': f"{concise_think_trace} </think> <answer> {concise_answer} </answer>",
                'ori_ds_answer': answer
            }) + '\n')
    except Exception as e:
        print(f"Error processing summarization for query: {query}\nError: {e}")
        pass


# async def main():
#     semaphore = asyncio.Semaphore(10)
#     tasks = []
#     with open("data/hendrycks_math/train.jsonl", "r") as f:
#         lines = f.readlines()
#     total = len(lines)
#     pbar = tqdm.tqdm(total=total, desc="Processing", ncols=80)

#     async def wrapped_process_one(query, gt_answer):
#         await process_one(query, gt_answer, semaphore)
#         pbar.update(1)

#     for line in lines:
#         data = json.loads(line)
#         query = data["problem"]
#         gt_answer = data["solution"]
#         tasks.append(asyncio.create_task(wrapped_process_one(query, gt_answer)))

#     await asyncio.gather(*tasks)
#     pbar.close()

async def main():
    semaphore = asyncio.Semaphore(20)
    tasks = []
    with open("data/hendrycks_math/sft.jsonl", "r") as f:
        lines = f.readlines()
    total = len(lines)
    pbar = tqdm.tqdm(total=total, desc="Processing", ncols=80)

    async def wrapped_process_one(query, gt_answer, ds_answer):
        await process_one(query, gt_answer, ds_answer, semaphore)
        pbar.update(1)

    for line in lines:
        data = json.loads(line)
        query = data["problem"]
        gt_answer = data["solution"]
        ds_answer = data["ds_answer"]
        tasks.append(asyncio.create_task(wrapped_process_one(query, gt_answer, ds_answer)))

    await asyncio.gather(*tasks)
    pbar.close()



if __name__ == "__main__":
    asyncio.run(main())