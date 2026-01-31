import json
import random

class SFTTrainDataset():
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
                answer = data["ds_answer"]
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
    
    def reset_pointer(self):
        self.ptr = 0
        # shuffle the data here if needed
        random.shuffle(self.samples)


if __name__ == "__main__":
    # Example usage
    train_set = SFTTrainDataset(
        data_path="/home/chenweicong/projects/assignment5-alignment/data/hendrycks_math/sft.jsonl",
        prompt_template_path="/home/chenweicong/projects/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
    )
    print("Total samples loaded:", len(train_set.samples))

    # batch_prompts, batch_answers = train_set.get_batch(batch_size=4)
    # for prompt, answer in zip(batch_prompts, batch_answers):
    #     print("Prompt:", prompt)
    #     print("Answer:", answer)
    #     print("-----")