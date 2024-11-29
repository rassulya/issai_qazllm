import os
import csv
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import yaml

# Import prompt functions from prompts.py
from prompts import (
    hellaswag_prompt,
    arc_prompt,
    gsm8k_prompt,
    truthfulqa_prompt,
    winogrande_prompt,
    mmlu_prompt
)

ROOT_DIR = Path(os.getenv("PROJECT_ROOT", "."))

# Path to the configuration file
CONFIG_PATH = ROOT_DIR / "conf" / "parameters_benchmark.yaml"


# Load the configuration
def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def get_prompt_function(task_name):
    prompt_functions = {
        "hellaswag": hellaswag_prompt,
        "arc": arc_prompt,
        "gsm8k": gsm8k_prompt,
        "truthfulqa": truthfulqa_prompt,
        "winogrande": winogrande_prompt,
        "mmlu": mmlu_prompt
    }
    return prompt_functions.get(task_name.split('_')[0], lambda row: row['question'])

class EvaluationDataset(Dataset):
    def __init__(self, dataset, prompt_function, language='en'):
        self.dataset = dataset
        self.prompt_function = prompt_function
        self.language = language

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        prompt = self.prompt_function(item)
        answer = item['answer']
        return {'prompt': prompt, 'answer': answer}

def evaluate_model(llm, dataset, task_name, batch_size=64, language='en'):
    prompt_function = get_prompt_function(task_name)
    eval_dataset = EvaluationDataset(dataset, prompt_function, language)
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1)

    for batch in tqdm(dataloader, desc=f"Evaluating {task_name} in {language}"):
        prompts = batch['prompt']
        answers = batch['answer']
        outputs = llm.generate(prompts, sampling_params)
        predicted = [output.outputs[0].text.strip() for output in outputs]
        correct += sum(str(pred) == str(ans) for pred, ans in zip(predicted, answers))
        total += len(answers)

    accuracy = correct / total if total > 0 else 0
    return accuracy

def get_output_filename(model_name):
    return f"evaluation_results_{model_name}.json"

def main(model_path, model_name, is_local_model, batch_size, tensor_parallel_size, languages, benchmarks, gpu_memory_utilization, datasets_dir):
    print(f"Loading model: {model_path}")
    print(f"Model name for results: {model_name}")
    if is_local_model:
        model_path = Path(model_path).expanduser().resolve().absolute()
    llm = LLM(
        model=model_path,
        trust_remote_code=False,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization
    )
    
    results = {}

    for language in languages:
        results[language] = {}
        for benchmark in benchmarks:
            # Construct filename with language suffix
            csv_file = f"{benchmark}_{language}.csv"
            file_path = os.path.join(datasets_dir, csv_file)
            
            if not os.path.exists(file_path):
                print(f"Warning: File not found - {file_path}")
                continue
                
            print(f"Processing task: {benchmark} in {language}")

            try:
                dataset = load_dataset(file_path)
                accuracy = evaluate_model(llm, dataset, benchmark, batch_size, language)
                results[language][benchmark] = {"acc": accuracy}

                # torch.cuda.empty_cache()
                # print(f"Cleared CUDA cache after {benchmark}")
                # time.sleep(5)
            except Exception as e:
                print(f"Error processing {benchmark} in {language}: {str(e)}")

    output = {"results": results}
    
    output_filename = get_output_filename(model_name)

    output_path = Path("data/evaluation") / output_filename

    output_path.parent.resolve().mkdir(parents=True, exist_ok=True)

    # Write to the file
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Evaluation complete. Results saved to {output_filename}")



if __name__ == "__main__":
    # Configuration
    config = load_config(CONFIG_PATH)
    main(**config['v1'])