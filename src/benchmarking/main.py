import os
import logging
from pathlib import Path
from typing import List, Dict
import yaml
import pandas as pd
from huggingface_hub import login
from evaluate import load
from vllm import LLM

from utils.utils import load_dataset, evaluate_model, save_results, save_accuracy_metrics, calculate_pass_at_k

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Constants
ROOT_DIR = Path(os.getenv("PROJECT_ROOT", "."))
CONFIG_PATH = ROOT_DIR / "conf" / "parameters_benchmark.yaml"
CREDENTIALS_PATH = ROOT_DIR / "conf" / "credentials.yaml"
OUTPUT_DIR = ROOT_DIR / "data" / "evaluation"
IS_CREDENTIALS=False

DEFAULT_MAX_TOKENS = {
    "mmlu": 15,
    "arc": 30,
    "drop": 40,
    "gsm8k": 512,
    "humaneval": 128,
    "hellaswag": 15,
    "winogrande": 15,
}


def load_config(config_path: Path) -> Dict:
    """Load configuration parameters from a YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)["params_benchmark"]


def load_credentials(credentials_path: Path) -> str:
    """Load Hugging Face token from a credentials YAML file."""
    with open(credentials_path, "r") as file:
        return yaml.safe_load(file).get("hf_token", "")


def initialize_llm(model_path: str, tensor_parallel_size: int, is_local_model: bool) -> LLM:
    """Initialize the LLM with the given parameters."""
    if is_local_model:
        model_path = Path(model_path).expanduser().resolve()
    

    logging.info("Initializing the LLM...")
    return LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.90,
    )


def evaluate_benchmark(
    llm: LLM,
    benchmark: str,
    languages: List[str],
    max_tokens: int,
    batch_size: int,
    data_portion: int,
    model_name: str,
) -> Dict[str, float]:
    """Evaluate a single benchmark across multiple languages."""
    accuracy_metrics = {}
    benchmark_results = []

    for lang in languages:
        try:
            dataset_path = f"data/datasets/{benchmark}_{lang}.csv"
            logging.info(f"Evaluating {benchmark.upper()} for language '{lang}' with {max_tokens} tokens.")

            dataset = load_dataset(dataset_path, data_portion)
            results = evaluate_model(llm, dataset, benchmark, lang, max_tokens, batch_size)
            benchmark_results.extend(results)

            if benchmark == "humaneval":
                df = pd.read_csv(dataset_path)
                df_generated = pd.DataFrame(results)
                if data_portion < 100:
                    df = df.iloc[:len(results)]
                accuracy = calculate_pass_at_k(df, df_generated)
            else:
                correct = sum(1 for r in results if r["status"] == "correct")
                total = len(results)
                accuracy = correct / total if total > 0 else 0

            accuracy_metrics[f"{benchmark}_{lang}"] = accuracy
            logging.info(f"Accuracy for {benchmark.upper()} ({lang}): {accuracy:.4f}")
        except Exception as e:
            logging.error(f"Error processing {benchmark} for {lang}: {e}")
            continue

    save_results(benchmark_results, benchmark, model_name)
    return accuracy_metrics


def main():
    """Main function to execute the benchmarking workflow."""
    try:
        # Load configurations and credentials
        config = load_config(CONFIG_PATH)
        
        if IS_CREDENTIALS:
            hf_token = load_credentials(CREDENTIALS_PATH)
            login(hf_token)
        # Prepare directories
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Initialize LLM
        llm = initialize_llm(
            model_path=config["model_path"],
            tensor_parallel_size=config["tensor_parallel_size"],
            is_local_model=config.get("is_local_model", False),
        )

        # Set max tokens for benchmarks
        max_tokens_map = config.get("max_tokens", DEFAULT_MAX_TOKENS)

        # Evaluate each benchmark
        all_accuracy_metrics = {}
        for benchmark in config["benchmarks"]:
            max_tokens = max_tokens_map.get(benchmark, 15)
            accuracy_metrics = evaluate_benchmark(
                llm=llm,
                benchmark=benchmark,
                languages=config["languages"],
                max_tokens=max_tokens,
                batch_size=config["batch_size"],
                data_portion=config["data_portion"],
                model_name=config["model_name"],
            )
            all_accuracy_metrics.update(accuracy_metrics)

        # Save overall accuracy metrics
        save_accuracy_metrics(all_accuracy_metrics, config["model_name"], OUTPUT_DIR)
        logging.info("All evaluations completed successfully.")
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
