import os
import logging
from pathlib import Path

import torch
import yaml
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset
from huggingface_hub import login

# Set up logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(levelname)s - %(message)s")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Define root and configuration paths
ROOT_DIR = Path(os.getenv("PROJECT_ROOT", "."))
CONFIG_PATH = ROOT_DIR / "conf" / "parameters_quantization.yaml"
CREDENTIALS_PATH = ROOT_DIR / "conf" / "credentials.yaml"

# Utility functions
def load_yaml(file_path: Path):
    """Utility function to load a YAML file."""
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

def load_hf_token(credentials_path: Path) -> str:
    """Load Hugging Face token from credentials YAML file."""
    credentials = load_yaml(credentials_path)
    return credentials.get("hf_token", "")

def initialize_and_move_meta_to_cuda(model, device_ids):
    """Initialize and move meta tensors to available CUDA devices."""
    available_devices = list(range(torch.cuda.device_count()))
    if not set(device_ids).issubset(available_devices):
        raise ValueError(f"Some device IDs in {device_ids} are not available. Available devices: {available_devices}")

    for name, param in model.named_parameters():
        if param.device == torch.device("meta"):
            device = torch.device(f"cuda:{device_ids[0]}")
            param.data = torch.zeros_like(param, device=device)
            device_ids.append(device_ids.pop(0))

def get_cuda_visible_devices():
    """Parse CUDA_VISIBLE_DEVICES environment variable."""
    return [int(i) for i in os.getenv("NVIDIA_VISIBLE_DEVICES", "").split(",") if i]

def load_kk_calib(dataset_name="akylbekmaxutov/KK_calibration_dataset"):
    """Load calibration dataset from a configurable source."""
    logger.info(f"Loading dataset: {dataset_name}")
    data = load_dataset(dataset_name, split="train")
    return [text for text in data["text"]]

def main():
    # Load configurations
    config = load_yaml(CONFIG_PATH)
    hf_token = load_hf_token(CREDENTIALS_PATH)
    login(hf_token)

    # Extract configuration details
    model_config = config["model"]
    quant_config = config["quant_config"]

    # Determine output path
    folder_name = (
        f"{model_config['path']}_AWQ-{quant_config['w_bit']}bit"
        f"-g{quant_config['q_group_size']}-{quant_config['version'].lower()}"
    )
    output_path = ROOT_DIR / "models" / "quantized" / folder_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Quantized model will be saved to: {output_path}")
    # Apply 777 permissions
    os.system(f"chmod -R 777 {output_path}")
    print(f"Permissions for {output_path} set to 777 recursively.")
    # Resolve model path
    if model_config["is_local"]:
        model_path = Path(model_config["path"]).expanduser().resolve()
    else:
        model_path = model_config["path"]

    # Load the model and tokenizer
    logger.info("Loading model and tokenizer...")
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_cache=False,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Log model distribution
    logger.info("Model distribution across GPUs:")
    for name, param in model.named_parameters():
        logger.info(f"{name}: {param.device}")

    # Initialize and move meta tensors to CUDA devices
    initialize_and_move_meta_to_cuda(model, get_cuda_visible_devices())

    # Enable gradient checkpointing if supported
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Quantize the model
    logger.info("Quantizing the model...")
    model.quantize(tokenizer, quant_config=quant_config)

    # Save quantized model and tokenizer
    logger.info("Saving quantized model and tokenizer...")
    output_path = str(output_path)
    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)
    logger.info(f"Quantized model and tokenizer saved to: {output_path}")

if __name__ == "__main__":
    main()
