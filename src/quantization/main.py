import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset
from pathlib import Path
import yaml  # Assuming you need to parse YAML

# Get the root directory of the project
ROOT_DIR = Path(os.getenv("PROJECT_ROOT", "."))

# Path to the configuration file
CONFIG_PATH = ROOT_DIR / "conf" / "parameters_quantization.yaml"


# Load the configuration
def load_config():
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)
    return config

config = load_config()
env_config = config["environment"]
cuda_visible_devices = env_config["cuda_visible_devices"]
model_path = config["model"]["path"]
quant_config = config["quant_config"]
max_memory = config["resources"]["max_memory"]
# Set environment variable to use specific GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = env_config  # Ensure only GPUs 4-7 are used




# Create a descriptive folder name based on the quantization config
folder_name = (
    f"Llama-3.1-8b-instruct-kk_bench_calibrated"
    f"_AWQ-{quant_config['w_bit']}bit"
    f"-g{quant_config['q_group_size']}"
    f"-{quant_config['version'].lower()}"
)


output_path = Path("models/quantized") / folder_name

output_path.parent.resolve().mkdir(parents=True, exist_ok=True)

print(f"Quantized model will be saved to: {output_path}")

# Load the original model and tokenizer
model = AutoAWQForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    # max_memory=max_memory,
    torch_dtype=torch.bfloat16,
    use_cache=False,  # Disable KV cache
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
# Print model distribution
print("Model distribution across GPUs:")
for name, param in model.named_parameters():
    print(f"{name}: {param.device}")


# Initialize and move meta tensors to CUDA devices
def initialize_and_move_meta_to_cuda(model, device_ids):
    # available_devices = list(range(torch.cuda.device_count()))
    # if not set(device_ids).issubset(set(available_devices)):
    #     raise ValueError(f"Some device IDs in {device_ids} are not available. Available devices: {available_devices}")

    for name, param in model.named_parameters():
        if param.device == torch.device("meta"):
            # Initialize the parameter with random data
            param.data = torch.zeros_like(param, device=torch.device(f'cuda:{device_ids[0]}'))
            device = torch.device(f'cuda:{device_ids.pop(0)}')
            param.data = param.to(device)
            device_ids.append(device.index)  # Append the device ID back to the end of the list

# initialize_and_move_meta_to_cuda(model, [0, 1, 2, 3])  # Use indices 0, 1, 2, 3 for GPUs 4, 5, 6, 7

# Enable gradient checkpointing to save memory (if supported by the model)
if hasattr(model, 'gradient_checkpointing_enable'):
    model.gradient_checkpointing_enable()

def load_kk_calib():
    data = load_dataset('akylbekmaxutov/kazakh_calibration_dataset', split="train")
    return [text for text in data["text"]]

# Quantize the model
model.quantize(
    tokenizer,
    quant_config=quant_config,
    #calib_data=load_kk_calib()
)

# Save the quantized model and tokenizer
model.save_quantized(output_path)
tokenizer.save_pretrained(output_path)

print(f"Quantized model and tokenizer saved to: {output_path}")
