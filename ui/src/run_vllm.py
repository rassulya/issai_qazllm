import os
import subprocess
import yaml


from pathlib import Path
import yaml  # Assuming you need to parse YAML

# Get the root directory of the project
ROOT_DIR = Path(os.getenv("PROJECT_ROOT", "."))

# Path to the configuration file
CONFIG_PATH = ROOT_DIR / "conf" / "parameters_ui.yaml"

# Load the configuration
def load_config():
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)
    return config


# Example usage
if __name__ == "__main__":
    config = load_config()
    print("Loaded Configuration:", config)
    
# Model and environment settings from YAML file
    model_config = config["model"]
    IS_LOCAL = model_config["is_local"]
    MODEL = model_config["model_path"]
    if IS_LOCAL: 
        print("LOCAL MODEL")
        MODEL = Path(ROOT_DIR) / MODEL    
    TOKENIZER = model_config["tokenizer"]
    PORT = model_config["port"]
    HOST = model_config["host"]
    SEED = model_config["seed"]

    # GPU settings from YAML file
    gpu_config = config["gpu"]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_config["cuda_visible_devices"])  # Set CUDA_VISIBLE_DEVICES
    GPU_MEMORY_UTILIZATION = gpu_config["gpu_memory_util"]
    MAX_NUM_BATCHED_TOKENS = gpu_config["max_num_batch_tokens"]
    MAX_MODEL_LEN = gpu_config["max_model_tokens"]
    MAX_NUM_SEQS = gpu_config["max_num_seq"]

    DTYPE = gpu_config["dtype"]
    TENSOR_PARALLEL_SIZE = gpu_config["gpu_number"]  # Use gpu_number for the tensor parallel size

    KV_CACHE_DTYPE = gpu_config["kv_cache_type"]
    SWAP_SPACE = gpu_config["swap_space"]
    BLOCK_SIZE = gpu_config["block_size"]

    # Debugging: Print each variable to ensure correct values
    print(f"MODEL: {MODEL}")
    print(f"TOKENIZER: {TOKENIZER}")
    print(f"PORT: {PORT}")
    print(f"HOST: {HOST}")
    print(f"SEED: {SEED}")
    print(f"GPU_MEMORY_UTILIZATION: {GPU_MEMORY_UTILIZATION}")
    print(f"MAX_NUM_BATCHED_TOKENS: {MAX_NUM_BATCHED_TOKENS}")
    print(f"MAX_MODEL_LEN: {MAX_MODEL_LEN}")
    print(f"MAX_NUM_SEQS: {MAX_NUM_SEQS}")
    print(f"DTYPE: {DTYPE}")
    print(f"TENSOR_PARALLEL_SIZE: {TENSOR_PARALLEL_SIZE}")
    print(f"KV_CACHE_DTYPE: {KV_CACHE_DTYPE}")
    print(f"SWAP_SPACE: {SWAP_SPACE}")
    print(f"BLOCK_SIZE: {BLOCK_SIZE}")

    # Construct the vLLM command
    CMD = [
        "vllm", "serve", str(MODEL),
        "--host", HOST,
        "--port", str(PORT),
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
        "--max-num-batched-tokens", str(MAX_NUM_BATCHED_TOKENS),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--trust-remote-code",
        "--tokenizer", TOKENIZER,
        "--seed", str(SEED),
        "--dtype", DTYPE,
        "--tensor-parallel-size", str(TENSOR_PARALLEL_SIZE),
        "--swap-space", str(SWAP_SPACE),
        "--block-size", str(BLOCK_SIZE),
        "--kv-cache-dtype", KV_CACHE_DTYPE,
        "--max-num-seqs", str(MAX_NUM_SEQS)
    ]

    # Debugging: Print the final command
    print(f"Running command: {' '.join(CMD)}")

    # Execute the command
    subprocess.run(CMD)

