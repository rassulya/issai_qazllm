import os
from huggingface_hub import snapshot_download, login
from pathlib import Path
import yaml

ROOT_DIR = Path(os.getenv("PROJECT_ROOT", "."))
CREDENTIALS_PATH = ROOT_DIR / "conf" / "credentials.yaml"

MODELS_DIR = "/models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Apply 777 permissions
os.system(f"chmod -R 777 {MODELS_DIR}")
print(f"Permissions for {MODELS_DIR} set to 777 recursively.")

# Utility functions
def load_yaml(file_path: Path):
    """Utility function to load a YAML file."""
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

def load_hf_token(credentials_path: Path) -> str:
    """Load Hugging Face token from credentials YAML file."""
    credentials = load_yaml(credentials_path)
    return credentials.get("hf_token", "")

hf_token = load_hf_token(CREDENTIALS_PATH)
login(hf_token)

# Download Datasets. Comment it below if you do not need it
# print("Downloading Dataset...")
# snapshot_download(
#     repo_id="issai/LLM_internvl_sft_datasets",
#     repo_type="dataset",
#     local_dir="data/LLM_internvl_sft_datasets",
#     local_dir_use_symlinks=False
# )

# Download Model. Comment it below if you do not need it


# Download the model
snapshot_download(
    repo_id="issai/LLama-3.1-KazLLM-1.0-8B",
    repo_type="model",
    local_dir=os.path.join(MODELS_DIR, "LLama-3.1-KazLLM-1.0-8B"),
    local_dir_use_symlinks=False,
)
print("Model downloaded successfully.")