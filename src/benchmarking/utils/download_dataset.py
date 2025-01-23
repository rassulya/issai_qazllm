import os
import logging
from pathlib import Path
import yaml

from datasets import load_dataset
from huggingface_hub import HfApi, login

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use absolute path instead of environment variable
ROOT_DIR = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT_DIR / "conf" / "parameters_benchmark.yaml"
CREDENTIALS_PATH = ROOT_DIR / "conf" / "credentials.yaml"
DATA_SAVE_DIR = ROOT_DIR / "data" / "datasets"

def load_yaml(file_path: Path) -> dict:
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

def filter_files(api: HfApi, repo_id: str) -> list:
    """
    Filter files from a Hugging Face repository based on keywords and prefixes.

    Args:
        api (HfApi): Hugging Face API instance.
        repo_id (str): ID of the repository.
        keyword (str): Keyword to match in file names.
        file_prefix (str): Prefix to match in file names.

    Returns:
        list: Filtered list of file names.
    """
    try:
        all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        # Keep only CSV files
        csv_files = [f for f in all_files if f.endswith('.csv')]
        logging.info(f"Found {len(csv_files)} CSV files in repository")
        return sorted(csv_files)
    except Exception as e:
        logging.error(f"Error listing repository files: {e}")
        return []

def process_and_save_dataset(config: dict, file_name: str, save_dir: Path):
    """Process a dataset file and save the filtered content to a CSV."""
    try:
        # Create directories if they don't exist
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the dataset
        logging.info(f"Loading dataset: {file_name}")
        dataset = load_dataset(config["data_repo"], data_files=file_name, split="train")
        
        save_path = save_dir / file_name
        dataset.to_csv(save_path)
        logging.info(f"Saved dataset to: {save_path}")
        
        # Verify file was created
        if not save_path.exists():
            raise FileNotFoundError(f"Failed to create dataset file: {save_path}")
            
    except Exception as e:
        logging.error(f"Error processing dataset {file_name}: {str(e)}")
        raise
        
def main():
    try:
        # Load credentials and configuration
        logging.info("Loading configuration...")
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
            
        with open(CREDENTIALS_PATH, "r") as f:
            credentials = yaml.safe_load(f)
        
        # Login to Hugging Face
        logging.info("Logging in to Hugging Face...")
        login(credentials["hf_token"])
        
        # Initialize Hugging Face API
        api = HfApi()
        
        # Get filtered files from the repository
        logging.info(f"Getting files from repository: {config['data_repo']}")
        all_files = api.list_repo_files(repo_id=config['data_repo'], repo_type="dataset")
        files_to_process = [f for f in all_files if f.endswith('.csv')]
        
        logging.info(f"Found {len(files_to_process)} dataset files to process")
        
        # Create save directory
        DATA_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Process and save each dataset
        for file_name in files_to_process:
            logging.info(f"Processing file: {file_name}")
            process_and_save_dataset(config, file_name, DATA_SAVE_DIR)
            
        logging.info("All datasets downloaded successfully")
        
    except Exception as e:
        logging.error(f"Error in dataset download: {str(e)}")
        raise

if __name__ == "__main__":
    main()