import sys
from pathlib import Path
from huggingface_hub import login
import yaml

# Add the src directory to Python path
src_dir = Path(__file__).parent.parent
sys.path.append(str(src_dir))

from services.vllm import vllm_server

def login_hf():
    with open("conf/credentials.yaml") as f:
        credentials = yaml.safe_load(f)
    token = credentials['hf_token']
    login(token)

if __name__ == "__main__":
    login_hf()
    vllm_server.start_server()