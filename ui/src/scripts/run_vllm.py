import sys
from pathlib import Path

# Add the src directory to Python path
src_dir = Path(__file__).parent.parent
sys.path.append(str(src_dir))

from services.vllm import vllm_server

if __name__ == "__main__":
    vllm_server.start_server()