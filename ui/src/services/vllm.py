from pathlib import Path
import subprocess
import logging
import yaml
import os
from openai import OpenAI
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()



class VLLMClient:
    def __init__(self):
        self.settings = settings
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=f"http://{settings.model_host}:{settings.model_port}/v1"
        )

    async def generate_stream(self, messages: list, params: dict) -> dict:
        return {
            "model": self.settings.model_path,
            "messages": messages,
            **params,
            "stream": True,
        }

# Create a client instance to export
vllm_client = VLLMClient()

# For running the server
class VLLMServer:
    def __init__(self):
        self.settings = settings
        self._setup_environment()

    def _setup_environment(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.settings.gpu_config["cuda_visible_devices"])

    def get_command(self):
        return [
            "vllm", "serve", str(self.settings.model_path),
            "--host", self.settings.model_host,
            "--port", str(self.settings.model_port),
            "--gpu-memory-utilization", str(self.settings.gpu_config["gpu_memory_util"]),
            "--max-num-batched-tokens", str(self.settings.gpu_config["max_num_batch_tokens"]),
            "--max-model-len", str(self.settings.gpu_config["max_model_tokens"]),
            "--trust-remote-code",
            "--tokenizer", self.settings.model_config["tokenizer"],
            "--seed", str(self.settings.model_config["seed"]),
            "--dtype", self.settings.gpu_config["dtype"],
            "--tensor-parallel-size", str(self.settings.gpu_config["gpu_number"]),
            "--swap-space", str(self.settings.gpu_config["swap_space"]),
            "--block-size", str(self.settings.gpu_config["block_size"]),
            "--kv-cache-dtype", self.settings.gpu_config["kv_cache_type"],
            "--max-num-seqs", str(self.settings.gpu_config["max_num_seq"])
        ]

    def start_server(self):
        cmd = self.get_command()
        logger.info(f"Starting VLLM server with command: {' '.join(cmd)}")
        subprocess.run(cmd)

vllm_server = VLLMServer()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    vllm_server.start_server()