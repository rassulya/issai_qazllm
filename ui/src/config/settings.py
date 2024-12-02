from pathlib import Path
import yaml
import os
from functools import lru_cache

class Settings:
    def __init__(self):
        self.root_dir = Path(os.getenv("PROJECT_ROOT", "."))
        self.config = self._load_config()
        
        # Model settings
        self.model_config = self.config["model"]
        self.model_path = self._get_model_path()
        self.model_port = self.model_config["port"]
        self.model_host = self.model_config["host"]
        
        # VLLM URL
        self.vllm_url = f"http://{self.model_host}:{self.model_port}/v1"
        
        # API settings
        self.api_port = self.config["api"]["port"]
        self.api_host = self.config["api"]["host"]
        
        # GPU settings
        self.gpu_config = self.config["gpu"]

    def _load_config(self) -> dict:
        config_path = self.root_dir / "conf" / "parameters_ui.yaml"
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def _get_model_path(self) -> str:
        model_path = self.model_config["model_path"]
        if self.model_config["is_local"]:
            return str(self.root_dir / model_path)
        return model_path

@lru_cache()
def get_settings() -> Settings:
    return Settings()