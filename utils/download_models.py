import os
from huggingface_hub import snapshot_download
from pathlib import Path

ROOT_DIR = Path(os.getenv("PROJECT_ROOT", "."))

# Download VOSK models to ASR directory
print("Downloading VOSK models...")
snapshot_download(
    repo_id="CCRss/qazllm_deployment",
    repo_type="model",
    allow_patterns="vosk_models/*",
    local_dir="models/asr_vosk",
    local_dir_use_symlinks=False
)

# Download Piper models to TTS directory
print("Downloading Piper models...")
snapshot_download(
    repo_id="CCRss/qazllm_deployment",
    repo_type="model",
    allow_patterns="piper_models/*",
    local_dir="models/tts_piper",
    local_dir_use_symlinks=False
)