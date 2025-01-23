#!/bin/bash
#SBATCH --job-name=4GPUs_test-benchmark
#SBATCH --nodelist=node006
#SBATCH --partition=defq
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=100000M
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS_DIR="$BASE_DIR/logs"
MODEL_PATH="/scratch/batyr_arystanbekov/output/qwen2_5_72B/epoch_20000"
CONFIG_PATH="${BASE_DIR}/conf/parameters_benchmark.yaml"
TEMP_CONFIG_DIR="${LOGS_DIR}/temp_configs"
DATASETS_DIR="${BASE_DIR}/data/datasets"
EVAL_DIR="${BASE_DIR}/data/evaluation"

mkdir -p "$LOGS_DIR" "$TEMP_CONFIG_DIR" "$DATASETS_DIR" "$EVAL_DIR"

# Extract model info from path
MODEL_SIZE=$(basename $(dirname "$MODEL_PATH") | sed 's/qwen2_5_//')
EPOCH_NUM=$(basename "$MODEL_PATH" | sed 's/epoch_//')

eval "$(conda shell.bash hook)"
conda activate qazllm_env

# Create YAML config
timestamp=$(date +%Y%m%d_%H%M%S)
temp_yaml="${TEMP_CONFIG_DIR}/config_qwen2_5_${MODEL_SIZE}_epoch${EPOCH_NUM}_${timestamp}.yaml"

cat > "$temp_yaml" << EOL
params_benchmark:
 model_path: "${MODEL_PATH}"
 model_name: "qwen2_5_${MODEL_SIZE}_epoch${EPOCH_NUM}"
 benchmarks:
   - "gsm8k"
   - "arc"
   - "mmlu"
   - "drop"
   - "hellaswag"
   - "humaneval"
   - "winogrande"
 languages:
   - "kk"
   - "ru"
   - "en"
 batch_size: 64
 tensor_parallel_size: 4
 data_portion: 100
 max_tokens:
   mmlu: 15
   arc: 30
   drop: 40
   gsm8k: 512
   humaneval: 512
   hellaswag: 15
   winogrande: 15
 is_local_model: true
data_repo: "issai/KazLLM_Benchmark_Dataset"
EOL

cd "$BASE_DIR"
python3 src/benchmarking/utils/download_dataset.py

export HF_ALLOW_CODE_EVAL="1"
echo "Starting benchmark for epoch_${EPOCH_NUM} at $(date)"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/benchmarking/main.py \
   --config "$temp_yaml" 2>&1 | tee "${LOGS_DIR}/benchmark_qwen2_5_${MODEL_SIZE}_epoch${EPOCH_NUM}_${timestamp}.log"

echo "Completed benchmark pipeline at $(date)"