#!/bin/bash

# Check if directory is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <model_directory> [starting_epoch]"
    echo "Example: $0 /scratch/batyr_arystanbekov/output/qwen2_5_72B 23000"
    exit 1
fi

MODEL_DIR="$1"
STARTING_EPOCH="${2:-0}"  # If not provided, start from 0
MAX_WAIT_TIME=300  # Maximum wait time in seconds (5 minutes)
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCHMARK_SCRIPT="${BASE_DIR}/scripts/run_single_benchmark.sh"
PROCESSED_FILE="${BASE_DIR}/logs/processed_checkpoints.txt"

# Create directories first
mkdir -p "$LOGS_DIR"
touch "$PROCESSED_FILE"

# Create processed checkpoints file if it doesn't exist
touch "$PROCESSED_FILE"

# Function to get next available epoch
get_next_epoch() {
    local current_epoch=$1
    if is_checkpoint_processed "${MODEL_DIR}/epoch_${current_epoch}"; then
        # If current is processed, find next
        local next_epoch=$(find "$MODEL_DIR" -maxdepth 1 -type d -name "epoch_*" | \
                          sed 's/.*epoch_//' | \
                          sort -n | \
                          awk -v curr="$current_epoch" '$1 > curr {print $1; exit}')
        echo "$next_epoch"
    else
        # If current not processed yet, return current
        echo "$current_epoch"
    fi
}

# Function to check if checkpoint was already processed
is_checkpoint_processed() {
    local epoch_dir="$1"
    if grep -Fxq "$epoch_dir" "$PROCESSED_FILE"; then
        return 0  # Found in processed file
    else
        return 1  # Not found in processed file
    fi
}

# Function to mark checkpoint as processed
mark_checkpoint_processed() {
    local epoch_dir="$1"
    echo "$epoch_dir" >> "$PROCESSED_FILE"
}

# Function to check if benchmark is already running
is_benchmark_running() {
    pgrep -f "run_single_benchmark.sh" > /dev/null
}

# Function to wait for new checkpoint
wait_for_checkpoint() {
    local epoch_dir="$1"
    local wait_time=0
    
    while [ ! -f "${epoch_dir}/model.safetensors.index.json" ]; do
        if [ $wait_time -ge $MAX_WAIT_TIME ]; then
            echo "Timeout waiting for checkpoint: ${epoch_dir}"
            return 1
        fi
        echo "Waiting for checkpoint: ${epoch_dir} (${wait_time}s/${MAX_WAIT_TIME}s)"
        sleep 30
        wait_time=$((wait_time + 30))
    done
    return 0
}

# Function to run benchmark for a specific epoch
run_epoch_benchmark() {
    local epoch_num="$1"
    local epoch_dir="${MODEL_DIR}/epoch_${epoch_num}"
    
    # Skip if already processed
    if is_checkpoint_processed "$epoch_dir"; then
        echo "Skipping epoch_${epoch_num} - already processed"
        return 0
    fi
    
    # Check if checkpoint exists or wait for it
    if [ ! -f "${epoch_dir}/model.safetensors.index.json" ]; then
        echo "Waiting for checkpoint epoch_${epoch_num}..."
        if ! wait_for_checkpoint "${epoch_dir}"; then
            echo "Skipping epoch_${epoch_num} - checkpoint not available"
            return 1
        fi
    fi
    
    # Create temporary script with updated MODEL_PATH
    temp_script=$(mktemp)
    cat "$BENCHMARK_SCRIPT" | sed "s|MODEL_PATH=.*|MODEL_PATH=\"${epoch_dir}\"|" > "$temp_script"
    chmod +x "$temp_script"
    
    echo "Starting benchmark for epoch_${epoch_num}"
    sbatch "$temp_script"
    
    # Wait for benchmark to complete
    while is_benchmark_running; do
        echo "Waiting for benchmark epoch_${epoch_num} to complete..."
        sleep 60
    done
    
    # Mark as processed only if benchmark completed successfully
    if [ $? -eq 0 ]; then
        mark_checkpoint_processed "$epoch_dir"
        echo "Successfully completed and marked as processed: epoch_${epoch_num}"
    else
        echo "Benchmark failed for epoch_${epoch_num}"
    fi
    
    rm "$temp_script"
}

# Main execution
echo "Starting benchmark sequence for ${MODEL_DIR}"
echo "Starting from epoch: ${STARTING_EPOCH}"
echo "Processed checkpoints will be tracked in: ${PROCESSED_FILE}"

current_epoch=$STARTING_EPOCH

while true; do
    # Get next available epoch
    next_epoch=$(get_next_epoch $current_epoch)
    
    if [ -z "$next_epoch" ]; then
        echo "No more checkpoints found after epoch_${current_epoch}"
        sleep 300  # Wait 5 minutes before checking again
        continue
    fi
    
    # Run benchmark for next epoch
    run_epoch_benchmark $next_epoch
    
    # Update current epoch
    current_epoch=$next_epoch
done

echo "Benchmark sequence completed"