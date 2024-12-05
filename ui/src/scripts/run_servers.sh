#!/bin/bash

# Export LD_LIBRARY_PATH if the library is in a non-standard location
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Function to cleanup
cleanup() {
    echo "Stopping all servers..."
    tmux kill-session -t qazllm_servers 2>/dev/null
    exit 0
}

# Set up cleanup on script exit (Ctrl+C)
trap cleanup EXIT

# Create new tmux session
SESSION_NAME="qazllm_servers"
tmux kill-session -t $SESSION_NAME 2>/dev/null
tmux new-session -d -s $SESSION_NAME

# Split window horizontally
tmux split-window -h -t $SESSION_NAME

# Start VLLM server in first pane
echo "Starting VLLM server..."
tmux send-keys -t "$SESSION_NAME:0.0" "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH && . $ENV_NAME_UI/bin/activate && PYTHONPATH=$PROJECT_ROOT/ui/src python $VLLM_SERVER" C-m

# Wait for VLLM to initialize
echo "Waiting for VLLM server to initialize..."
sleep 10  # Или используйте проверку готовности сервера

# Start UI server in second pane
echo "Starting UI server..."
tmux send-keys -t "$SESSION_NAME:0.1" "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH && . $ENV_NAME_UI/bin/activate && PYTHONPATH=$PROJECT_ROOT/ui/src python $UI_SERVER" C-m

# Wait for UI server to be available
echo "Waiting for UI server to be available..."
until curl -s http://0.0.0.0:8035 > /dev/null; do
    sleep 1
done

# Print success message
echo "INFO:     Uvicorn running on http://0.0.0.0:8035"

echo "All servers are running. Press Ctrl+C to stop everything."

# Keep script running to maintain control
while true; do
    sleep 1
done
