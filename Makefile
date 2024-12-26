PROJECT_ROOT = $(shell pwd)
ENV_NAME_MODEL = envs/qazllm_model_env
REQUIREMENTS_FILE_MODEL = requirements.txt
BENCHMARK_V1 = src/benchmarking/llm_bench_version1/main.py
BENCHMARK_V2 = src/benchmarking/llm_bench_version2/main.py
MODEL_IMAGE_TAG=issai_qazllm:latest
TRAINING_IMAGE=training_image:latest
# Variables
ENV_NAME_UI = envs/qazllm_ui_env
REQUIREMENTS_FILE_UI = ui/environment/requirements.txt
UI_SERVER = ui/src/server.py
VLLM_SERVER = ui/src/run_vllm.py
UI_MODELS = utils/download_models.py


# Targets
.PHONY:  create_model_env install_model_requirements \
		create_ui_env install_ui_requirements download_ui_models \
		run_model run training build_docker docker_down install_nvidia_docker run_ui
# Target to install NVIDIA Docker
install_nvidia_docker:
	./install_nvidia_docker.sh

# Build the Docker image
build_docker:
	docker-compose build

# Run the Docker container with a custom command
run_model:
	@echo "Running $(DIR)"
	@DIR=$(DIR) docker-compose up 

run_training:
	docker run -it --runtime=nvidia \
		-v "$(shell pwd)":/issai_qazllm \
		-w /issai_qazllm \
		-e PROJECT_ROOT="$(shell pwd)" \
		$(TRAINING_IMAGE) \
		bash -c 'echo "Host project root: $$HOST_PROJECT_ROOT"; export PROJECT_ROOT=$(pwd); echo "Container project root: $$PROJECT_ROOT"; exec bash'
		
docker_down:
	docker-compose down

export_env_vars:
	@echo "export PROJECT_ROOT=$(PROJECT_ROOT)" 
	@echo "Environment variable PROJECT_ROOT exported."

create_model_env:
	@if [ ! -d $(ENV_NAME_MODEL) ]; then \
		python3 -m venv $(ENV_NAME_MODEL); \
		echo "Virtual environment $(ENV_NAME_MODEL) created."; \
	else \
		echo "Virtual environment $(ENV_NAME_MODEL) already exists."; \
	fi

install_model_requirements: 
	@. $(ENV_NAME_MODEL)/bin/activate && pip install -r $(REQUIREMENTS_FILE_MODEL)
	@echo "Requirements installed."


# Create the Python virtual environment
create_ui_env:
	@if [ ! -d $(ENV_NAME_UI) ]; then \
		python3 -m venv $(ENV_NAME_UI); \
		echo "Virtual environment $(ENV_NAME_UI) created."; \
	else \
		echo "Virtual environment $(ENV_NAME_UI) already exists."; \
	fi

# Install requirements and export environment variables
install_ui_requirements: 
	@. $(ENV_NAME_UI)/bin/activate && pip install -r $(REQUIREMENTS_FILE_UI)
	@echo "Requirements installed."
	
download_ui_models:
	python3 -m venv envs/tmp_env && \
	. envs/tmp_env/bin/activate && \
	pip install huggingface_hub && \
	python $(UI_MODELS) && \
	deactivate && \
	rm -rf envs/tmp_env

run_ui:
	cd ui && docker-compose up