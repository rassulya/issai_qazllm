PROJECT_ROOT = $(shell pwd)
ENV_NAME_MODEL = envs/qazllm_model_env
REQUIREMENTS_FILE_MODEL = requirements.txt
BENCHMARK_V1 = src/benchmarking/llm_bench_version1/main.py
BENCHMARK_V2 = src/benchmarking/llm_bench_version2/main.py
MODEL_IMAGE_TAG=issai_qazllm:latest

# Variables
ENV_NAME_UI = envs/qazllm_ui_env
REQUIREMENTS_FILE_UI = ui/environment/requirements.txt
UI_SERVER = ui/src/server.py
VLLM_SERVER = ui/src/run_vllm.py
UI_MODELS = utils/download_models.py




# Define the image name/tag



# Targets
.PHONY: export_env_vars create_model_env install_model_requirements \
		deploy_ui create_ui_env install_ui_requirements run_ui_server run_vllm_server download_ui_models \
		model_docker_build model_docker_run model_docker_run_default model_docker_tag model_docker_down \
		install_nvidia_docker

# Target to install NVIDIA Docker
install_nvidia_docker:
	./install_nvidia_docker.sh

# Build the Docker image
model_docker_build:
	docker-compose build

# Run the Docker container with a custom command
model_docker_run:
	@echo "Running $(DIR)"
	@DIR=$(DIR) docker-compose up --build


# Tag the image (optional, if you want to re-tag manually)
model_docker_tag:
	docker tag my-cogvlm-image:latest $(IMAGE_TAG)

# Bring down any running containers
model_docker_down:
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

run_benchmark_v1: 
	@. $(ENV_NAME_MODEL)/bin/activate && python $(BENCHMARK_V1)

run_benchmark_v2: 
	@. $(ENV_NAME_MODEL)/bin/activate && python $(BENCHMARK_V2)

create_ui_env:
	@if [ ! -d $(ENV_NAME_UI) ]; then \
		python3 -m venv $(ENV_NAME_UI); \
		echo "Virtual environment $(ENV_NAME_UI) created."; \
	else \
		echo "Virtual environment $(ENV_NAME_UI) already exists."; \
	fi

####UI
deploy_ui: export_env_vars create_ui_env install_ui_requirements download_ui_models run_ui_server run_vllm_server
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

run_ui_server: 
	@. $(ENV_NAME_UI)/bin/activate && python $(UI_SERVER)

run_vllm_server: 
	@. $(ENV_NAME_UI)/bin/activate && python $(VLLM_SERVER)


# Clean up the environment

