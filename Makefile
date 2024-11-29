PROJECT_ROOT = $(shell pwd)
ENV_NAME_MODEL = envs/qazllm_model_env
REQUIREMENTS_FILE_MODEL = requirements.txt
BENCHMARK_V1 = src/benchmarking/llm_bench_version1/main.py
BENCHMARK_V2 = src/benchmarking/llm_bench_version2/main.py

# Variables
ENV_NAME_UI = envs/qazllm_ui_env
REQUIREMENTS_FILE_UI = ui/environment/requirements.txt
UI_SERVER = ui/src/server.py
VLLM_SERVER = ui/src/run_vllm.py
UI_MODELS = utils/download_models.py

# Targets
.PHONY: export_env_vars create_model_env install_model_requirements \
		deploy_ui create_ui_env install_ui_requirements run_ui_server run_vllm_server download_ui_models 

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
	@. $(ENV_NAME_UI)/bin/activate && python $(UI_MODELS)

run_ui_server: 
	@. $(ENV_NAME_UI)/bin/activate && python $(UI_SERVER)

run_vllm_server: 
	@. $(ENV_NAME_UI)/bin/activate && python $(VLLM_SERVER)


# Clean up the environment

