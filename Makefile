# Variables
ENV_NAME = qazllm_ui_env
REQUIREMENTS_FILE = ui/environment/requirements.txt
UI_SERVER = ui/src/server.py
VLLM_SERVER = ui/src/run_vllm.py
UI_MODELS = utils/download_models.py
PROJECT_ROOT = $(shell pwd)

# Targets
.PHONY: create_ui_env install_ui_requirements run_ui_server run_vllm_server download_ui_models clean

# Create the Python virtual environment
create_ui_env:
	@if [ ! -d $(ENV_NAME) ]; then \
		python3 -m venv $(ENV_NAME); \
		echo "Virtual environment $(ENV_NAME) created."; \
	else \
		echo "Virtual environment $(ENV_NAME) already exists."; \
	fi

# Install requirements and export environment variables
install_ui_requirements: 
	@. $(ENV_NAME)/bin/activate && pip install -r $(REQUIREMENTS_FILE)
	@echo "Requirements installed."
	@echo "export PROJECT_ROOT=$(PROJECT_ROOT)" > $(ENV_NAME)/bin/activate_env
	@echo "Environment variable PROJECT_ROOT exported."

download_ui_models:
	@. $(ENV_NAME)/bin/activate && . $(ENV_NAME)/bin/activate_env && python $(UI_MODELS)

run_ui_server:
	@. $(ENV_NAME)/bin/activate && . $(ENV_NAME)/bin/activate_env && python $(UI_SERVER)

run_vllm_server:
	@. $(ENV_NAME)/bin/activate && . $(ENV_NAME)/bin/activate_env && python $(VLLM_SERVER)


# Clean up the environment
clean:
	@rm -rf $(ENV_NAME)
	@echo "Cleaned up virtual environment $(ENV_NAME)."
