# Qaz LLM
![image](https://github.com/user-attachments/assets/a0420652-4e7c-483b-a902-ca10ac73b7c8)

Made in Kazakhstan - Қазақстанда жасалған
## Overview


## Usage

### Prerequisites

1. **Check if Docker is installed**:
   ```bash
   docker --version
   ```
   If Docker is not installed, refer to the [official Docker installation guide](https://docs.docker.com/get-docker/).

2. **Check if Docker Compose is installed**:
   ```bash
   docker-compose --version
   ```
   If Docker Compose is not installed, refer to the [official Docker Compose installation guide](https://docs.docker.com/compose/install/).

3. **Check if CUDA and GPUs are available**:
   ```bash
   nvidia-smi
   ```
   If CUDA is not configured or GPUs are not detected, refer to the [CUDA Toolkit Installation Guide](https://developer.nvidia.com/cuda-toolkit).

4. **Check if NVIDIA Docker is installed**:
   ```bash
   nvidia-docker --version
   ```
   If NVIDIA Docker is not installed, run the following command in the project root directory (where the `Makefile` is located):
   ```bash
   make install_nvidia_docker
   ```
   For further information, refer to the [NVIDIA Docker installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

5. **Configure GPU access**:
   In the `docker-compose.yaml` file, set the `NVIDIA_VISIBLE_DEVICES` environment variable to specify the GPUs you want to use.
---

### Setting Configurations

- Edit the `conf/parameters_benchmark.yaml` file to set your desired configurations for benchmarking. If you changed number gpus in docker-compose, set number of tensor_parallel_size to be equal.
- Edit the `conf/parameters_quantization.yaml` file to set your desired configurations for quantization.
- Edit the `conf/parameters_ui.yaml` file to set your desired configurations for ui deployment.

---

### Setting credentials

Credentials are required to access datasets and model from huggingface. Make sure firstly you got gated access to models in the [KazLLM collection](https://huggingface.co/collections/issai/issai-kazllm-10-6732d58c81bcaf177442c362)
Datasets that were used to train models are private. You can access them via token provided. Name of datasets: issai/KazLLM_SFT_Dataset and issai/KazLLM_SFT_Dataset.
 
- Create file  `conf/credentials.yaml`. Include inside your credentials as *hf_token: <your_huggingface_token>* 

---

### Build Docker Images for Benchmarking and Quantization

If Docker images need to be built, run:
```bash
make build_docker
```

### Build Docker Images for Training
Training of a model requires different packages. Therefore make following changes:

1) In Dockefile change line ```COPY requirements.txt ./requirements.txt``` to ```COPY src/training/requirements.txt ./requirements.txt``` 
2) Change ```image```(Image name) in docker-compose to desired image name and image tag. Put the same name into Makefile ```TRAINING_IMAGE``` variable
3) Run ```make build_docker```

---

### Download Datasets for Benchmarking

To download the required datasets for benchmarking, run:
```bash
make run_model DIR=src/benchmarking/utils/download_dataset.py
```
---


### Download Datasets and Models for Training

Make sure you requested access for gated models in [KazLLM Models Collections](https://huggingface.co/collections/issai/issai-kazllm-10-6732d58c81bcaf177442c362).

Set required paths inside of *utils/download_training.py*. Comment out if you don't need either dataset or model
To download the required datasets for benchmarking, run:
```bash
make run_model DIR=utils/download_training.py
```
---

### Run Benchmark

```bash
make run_model DIR="src/benchmarking/main.py"
```

### Run Quantization

```bash
make run_model DIR="src/quantization/main.py"
```

### Run Training
1) Run docker container
```bash
make run_training
```
2) Setup required configs in *src/training/config_train/* based on your model and train type. Change dataset and mode paths. You can stick default one.
3) Go to *src/training*
4) Run *./start.sh* if you are training 8b and *./start_70b.sh* if 70b model

### Run UI
```bash
make run_ui
```
wait until model server is deployed
