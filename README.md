# LLM Benchmarking Framework
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

Edit the `conf/parameters_benchmark.yaml` file to set your desired configurations for benchmarking.
Edit the `conf/parameters_quantization.yaml` file to set your desired configurations for quantization.
Edit the `conf/parameters_ui.yaml` file to set your desired configurations for ui deployment.


---

### Download Datasets

To download the required datasets for benchmarking, run:
```bash
make run_model DIR=src/utils/download_dataset.py
```

---

### Build Docker Images (if required)

If Docker images need to be built, run:
```bash
make build_docker
```

---

### Run Benchmark

To start the benchmarking process, run:
```bash
make run_via_compose DIR=src/main.py
```


### Run Benchmark

```bash
make run_model DIR="src/benchmarking/main.py"
```

### Run Quantization

```bash
make run_model DIR="src/quatization/main.py"
```

### Run Training

```bash
make run_model DIR="src/quatization/main.py"
```

### Run UI
```bash
make run_ui
```
