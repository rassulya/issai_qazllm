# Stage 1: Base CUDA image
FROM nvidia/cuda:12.0.1-cudnn8-devel-ubuntu20.04 AS cuda-base

# Set environment variables for CUDA and Python
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/usr/local/cuda/bin:/usr/local/bin:$PATH


# Install dependencies and Python 3.11 in one step to reduce layers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    python3.11 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Stage 2: Build stage
FROM cuda-base AS builder

WORKDIR /issai_qazllm

# Create and use a virtual environment for Python dependencies
# RUN python3.11 -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Final runtime stage
FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu20.04 AS runtime

WORKDIR /issai_qazllm

# Copy the virtual environment from the builder stage
# COPY --from=builder /opt/venv /opt/venv

# Set up the Python environment in the runtime
# ENV PATH="/opt/venv/bin:$PATH"
