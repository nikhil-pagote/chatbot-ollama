# Use the official Python DevContainer image with NVIDIA CUDA base
FROM mcr.microsoft.com/devcontainers/python:1-3.11

# Install required system packages for GPU, pipx, and Poetry
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    xdg-utils \
    lsb-release \
    ca-certificates \
    build-essential \
    git \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Optional: Install NVIDIA container toolkit (note: only works with WSL2 + proper host setup)
# Leaving this here for documentation — installation should be handled on the host
# RUN distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
#   && curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
#   && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
#     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' | \
#     tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
#   && apt-get update \
#   && apt-get install -y nvidia-container-toolkit

# Install pipx and Poetry
RUN pip install --no-cache-dir pipx && \
    pipx ensurepath && \
    pipx install poetry

# Ensure Poetry is available on PATH
ENV PATH="/home/vscode/.local/bin:$PATH"

# Set workspace
WORKDIR /workspace

# Set permissions for the vscode user
RUN chown -R vscode:vscode /workspace
