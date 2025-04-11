# Use official Python base image with DevContainer features
FROM mcr.microsoft.com/devcontainers/python:1-3.11

# Install pipx & Poetry
RUN apt-get update && apt-get install -y curl && \
    pip install --no-cache-dir pipx && \
    pipx ensurepath && \
    pipx install poetry

# Make Poetry available to the vscode user
ENV PATH="/home/vscode/.local/bin:$PATH"

# Set the workspace directory (shared with host via Docker volume)
WORKDIR /workspace

# Ensure correct ownership
RUN chown -R vscode:vscode /workspace
