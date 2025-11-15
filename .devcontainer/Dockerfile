FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

RUN apt update -y && \
    apt install -y --no-install-recommends \
    cmake \
    ninja-build \
    gcc \
    g++ \
    gdb \
    clangd \
    libgtest-dev \
    curl && \
    rm -rf /var/lib/apt/lists/*

USER ubuntu

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /workspace