FROM python:3.9-slim

# Install development dependencies using uv
COPY dev-requirements.txt /tmp/dev-requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/dev-requirements.txt

# Install system dependencies
RUN set -ex && \
    apt-get update && \
    apt-get install -y \
    build-essential \
    bash-completion \
    git \
    openssh-client \
    ca-certificates \
    rsync \
    vim \
    nano \
    wget \
    curl \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /workspaces/dlomix

USER root