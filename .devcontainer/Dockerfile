FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install extra development dependencies
COPY dev-requirements.txt /tmp/dev-requirements.txt
RUN pip install --upgrade --no-cache-dir pip && \
    pip install --no-cache-dir -r /tmp/dev-requirements.txt

# Install system dependencies
RUN set -ex && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
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
