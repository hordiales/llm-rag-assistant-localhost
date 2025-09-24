FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

# System dependencies for building wheels and numeric libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./

# Install PyTorch CPU wheels first (requires custom index), then remaining deps
RUN pip install --upgrade pip \
    && pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt \
    && python -m nltk.downloader punkt

COPY . .

# Default command launches the console chatbot; override as needed
ENTRYPOINT ["python", "chatbot_rag_local.py"]
