FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        libgl1-mesa-glx \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install uv for high-speed package installation
RUN pip install --no-cache-dir uv

COPY pyproject.toml requirements.txt README.md ./
COPY src/ src/
COPY configs/ configs/
RUN uv pip install --system --no-cache -r requirements.txt && \
    uv pip install --system --no-cache -e .

ENTRYPOINT ["python", "src/training/train.py"]
