# Use an official PyTorch runtime with CUDA & cuDNN
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

# Ensure real‑time logs
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy & install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy entire repo (src/, configs/, etc.)
COPY . /app

# Entrypoint for training
ENTRYPOINT ["python", "src/training/train.py"]
