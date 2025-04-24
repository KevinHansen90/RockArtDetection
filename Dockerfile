# Dockerfile

# 1) Base image with PyTorch, CUDA & cuDNN
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

# 2) Make Python logging unbuffered so you see logs in real time
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 3) Install only the Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 4) Copy only your training code (no data or configs)
COPY src/ src/

# 5) The entrypoint will kick off your train.py
ENTRYPOINT ["python", "src/training/train.py"]