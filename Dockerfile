# Dockerfile for custom RockArtDetection training (Faster R-CNN, RetinaNet, Deformable DETR)

# Use a PyTorch base image matching your required versions (PyTorch 2.2.x, CUDA 11.8 recommended)
# Check Docker Hub for exact tags: https://hub.docker.com/r/pytorch/pytorch/tags
# Example tag:
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies from your requirements file
# Ensure scikit-learn, pandas, gcsfs are added to requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire source code directory into the container
COPY src/ /app/src/

# (Optional) Copy configs if needed locally within container,
# but mounting/passing GCS paths is generally preferred for cloud runs
# COPY configs/ /app/configs/

# Define the entrypoint for Vertex AI to run your training script
# Ensure train.py is executable and accepts necessary args (like --config)
ENTRYPOINT ["python", "src/training/train.py"]