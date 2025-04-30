FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install google-cloud-storage
COPY src/ src/
COPY configs/ configs/
ENTRYPOINT ["python", "src/training/train.py"]
