FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

COPY ../../requirements.txt .

RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

COPY ../ /app/src/
COPY ../../configs /app/configs/

ENTRYPOINT ["python", "src/training/train.py"]