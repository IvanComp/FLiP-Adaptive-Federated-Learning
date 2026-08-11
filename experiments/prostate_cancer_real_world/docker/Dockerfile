FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY experiments/prostate_cancer_real_world/docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY experiments/prostate_cancer_real_world/docker/client.py /app/client.py
COPY experiments/prostate_cancer_real_world/docker/server.py /app/server.py
COPY experiments/prostate_cancer_real_world/src /app/src
COPY experiments/prostate_cancer_real_world/config /app/config

CMD ["python", "/app/client.py"]
