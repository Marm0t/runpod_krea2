FROM runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2204

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/huggingface-cache \
    MODEL_ID=krea/Krea-2-Turbo \
    LOAD_MODE=cuda

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

COPY worker_core.py handler.py api.py /app/

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80"]
