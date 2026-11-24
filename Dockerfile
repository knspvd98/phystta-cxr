FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    MPLCONFIGDIR=/workspace/.cache/matplotlib

WORKDIR /workspace/phystta

COPY requirements.lock /workspace/phystta/requirements.lock
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.lock

COPY . /workspace/phystta

RUN python3 -m pip install --no-deps -e .

CMD ["phystta", "list"]
