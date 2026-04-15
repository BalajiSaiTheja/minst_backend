# --- Stage 1: Builder ---
FROM python:3.12-slim as builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install heavy deps
RUN pip install --no-cache-dir --prefix=/install \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install app deps
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Stage 2: Runtime ---
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 10000

CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", \
     "server.endpoints:app", "--bind", "0.0.0.0:10000", "--timeout", "120"]