FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /audio_app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    gcc \
    g++ \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 🔹 ОБЯЗАТЕЛЬНО: обновляем pip
RUN python -m pip install --upgrade pip setuptools wheel

# 🔹 PyTorch CPU (теперь pip его увидит)
RUN pip install --no-cache-dir \
    torch==2.9.1 \
    torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cpu

# 🔹 Остальные зависимости (без torch)
COPY req.txt .
RUN pip install --no-cache-dir -r req.txt --no-deps

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
