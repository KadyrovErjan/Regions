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

COPY req.txt .
RUN pip install --no-cache-dir -r req.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
