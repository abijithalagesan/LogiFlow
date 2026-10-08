FROM python:3.9-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY src ./src
COPY models ./models
COPY dataset/route_features.csv ./dataset/route_features.csv

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
