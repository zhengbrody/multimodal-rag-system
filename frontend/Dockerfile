FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY frontend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY frontend/ /app/frontend/

EXPOSE 8501

CMD ["streamlit", "run", "/app/frontend/personal_app.py", "--server.address=0.0.0.0", "--server.port=8501"]

