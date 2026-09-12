FROM python:3.11-slim

WORKDIR /app

COPY requirements-cloud.txt .
RUN pip install --no-cache-dir -r requirements-cloud.txt

COPY server/ server/

ENV PERCEPTION_ENABLED=0
ENV OCR_ENABLED=0
ENV VLM_BACKEND=openai
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=8000

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "server.server:app", "--host", "0.0.0.0", "--port", "8000"]
