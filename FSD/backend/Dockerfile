FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY FSD/backend/requirements.txt /app/FSD/backend/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && python -m pip install --no-cache-dir -r /app/FSD/backend/requirements.txt

COPY src/ /app/src/
COPY FSD/backend/app/ /app/FSD/backend/app/

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--app-dir", "/app/FSD/backend", "--host", "0.0.0.0", "--port", "8000"]
