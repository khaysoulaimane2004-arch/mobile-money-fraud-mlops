FROM python:3.11-slim

WORKDIR /app

# Copy slim requirements for API only
COPY requirements-api.txt .

# Install with timeout and retries to handle slow connections
RUN pip install --no-cache-dir \
    --timeout=120 \
    --retries=5 \
    -r requirements-api.txt

# Copy the API code
COPY src/api ./src/api
RUN touch src/__init__.py

# Copy model files
COPY models/model.json ./models/model.json
COPY models/model_metadata.json ./models/model_metadata.json

ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]