FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Set environment variables for MLflow tracking
ENV MLFLOW_TRACKING_URI=http://local:5000
ENV MLFLOW_S3_ENDPOINT_URL=
ENV AWS_ACCESS_KEY_ID=
ENV AWS_SECRET_ACCESS_KEY=

# Run FastAPI app
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
