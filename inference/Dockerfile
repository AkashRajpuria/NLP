# Use Python 3.12.3 as the base image
FROM python:3.12.3

# Set working directory for the application
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and necessary directories
COPY training ./training
COPY inference ./inference
COPY data /app/data
COPY outputs /app/outputs

# Define the entry point for inference
CMD ["python", "inference/inference.py"]
