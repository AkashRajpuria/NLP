# Use the official Python image
FROM python:3.12.3

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Run the training script when the container starts
CMD ["python", "training/train.py"]
