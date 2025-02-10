# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Install system dependencies for machine learning and data processing
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Create writable cache directories for Hugging Face and Matplotlib
RUN mkdir -p /tmp/huggingface_cache /tmp/matplotlib

# Set environment variables for cache paths
ENV HF_HOME=/tmp/huggingface_cache
ENV MPLCONFIGDIR=/tmp/matplotlib

# Copy requirements.txt to leverage Docker caching
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch separately to ensure compatibility with the correct architecture
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the application code
COPY . /app

# Expose port 7860 for Flask app
EXPOSE 7860

# Run the Flask application
CMD ["python", "main.py"]
