﻿FROM python:3.9-slim

WORKDIR /app

# Install system dependencies including FFmpeg and Open3D dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    gcc \
    g++ \
    libglu1-mesa-dev \
    libglfw3-dev \
    libglew-dev \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8080

# Run the application
CMD ["python3", "panorama_api.py"]
