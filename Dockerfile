FROM python:3.11-slim

# Set the working directory
WORKDIR /app
RUN mkdir -p /models
# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


