# Use official Python runtime as a parent image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy local code to the container image
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port 8000 for the FastAPI dashboard
EXPOSE 8000

# Run the app. Note: The app will run on 0.0.0.0 allowing external access
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
