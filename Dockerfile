# Dockerfile
# ----------
# This recipe builds a self-contained "container" image of our app, so it runs
# the same way on any machine (your laptop, a server, the cloud).

# Start from a small official Python image
FROM python:3.11-slim

# Don't write .pyc files; show logs immediately
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python dependencies first (this layer is cached for faster rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project in
COPY . .

# Generate sample data + train the model at build time so the image is ready
RUN python src/make_sample_data.py && python src/train_model.py

# The API listens on port 8000
EXPOSE 8000

# Start the server. 0.0.0.0 = "accept connections from outside the container".
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
