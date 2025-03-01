# Use Python slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy all files (including the model)
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask port
EXPOSE 5000

# Set environment variables
ENV MODEL_PATH="/app/lung_disease_model.keras"

# Start the Flask app
CMD ["python", "app.py"]
