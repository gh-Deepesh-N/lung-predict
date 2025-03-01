# Use Python slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy application files (including model, .env, and requirements)
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure the model file is present
RUN if [ ! -f "/app/lung_disease_model.keras" ]; then echo "Model file missing!"; exit 1; fi

# Expose Flask port
EXPOSE 5000

# Set environment variables
ENV MODEL_PATH="/app/lung_disease_model.keras"

# Start the Flask app
CMD ["python", "app.py"]
