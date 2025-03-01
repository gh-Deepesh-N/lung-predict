# Use lightweight Python image
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Copy app files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask port
EXPOSE 5000

# Run Flask application
CMD ["python", "app.py"]
