# Use a more specific Python base image
FROM python:3.8-slim

# Set the working directory
WORKDIR /opt

# Copy and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY . /opt/

# Copy the model directory
COPY app/model /opt/app/model

# Expose the port
EXPOSE 8000

# Specify the Python unbuffered mode
ENV PYTHONUNBUFFERED 1

# Create a non-root user
RUN useradd -m myuser
USER myuser

# Run the FastAPI application using Gunicorn
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "app.main:app"]

# Note: Adjust the last line based on the location of your FastAPI application entry point.
# If your FastAPI app is in a directory named 'app', then the line should be:
# CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "app.main:app"]
