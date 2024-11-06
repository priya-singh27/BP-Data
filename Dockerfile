# Base stage with Python and pipenv
FROM python:3.11-slim as python-base

# Install pipenv
RUN pip install pipenv

# Install CMake and build essentials
RUN apt-get update && apt-get install -y cmake build-essential libgl1-mesa-glx libglib2.0-0

# Builder stage for dependencies
FROM python-base as builder

# Set working directory
WORKDIR /app

# Copy only dependency files first
COPY Pipfile Pipfile.lock ./

# Install dependencies into a virtual environment
RUN PIPENV_VENV_IN_PROJECT=1 pipenv install --deploy

# Final stage
FROM python-base as final

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /app/.venv ./.venv

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"

# Command to run the application
CMD ["python", "app.py"]
