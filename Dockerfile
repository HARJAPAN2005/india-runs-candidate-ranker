FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY precompute.py rank.py ./
COPY India_runs_data_and_ai_challenge/ India_runs_data_and_ai_challenge/

# Default: build artifacts then rank.
# Override CMD to run steps individually, e.g.:
#   docker run ... python precompute.py
#   docker run ... python rank.py
CMD ["bash", "-c", "python precompute.py && python rank.py"]
