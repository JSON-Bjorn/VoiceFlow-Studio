FROM python:3.10-slim

WORKDIR /app

# Copy and install backend requirements
COPY backend/requirements.txt ./backend-requirements.txt
# Copy and install frontend requirements
COPY frontend/requirements.txt ./frontend-requirements.txt
RUN pip install --no-cache-dir -r backend-requirements.txt -r frontend-requirements.txt

# Copy all code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Install supervisor to run both processes
RUN pip install supervisor

# Copy supervisor config
COPY supervisord.conf .

EXPOSE 8000 8501

CMD ["supervisord", "-c", "supervisord.conf"] 