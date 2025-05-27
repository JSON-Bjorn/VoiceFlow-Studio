# Root Dockerfile for VoiceFlow Studio (multi-service)

# --- Backend Build Stage ---
FROM python:3.10 AS backend-build
WORKDIR /app/backend
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ .

# --- Frontend Build Stage ---
FROM node:20 AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json ./
COPY frontend/package-lock.json ./
RUN npm install
COPY frontend/ .
RUN npm run build

# --- Final Stage ---
FROM python:3.10
WORKDIR /app
# Copy backend from backend-build
COPY --from=backend-build /app/backend /app/backend
# Copy frontend build from frontend-build
COPY --from=frontend-build /app/frontend/build /app/frontend/build
# Copy supervisord config
COPY supervisord.conf /etc/supervisord.conf
# Install backend dependencies
RUN pip install --no-cache-dir -r /app/backend/requirements.txt
# Install supervisor
RUN pip install supervisor
# Expose backend and frontend ports
EXPOSE 8000 3000
# Start both backend and frontend using supervisord
CMD ["supervisord", "-c", "/etc/supervisord.conf"] 