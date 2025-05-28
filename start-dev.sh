#!/bin/bash

echo "Starting VoiceFlow Studio Development Servers..."
echo

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to cleanup background processes on exit
cleanup() {
    echo "Stopping servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit
}

# Set up trap to cleanup on script exit
trap cleanup EXIT INT TERM

echo "Starting Backend (FastAPI)..."
cd "$SCRIPT_DIR/backend"
# Try different virtual environment activation methods
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    echo "Warning: Virtual environment not found. Make sure to set it up first."
fi
python run.py &
BACKEND_PID=$!
cd "$SCRIPT_DIR"

echo "Waiting 3 seconds for backend to start..."
sleep 3

echo "Starting Frontend (Next.js)..."
cd "$SCRIPT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!
cd "$SCRIPT_DIR"

echo
echo "Both servers are running:"
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:3000 (or next available port)"
echo
echo "Press Ctrl+C to stop both servers"

# Wait for background processes
wait 