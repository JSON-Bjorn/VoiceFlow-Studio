@echo off
echo Starting VoiceFlow Studio Development Servers...
echo.

echo Starting Backend (FastAPI)...
start "VoiceFlow Backend" cmd /k "cd /d "%~dp0backend" && call venv\Scripts\activate.bat && python run.py"

echo Waiting 3 seconds...
timeout /t 3 /nobreak > nul

echo Starting Frontend (Next.js)...
start "VoiceFlow Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev"

echo.
echo Both servers are starting...
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000 (or next available port)
echo.
echo Press any key to exit...
pause > nul 