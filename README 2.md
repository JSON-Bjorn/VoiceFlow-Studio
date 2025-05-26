# VoiceFlow Studio

This repository contains the code for VoiceFlow Studio, an AI-powered podcast generation platform. The project is structured into a FastAPI backend and a Streamlit frontend.

## Project Structure

- `backend/` — FastAPI backend for podcast generation
- `frontend/` — Streamlit frontend for user interaction
- `prd.txt` — Product requirements document

## Getting Started

1. Install backend dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
2. Start the backend server:
   ```bash
   uvicorn main:app --reload
   ```
3. Install frontend dependencies:
   ```bash
   cd ../frontend
   pip install -r requirements.txt
   ```
4. Start the frontend app:
   ```bash
   streamlit run app.py
   ```

---

Follow the steps in the Product Requirements Document (`prd.txt`) for development milestones and features. 