# VoiceFlow Studio

This project is a full-scale, AI-powered podcast generation platform. See `prd.txt` for the full product requirements.

## Current Project Flow

### Backend (FastAPI)
- **Status:** The backend is modularized and partially functional.
- **What works:**
  - The FastAPI server starts and exposes the following endpoints:
    - `/` — Returns a JSON message confirming the backend is running.
    - `/test_import` — Tests Wikipedia API integration and returns a summary of the Python programming language.
    - `/generate_podcast` — Accepts a topic and returns a generated podcast script (requires OpenAI API key in environment).
    - `/generate_audio` — Accepts a script and returns a WAV file using local TTS (pyttsx3).
- **What is stubbed/not yet implemented:**
  - User registration, login, authentication (API stubs only)
  - Payment/Stripe integration (API stubs only)
  - Admin endpoints (API stubs only)
  - Podcast credit management
  - Full agent pipeline (Script Agent, Voice Agent, Audio Agent)

### Frontend (React)
- **Status:** The frontend is scaffolded with all main pages/components as stubs.
- **What works:**
  - Navigation between stub pages: Registration, Login, Dashboard, Podcast Generation, Payment, Admin, Example Podcast Library.
- **What is not yet functional:**
  - No real backend integration or user flows implemented yet.

## How to Build and Run the Project (Docker)

1. **Build the full project (backend + frontend) from the root directory:**
   ```bash
   docker build -t voiceflow-studio .
   ```
2. **Run the container:**
   ```bash
   docker run -p 8000:8000 -p 3000:3000 voiceflow-studio
   ```
   - The backend will be available at [http://localhost:8000](http://localhost:8000)
   - The frontend will be available at [http://localhost:3000](http://localhost:3000)

## How to Run the Backend or Frontend Separately (Development)

- **Backend:**
  ```bash
  cd backend
  uvicorn main:app --reload
  ```
- **Frontend:**
  ```bash
  cd frontend
  npm install
  npm start
  ```

## Project Structure

- `backend/` - FastAPI backend with modular agent pipeline
- `frontend/` - React frontend with user dashboard, payment, and podcast generation
- `Dockerfile` - Root Dockerfile to build and run both backend and frontend
- `supervisord.conf` - Supervisor config to run both services in one container
- `prd.txt` - Product requirements document
- `tasks.txt` - Project task tracking
- `.env`, `.gitignore`, `README.md`, etc.

**Note:**
- The `backend/compose.yml` file is deprecated and can be deleted. The root Dockerfile now handles multi-service builds and runs.

## Key Features (Planned)
- User registration, login, dashboard
- Payment integration (Stripe)
- Podcast generation using OpenAI and ElevenLabs
- Example podcast library
- Admin dashboard

See the PRD for more details and the current `tasks.txt` for up-to-date progress and next steps.

## Project Vision
VoiceFlow Studio aims to democratize podcast creation by enabling anyone to generate professional, engaging audio content through AI. Users will input a topic and receive a complete podcast episode featuring a natural conversation between two AI hosts.

## Planned Features
- **Text-to-Podcast Generation:** Enter a topic and get a 5–15 minute podcast episode, with a natural-sounding conversation between two distinct AI hosts (Björn & Felix).
- **AI Host Personalities:** Consistent, engaging personalities for each host—one curious and enthusiastic, the other knowledgeable and analytical.
- **Voice Synthesis:** High-quality, human-like voices using premium TTS APIs (e.g., ElevenLabs).
- **Professional Audio Output:** Automatic background music, intro/outro, and audio post-production for a polished result.
- **User-Friendly Interface:** Simple web interface for entering topics, adjusting settings, and downloading or listening to generated podcasts.
- **Automated Research:** The system will gather relevant information from sources like Wikipedia to ensure accurate and engaging content.
- **Multi-Language Support:** Initial support for Swedish and English.

## Target Users
- **Students & Educators:** For creating educational audio content.
- **Content Creators & Hobbyists:** For quickly generating podcast drafts and ideas.
- **Professionals & Researchers:** For summarizing topics and sharing knowledge in audio format.

## Technology Stack (Planned)
- **Backend:** Python (FastAPI)
- **Frontend:** Streamlit or similar (for rapid prototyping, UI isnt as important)
- **AI/ML:** Claude, OpenAI, Ollama, Groq, ElevenLabs TTS (whatever costs the least and gives the best result)
- **Audio Processing:** pydub, librosa (needs research)
- **Deployment:** Local development environment

## Project Timeline
- **Total Duration:** 4 weeks
- **MVP Deadline:** End of week 2 (core podcast generation working)
- **Polish & Testing:** Week 4

## Follow the Project
This repository will be updated as development progresses. Planned updates include:
- Progress on core podcast generation features
- Demos and sample outputs
- User interface previews
- Technical architecture and design notes

// Björn
---

**Stay tuned for updates as we build VoiceFlow Studio!** 