from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import sys

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import API routes
from app.api import (
    auth,
    users,
    credits,
    podcasts,
    stripe_api,
    ai_pipeline,
    elevenlabs,
    storage,
)

# Load environment variables
load_dotenv()

app = FastAPI(
    title="VoiceFlow Studio API",
    description="AI-powered podcast generation platform",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(auth.router, prefix="/api")
app.include_router(users.router, prefix="/api")
app.include_router(credits.router, prefix="/api")
app.include_router(podcasts.router)
app.include_router(stripe_api.router)
app.include_router(ai_pipeline.router)
app.include_router(elevenlabs.router)
app.include_router(storage.router)


@app.get("/")
async def root():
    return {"message": "VoiceFlow Studio API is running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
