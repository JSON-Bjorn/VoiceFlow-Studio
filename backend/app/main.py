"""
VoiceFlow Studio Backend

FastAPI application for the VoiceFlow Studio podcast generation platform.
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import sys

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import init_db

# Import API routes
from app.api import (
    auth,
    users,
    chatterbox,
    credits,
    podcasts,
    stripe_api,
    ai_pipeline,
    storage,
    audio,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="VoiceFlow Studio API",
    description="Backend API for VoiceFlow Studio - AI-powered podcast generation platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
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
app.include_router(chatterbox.router)
app.include_router(credits.router, prefix="/api")
app.include_router(podcasts.router)
app.include_router(stripe_api.router)
app.include_router(ai_pipeline.router)
app.include_router(storage.router)
app.include_router(audio.router)


@app.on_event("startup")
async def startup_event():
    """Initialize the database on startup"""
    logger.info("Starting VoiceFlow Studio Backend...")
    await init_db()
    logger.info("Database initialized successfully")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "VoiceFlow Studio API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
