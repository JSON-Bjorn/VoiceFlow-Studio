from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import FileResponse, JSONResponse
import pyttsx3
import uuid
import os
import wikipediaapi
from pydub import AudioSegment
import traceback
import sys
from orchestrator import ResearchAgent, ConversationOrchestrator
from app.api import auth, payment, podcast, admin

app = FastAPI()

print("VoiceFlow Studio Backend starting up...", flush=True)


class PodcastRequest(BaseModel):
    topic: str


class AudioRequest(BaseModel):
    script: str


@app.get("/")
def read_root():
    return {"message": "VoiceFlow Studio Backend is running."}


@app.get("/test_import")
def test_import():
    try:
        wiki = wikipediaapi.Wikipedia(
            user_agent="voiceflow-studio/1.0 (contact@example.com)", language="en"
        )
        page = wiki.page("Python (programming language)")
        summary = page.summary[0:100] if page.exists() else "Page not found."
        return {"import": "success", "summary": summary}
    except Exception as e:
        print("Exception in /test_import:", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/generate_podcast")
def generate_podcast(request: PodcastRequest):
    try:
        topic = request.topic
        research_agent = ResearchAgent()
        subtopics, summaries = research_agent.get_subtopics_and_summaries(
            topic, max_subtopics=5
        )
        orchestrator = ConversationOrchestrator(topic, subtopics, summaries)
        script = orchestrator.generate_full_script()
        return {"script": script.strip()}
    except Exception as e:
        print("Exception in /generate_podcast:", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/generate_audio")
def generate_audio(request: AudioRequest):
    script = request.script
    lines = [line.strip() for line in script.split("\n") if line.strip()]
    filename = f"audio_{uuid.uuid4().hex}.wav"
    filepath = os.path.join("/tmp", filename) if os.name != "nt" else filename
    temp_files = []
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    # Assign first two available voices
    bjorn_voice = voices[0].id if len(voices) > 0 else None
    felix_voice = voices[1].id if len(voices) > 1 else None
    segments = []
    for i, line in enumerate(lines):
        if line.startswith("Björn:") and bjorn_voice:
            engine.setProperty("voice", bjorn_voice)
        elif line.startswith("Felix:") and felix_voice:
            engine.setProperty("voice", felix_voice)
        else:
            engine.setProperty("voice", voices[0].id)
        temp_wav = f"temp_{uuid.uuid4().hex}.wav"
        engine.save_to_file(line, temp_wav)
        engine.runAndWait()
        segments.append(AudioSegment.from_wav(temp_wav))
        temp_files.append(temp_wav)
    # Concatenate all segments
    if segments:
        combined = segments[0]
        for seg in segments[1:]:
            combined += seg
        combined.export(filepath, format="wav")
    # Cleanup temp files
    for f in temp_files:
        if os.path.exists(f):
            os.remove(f)
    return FileResponse(filepath, media_type="audio/wav", filename=filename)


app.include_router(auth.router)
app.include_router(payment.router)
app.include_router(podcast.router)
app.include_router(admin.router)
