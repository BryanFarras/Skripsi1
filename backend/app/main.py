from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from .config import STATIC_DIR, EXPORTS_DIR
from .api.routes import router as api_router

app = FastAPI(
    title="AI Music Detector - Forensic Audio Visualizer",
    description="High-resolution audio spectral analysis and AI artifact detection for music forensics.",
    version="1.0.0"
)

# Enable CORS for local cross-origin development if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API endpoints
app.include_router(api_router)

# Mount static files and exports
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if EXPORTS_DIR.exists():
    app.mount("/exports", StaticFiles(directory=str(EXPORTS_DIR)), name="exports")

@app.get("/")
async def root():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "status": "online",
        "app": "AI Music Detector Visualizer",
        "docs": "/docs",
        "endpoints": ["/api/upload", "/api/analyze-local-path", "/api/audio/{id}", "/api/export-plot/{id}"]
    }
