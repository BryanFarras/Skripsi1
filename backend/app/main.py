from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from .config import FRONTEND_DIR, EXPORTS_DIR
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

@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.endswith((".html", ".js", ".css")):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# Include API endpoints first so they take precedence
app.include_router(api_router)

@app.on_event("startup")
async def startup_event():
    print("\n" + "=" * 65)
    print(" [XAI SYSTEM] Initializing Machine Learning & TreeSHAP Engine...")
    print("=" * 65)
    try:
        from ml_baseline.baseline_adapter import get_baseline_adapter
        adapter = get_baseline_adapter()
        if adapter.is_loaded:
            print("[XAI SYSTEM OK] Champion Baseline Model (Random Forest) & TreeSHAP Loaded!")
        else:
            print("[XAI SYSTEM NOTICE] Model adapter initialized in standby mode.")
    except Exception as e:
        print(f"[XAI SYSTEM NOTE] Startup initialization notice: {e}")


# Mount exports directory
if EXPORTS_DIR.exists():
    app.mount("/exports", StaticFiles(directory=str(EXPORTS_DIR)), name="exports")

# Mount frontend static directories
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    if (FRONTEND_DIR / "css").exists():
        app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
    if (FRONTEND_DIR / "js").exists():
        app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")

@app.get("/")
async def root():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "status": "online",
        "app": "AI Music Detector Visualizer",
        "docs": "/docs",
        "endpoints": ["/api/upload", "/api/analyze-local-path", "/api/audio/{id}", "/api/export-plot/{id}"]
    }
