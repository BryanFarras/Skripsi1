import uuid
import shutil
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Request, Query
from fastapi.responses import FileResponse, JSONResponse

from ..config import UPLOADS_DIR, EXPORTS_DIR, ALLOWED_EXTENSIONS
from ..services.audio_loader import load_audio
from ..services.visualizer_service import (
    compute_fft_spectrum,
    compute_tonal_balance,
    compute_spectrogram,
    render_publication_plot
)
from ..services.detector_service import extract_forensic_metrics
from ..models.schemas import AnalysisResponse, AudioMetadata

router = APIRouter(prefix="/api", tags=["Audio Visualizer & Forensics"])

# In-memory registry of processed files to avoid re-decoding
_AUDIO_CACHE = {}

@router.post("/upload")
async def upload_audio_file(request: Request):
    """
    Accepts audio upload either via multipart form-data or raw octet-stream bytes.
    """
    content_type = request.headers.get("content-type", "")
    file_id = str(uuid.uuid4())[:12]
    original_filename = "uploaded_audio.wav"

    if "multipart/form-data" in content_type:
        form = await request.form()
        uploaded_file = form.get("file")
        if not uploaded_file:
            raise HTTPException(status_code=400, detail="No file found in form data.")
        original_filename = getattr(uploaded_file, "filename", original_filename)
        ext = Path(original_filename).suffix.lower()
        if not ext:
            ext = ".wav"
        save_path = UPLOADS_DIR / f"{file_id}{ext}"
        
        with open(save_path, "wb") as f:
            content = await uploaded_file.read()
            f.write(content)
    else:
        # Raw bytes stream
        original_filename = request.headers.get("x-filename", "audio.wav")
        ext = Path(original_filename).suffix.lower() or ".wav"
        save_path = UPLOADS_DIR / f"{file_id}{ext}"
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="Empty audio payload received.")
        with open(save_path, "wb") as f:
            f.write(body)

    # Process and analyze immediately
    try:
        audio, sr, duration = load_audio(save_path)
    except Exception as e:
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to decode audio: {str(e)}")

    metadata = AudioMetadata(
        file_id=file_id,
        original_filename=original_filename,
        duration_seconds=round(duration, 2),
        sample_rate=sr,
        channels=1,
        total_samples=len(audio)
    )

    # Cache waveform
    _AUDIO_CACHE[file_id] = {
        "audio": audio,
        "sample_rate": sr,
        "metadata": metadata,
        "file_path": save_path
    }

    # Perform full analysis
    fft_spec = compute_fft_spectrum(audio, sr)
    tonal_bal = compute_tonal_balance(audio, sr)
    spectro = compute_spectrogram(audio, sr)
    forensics = extract_forensic_metrics(audio, sr)

    return AnalysisResponse(
        metadata=metadata,
        fft_spectrum=fft_spec,
        tonal_balance=tonal_bal,
        spectrogram=spectro,
        forensics=forensics
    )

@router.post("/analyze-local-path")
async def analyze_local_path(payload: dict):
    """
    Analyzes an audio file directly from a local file path on disk without re-uploading.
    Useful for batch processing local datasets (e.g. SingFake, FMA, ACE-Step outputs).
    """
    path_str = payload.get("file_path")
    if not path_str:
        raise HTTPException(status_code=400, detail="Missing 'file_path' in request body.")
    
    file_path = Path(path_str)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Local file not found: {file_path}")

    file_id = str(uuid.uuid4())[:12]
    audio, sr, duration = load_audio(file_path)

    metadata = AudioMetadata(
        file_id=file_id,
        original_filename=file_path.name,
        duration_seconds=round(duration, 2),
        sample_rate=sr,
        channels=1,
        total_samples=len(audio)
    )

    _AUDIO_CACHE[file_id] = {
        "audio": audio,
        "sample_rate": sr,
        "metadata": metadata,
        "file_path": file_path
    }

    fft_spec = compute_fft_spectrum(audio, sr)
    tonal_bal = compute_tonal_balance(audio, sr)
    spectro = compute_spectrogram(audio, sr)
    forensics = extract_forensic_metrics(audio, sr)

    return AnalysisResponse(
        metadata=metadata,
        fft_spectrum=fft_spec,
        tonal_balance=tonal_bal,
        spectrogram=spectro,
        forensics=forensics
    )

@router.get("/audio/{file_id}")
async def stream_audio_file(file_id: str):
    """
    Streams the uploaded audio file for playback in the web player.
    """
    cached = _AUDIO_CACHE.get(file_id)
    if cached and cached["file_path"].exists():
        return FileResponse(
            path=cached["file_path"],
            media_type="audio/wav",
            filename=cached["metadata"].original_filename
        )
    
    # Check uploads directory directly
    matches = list(UPLOADS_DIR.glob(f"{file_id}.*"))
    if matches and matches[0].exists():
        return FileResponse(path=matches[0])
    
    raise HTTPException(status_code=404, detail="Audio file not found or expired.")

@router.post("/export-plot/{file_id}")
async def export_figure(file_id: str):
    """
    Generates a publication-ready 3-panel PNG plot (Equalizer, Tonal Balance, Spectrogram)
    and saves to backend/exports/.
    """
    cached = _AUDIO_CACHE.get(file_id)
    if not cached:
        raise HTTPException(status_code=404, detail="Audio data not found in cache. Please analyze first.")

    audio = cached["audio"]
    sr = cached["sample_rate"]
    meta = cached["metadata"]
    forensics = extract_forensic_metrics(audio, sr)

    export_filename = f"forensic_plot_{meta.original_filename}_{file_id}.png"
    export_path = EXPORTS_DIR / export_filename

    render_publication_plot(
        audio=audio,
        sample_rate=sr,
        filename=meta.original_filename,
        output_path=export_path,
        forensics=forensics
    )

    return FileResponse(
        path=export_path,
        media_type="image/png",
        filename=export_filename
    )
