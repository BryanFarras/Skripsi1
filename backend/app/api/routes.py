import time
import uuid
import shutil
import urllib.parse
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Request, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from ..config import UPLOADS_DIR, EXPORTS_DIR, STEMS_TEMP_DIR, STEMS_EXPIRE_SECONDS, ALLOWED_EXTENSIONS
from ..services.audio_loader import load_audio
from ..services.visualizer_service import (
    compute_fft_spectrum,
    compute_tonal_balance,
    compute_spectrogram,
    compute_waterfall_3d,
    render_publication_plot
)
from ..services.detector_service import extract_forensic_metrics
from ..services.stem_service import StemService
from ..models.schemas import AnalysisResponse, AudioMetadata

router = APIRouter(prefix="/api", tags=["Audio Visualizer & Forensics"])

# In-memory registry of processed files
_AUDIO_CACHE = {}

@router.post("/upload")
async def upload_audio_file(request: Request):
    """
    Accepts audio upload via direct binary octet-stream or multipart form-data.
    """
    content_type = request.headers.get("content-type", "")
    file_id = str(uuid.uuid4())[:12]
    original_filename = "uploaded_audio.wav"

    # Priority 1: Check if client provided filename via header (Direct Binary Stream)
    if "x-filename" in request.headers or "application/octet-stream" in content_type:
        raw_name = request.headers.get("x-filename", "audio.wav")
        original_filename = urllib.parse.unquote(raw_name)
        ext = Path(original_filename).suffix.lower() or ".wav"
        save_path = UPLOADS_DIR / f"{file_id}{ext}"
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="Empty audio payload received.")
        with open(save_path, "wb") as f:
            f.write(body)
    elif "multipart/form-data" in content_type:
        try:
            form = await request.form()
            uploaded_file = form.get("file")
            if not uploaded_file:
                raise HTTPException(status_code=400, detail="No audio file received in form data.")
            original_filename = getattr(uploaded_file, "filename", original_filename)
            ext = Path(original_filename).suffix.lower() or ".wav"
            save_path = UPLOADS_DIR / f"{file_id}{ext}"
            with open(save_path, "wb") as f:
                content = await uploaded_file.read()
                f.write(content)
        except Exception:
            # Fallback to direct raw body
            raw_body = await request.body()
            ext = Path(original_filename).suffix.lower() or ".wav"
            save_path = UPLOADS_DIR / f"{file_id}{ext}"
            with open(save_path, "wb") as f:
                f.write(raw_body)
    else:
        # Default binary read
        raw_name = request.headers.get("x-filename", "audio.wav")
        original_filename = urllib.parse.unquote(raw_name)
        ext = Path(original_filename).suffix.lower() or ".wav"
        save_path = UPLOADS_DIR / f"{file_id}{ext}"
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="Empty audio payload received.")
        with open(save_path, "wb") as f:
            f.write(body)

    try:
        audio, sr, duration = load_audio(save_path)
    except Exception as e:
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(status_code=400, detail=f"Failed to decode audio file ({original_filename}): {str(e)}")

    metadata = AudioMetadata(
        file_id=file_id,
        original_filename=original_filename,
        duration_seconds=round(duration, 2),
        sample_rate=sr,
        channels=1,
        total_samples=len(audio)
    )

    _AUDIO_CACHE[file_id] = {
        "audio": audio,
        "sample_rate": sr,
        "metadata": metadata,
        "file_path": save_path
    }

    fft_spec = compute_fft_spectrum(audio, sr)
    tonal_bal = compute_tonal_balance(audio, sr)
    spectro = compute_spectrogram(audio, sr)
    waterfall_3d = compute_waterfall_3d(audio, sr)
    forensics = extract_forensic_metrics(audio, sr)

    return AnalysisResponse(
        metadata=metadata,
        fft_spectrum=fft_spec,
        tonal_balance=tonal_bal,
        spectrogram=spectro,
        waterfall_3d=waterfall_3d,
        forensics=forensics
    )

@router.post("/analyze-local-path")
async def analyze_local_path(payload: dict):
    """
    Analyzes an audio file directly from a local file path on disk without re-uploading.
    """
    path_str = payload.get("file_path")
    if not path_str:
        raise HTTPException(status_code=400, detail="Missing 'file_path' in request body.")
    
    file_path = Path(path_str)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Local file not found: {file_path}")

    file_id = str(uuid.uuid4())[:12]
    try:
        audio, sr, duration = load_audio(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode audio ({file_path.name}): {str(e)}")

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
    waterfall_3d = compute_waterfall_3d(audio, sr)
    forensics = extract_forensic_metrics(audio, sr)

    return AnalysisResponse(
        metadata=metadata,
        fft_spectrum=fft_spec,
        tonal_balance=tonal_bal,
        spectrogram=spectro,
        waterfall_3d=waterfall_3d,
        forensics=forensics
    )

@router.get("/audio/{file_id}")
async def stream_audio_file(file_id: str, request: Request):
    """
    Streams audio supporting HTTP 206 Partial Content Range requests.
    Enables smooth click-to-seek without resetting playback to 0!
    """
    cached = _AUDIO_CACHE.get(file_id)
    if not cached or not cached["file_path"].exists():
        matches = list(UPLOADS_DIR.glob(f"{file_id}.*"))
        if matches and matches[0].exists():
            file_path = matches[0]
        else:
            raise HTTPException(status_code=404, detail="Audio file not found or expired.")
    else:
        file_path = cached["file_path"]

    file_size = file_path.stat().st_size
    range_header = request.headers.get("range")
    content_type = "audio/wav" if file_path.suffix.lower() == ".wav" else "audio/mpeg"

    if range_header:
        # Parse range header: e.g. "bytes=12345-" or "bytes=100-200"
        bytes_str = range_header.replace("bytes=", "").strip()
        parts = bytes_str.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if (len(parts) > 1 and parts[1]) else file_size - 1
        end = min(end, file_size - 1)
        length = end - start + 1

        def iter_range():
            with open(file_path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
            "Content-Type": content_type
        }
        return StreamingResponse(iter_range(), status_code=206, headers=headers)
    else:
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Content-Type": content_type
        }
        def iter_full():
            with open(file_path, "rb") as f:
                while chunk := f.read(65536):
                    yield chunk
        return StreamingResponse(iter_full(), status_code=200, headers=headers)

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


def cleanup_expired_stems(max_age_seconds: int = STEMS_EXPIRE_SECONDS):
    """
    Scans STEMS_TEMP_DIR and deletes any temporary stem folders or files
    older than max_age_seconds (default: 1 hour / 3600s).
    """
    now = time.time()
    if not STEMS_TEMP_DIR.exists():
        return

    for entry in STEMS_TEMP_DIR.iterdir():
        try:
            mtime = entry.stat().st_mtime
            if (now - mtime) > max_age_seconds:
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    entry.unlink(missing_ok=True)
                print(f"[StemService] Cleaned up expired temporary stem: {entry.name}")
        except Exception as e:
            print(f"[StemService] Cleanup error on {entry}: {e}")


@router.post("/split-stems")
async def split_audio_stems(payload: dict):
    """
    Splits the currently loaded or specified audio file into 5 forensic stems:
    mixture, vocals, drums, bass, other.
    Results are saved in a temporary folder and automatically deleted after 1 hour.
    """
    # 1. Clean up old stems older than 1 hour
    cleanup_expired_stems()

    file_id = payload.get("file_id")
    file_path_str = payload.get("file_path")

    audio_file = None
    if file_id and file_id in _AUDIO_CACHE:
        audio_file = _AUDIO_CACHE[file_id]["file_path"]
    elif file_id:
        matches = list(UPLOADS_DIR.glob(f"{file_id}.*"))
        if matches and matches[0].exists():
            audio_file = matches[0]

    if not audio_file and file_path_str:
        candidate = Path(file_path_str)
        if candidate.exists():
            audio_file = candidate

    if not audio_file or not Path(audio_file).exists():
        raise HTTPException(status_code=404, detail="Audio file not found in active cache or specified path.")

    audio_file = Path(audio_file)
    stem_token = f"stem_{file_id or 'trk'}_{uuid.uuid4().hex[:6]}"
    target_dir = STEMS_TEMP_DIR / stem_token

    try:
        service = StemService(model_name="htdemucs")
        result = service.separate_track(audio_file, output_dir=target_dir, save_wavs=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stem splitting failed: {str(e)}")

    output_dir = result["output_dir"]
    stems_meta = {}

    for stem_name, file_path in result["file_paths"].items():
        size_bytes = file_path.stat().st_size if file_path.exists() else 0
        stems_meta[stem_name] = {
            "name": stem_name,
            "filename": file_path.name,
            "size_bytes": size_bytes,
            "size_formatted": f"{size_bytes / (1024 * 1024):.2f} MB",
            "stream_url": f"/api/stems/{stem_token}/{stem_name}",
            "file_path": str(file_path)
        }

    return {
        "status": "success",
        "stem_token": stem_token,
        "original_filename": audio_file.name,
        "duration_seconds": round(result["duration"], 2),
        "expires_in_seconds": STEMS_EXPIRE_SECONDS,
        "expires_in_text": "1 hour",
        "stems": stems_meta
    }


@router.get("/stems/{token}/{stem_name}")
async def stream_stem_file(token: str, stem_name: str, request: Request):
    """
    Streams an individual temporary stem WAV file with HTTP 206 Partial Content Range support.
    """
    # Clean up expired stems
    cleanup_expired_stems()

    # Normalize stem name (strip .wav if sent)
    clean_stem = stem_name.replace(".wav", "").lower()
    if clean_stem not in StemService.STEM_NAMES:
        raise HTTPException(status_code=400, detail=f"Invalid stem name '{clean_stem}'. Allowed: {StemService.STEM_NAMES}")

    token_dir = STEMS_TEMP_DIR / token
    if not token_dir.exists():
        raise HTTPException(status_code=404, detail="Temporary stem session expired or not found.")

    # Find the stem file inside token_dir (or its subfolder)
    matches = list(token_dir.rglob(f"*_{clean_stem}.wav")) + list(token_dir.rglob(f"{clean_stem}.wav"))
    if not matches or not matches[0].exists():
        raise HTTPException(status_code=404, detail=f"Stem file '{clean_stem}' not found in session.")

    file_path = matches[0]

    # Verify expiration (1 hour)
    now = time.time()
    if (now - file_path.stat().st_mtime) > STEMS_EXPIRE_SECONDS:
        cleanup_expired_stems()
        raise HTTPException(status_code=410, detail="This temporary stem file has expired (>1 hour) and was deleted.")

    file_size = file_path.stat().st_size
    range_header = request.headers.get("range")
    content_type = "audio/wav"

    if range_header:
        bytes_str = range_header.replace("bytes=", "").strip()
        parts = bytes_str.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if (len(parts) > 1 and parts[1]) else file_size - 1
        end = min(end, file_size - 1)
        length = end - start + 1

        def iter_range():
            with open(file_path, "rb") as f:
                f.seek(start)
                rem = length
                while rem > 0:
                    chunk = f.read(min(65536, rem))
                    if not chunk:
                        break
                    rem -= len(chunk)
                    yield chunk

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
            "Content-Type": content_type
        }
        return StreamingResponse(iter_range(), status_code=206, headers=headers)
    else:
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Content-Type": content_type
        }
        def iter_full():
            with open(file_path, "rb") as f:
                while chunk := f.read(65536):
                    yield chunk
        return StreamingResponse(iter_full(), status_code=200, headers=headers)


@router.post("/load-stem")
async def load_stem_to_visualizer(payload: dict):
    """
    Loads an extracted stem file directly into the visualizer (updating 3D waterfall,
    spectrogram, equalizer, and forensic metrics for that specific component).
    """
    token = payload.get("stem_token")
    stem_name = payload.get("stem_name", "").replace(".wav", "").lower()

    if not token or not stem_name:
        raise HTTPException(status_code=400, detail="Missing 'stem_token' or 'stem_name'.")

    token_dir = STEMS_TEMP_DIR / token
    matches = list(token_dir.rglob(f"*_{stem_name}.wav")) + list(token_dir.rglob(f"{stem_name}.wav"))
    if not matches or not matches[0].exists():
        raise HTTPException(status_code=404, detail="Stem file not found or expired.")

    stem_path = matches[0]
    file_id = f"stem_{token}_{stem_name}"

    try:
        audio, sr, duration = load_audio(stem_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load stem: {e}")

    metadata = AudioMetadata(
        file_id=file_id,
        original_filename=f"[{stem_name.upper()}] {stem_path.name}",
        duration_seconds=round(duration, 2),
        sample_rate=sr,
        channels=1,
        total_samples=len(audio)
    )

    _AUDIO_CACHE[file_id] = {
        "audio": audio,
        "sample_rate": sr,
        "metadata": metadata,
        "file_path": stem_path
    }

    fft_spec = compute_fft_spectrum(audio, sr)
    tonal_bal = compute_tonal_balance(audio, sr)
    spectro = compute_spectrogram(audio, sr)
    waterfall_3d = compute_waterfall_3d(audio, sr)
    forensics = extract_forensic_metrics(audio, sr)

    return AnalysisResponse(
        metadata=metadata,
        fft_spectrum=fft_spec,
        tonal_balance=tonal_bal,
        spectrogram=spectro,
        waterfall_3d=waterfall_3d,
        forensics=forensics
    )

