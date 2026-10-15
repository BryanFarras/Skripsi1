"""
Online Audio Service for AI Music Forensics & XAI
Fetches and streams audio directly from online platforms:
  - YouTube & YouTube Music
  - Spotify (via track resolution & high-fidelity stream retrieval)
  - Apple Music / iTunes (instant studio preview & full audio retrieval)
  - SoundCloud (native audio stream)
  - Direct audio URLs (MP3, WAV, AAC, M4A, FLAC, OGG)
  - Keyword / Artist / Song Title search queries
"""

import os
import re
import sys
import uuid
import json
import shutil
import subprocess
import urllib.parse
import urllib.request
import requests
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from ..config import UPLOADS_DIR

# Locate and configure ffmpeg
def get_ffmpeg_exe() -> str:
    """Locates ffmpeg.exe from PATH or virtual environment by searching parents."""
    which_ffmpeg = shutil.which("ffmpeg")
    if which_ffmpeg:
        return which_ffmpeg
    for p in Path(__file__).resolve().parents:
        cand = p / ".venv" / "Scripts" / "ffmpeg.exe"
        if cand.exists():
            return str(cand)
    return "ffmpeg"

def convert_to_mp3(input_path: Path, output_path: Path) -> Path:
    """Converts any audio/video container (webm, m4a, ogg, etc.) to 192k MP3 via ffmpeg."""
    if input_path.suffix.lower() == ".mp3" and input_path.exists():
        return input_path

    ffmpeg_bin = get_ffmpeg_exe()
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", str(input_path),
        "-vn",
        "-acodec", "libmp3lame",
        "-b:a", "192k",
        str(output_path)
    ]

    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
        if proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
            if input_path != output_path and input_path.exists():
                input_path.unlink(missing_ok=True)
            return output_path
    except Exception as e:
        print(f"[FFmpeg Conversion Notice] {e}")

    # Fallback to copy if input exists
    return input_path

def detect_platform_source(url_or_query: str) -> str:
    """Classifies the URL or query into its source category."""
    text = url_or_query.strip().lower()
    if not (text.startswith("http://") or text.startswith("https://") or text.startswith("spotify:")):
        return "search"
    if "spotify.com" in text or text.startswith("spotify:"):
        return "spotify"
    if "apple.com" in text or "itunes.apple.com" in text:
        return "itunes"
    if "soundcloud.com" in text:
        return "soundcloud"
    if "youtube.com" in text or "youtu.be" in text:
        return "youtube"
    if any(text.endswith(ext) for ext in [".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus"]):
        return "direct_url"
    return "web_url"

def fetch_spotify_metadata(url: str) -> Dict[str, Any]:
    """Fetches track title, artist, and artwork from Spotify oEmbed endpoint without API keys."""
    oembed_url = f"https://open.spotify.com/oembed?url={urllib.parse.quote(url)}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0"}
    try:
        res = requests.get(oembed_url, headers=headers, timeout=8)
        if res.status_code == 200:
            data = res.json()
            title = data.get("title", "Spotify Track")
            artwork = data.get("thumbnail_url", "")
            return {
                "title": title,
                "artist": data.get("author_name", "Spotify Artist"),
                "artwork_url": artwork,
                "original_url": url
            }
    except Exception as e:
        print(f"[Spotify oEmbed Note] {e}")
    return {"title": "Spotify Song", "artist": "", "artwork_url": "", "original_url": url}

def fetch_itunes_track_info(url_or_id: str) -> Dict[str, Any]:
    """Retrieves track metadata and instant high-fidelity 30s preview URL from iTunes."""
    track_id = None
    if url_or_id.isdigit():
        track_id = url_or_id
    else:
        match_i = re.search(r"[?&]i=(\d+)", url_or_id)
        if match_i:
            track_id = match_i.group(1)
        else:
            match_id = re.search(r"/(\d+)(?:[?/#]|$)", url_or_id)
            if match_id:
                track_id = match_id.group(1)

    if track_id:
        api_url = f"https://itunes.apple.com/lookup?id={track_id}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        try:
            res = requests.get(api_url, headers=headers, timeout=8)
            if res.status_code == 200:
                data = res.json()
                results = data.get("results", [])
                if results:
                    item = results[0]
                    artwork = item.get("artworkUrl100", "").replace("100x100bb", "300x300bb")
                    return {
                        "title": item.get("trackName", "iTunes Track"),
                        "artist": item.get("artistName", "Artist"),
                        "album": item.get("collectionName", ""),
                        "preview_url": item.get("previewUrl", ""),
                        "artwork_url": artwork,
                        "genre": item.get("primaryGenreName", "Music"),
                        "original_url": url_or_id
                    }
        except Exception as e:
            print(f"[iTunes Lookup Error] {e}")

    return {"title": "iTunes Track", "artist": "", "preview_url": "", "artwork_url": "", "original_url": url_or_id}

def search_itunes_preview(query: str) -> Optional[Dict[str, Any]]:
    """Searches iTunes for instant 30-second studio master audio preview."""
    encoded = urllib.parse.quote_plus(query)
    api_url = f"https://itunes.apple.com/search?term={encoded}&entity=song&limit=1&media=music"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        res = requests.get(api_url, headers=headers, timeout=6)
        if res.status_code == 200:
            data = res.json()
            results = data.get("results", [])
            if results and results[0].get("previewUrl"):
                item = results[0]
                artwork = item.get("artworkUrl100", "").replace("100x100bb", "300x300bb")
                return {
                    "title": item.get("trackName", query),
                    "artist": item.get("artistName", ""),
                    "album": item.get("collectionName", ""),
                    "preview_url": item.get("previewUrl", ""),
                    "artwork_url": artwork,
                    "genre": item.get("primaryGenreName", "Music")
                }
    except Exception:
        pass
    return None

def download_direct_audio(url: str, save_path: Path) -> Path:
    """Downloads audio file from a direct HTTP/HTTPS audio URL."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0"}
    with requests.get(url, headers=headers, stream=True, timeout=25) as r:
        r.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
    return save_path

def download_via_ytdlp(query_or_url: str, output_base: Path) -> Tuple[Path, Dict[str, Any]]:
    """
    Downloads audio stream using yt-dlp (YouTube, SoundCloud, and web sources)
    without requiring postprocessor dependencies, then converts cleanly to MP3.
    """
    import yt_dlp

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f"{output_base}.%(ext)s",
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
        'max_filesize': 70 * 1024 * 1024,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(query_or_url, download=True)
        if 'entries' in info and len(info['entries']) > 0:
            info = info['entries'][0]

        title = info.get('title', 'Online Audio')
        artist = info.get('artist') or info.get('uploader') or info.get('channel') or ''
        thumbnail = info.get('thumbnail', '')
        duration = info.get('duration', 0)

        # Find whatever file was downloaded
        candidates = list(output_base.parent.glob(f"{output_base.name}.*"))
        if not candidates:
            raise FileNotFoundError(f"yt-dlp could not locate downloaded file at {output_base}")
        
        raw_downloaded = candidates[0]
        final_mp3 = output_base.parent / f"{output_base.name}.mp3"
        saved_mp3 = convert_to_mp3(raw_downloaded, final_mp3)

        meta = {
            "title": title,
            "artist": artist,
            "artwork_url": thumbnail,
            "duration": duration,
            "uploader": info.get('uploader', ''),
            "view_count": info.get('view_count', 0),
            "webpage_url": info.get('webpage_url', query_or_url)
        }
        return saved_mp3, meta

def fetch_and_process_online_audio(url_or_query: str) -> Dict[str, Any]:
    """
    Unified entry point for online audio analysis.
    Supports YouTube, Spotify, Apple Music / iTunes, SoundCloud, Direct URLs, and Search.
    Returns metadata and local file path ready for full 110-D TreeSHAP forensic analysis.
    """
    clean_input = url_or_query.strip()
    if not clean_input:
        raise ValueError("URL or search query cannot be empty.")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    file_id = f"online_{uuid.uuid4().hex[:10]}"
    source = detect_platform_source(clean_input)

    print(f"\n[Online Audio Service] Input: '{clean_input[:65]}' | Source: {source}")

    title = "Online Audio Track"
    artist = ""
    artwork_url = ""
    original_url = clean_input
    saved_file: Optional[Path] = None

    # 1. SPOTIFY URL
    if source == "spotify":
        sp_meta = fetch_spotify_metadata(clean_input)
        title = sp_meta.get("title", "Spotify Track")
        artist = sp_meta.get("artist", "")
        artwork_url = sp_meta.get("artwork_url", "")

        query = f"{title} {artist}".strip()
        itunes_match = search_itunes_preview(query)
        if itunes_match and itunes_match.get("preview_url"):
            dest_raw = UPLOADS_DIR / f"{file_id}.m4a"
            dest_mp3 = UPLOADS_DIR / f"{file_id}.mp3"
            download_direct_audio(itunes_match["preview_url"], dest_raw)
            saved_file = convert_to_mp3(dest_raw, dest_mp3)
            if not artwork_url:
                artwork_url = itunes_match.get("artwork_url", "")
        else:
            output_base = UPLOADS_DIR / file_id
            saved_file, yt_meta = download_via_ytdlp(f"ytsearch1:{query} audio", output_base)
            if not artwork_url:
                artwork_url = yt_meta.get("artwork_url", "")

    # 2. ITUNES / APPLE MUSIC URL
    elif source == "itunes":
        it_meta = fetch_itunes_track_info(clean_input)
        title = it_meta.get("title", "Apple Music Track")
        artist = it_meta.get("artist", "")
        artwork_url = it_meta.get("artwork_url", "")

        if it_meta.get("preview_url"):
            dest_raw = UPLOADS_DIR / f"{file_id}.m4a"
            dest_mp3 = UPLOADS_DIR / f"{file_id}.mp3"
            download_direct_audio(it_meta["preview_url"], dest_raw)
            saved_file = convert_to_mp3(dest_raw, dest_mp3)
        else:
            query = f"{title} {artist}".strip()
            output_base = UPLOADS_DIR / file_id
            saved_file, yt_meta = download_via_ytdlp(f"ytsearch1:{query} audio", output_base)
            if not artwork_url:
                artwork_url = yt_meta.get("artwork_url", "")

    # 3. DIRECT AUDIO URL
    elif source == "direct_url":
        ext = Path(urllib.parse.urlparse(clean_input).path).suffix.lower() or ".mp3"
        dest_raw = UPLOADS_DIR / f"{file_id}{ext}"
        dest_mp3 = UPLOADS_DIR / f"{file_id}.mp3"
        download_direct_audio(clean_input, dest_raw)
        saved_file = convert_to_mp3(dest_raw, dest_mp3)
        title = Path(urllib.parse.urlparse(clean_input).path).stem or "Direct Audio Stream"

    # 4. YOUTUBE, SOUNDCLOUD, OR GENERIC WEB MEDIA URL
    elif source in ("youtube", "soundcloud", "web_url"):
        output_base = UPLOADS_DIR / file_id
        saved_file, yt_meta = download_via_ytdlp(clean_input, output_base)
        title = yt_meta.get("title", title)
        artist = yt_meta.get("artist", "")
        artwork_url = yt_meta.get("artwork_url", "")
        original_url = yt_meta.get("webpage_url", clean_input)

    # 5. KEYWORD SEARCH QUERY
    else: # source == "search"
        output_base = UPLOADS_DIR / file_id
        saved_file, yt_meta = download_via_ytdlp(f"ytsearch1:{clean_input} audio", output_base)
        title = yt_meta.get("title", clean_input)
        artist = yt_meta.get("artist", "")
        artwork_url = yt_meta.get("artwork_url", "")
        original_url = yt_meta.get("webpage_url", clean_input)

    if not saved_file or not saved_file.exists():
        raise RuntimeError("Failed to retrieve or process online audio.")

    clean_title = re.sub(r'[\\/*?:"<>|]', "", title)[:60]
    display_filename = f"{clean_title}.mp3" if not clean_title.endswith((".mp3", ".wav", ".m4a")) else clean_title

    return {
        "file_path": saved_file,
        "file_id": file_id,
        "original_filename": display_filename,
        "title": title,
        "artist": artist,
        "source": source,
        "artwork_url": artwork_url,
        "original_url": original_url,
        "file_size_bytes": saved_file.stat().st_size
    }
