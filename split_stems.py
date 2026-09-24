#!/usr/bin/env python3
"""
AI Music Detector - Audio Stem Splitter CLI
Splits any audio file (WAV, MP3, FLAC, M4A, STEM.MP4) into 5 separate components:
  1. mixture (Full Song)
  2. vocals  (Singing voice / speech)
  3. drums   (Percussion, kick, snare, hi-hats)
  4. bass    (Bassline, sub-frequencies)
  5. other   (Guitars, synths, pianos, reverberation)
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Auto-relaunch inside workspace .venv if torch or demucs are not in current Python
def ensure_venv():
    try:
        import torch
        import demucs
    except ImportError:
        venv_python = PROJECT_ROOT.parent.parent / ".venv" / "Scripts" / "python.exe"
        if venv_python.exists() and str(venv_python) != sys.executable:
            print(f"[*] Switching to virtual environment Python: {venv_python}")
            import subprocess
            cmd = [str(venv_python), str(__file__)] + sys.argv[1:]
            res = subprocess.run(cmd)
            sys.exit(res.returncode)
        else:
            print("[ERROR] Required packages 'torch' or 'demucs' are not installed.")
            print("Please run: pip install torch torchaudio demucs soundfile")
            sys.exit(1)

ensure_venv()

from backend.app.services.stem_service import StemService, ensure_ffmpeg_in_path

def format_size(num_bytes: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"

def process_single_file(audio_path: Path, output_dir: Optional[Path], service: StemService):
    stem_folder_name = f"{audio_path.stem}_stems"
    if output_dir is None:
        target_dir = audio_path.parent / stem_folder_name
    elif output_dir.name == stem_folder_name:
        target_dir = output_dir
    else:
        target_dir = output_dir / stem_folder_name

    print("=" * 65)
    print(f"  PROCESSING TRACK: {audio_path.name}")
    print("=" * 65)
    print(f"[*] Input File:    {audio_path}")
    print(f"[*] Output Folder: {target_dir}")
    print(f"[*] Stem Model:    {service.model_name} ({service._get_device().upper()})")
    print("")

    start_time = time.time()
    result = service.separate_track(audio_path, output_dir=target_dir, save_wavs=True)
    elapsed = time.time() - start_time

    print("\n" + "-" * 65)
    print(f"  SEPARATION COMPLETE ({elapsed:.2f}s | Audio Duration: {result['duration']:.2f}s)")
    print("-" * 65)
    
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    for stem_name, file_path in result["file_paths"].items():
        size_str = format_size(file_path.stat().st_size) if file_path.exists() else "N/A"
        tag = f"[{stem_name.upper()}]"
        print(f"  {tag:<10} -> {file_path.name} ({size_str})")

    print(f"\n[SUCCESS] All 5 stems saved to:\n{result['output_dir']}\n")
    return result

def main():
    parser = argparse.ArgumentParser(
        description="AI Music Detector - Multi-Stem Audio Splitter (Demucs & MUSDB18)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python split_stems.py "song1.wav"
    -> Creates folder: song1_stems/
  python split_stems.py "song1.wav" --output-dir "my_folder"
    -> Creates folder: my_folder/song1_stems/
  python split_stems.py --batch-dir "Datasets/raw/ai_suno" --output-dir "Datasets/stems/ai_suno"
    -> Creates folder for each track: Datasets/stems/ai_suno/<track>_stems/
        """
    )
    parser.add_argument("input_file", nargs="?", default=None, help="Path to single audio file (WAV, MP3, FLAC, M4A, STEM.MP4)")
    parser.add_argument("--output-dir", "-o", default=None, help="Directory to save extracted stem WAV files")
    parser.add_argument("--batch-dir", "-b", default=None, help="Directory containing multiple audio files to process in batch")
    parser.add_argument("--model", "-m", default="htdemucs", help="Demucs model name (default: htdemucs)")
    parser.add_argument("--device", "-d", default=None, help="Device to use ('cuda' or 'cpu', default: auto-detect)")

    args = parser.parse_args()

    if not args.input_file and not args.batch_dir:
        parser.print_help()
        print("\n[NOTE] Please provide an input file or a batch directory.")
        sys.exit(1)

    # Initialize stem service
    service = StemService(model_name=args.model, device=args.device)

    # Single File Mode
    if args.input_file:
        file_path = Path(args.input_file).resolve()
        if not file_path.exists():
            print(f"[ERROR] Input audio file not found: {file_path}")
            sys.exit(1)

        out_dir = Path(args.output_dir).resolve() if args.output_dir else None
        process_single_file(file_path, out_dir, service)

    # Batch Directory Mode
    elif args.batch_dir:
        batch_path = Path(args.batch_dir).resolve()
        if not batch_path.exists() or not batch_path.is_dir():
            print(f"[ERROR] Batch directory not found: {batch_path}")
            sys.exit(1)

        exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".mp4"}
        audio_files = [p for p in batch_path.rglob("*") if p.suffix.lower() in exts and not p.name.endswith(tuple(f"_{s}.wav" for s in StemService.STEM_NAMES))]

        if not audio_files:
            print(f"[ERROR] No audio files found in: {batch_path}")
            sys.exit(1)

        print(f"[*] Found {len(audio_files)} audio file(s) in batch folder: {batch_path}")
        master_out = Path(args.output_dir).resolve() if args.output_dir else batch_path / "separated_stems"

        for idx, af in enumerate(audio_files, 1):
            print(f"\n[{idx}/{len(audio_files)}] Processing {af.name}...")
            target_stem_dir = master_out / f"{af.stem}_stems"
            try:
                service.separate_track(af, output_dir=target_stem_dir, save_wavs=True)
            except Exception as e:
                print(f"[ERROR] Failed to process {af.name}: {e}")

        print(f"\n[SUCCESS] Batch processing finished! Stems saved to:\n{master_out}")

if __name__ == "__main__":
    main()
