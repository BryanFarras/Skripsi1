"""
Dataset Builder for Baseline AI Audio Classifier
Processes Bona-fide (MUSDB18) and Spoof (Suno AI) tracks into balanced,
structured tabular datasets with track-level grouping to prevent data leakage.
"""

import os
import sys
import csv
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

# Ensure package importability
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ml_baseline.config import (
        SUNO_DIR,
        MUSDB18_DIR,
        DATA_DIR,
        LABEL_BONAFIDE,
        LABEL_SPOOF,
        RANDOM_STATE,
        SLICE_DURATION,
        SLICE_HOP
    )
    from ml_baseline.features import AudioFeatureExtractor
else:
    from .config import (
        SUNO_DIR,
        MUSDB18_DIR,
        DATA_DIR,
        LABEL_BONAFIDE,
        LABEL_SPOOF,
        RANDOM_STATE,
        SLICE_DURATION,
        SLICE_HOP
    )
    from .features import AudioFeatureExtractor

def discover_audio_files(max_bonafide_tracks: int = 15) -> Tuple[List[Path], List[Path]]:
    """
    Discovers Spoof (Suno AI) and Bona-fide (MUSDB18) audio files.
    Limits bonafide tracks if requested to maintain class balance.
    """
    # 1. Discover Suno tracks (Spoof)
    suno_files = sorted(list(SUNO_DIR.glob("*.mp3")) + list(SUNO_DIR.glob("*.wav")))
    if not suno_files:
        raise FileNotFoundError(f"No Suno tracks found in {SUNO_DIR}")

    # 2. Discover MUSDB18 tracks (Bona-fide)
    musdb_files = sorted(list(MUSDB18_DIR.glob("*.stem.mp4")))
    if not musdb_files:
        raise FileNotFoundError(f"No MUSDB18 tracks found in {MUSDB18_DIR}")

    # Sample a balanced subset of bona-fide tracks if requested
    rng = np.random.RandomState(RANDOM_STATE)
    if max_bonafide_tracks and len(musdb_files) > max_bonafide_tracks:
        selected_musdb = list(rng.choice(musdb_files, size=max_bonafide_tracks, replace=False))
        selected_musdb.sort()
    else:
        selected_musdb = musdb_files

    return suno_files, selected_musdb

def build_dataset(
    max_bonafide_tracks: int = 12,
    max_slices_per_track: int = 40,
    force_rebuild: bool = False
) -> Dict[str, Any]:
    """
    Extracts features across spoof and bona-fide tracks and persists data files.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "dataset.csv"
    npz_path = DATA_DIR / "dataset.npz"
    meta_path = DATA_DIR / "metadata.json"

    if not force_rebuild and npz_path.exists() and meta_path.exists():
        print(f"[*] Found existing cached dataset at {npz_path}")
        return load_dataset()

    print("[*] Discovering tracks for dataset extraction...")
    suno_tracks, musdb_tracks = discover_audio_files(max_bonafide_tracks=max_bonafide_tracks)
    print(f"    - Found {len(suno_tracks)} Spoof (Suno AI) tracks")
    print(f"    - Selected {len(musdb_tracks)} Bona-fide (MUSDB18) tracks")

    extractor = AudioFeatureExtractor()
    feature_names = extractor.get_feature_names()

    records = []
    features_list = []
    track_id_counter = 0

    total_start = time.time()

    # Process Spoof Tracks (Suno)
    print("\n[*] Processing Spoof Tracks (Suno AI)...")
    for track_idx, track_path in enumerate(suno_tracks, 1):
        t0 = time.time()
        track_id_counter += 1
        curr_track_id = f"spoof_{track_id_counter}"
        
        try:
            y, sr = extractor.load_audio_file(track_path)
            slices = list(extractor.slice_waveform(y, duration=SLICE_DURATION, hop=SLICE_HOP))
            if max_slices_per_track and len(slices) > max_slices_per_track:
                # Evenly sample slices across the song
                indices = np.linspace(0, len(slices) - 1, max_slices_per_track, dtype=int)
                slices = [slices[i] for i in indices]

            for slice_idx, start_sec, chunk in slices:
                feat = extractor.extract_from_waveform(chunk, sr=sr)
                features_list.append(feat)
                records.append({
                    "track_id": curr_track_id,
                    "track_name": track_path.stem,
                    "slice_idx": slice_idx,
                    "start_sec": round(start_sec, 2),
                    "label": LABEL_SPOOF,
                    "label_name": "Spoof (AI)"
                })
            print(f"    [{track_idx}/{len(suno_tracks)}] {track_path.stem[:35]:<35} -> {len(slices)} slices ({time.time()-t0:.2f}s)")
        except Exception as e:
            print(f"    [!] Error reading {track_path.name}: {e}")

    # Process Bona-fide Tracks (MUSDB18)
    print("\n[*] Processing Bona-fide Tracks (MUSDB18 Human Studio)...")
    for track_idx, track_path in enumerate(musdb_tracks, 1):
        t0 = time.time()
        track_id_counter += 1
        curr_track_id = f"bonafide_{track_id_counter}"

        try:
            y, sr = extractor.load_audio_file(track_path)
            slices = list(extractor.slice_waveform(y, duration=SLICE_DURATION, hop=SLICE_HOP))
            if max_slices_per_track and len(slices) > max_slices_per_track:
                indices = np.linspace(0, len(slices) - 1, max_slices_per_track, dtype=int)
                slices = [slices[i] for i in indices]

            for slice_idx, start_sec, chunk in slices:
                feat = extractor.extract_from_waveform(chunk, sr=sr)
                features_list.append(feat)
                records.append({
                    "track_id": curr_track_id,
                    "track_name": track_path.stem.replace(".stem", ""),
                    "slice_idx": slice_idx,
                    "start_sec": round(start_sec, 2),
                    "label": LABEL_BONAFIDE,
                    "label_name": "Bona-fide (Human)"
                })
            print(f"    [{track_idx}/{len(musdb_tracks)}] {track_path.stem[:35]:<35} -> {len(slices)} slices ({time.time()-t0:.2f}s)")
        except Exception as e:
            print(f"    [!] Error reading {track_path.name}: {e}")

    X = np.array(features_list, dtype=np.float32)
    y = np.array([r["label"] for r in records], dtype=int)
    groups = np.array([r["track_id"] for r in records])

    print(f"\n[*] Feature extraction complete in {time.time()-total_start:.1f}s!")
    print(f"    Total samples: {len(X)} | Feature dimension: {X.shape[1]}")
    print(f"    Spoof slices: {np.sum(y == LABEL_SPOOF)} | Bona-fide slices: {np.sum(y == LABEL_BONAFIDE)}")

    # Save CSV
    meta_headers = ["track_id", "track_name", "slice_idx", "start_sec", "label", "label_name"]
    full_headers = meta_headers + feature_names
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(full_headers)
        for r, feat_row in zip(records, X):
            row = [r[k] for k in meta_headers] + [float(val) for val in feat_row]
            writer.writerow(row)
    print(f"[*] Saved dataset CSV: {csv_path}")

    # Save NPZ
    np.savez_compressed(
        npz_path,
        X=X,
        y=y,
        groups=groups,
        feature_names=np.array(feature_names)
    )
    print(f"[*] Saved dataset NPZ: {npz_path}")

    # Save Metadata JSON
    metadata = {
        "num_samples": int(len(X)),
        "num_features": int(X.shape[1]),
        "num_spoof_samples": int(np.sum(y == LABEL_SPOOF)),
        "num_bonafide_samples": int(np.sum(y == LABEL_BONAFIDE)),
        "num_tracks": int(len(set(groups))),
        "feature_names": feature_names,
        "sample_rate": extractor.sample_rate,
        "slice_duration_sec": SLICE_DURATION,
        "slice_hop_sec": SLICE_HOP,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[*] Saved metadata: {meta_path}")

    return {
        "X": X,
        "y": y,
        "groups": groups,
        "feature_names": feature_names,
        "metadata": metadata
    }

def load_dataset() -> Dict[str, Any]:
    """Loads cached dataset from disk."""
    npz_path = DATA_DIR / "dataset.npz"
    meta_path = DATA_DIR / "metadata.json"
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset not built yet! Please run build_dataset().")

    data = np.load(npz_path, allow_pickle=True)
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    return {
        "X": data["X"],
        "y": data["y"],
        "groups": data["groups"],
        "feature_names": list(data["feature_names"]),
        "metadata": metadata
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and build audio dataset")
    parser.add_argument("--max-bonafide", type=int, default=12, help="Max bona-fide tracks to sample")
    parser.add_argument("--max-slices", type=int, default=40, help="Max slices per track")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild dataset")
    args = parser.parse_args()

    build_dataset(
        max_bonafide_tracks=args.max_bonafide,
        max_slices_per_track=args.max_slices,
        force_rebuild=args.rebuild
    )
