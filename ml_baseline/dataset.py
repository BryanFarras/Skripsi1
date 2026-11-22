"""
Dataset Builder for Baseline AI Audio Classifier
Processes Bona-fide (MUSDB18 Master Mixes / bonafide_30) and 
Spoof (4 Open-Source Generators: ACE-Step, HeartMuLa, DiffRhythm, Muse)
into balanced, structured tabular datasets with track-level grouping to prevent data leakage.
(Suno AI is strictly excluded and deprecated per thesis supervisor directive).
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
        BONAFIDE_DIR,
        MUSDB18_DIR,
        OPEN_SOURCE_GENERATOR_DIRS,
        DATA_DIR,
        LABEL_BONAFIDE,
        LABEL_SPOOF,
        RANDOM_STATE,
        SLICE_DURATION,
        SLICE_HOP
    )
    from ml_baseline.features import AudioFeatureExtractor
    from ml_baseline.normalization import normalize_audio_loudness
else:
    from .config import (
        BONAFIDE_DIR,
        MUSDB18_DIR,
        OPEN_SOURCE_GENERATOR_DIRS,
        DATA_DIR,
        LABEL_BONAFIDE,
        LABEL_SPOOF,
        RANDOM_STATE,
        SLICE_DURATION,
        SLICE_HOP
    )
    from .features import AudioFeatureExtractor
    from .normalization import normalize_audio_loudness

def discover_audio_files(
    max_bonafide_tracks: int = 30,
    max_spoof_per_tool: int = 30
) -> Tuple[Dict[str, List[Path]], List[Path]]:
    """
    Discovers Spoof across the 4 Open-Source Generators (ACE-Step, HeartMuLa, DiffRhythm, Muse)
    and Bona-fide (MUSDB18 / bonafide_30) audio files.
    Returns:
        spoof_by_tool: Dict mapping tool_name -> List of Paths
        bonafide_files: List of Paths
    """
    rng = np.random.RandomState(RANDOM_STATE)

    # 1. Discover Spoof tracks across all 4 open-source generators
    spoof_by_tool: Dict[str, List[Path]] = {}
    for tool_name, tool_dir in OPEN_SOURCE_GENERATOR_DIRS.items():
        if not tool_dir.exists():
            print(f"    [!] Warning: Tool directory not found: {tool_dir}")
            continue
        audio_files = sorted(list(tool_dir.glob("*.mp3")) + list(tool_dir.glob("*.wav")))
        if max_spoof_per_tool and len(audio_files) > max_spoof_per_tool:
            selected = list(rng.choice(audio_files, size=max_spoof_per_tool, replace=False))
            selected.sort()
            spoof_by_tool[tool_name] = selected
        else:
            spoof_by_tool[tool_name] = audio_files

    # 2. Discover Bona-fide tracks (prefer bonafide_30, fallback to musdb18)
    bonafide_files: List[Path] = []
    if BONAFIDE_DIR.exists():
        bf_candidates = sorted(list(BONAFIDE_DIR.glob("*.wav")) + list(BONAFIDE_DIR.glob("*.mp3")))
        if bf_candidates:
            bonafide_files = bf_candidates

    if not bonafide_files and MUSDB18_DIR.exists():
        musdb_train = sorted(list(MUSDB18_DIR.glob("*.stem.mp4")))
        musdb_test = sorted(list((MUSDB18_DIR.parent / "test").glob("*.stem.mp4")))
        bonafide_files = musdb_train + musdb_test

    if max_bonafide_tracks and len(bonafide_files) > max_bonafide_tracks:
        selected_bf = list(rng.choice(bonafide_files, size=max_bonafide_tracks, replace=False))
        selected_bf.sort()
        bonafide_files = selected_bf

    return spoof_by_tool, bonafide_files

def build_dataset(
    max_bonafide_tracks: int = 30,
    max_spoof_per_tool: int = 30,
    max_slices_per_track: int = 25,
    force_rebuild: bool = False,
    normalize_loudness: bool = True,
    target_lufs: float = -14.0,
    output_name: str = "dataset"
) -> Dict[str, Any]:
    """
    Extracts features across spoof (4 open-source generators) and bona-fide tracks.
    Optionally standardizes loudness to target_lufs (EBU R128 / ITU-R BS.1770).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / f"{output_name}.csv"
    npz_path = DATA_DIR / f"{output_name}.npz"
    meta_path = DATA_DIR / f"{output_name}_metadata.json"

    if not force_rebuild and npz_path.exists() and meta_path.exists():
        print(f"[*] Found existing cached dataset at {npz_path}")
        return load_dataset(output_name=output_name)

    print(f"[*] Discovering tracks for dataset extraction ('{output_name}')...")
    print(f"    - Loudness normalization (EBU R128): {'ENABLED (-14 LUFS)' if normalize_loudness else 'DISABLED (Raw)'}")
    spoof_by_tool, bonafide_tracks = discover_audio_files(
        max_bonafide_tracks=max_bonafide_tracks,
        max_spoof_per_tool=max_spoof_per_tool
    )

    total_spoof_tracks = sum(len(files) for files in spoof_by_tool.values())
    print(f"    - Found {total_spoof_tracks} Spoof tracks across {list(spoof_by_tool.keys())}:")
    for t_name, files in spoof_by_tool.items():
        print(f"      * {t_name}: {len(files)} tracks")
    print(f"    - Selected {len(bonafide_tracks)} Bona-fide reference tracks")

    extractor = AudioFeatureExtractor()
    feature_names = extractor.get_feature_names()

    records = []
    features_list = []
    track_id_counter = 0

    total_start = time.time()

    # Process Spoof Tracks across each open-source tool
    for tool_name, track_list in spoof_by_tool.items():
        print(f"\n[*] Processing Spoof Tracks for [{tool_name.upper()}] ({len(track_list)} tracks)...")
        for track_idx, track_path in enumerate(track_list, 1):
            t0 = time.time()
            track_id_counter += 1
            curr_track_id = f"spoof_{tool_name}_{track_id_counter}"

            try:
                y, sr = extractor.load_audio_file(track_path)
                if normalize_loudness:
                    y, _, _ = normalize_audio_loudness(y, sr=sr, target_lufs=target_lufs)
                slices = list(extractor.slice_waveform(y, duration=SLICE_DURATION, hop=SLICE_HOP))
                if max_slices_per_track and len(slices) > max_slices_per_track:
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
                        "label_name": "Spoof (AI)",
                        "generator_tool": tool_name
                    })
                print(f"    [{track_idx}/{len(track_list)}] {track_path.stem[:35]:<35} -> {len(slices)} slices ({time.time()-t0:.2f}s)")
            except Exception as e:
                print(f"    [!] Error reading {track_path.name}: {e}")

    # Process Bona-fide Tracks
    print(f"\n[*] Processing Bona-fide Tracks ({len(bonafide_tracks)} tracks)...")
    for track_idx, track_path in enumerate(bonafide_tracks, 1):
        t0 = time.time()
        track_id_counter += 1
        curr_track_id = f"bonafide_{track_id_counter}"

        try:
            y, sr = extractor.load_audio_file(track_path)
            if normalize_loudness:
                y, _, _ = normalize_audio_loudness(y, sr=sr, target_lufs=target_lufs)
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
                    "label_name": "Bona-fide (Human)",
                    "generator_tool": "bonafide"
                })
            print(f"    [{track_idx}/{len(bonafide_tracks)}] {track_path.stem[:35]:<35} -> {len(slices)} slices ({time.time()-t0:.2f}s)")
        except Exception as e:
            print(f"    [!] Error reading {track_path.name}: {e}")

    X = np.array(features_list, dtype=np.float32)
    y = np.array([r["label"] for r in records], dtype=int)
    groups = np.array([r["track_id"] for r in records])
    generator_tools = np.array([r["generator_tool"] for r in records])

    print(f"\n[*] Feature extraction complete in {time.time()-total_start:.1f}s!")
    print(f"    Total samples: {len(X)} | Feature dimension: {X.shape[1]}")
    print(f"    Spoof slices: {np.sum(y == LABEL_SPOOF)} | Bona-fide slices: {np.sum(y == LABEL_BONAFIDE)}")

    # Save CSV
    meta_headers = ["track_id", "track_name", "slice_idx", "start_sec", "label", "label_name", "generator_tool"]
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
        generator_tools=generator_tools,
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
        "tools": list(spoof_by_tool.keys()),
        "suno_status": "REMOVED & REPLACED BY ACE-STEP (Open-Source Thesis Scope)",
        "loudness_normalized": bool(normalize_loudness),
        "target_lufs": float(target_lufs) if normalize_loudness else None,
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
        "generator_tools": generator_tools,
        "feature_names": feature_names,
        "metadata": metadata
    }

def load_dataset(output_name: str = "dataset") -> Dict[str, Any]:
    """Loads cached dataset from disk by output_name."""
    npz_path = DATA_DIR / f"{output_name}.npz"
    meta_path = DATA_DIR / f"{output_name}_metadata.json"
    if not npz_path.exists():
        legacy_meta = DATA_DIR / "metadata.json"
        if legacy_meta.exists() and output_name == "dataset":
            meta_path = legacy_meta
        else:
            raise FileNotFoundError(f"Dataset '{output_name}' not found at {npz_path}!")

    data = np.load(npz_path, allow_pickle=True)
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    return {
        "X": data["X"],
        "y": data["y"],
        "groups": data["groups"],
        "generator_tools": data["generator_tools"] if "generator_tools" in data else None,
        "feature_names": list(data["feature_names"]),
        "metadata": metadata
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and build audio dataset")
    parser.add_argument("--max-bonafide", type=int, default=30, help="Max bona-fide tracks to sample")
    parser.add_argument("--max-spoof-per-tool", type=int, default=30, help="Max spoof tracks per tool")
    parser.add_argument("--max-slices", type=int, default=25, help="Max slices per track")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild dataset")
    parser.add_argument("--no-norm", action="store_true", help="Disable loudness normalization (extract raw)")
    parser.add_argument("--output-name", type=str, default="dataset", help="Dataset name prefix")
    args = parser.parse_args()

    build_dataset(
        max_bonafide_tracks=args.max_bonafide,
        max_spoof_per_tool=args.max_spoof_per_tool,
        max_slices_per_track=args.max_slices,
        force_rebuild=args.rebuild,
        normalize_loudness=not args.no_norm,
        output_name=args.output_name
    )
