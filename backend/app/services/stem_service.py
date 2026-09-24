import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

# Ensure ffmpeg in .venv or system is discovered
def ensure_ffmpeg_in_path() -> Optional[str]:
    """Finds ffmpeg and injects its directory into os.environ['PATH'] so Demucs/Torchaudio can use it."""
    # Check if already in PATH
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        return ffmpeg_bin

    # Search common project locations (.venv/Scripts/ffmpeg.exe)
    curr = Path(__file__).resolve()
    for parent in [curr.parent.parent.parent.parent.parent, curr.parent.parent.parent]:
        venv_ffmpeg = parent / ".venv" / "Scripts" / "ffmpeg.exe"
        if venv_ffmpeg.exists():
            bin_dir = str(venv_ffmpeg.parent)
            if bin_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            return str(venv_ffmpeg)

    return None

class StemService:
    """
    Stem Separation Service for the AI Music Detector.
    Splits any full audio track into 5 components:
      1. mixture (Full Song)
      2. vocals  (Singing voice / speech)
      3. drums   (Percussion, kick, snare, hi-hats)
      4. bass    (Bassline, sub-frequencies)
      5. other   (Harmonic accompaniment: guitars, keys, synths, reverb)
    """

    STEM_NAMES = ["mixture", "vocals", "drums", "bass", "other"]

    def __init__(self, model_name: str = "htdemucs", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._demucs_model = None
        self.ffmpeg_path = ensure_ffmpeg_in_path()

    def _get_device(self) -> str:
        if self.device:
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load_demucs_model(self):
        if self._demucs_model is not None:
            return self._demucs_model

        try:
            import torch
            from demucs.pretrained import get_model
            dev = self._get_device()
            print(f"[StemService] Loading Demucs ({self.model_name}) on device: {dev}...")
            model = get_model(name=self.model_name)
            model.to(dev)
            model.eval()
            self._demucs_model = model
            return self._demucs_model
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Demucs model '{self.model_name}': {e}")

    def is_stem_mp4(self, file_path: Path) -> bool:
        """Check if file is a Native Instruments STEMS format (.stem.mp4)."""
        suffix = file_path.suffix.lower()
        if suffix in [".mp4", ".stem.mp4", ".m4a"]:
            return ".stem." in file_path.name.lower() or file_path.name.lower().endswith(".stem.mp4")
        return False

    def separate_musdb_stem_mp4(self, file_path: Path, output_dir: Path) -> Dict[str, Path]:
        """
        Fast extraction for Native Instruments STEMS format using ffmpeg stream mapping:
          Stream 0: mixture
          Stream 1: drums
          Stream 2: bass
          Stream 3: other
          Stream 4: vocals
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        stem_map = {
            "mixture": 0,
            "drums": 1,
            "bass": 2,
            "other": 3,
            "vocals": 4
        }
        stem_files = {}
        base_name = file_path.stem.replace(".stem", "")

        ffmpeg_cmd = self.ffmpeg_path or "ffmpeg"

        for stem_name, stream_idx in stem_map.items():
            out_file = output_dir / f"{base_name}_{stem_name}.wav"
            cmd = [
                ffmpeg_cmd, "-y",
                "-i", str(file_path),
                "-map", f"0:a:{stream_idx}",
                "-c:a", "pcm_s16le",
                "-ar", "44100",
                str(out_file)
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode != 0:
                raise RuntimeError(f"FFmpeg extraction of stem '{stem_name}' failed: {res.stderr.decode('utf-8', errors='ignore')}")
            stem_files[stem_name] = out_file

        return stem_files

    def separate_track(
        self,
        audio_path: Path,
        output_dir: Optional[Path] = None,
        save_wavs: bool = True
    ) -> Dict[str, Any]:
        """
        Separates any audio track (WAV, MP3, FLAC, M4A, STEM.MP4) into 5 stems.
        
        Returns:
            Dict containing:
              - 'stems': dict of {stem_name: numpy_waveform_float32}
              - 'file_paths': dict of {stem_name: Path} (if save_wavs=True)
              - 'sample_rate': int (44100)
              - 'duration': float (seconds)
        """
        audio_path = Path(audio_path).resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        stem_folder_name = f"{audio_path.stem}_stems"
        if output_dir is None:
            output_dir = audio_path.parent / stem_folder_name
        else:
            output_dir = Path(output_dir).resolve()
            if output_dir.name != stem_folder_name:
                output_dir = output_dir / stem_folder_name

        output_dir = Path(output_dir).resolve()
        if save_wavs:
            output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Quick check for Native Instruments .stem.mp4 format
        if self.is_stem_mp4(audio_path):
            print(f"[StemService] Detected Native Instruments STEMS container. Using multi-stream demuxing...")
            file_paths = self.separate_musdb_stem_mp4(audio_path, output_dir)
            stems_numpy = {}
            sr = 44100
            import soundfile as sf
            for name, path in file_paths.items():
                data, sr = sf.read(str(path), dtype="float32")
                if data.ndim > 1:
                    data = np.mean(data, axis=1) # Mono
                stems_numpy[name] = data

            dur = float(len(stems_numpy["mixture"]) / sr)
            return {
                "stems": stems_numpy,
                "file_paths": file_paths,
                "sample_rate": sr,
                "duration": dur,
                "output_dir": output_dir
            }

        # 2. Check if torch & demucs are available in current interpreter
        try:
            import torch
            import demucs
            has_torch = True
        except ImportError:
            has_torch = False

        if not has_torch:
            curr = Path(__file__).resolve()
            venv_python = None
            for p in curr.parents:
                vp = p / ".venv" / "Scripts" / "python.exe"
                if vp.exists():
                    venv_python = vp
                    break

            if venv_python is None:
                raise RuntimeError("Torch / Demucs are not installed in the current environment and .venv was not found.")

            # split_stems.py is at MY-AIDETECTOR root (curr.parents[2])
            split_script = curr.parents[2] / "split_stems.py"
            if not split_script.exists():
                split_script = curr.parent.parent.parent.parent / "split_stems.py"

            print(f"[StemService] Running Demucs via subprocess with {venv_python}...")
            cmd = [str(venv_python), str(split_script), str(audio_path), "--output-dir", str(output_dir)]
            sub_res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
            if sub_res.returncode != 0:
                raise RuntimeError(f"Subprocess stem splitting failed: {sub_res.stderr or sub_res.stdout}")

            # Collect stems created by split_stems.py
            import wave
            saved_paths = {}
            stems_numpy = {}
            sr = 44100
            for name in self.STEM_NAMES:
                matches = list(output_dir.rglob(f"*_{name}.wav")) + list(output_dir.rglob(f"{name}.wav"))
                if matches and matches[0].exists():
                    f = matches[0]
                    saved_paths[name] = f
                    try:
                        with wave.open(str(f), "rb") as wf:
                            sr = wf.getframerate()
                            n_channels = wf.getnchannels()
                            n_frames = wf.getnframes()
                            raw_bytes = wf.readframes(n_frames)
                            data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                            if n_channels > 1:
                                data = data.reshape(-1, n_channels)
                                mono = np.mean(data, axis=1)
                            else:
                                mono = data
                            stems_numpy[name] = mono
                    except Exception as err:
                        print(f"[StemService] Note: wave read error on {f}: {err}")

            dur = float(len(stems_numpy.get("mixture", [])) / sr) if "mixture" in stems_numpy and len(stems_numpy["mixture"]) > 0 else 0.0
            return {
                "stems": stems_numpy,
                "file_paths": saved_paths,
                "sample_rate": sr,
                "duration": dur,
                "output_dir": output_dir
            }

        # 3. Direct in-process Demucs execution (when running inside torch environment)
        import torchaudio
        from demucs.apply import apply_model
        import soundfile as sf

        ensure_ffmpeg_in_path()
        model = self._load_demucs_model()
        target_sr = model.samplerate # 44100 Hz

        print(f"[StemService] Loading audio: {audio_path.name}")
        try:
            data, sr = sf.read(str(audio_path), dtype="float32")
            if data.ndim == 1:
                wav = torch.from_numpy(data).unsqueeze(0)
            else:
                wav = torch.from_numpy(data.T)
        except Exception:
            try:
                wav, sr = torchaudio.load(str(audio_path))
            except Exception:
                from .audio_loader import load_audio
                raw_data, sr, _ = load_audio(audio_path, target_sr=target_sr)
                wav = torch.from_numpy(raw_data).unsqueeze(0)

        # Ensure stereo for Demucs
        if wav.ndim == 1:
            wav = wav.unsqueeze(0).repeat(2, 1)
        elif wav.shape[0] == 1:
            wav = wav.repeat(2, 1)

        # Resample if needed
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            wav = resampler(wav)

        dur_sec = float(wav.shape[-1] / target_sr)
        print(f"[StemService] Running Demucs separation on {dur_sec:.1f}s audio...")

        # Normalize reference
        ref = wav.mean(0)
        norm_wav = (wav - ref.mean()) / (ref.std() + 1e-8)

        dev = self._get_device()
        with torch.no_grad():
            # sources shape: [4, Channels, Samples] -> drums, bass, other, vocals
            sources = apply_model(model, norm_wav.unsqueeze(0).to(dev), device=dev, progress=True)[0]

        # Build stems dictionary: mixture + 4 demucs sources
        stems_dict = {
            "mixture": wav.cpu(),
        }
        for idx, stem_name in enumerate(model.sources):
            stems_dict[stem_name] = sources[idx].cpu()

        # Convert and save stems preserving full stereo channels
        stems_numpy = {}
        saved_paths = {}
        base_name = audio_path.stem

        for name in self.STEM_NAMES:
            stem_tensor = stems_dict[name]  # Shape: [Channels, Samples]
            stereo_audio = stem_tensor.numpy().astype(np.float32)

            # Normalize peak to 0.95 to prevent clipping while preserving channel balance
            peak = np.max(np.abs(stereo_audio))
            if peak > 0.95:
                stereo_audio = stereo_audio / peak * 0.95

            # Mono projection for internal 1D acoustic feature routines
            if stereo_audio.ndim > 1 and stereo_audio.shape[0] > 1:
                mono = np.mean(stereo_audio, axis=0)
            else:
                mono = stereo_audio.squeeze(0) if stereo_audio.ndim > 1 else stereo_audio

            stems_numpy[name] = mono

            if save_wavs:
                stem_file = output_dir / f"{base_name}_{name}.wav"
                # soundfile expects [Samples, Channels] for multi-channel stereo audio
                if stereo_audio.ndim > 1 and stereo_audio.shape[0] > 1:
                    save_data = stereo_audio.T  # Transpose [2, N] -> [N, 2]
                else:
                    save_data = mono

                sf.write(str(stem_file), save_data, target_sr, subtype="PCM_16")
                saved_paths[name] = stem_file

        print(f"[StemService] Successfully split 5 stereo stems into: {output_dir}")
        return {
            "stems": stems_numpy,
            "file_paths": saved_paths if save_wavs else {},
            "sample_rate": target_sr,
            "duration": dur_sec,
            "output_dir": output_dir
        }
