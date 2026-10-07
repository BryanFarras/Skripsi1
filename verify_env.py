#!/usr/bin/env python3
"""
AI Music Detector — Environment & Dependency Verification Script
Validates all required audio DSP, machine learning, XAI, and REST modules.
"""

import sys
import shutil

MODULES_TO_CHECK = [
    ("FastAPI Backend", "fastapi"),
    ("Uvicorn Server", "uvicorn"),
    ("Pydantic", "pydantic"),
    ("NumPy", "numpy"),
    ("SciPy", "scipy"),
    ("SoundFile", "soundfile"),
    ("Librosa (Audio DSP)", "librosa"),
    ("Matplotlib", "matplotlib"),
    ("Scikit-Learn (ML)", "sklearn"),
    ("SHAP (TreeSHAP XAI)", "shap"),
    ("PyTorch", "torch"),
    ("TorchAudio", "torchaudio"),
    ("Meta Demucs (Stems)", "demucs"),
    ("Requests", "requests"),
]

def main():
    print("\n" + "=" * 65)
    print("   AI MUSIC DETECTOR — ENVIRONMENT INTEGRITY VERIFICATION")
    print("=" * 65 + "\n")

    all_passed = True

    # 1. Python Version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 10):
        print(f" [PASS] Python Version:        {py_ver} (Supported)")
    else:
        print(f" [FAIL] Python Version:        {py_ver} (Requires 3.10+)")
        all_passed = False

    # 2. FFmpeg Executable
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        print(f" [PASS] FFmpeg Executable:     {ffmpeg_bin}")
    else:
        print(" [WARN] FFmpeg Executable:     NOT FOUND in system PATH")
        print("        (Note: Required for MP3/M4A decoding and stem separation)")

    # 3. Python Modules
    print("\n [*] Verifying Core Modules:")
    for label, mod_name in MODULES_TO_CHECK:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "Installed")
            print(f"   - {label:<24} [OK] (v{ver})")
        except Exception as e:
            print(f"   - {label:<24} [FAIL] ({e})")
            all_passed = False

    # 4. Hardware Acceleration (CUDA)
    print("\n [*] Hardware Acceleration Profile:")
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            dev_name = torch.cuda.get_device_name(0)
            dev_count = torch.cuda.device_count()
            print(f"   - CUDA Acceleration:      [ENABLED] ({dev_count} GPU found)")
            print(f"   - Active GPU Device:      {dev_name}")
        else:
            print("   - CUDA Acceleration:      [DISABLED] (Running on CPU)")
    except Exception as e:
        print(f"   - CUDA Check:             [SKIPPED] ({e})")

    print("\n" + "=" * 65)
    if all_passed:
        print(" [RESULT] All required dependencies are installed and operational!")
        print("=" * 65 + "\n")
        return 0
    else:
        print(" [RESULT] Some dependencies are missing. Run install_dependencies.ps1.")
        print("=" * 65 + "\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
