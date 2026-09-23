import sys
import time
import webbrowser
import threading
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

def open_browser(port: int):
    time.sleep(1.2)
    url = f"http://localhost:{port}"
    print(f"\n[AI Detector] Opening visualizer in your browser: {url}")
    webbrowser.open(url)

def main():
    import uvicorn
    from backend.app.config import HOST, PORT

    print("=" * 65)
    print("  AI MUSIC DETECTOR - FORENSIC VISUALIZER & BACKEND SERVER")
    print("=" * 65)
    print(f"[*] Starting local server on http://{HOST}:{PORT}")
    print(f"[*] Serving web dashboard at: http://localhost:{PORT}")
    print(f"[*] Interactive API docs at:  http://localhost:{PORT}/docs")
    print("[*] Press Ctrl+C to stop the server.\n")

    # Launch browser in a background thread
    threading.Thread(target=open_browser, args=(PORT,), daemon=True).start()

    # Run Uvicorn server
    uvicorn.run("backend.app.main:app", host=HOST, port=PORT, reload=False)

if __name__ == "__main__":
    main()
