import sys
import time
import socket
import webbrowser
import threading
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

def find_available_port(host: str, starting_port: int) -> int:
    port = starting_port
    while port < starting_port + 50:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                port += 1
    return starting_port

def open_browser(port: int):
    time.sleep(1.2)
    url = f"http://localhost:{port}"
    print(f"\n[AI Detector] Opening web dashboard in your browser: {url}")
    webbrowser.open(url)

def main():
    import uvicorn
    from backend.app.config import HOST, PORT

    port = find_available_port(HOST, PORT)

    print("=" * 65)
    print("  AI MUSIC DETECTOR - FORENSIC VISUALIZER & BACKEND SERVER")
    print("=" * 65)
    print(f"[*] Server Address:  http://{HOST}:{port}")
    print(f"[*] Web Dashboard:   http://localhost:{port}")
    print(f"[*] API Docs:        http://localhost:{port}/docs")
    print("[*] Press Ctrl+C in this console to stop the server.\n")

    # Launch browser in a background thread
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()

    # Run Uvicorn server with auto-reload enabled
    uvicorn.run("backend.app.main:app", host=HOST, port=port, reload=True, reload_dirs=[str(PROJECT_ROOT / "backend")])

if __name__ == "__main__":
    main()
