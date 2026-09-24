// Audio Uploader, Local File Path Analyzer & Demo Loader
import { state, elements, showLoading, hideLoading, formatTimecode } from './state.js';
import { updateDiagnostics } from './components/forensics.js';
import { switchView } from './components/controls.js';
import { stopLiveRenderLoop } from './audioPlayer.js';

// Upload Audio File
export async function uploadAudio(file) {
  showLoading(`Analyzing "${file.name}"... Computing 3D Waterfall & AI Forensics...`);
  try {
    let res;
    try {
      const formData = new FormData();
      formData.append('file', file);
      res = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });
    } catch (netErr) {
      console.warn('Multipart upload failed at network level, trying direct stream:', netErr);
      res = await fetch('/api/upload', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/octet-stream',
          'x-filename': encodeURIComponent(file.name)
        },
        body: file
      });
    }

    if (!res || !res.ok) {
      const err = await res.json().catch(() => ({ detail: res ? res.statusText : 'Upload failed' }));
      throw new Error(err.detail || 'Upload failed');
    }

    const data = await res.json();
    handleAnalysisLoaded(data);
  } catch (err) {
    alert('Error analyzing audio: ' + err.message);
  } finally {
    hideLoading();
  }
}

// Analyze Local Path on Disk
export async function analyzeLocalPath() {
  const p = elements.localPathInput ? elements.localPathInput.value.trim() : '';
  if (!p) {
    alert('Please enter a valid local audio path.');
    return;
  }
  showLoading('Loading local file from disk...');
  try {
    const res = await fetch('/api/analyze-local-path', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file_path: p })
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Failed to analyze path');
    }
    const data = await res.json();
    handleAnalysisLoaded(data);
  } catch (err) {
    alert('Error: ' + err.message);
  } finally {
    hideLoading();
  }
}

// Load Bundled Sample Audio Demo
export async function loadDemoAudio() {
  if (elements.localPathInput) {
    elements.localPathInput.value = 'backend/uploads/sample_ai_test.wav';
  }
  analyzeLocalPath();
}

// Handler when analysis is loaded
export function handleAnalysisLoaded(data) {
  state.analysisData = data;
  state.currentFileId = data.metadata.file_id;
  state.currentDuration = data.metadata.duration_seconds;
  state.trimStartSec = 0;
  state.trimEndSec = state.currentDuration;

  // Toggle Viewport Screen
  if (elements.dropPrompt) elements.dropPrompt.style.display = 'none';
  if (elements.mediaScreen) elements.mediaScreen.style.display = 'flex';
  if (elements.trackStats) elements.trackStats.style.display = 'block';
  if (elements.forensicsSummaryBox) elements.forensicsSummaryBox.style.display = 'block';
  if (elements.exportTopBtn) elements.exportTopBtn.disabled = false;
  if (elements.primaryActionBtn) elements.primaryActionBtn.disabled = false;

  // Update Stats
  if (elements.statRate) elements.statRate.innerText = `${(data.metadata.sample_rate / 1000).toFixed(1)} kHz`;
  if (elements.statChannels) elements.statChannels.innerText = data.metadata.channels === 1 ? 'Mono' : 'Stereo';
  if (elements.statDuration) elements.statDuration.innerText = `${state.currentDuration.toFixed(2)}s`;

  // Timecodes
  if (elements.trimStartInput) elements.trimStartInput.value = formatTimecode(0);
  if (elements.trimEndInput) elements.trimEndInput.value = formatTimecode(state.currentDuration);
  if (elements.timecodeDisplay) elements.timecodeDisplay.innerText = `${formatTimecode(0)} / ${formatTimecode(state.currentDuration)}`;

  // Reset live buffers for clean visualization
  state.liveMagnitudesDb = null;
  state.livePeakHoldDb = null;
  if (state.isLiveRendering) stopLiveRenderLoop();

  // Set audio source
  if (elements.audioElement) {
    elements.audioElement.crossOrigin = 'anonymous';
    elements.audioElement.src = `/api/audio/${state.currentFileId}`;
    elements.audioElement.load();
  }

  // Populate Diagnostics
  updateDiagnostics(data.forensics);

  // Reset spectrogram offscreen cache for new audio file
  state.specOffscreenCanvas = null;
  state.specOffscreenKey = null;

  // Switch to spectrogram view to ensure toolbar and playhead are active
  switchView('spectrogram');
}

// Setup Upload & Drag/Drop Listeners
export function setupUploader() {
  if (elements.browseBtn) elements.browseBtn.addEventListener('click', () => elements.audioFileInput.click());
  if (elements.fabBtn) elements.fabBtn.addEventListener('click', () => elements.audioFileInput.click());
  if (elements.audioFileInput) {
    elements.audioFileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) uploadAudio(e.target.files[0]);
    });
  }

  if (elements.mediaDropZone) {
    elements.mediaDropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      elements.mediaDropZone.classList.add('dragover');
    });
    elements.mediaDropZone.addEventListener('dragleave', () => elements.mediaDropZone.classList.remove('dragover'));
    elements.mediaDropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      elements.mediaDropZone.classList.remove('dragover');
      if (e.dataTransfer.files.length > 0) {
        uploadAudio(e.dataTransfer.files[0]);
      }
    });
  }

  if (elements.analyzePathBtn) elements.analyzePathBtn.addEventListener('click', analyzeLocalPath);
  if (elements.localPathInput) {
    elements.localPathInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') analyzeLocalPath();
    });
  }
  if (elements.loadDemoBtn) elements.loadDemoBtn.addEventListener('click', loadDemoAudio);
}
