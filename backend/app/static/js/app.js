let currentFileId = null;
let currentDuration = 0;
let analysisData = null;
let currentView = 'spectrogram';
let trimStartSec = 0;
let trimEndSec = 0;

// DOM Elements
const mediaDropZone = document.getElementById('mediaDropZone');
const dropPrompt = document.getElementById('dropPrompt');
const mediaScreen = document.getElementById('mediaScreen');
const audioFileInput = document.getElementById('audioFileInput');
const browseBtn = document.getElementById('browseBtn');
const fabBtn = document.getElementById('fabBtn');
const loadDemoBtn = document.getElementById('loadDemoBtn');

const localPathInput = document.getElementById('localPathInput');
const analyzePathBtn = document.getElementById('analyzePathBtn');
const exportTopBtn = document.getElementById('exportTopBtn');
const primaryActionBtn = document.getElementById('primaryActionBtn');

// Player Controls
const audioElement = document.getElementById('audioElement');
const bigPlayBtn = document.getElementById('bigPlayBtn');
const bigPlayIcon = document.getElementById('bigPlayIcon');
const playOverlay = document.getElementById('playOverlay');
const miniPlayBtn = document.getElementById('miniPlayBtn');
const miniPlayIcon = document.getElementById('miniPlayIcon');
const progressBarBg = document.getElementById('progressBarBg');
const progressBarFill = document.getElementById('progressBarFill');
const trimRangeHighlight = document.getElementById('trimRangeHighlight');
const timecodeDisplay = document.getElementById('timecodeDisplay');
const playheadCursor = document.getElementById('playheadCursor');
const canvasCutoffBadge = document.getElementById('canvasCutoffBadge');
const cutoffValText = document.getElementById('cutoffValText');

// Position Buttons
const useTrimStartBtn = document.getElementById('useTrimStartBtn');
const useTrimEndBtn = document.getElementById('useTrimEndBtn');
const resetTrimBtn = document.getElementById('resetTrimBtn');
const trackStats = document.getElementById('trackStats');
const statRate = document.getElementById('statRate');
const statChannels = document.getElementById('statChannels');
const statDuration = document.getElementById('statDuration');

// Sidebar Options
const trimStartInput = document.getElementById('trimStartInput');
const trimEndInput = document.getElementById('trimEndInput');
const copyStartBtn = document.getElementById('copyStartBtn');
const copyEndBtn = document.getElementById('copyEndBtn');
const fftSizeSelect = document.getElementById('fftSizeSelect');
const colorPaletteSelect = document.getElementById('colorPaletteSelect');
const aiCutoffCheck = document.getElementById('aiCutoffCheck');

// Forensics Summary
const forensicsSummaryBox = document.getElementById('forensicsSummaryBox');
const cutoffBadge = document.getElementById('cutoffBadge');
const valCutoff = document.getElementById('valCutoff');
const valRolloff = document.getElementById('valRolloff');
const valCentroid = document.getElementById('valCentroid');
const valAirPower = document.getElementById('valAirPower');
const forensicNotes = document.getElementById('forensicNotes');

// Canvases
const mainSpecCanvas = document.getElementById('mainSpecCanvas');
const mainEqCanvas = document.getElementById('mainEqCanvas');
const mainTonalCanvas = document.getElementById('mainTonalCanvas');
const stackedVisualizersCard = document.getElementById('stackedVisualizersCard');
const stackedEqCanvas = document.getElementById('stackedEqCanvas');
const stackedTonalCanvas = document.getElementById('stackedTonalCanvas');

const loadingOverlay = document.getElementById('loadingOverlay');

function init() {
  // File Upload Handlers
  browseBtn.addEventListener('click', () => audioFileInput.click());
  fabBtn.addEventListener('click', () => audioFileInput.click());
  audioFileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) uploadAudio(e.target.files[0]);
  });

  // Drag & Drop
  mediaDropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    mediaDropZone.classList.add('dragover');
  });
  mediaDropZone.addEventListener('dragleave', () => mediaDropZone.classList.remove('dragover'));
  mediaDropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    mediaDropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      uploadAudio(e.dataTransfer.files[0]);
    }
  });

  // Local Path Analyze
  analyzePathBtn.addEventListener('click', analyzeLocalPath);
  localPathInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') analyzeLocalPath();
  });

  // Demo Audio Button
  loadDemoBtn.addEventListener('click', loadDemoAudio);

  // Playback Handlers
  bigPlayBtn.addEventListener('click', togglePlayback);
  miniPlayBtn.addEventListener('click', togglePlayback);
  audioElement.addEventListener('timeupdate', onTimeUpdate);
  audioElement.addEventListener('ended', onEnded);
  audioElement.addEventListener('play', () => updatePlayIcons(true));
  audioElement.addEventListener('pause', () => updatePlayIcons(false));

  progressBarBg.addEventListener('click', seekAudio);
  mainSpecCanvas.addEventListener('click', handleCanvasClick);

  // View Switcher Tabs
  document.querySelectorAll('.screen-tab').forEach(tab => {
    tab.addEventListener('click', (e) => switchView(e.target.dataset.view));
  });

  // Position Buttons & Sidebar Buttons
  useTrimStartBtn.addEventListener('click', setTrimStartFromCurrent);
  useTrimEndBtn.addEventListener('click', setTrimEndFromCurrent);
  copyStartBtn.addEventListener('click', setTrimStartFromCurrent);
  copyEndBtn.addEventListener('click', setTrimEndFromCurrent);
  resetTrimBtn.addEventListener('click', resetTrim);

  // Colormap & AI Cutoff Select
  colorPaletteSelect.addEventListener('change', () => redrawCurrentView());
  aiCutoffCheck.addEventListener('change', () => redrawCurrentView());

  // Export Buttons
  exportTopBtn.addEventListener('click', exportPlot);
  primaryActionBtn.addEventListener('click', exportPlot);

  window.addEventListener('resize', () => redrawCurrentView());
}

function showLoading(text) {
  document.getElementById('loadingText').innerText = text || 'Processing audio...';
  loadingOverlay.style.display = 'flex';
}

function hideLoading() {
  loadingOverlay.style.display = 'none';
}

// Upload Audio File
async function uploadAudio(file) {
  showLoading(`Analyzing "${file.name}"... Computing STFT & AI Forensics...`);
  try {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch('/api/upload', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      const err = await res.json();
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

// Analyze Local Path
async function analyzeLocalPath() {
  const p = localPathInput.value.trim();
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

// Load Bundled Sample
async function loadDemoAudio() {
  localPathInput.value = 'backend/uploads/sample_ai_test.wav';
  analyzeLocalPath();
}

// Handler when analysis is loaded
function handleAnalysisLoaded(data) {
  analysisData = data;
  currentFileId = data.metadata.file_id;
  currentDuration = data.metadata.duration_seconds;
  trimStartSec = 0;
  trimEndSec = currentDuration;

  // Toggle Viewport Screen
  dropPrompt.style.display = 'none';
  mediaScreen.style.display = 'flex';
  trackStats.style.display = 'block';
  forensicsSummaryBox.style.display = 'block';
  playheadCursor.style.display = 'block';
  exportTopBtn.disabled = false;
  primaryActionBtn.disabled = false;

  // Update Stats
  statRate.innerText = `${(data.metadata.sample_rate / 1000).toFixed(1)} kHz`;
  statChannels.innerText = data.metadata.channels === 1 ? 'Mono' : 'Stereo';
  statDuration.innerText = `${currentDuration.toFixed(2)}s`;

  // Timecodes
  trimStartInput.value = formatTimecode(0);
  trimEndInput.value = formatTimecode(currentDuration);
  timecodeDisplay.innerText = `${formatTimecode(0)} / ${formatTimecode(currentDuration)}`;

  // Set audio source
  audioElement.src = `/api/audio/${currentFileId}`;
  audioElement.load();

  // Populate Diagnostics
  updateDiagnostics(data.forensics);

  // Redraw
  redrawCurrentView();
}

function updateDiagnostics(forensics) {
  if (forensics.cutoff_detected) {
    cutoffBadge.innerText = forensics.cutoff_severity;
    cutoffBadge.className = forensics.estimated_cutoff_hz <= 16500 ? 'badge badge-danger' : 'badge badge-warning';
    valCutoff.innerText = `${forensics.estimated_cutoff_hz.toFixed(0)} Hz`;
    canvasCutoffBadge.style.display = 'flex';
    cutoffValText.innerText = `${forensics.estimated_cutoff_hz.toFixed(0)} Hz`;
  } else {
    cutoffBadge.innerText = 'Natural Spectrum';
    cutoffBadge.className = 'badge badge-normal';
    valCutoff.innerText = 'None (>20kHz)';
    canvasCutoffBadge.style.display = 'none';
  }

  valRolloff.innerText = `${forensics.spectral_rolloff_95.toFixed(0)} Hz`;
  valCentroid.innerText = `${forensics.spectral_centroid_hz.toFixed(0)} Hz`;
  valAirPower.innerText = `${(forensics.high_freq_energy_ratio * 100).toFixed(2)}%`;
  forensicNotes.innerHTML = forensics.forensic_notes.map(n => `<p>• ${n}</p>`).join('');
}

// View Switcher (Spectrogram, Equalizer, Tonal, Stacked)
function switchView(view) {
  currentView = view;
  document.querySelectorAll('.screen-tab').forEach(t => {
    t.classList.toggle('active', t.dataset.view === view);
  });

  mainSpecCanvas.style.display = (view === 'spectrogram' || view === 'stacked') ? 'block' : 'none';
  mainEqCanvas.style.display = view === 'equalizer' ? 'block' : 'none';
  mainTonalCanvas.style.display = view === 'tonal' ? 'block' : 'none';
  stackedVisualizersCard.style.display = view === 'stacked' ? 'flex' : 'none';

  redrawCurrentView();
}

function redrawCurrentView() {
  if (!analysisData) return;
  const palette = colorPaletteSelect.value;
  const showCutoff = aiCutoffCheck.checked;

  if (currentView === 'spectrogram' || currentView === 'stacked') {
    renderSpectrogramCanvas(mainSpecCanvas, analysisData.spectrogram, analysisData.forensics, palette, showCutoff);
  }
  if (currentView === 'equalizer') {
    renderEqualizerCanvas(mainEqCanvas, analysisData.fft_spectrum);
  }
  if (currentView === 'tonal') {
    renderTonalCanvas(mainTonalCanvas, analysisData.tonal_balance);
  }
  if (currentView === 'stacked') {
    renderEqualizerCanvas(stackedEqCanvas, analysisData.fft_spectrum);
    renderTonalCanvas(stackedTonalCanvas, analysisData.tonal_balance);
  }
}

// 1. Render Spectrogram Canvas (Wave Candy style)
function renderSpectrogramCanvas(canvas, specData, forensics, palette, showCutoff) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const w = rect.width;
  const h = rect.height;

  const matrix = specData.matrix_db;
  const numFreqs = matrix.length;
  const numTimes = matrix[0].length;

  const off = document.createElement('canvas');
  off.width = numTimes;
  off.height = numFreqs;
  const offCtx = off.getContext('2d');
  const imgData = offCtx.createImageData(numTimes, numFreqs);
  const data = imgData.data;

  for (let f = 0; f < numFreqs; f++) {
    const row = matrix[numFreqs - 1 - f];
    for (let t = 0; t < numTimes; t++) {
      const db = row[t];
      const norm = Math.max(0, Math.min(1, (db + 80) / 80));
      const [r, g, b] = getColorForPalette(norm, palette);
      const p = (f * numTimes + t) * 4;
      data[p] = r;
      data[p + 1] = g;
      data[p + 2] = b;
      data[p + 3] = 255;
    }
  }

  offCtx.putImageData(imgData, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, 0, 0, w, h);

  // AI Cutoff line
  if (showCutoff && forensics && forensics.cutoff_detected && forensics.estimated_cutoff_hz) {
    const maxHz = specData.frequencies[specData.frequencies.length - 1];
    const cutoffY = h - (forensics.estimated_cutoff_hz / maxHz) * h;

    ctx.strokeStyle = '#00ffcc';
    ctx.lineWidth = 2.0;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(0, cutoffY);
    ctx.lineTo(w, cutoffY);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Frequency tick labels on left side (Wave Candy style)
  const maxHz = specData.frequencies[specData.frequencies.length - 1];
  [20000, 15000, 10000, 5000, 1000].forEach(hz => {
    if (hz < maxHz) {
      const y = h - (hz / maxHz) * h;
      ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
      ctx.font = '9px JetBrains Mono';
      ctx.fillText(`${hz >= 1000 ? hz / 1000 + 'kHz' : hz + 'Hz'}`, 6, y - 3);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(42, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }
  });
}

// 2. Render Equalizer Canvas
function renderEqualizerCanvas(canvas, eqData) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const w = rect.width;
  const h = rect.height;

  ctx.fillStyle = '#080a0f';
  ctx.fillRect(0, 0, w, h);

  // Freq grid
  [100, 1000, 10000].forEach(f => {
    const x = freqToX(f, w);
    ctx.strokeStyle = '#1a2233';
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
    ctx.fillStyle = '#64748b';
    ctx.font = '10px JetBrains Mono';
    ctx.fillText(`${f >= 1000 ? f/1000 + 'k' : f}Hz`, x + 4, h - 6);
  });

  const freqs = eqData.frequencies;
  const peaks = eqData.peaks_db;
  const avgs = eqData.magnitudes_db;

  // Peak hold line
  ctx.strokeStyle = '#475569';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  for (let i = 0; i < freqs.length; i++) {
    const x = freqToX(freqs[i], w);
    const y = dbToY(peaks[i], h);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Cyan gradient fill
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, 'rgba(0, 229, 255, 0.35)');
  grad.addColorStop(1, 'rgba(0, 229, 255, 0.0)');

  ctx.beginPath();
  for (let i = 0; i < freqs.length; i++) {
    const x = freqToX(freqs[i], w);
    const y = dbToY(avgs[i], h);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.lineTo(w, h);
  ctx.lineTo(0, h);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Stroke on top
  ctx.strokeStyle = '#00e5ff';
  ctx.lineWidth = 2.0;
  ctx.beginPath();
  for (let i = 0; i < freqs.length; i++) {
    const x = freqToX(freqs[i], w);
    const y = dbToY(avgs[i], h);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

// 3. Render Tonal Balance Canvas
function renderTonalCanvas(canvas, tbData) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const w = rect.width;
  const h = rect.height;

  ctx.fillStyle = '#080a0f';
  ctx.fillRect(0, 0, w, h);

  const curveF = tbData.curve_freqs;
  const curveL = tbData.curve_levels;

  tbData.bands.forEach(b => {
    const fMin = b.freq_range[0];
    const fMax = b.freq_range[1];
    ctx.beginPath();
    let started = false, firstX = 0, lastX = 0;

    for (let i = 0; i < curveF.length; i++) {
      if (curveF[i] >= fMin && curveF[i] <= fMax) {
        const x = freqToX(curveF[i], w);
        const y = dbToY(curveL[i], h);
        if (!started) { ctx.moveTo(x, y); firstX = x; started = true; }
        else { ctx.lineTo(x, y); }
        lastX = x;
      }
    }
    if (started) {
      ctx.lineTo(lastX, h);
      ctx.lineTo(firstX, h);
      ctx.closePath();
      ctx.fillStyle = b.color_hex + '66';
      ctx.fill();
    }
  });

  // White smooth curve
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  for (let i = 0; i < curveF.length; i++) {
    const x = freqToX(curveF[i], w);
    const y = dbToY(curveL[i], h);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

// Helpers
function freqToX(freq, w) {
  const minF = Math.log10(20);
  const maxF = Math.log10(20000);
  const clamped = Math.max(20, Math.min(20000, freq));
  return ((Math.log10(clamped) - minF) / (maxF - minF)) * w;
}

function dbToY(db, h) {
  const minDb = -80.0;
  const maxDb = 0.0;
  const clamped = Math.max(minDb, Math.min(maxDb, db));
  return h - ((clamped - minDb) / (maxDb - minDb)) * h;
}

function getColorForPalette(t, palette) {
  if (palette === 'cyan') {
    return [Math.round(0 * t), Math.round(229 * t), Math.round(255 * t)];
  }
  if (palette === 'plasma') {
    // Plasma: dark blue -> violet -> pink -> yellow
    return [Math.round(255 * Math.pow(t, 0.7)), Math.round(180 * Math.pow(t, 1.5)), Math.round(255 * (1 - t * 0.5))];
  }
  if (palette === 'magma') {
    return [Math.round(255 * t), Math.round(120 * Math.pow(t, 2)), Math.round(200 * Math.pow(1-t, 2))];
  }
  // Default Inferno / Wave Candy
  const colors = [
    [0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
    [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]
  ];
  const idx = t * (colors.length - 1);
  const i = Math.floor(idx);
  const f = idx - i;
  if (i >= colors.length - 1) return colors[colors.length - 1];
  const p1 = colors[i], p2 = colors[i + 1];
  return [
    Math.round(p1[0] + (p2[0] - p1[0]) * f),
    Math.round(p1[1] + (p2[1] - p1[1]) * f),
    Math.round(p1[2] + (p2[2] - p1[2]) * f)
  ];
}

// Playback Logic
function togglePlayback() {
  if (audioElement.paused) {
    audioElement.play();
  } else {
    audioElement.pause();
  }
}

function updatePlayIcons(isPlaying) {
  if (isPlaying) {
    bigPlayIcon.innerHTML = '<rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect>';
    miniPlayIcon.innerHTML = '<rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect>';
    playOverlay.classList.add('hidden');
  } else {
    bigPlayIcon.innerHTML = '<polygon points="6 3 20 12 6 21 6 3"></polygon>';
    miniPlayIcon.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"></polygon>';
    playOverlay.classList.remove('hidden');
  }
}

function onTimeUpdate() {
  if (!currentDuration) return;
  const cur = audioElement.currentTime;
  const pct = (cur / currentDuration) * 100;
  progressBarFill.style.width = `${pct}%`;
  playheadCursor.style.left = `${pct}%`;
  timecodeDisplay.innerText = `${formatTimecode(cur)} / ${formatTimecode(currentDuration)}`;
}

function onEnded() {
  updatePlayIcons(false);
  progressBarFill.style.width = '0%';
  playheadCursor.style.left = '0%';
}

function seekAudio(e) {
  if (!currentDuration) return;
  const rect = progressBarBg.getBoundingClientRect();
  const pct = (e.clientX - rect.left) / rect.width;
  audioElement.currentTime = pct * currentDuration;
}

function handleCanvasClick(e) {
  if (!currentDuration) return;
  const rect = mainSpecCanvas.getBoundingClientRect();
  const pct = (e.clientX - rect.left) / rect.width;
  audioElement.currentTime = pct * currentDuration;
}

// Timecode Formatting (hh:mm:ss.ms)
function formatTimecode(totalSec) {
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = Math.floor(totalSec % 60);
  const ms = Math.floor((totalSec - Math.floor(totalSec)) * 100);

  const pad = (n) => String(n).padStart(2, '0');
  return `${pad(h)} : ${pad(m)} : ${pad(s)} . ${pad(ms)}`;
}

// Trim Positions
function setTrimStartFromCurrent() {
  if (!audioElement) return;
  trimStartSec = audioElement.currentTime;
  trimStartInput.value = formatTimecode(trimStartSec);
  updateTrimHighlight();
}

function setTrimEndFromCurrent() {
  if (!audioElement) return;
  trimEndSec = audioElement.currentTime;
  trimEndInput.value = formatTimecode(trimEndSec);
  updateTrimHighlight();
}

function resetTrim() {
  trimStartSec = 0;
  trimEndSec = currentDuration;
  trimStartInput.value = formatTimecode(0);
  trimEndInput.value = formatTimecode(currentDuration);
  updateTrimHighlight();
}

function updateTrimHighlight() {
  if (!currentDuration) return;
  const sPct = (trimStartSec / currentDuration) * 100;
  const ePct = (trimEndSec / currentDuration) * 100;
  trimRangeHighlight.style.display = 'block';
  trimRangeHighlight.style.left = `${sPct}%`;
  trimRangeHighlight.style.width = `${Math.max(0, ePct - sPct)}%`;
}

// Export Publication Plot
async function exportPlot() {
  if (!currentFileId) return;
  showLoading('Rendering 200 DPI Skripsi publication figure...');
  try {
    const res = await fetch(`/api/export-plot/${currentFileId}`, { method: 'POST' });
    if (!res.ok) throw new Error('Export failed');
    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `skripsi_visualizer_${analysisData.metadata.original_filename}.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  } catch (err) {
    alert('Error exporting plot: ' + err.message);
  } finally {
    hideLoading();
  }
}

window.addEventListener('DOMContentLoaded', init);
