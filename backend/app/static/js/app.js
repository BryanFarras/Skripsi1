let currentFileId = null;
let currentDuration = 0;
let analysisData = null;

// DOM Elements
const dropZone = document.getElementById('dropZone');
const audioFileInput = document.getElementById('audioFileInput');
const localPathInput = document.getElementById('localPathInput');
const analyzePathBtn = document.getElementById('analyzePathBtn');
const exportPlotBtn = document.getElementById('exportPlotBtn');
const loadingOverlay = document.getElementById('loadingOverlay');

const playerCard = document.getElementById('playerCard');
const trackTitle = document.getElementById('trackTitle');
const trackDetails = document.getElementById('trackDetails');
const audioElement = document.getElementById('audioElement');
const playBtn = document.getElementById('playBtn');
const playIcon = document.getElementById('playIcon');
const timeDisplay = document.getElementById('timeDisplay');
const playheadCursor = document.getElementById('playheadCursor');

const forensicsCard = document.getElementById('forensicsCard');
const cutoffBadge = document.getElementById('cutoffBadge');
const valCutoff = document.getElementById('valCutoff');
const valRolloff = document.getElementById('valRolloff');
const valCentroid = document.getElementById('valCentroid');
const valHighFreq = document.getElementById('valHighFreq');
const forensicNotes = document.getElementById('forensicNotes');
const cutoffIndicator = document.getElementById('cutoffIndicator');
const bandPills = document.getElementById('bandPills');

// Canvases
const eqCanvas = document.getElementById('eqCanvas');
const tonalCanvas = document.getElementById('tonalCanvas');
const specCanvas = document.getElementById('specCanvas');

// Initialize event listeners
function init() {
  dropZone.addEventListener('click', () => audioFileInput.click());
  audioFileInput.addEventListener('change', handleFileSelect);

  // Drag and Drop
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });

  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      uploadFile(e.dataTransfer.files[0]);
    }
  });

  analyzePathBtn.addEventListener('click', analyzeLocalPath);
  localPathInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') analyzeLocalPath();
  });

  exportPlotBtn.addEventListener('click', exportPlot);

  // Audio Playback
  playBtn.addEventListener('click', togglePlay);
  audioElement.addEventListener('timeupdate', updatePlaybackCursor);
  audioElement.addEventListener('ended', onPlaybackEnded);

  // Canvas click to seek
  specCanvas.addEventListener('click', handleSpecClick);

  // Resize listener
  window.addEventListener('resize', redrawAll);
}

function showLoading(text) {
  document.getElementById('loadingText').innerText = text || 'Processing audio...';
  loadingOverlay.style.display = 'flex';
}

function hideLoading() {
  loadingOverlay.style.display = 'none';
}

// Upload via Drag / Browse
async function handleFileSelect(e) {
  const file = e.target.files[0];
  if (file) uploadFile(file);
}

async function uploadFile(file) {
  showLoading(`Analyzing "${file.name}"... Computing FFT & Spectrogram...`);
  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/api/upload', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || 'Analysis failed.');
    }

    const data = await response.json();
    handleAnalysisSuccess(data);
  } catch (err) {
    alert('Error analyzing audio: ' + err.message);
  } finally {
    hideLoading();
  }
}

// Analyze via local disk path
async function analyzeLocalPath() {
  const path = localPathInput.value.trim();
  if (!path) {
    alert('Please enter a valid file path.');
    return;
  }

  showLoading(`Analyzing local audio file from disk...`);
  try {
    const response = await fetch('/api/analyze-local-path', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file_path: path })
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || 'Failed to analyze path.');
    }

    const data = await response.json();
    handleAnalysisSuccess(data);
  } catch (err) {
    alert('Error analyzing file: ' + err.message);
  } finally {
    hideLoading();
  }
}

function handleAnalysisSuccess(data) {
  analysisData = data;
  currentFileId = data.metadata.file_id;
  currentDuration = data.metadata.duration_seconds;

  // Update Track Info
  trackTitle.innerText = data.metadata.original_filename;
  trackDetails.innerText = `${(data.metadata.sample_rate / 1000).toFixed(1)} kHz • Mono • ${formatTime(currentDuration)}`;
  timeDisplay.innerText = `0:00 / ${formatTime(currentDuration)}`;
  
  // Set Audio Source
  audioElement.src = `/api/audio/${currentFileId}`;
  audioElement.load();

  playerCard.style.display = 'block';
  forensicsCard.style.display = 'block';
  exportPlotBtn.disabled = false;
  playheadCursor.style.display = 'block';

  // Populate Forensic Badge & Metrics
  updateForensicsUI(data.forensics);

  // Render Visualizers
  redrawAll();
}

function updateForensicsUI(forensics) {
  if (forensics.cutoff_detected) {
    cutoffBadge.innerText = forensics.cutoff_severity;
    if (forensics.estimated_cutoff_hz <= 16500) {
      cutoffBadge.className = 'badge badge-danger';
    } else {
      cutoffBadge.className = 'badge badge-warning';
    }
    valCutoff.innerText = `${forensics.estimated_cutoff_hz.toFixed(0)} Hz`;
    cutoffIndicator.style.display = 'flex';
  } else {
    cutoffBadge.innerText = 'Natural Spectrum';
    cutoffBadge.className = 'badge badge-normal';
    valCutoff.innerText = 'Full Range (20k+)';
    cutoffIndicator.style.display = 'none';
  }

  valRolloff.innerText = `${forensics.spectral_rolloff_95.toFixed(0)} Hz`;
  valCentroid.innerText = `${forensics.spectral_centroid_hz.toFixed(0)} Hz`;
  valHighFreq.innerText = `${(forensics.high_freq_energy_ratio * 100).toFixed(2)}%`;

  forensicNotes.innerHTML = forensics.forensic_notes.map(n => `<p>• ${n}</p>`).join('');

  // Tonal Pills
  if (analysisData && analysisData.tonal_balance) {
    bandPills.innerHTML = analysisData.tonal_balance.bands.map(b => `
      <span class="band-pill" style="border-color: ${b.color_hex}; color: ${b.color_hex};">
        ${b.name}: ${b.energy_percent.toFixed(1)}%
      </span>
    `).join('');
  }
}

// Redraw all 3 canvases
function redrawAll() {
  if (!analysisData) return;
  renderEqualizer(analysisData.fft_spectrum);
  renderTonalBalance(analysisData.tonal_balance);
  renderSpectrogram(analysisData.spectrogram, analysisData.forensics);
}

// 1. Equalizer Curve
function renderEqualizer(eqData) {
  const dpr = window.devicePixelRatio || 1;
  const rect = eqCanvas.getBoundingClientRect();
  eqCanvas.width = rect.width * dpr;
  eqCanvas.height = rect.height * dpr;

  const ctx = eqCanvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const w = rect.width;
  const h = rect.height;

  ctx.fillStyle = '#12141a';
  ctx.fillRect(0, 0, w, h);

  // Draw Grid Lines (100Hz, 1kHz, 10kHz)
  const gridFreqs = [100, 1000, 10000];
  ctx.strokeStyle = '#1e2430';
  ctx.lineWidth = 1;
  ctx.fillStyle = '#64748b';
  ctx.font = '10px JetBrains Mono';

  gridFreqs.forEach(f => {
    const x = freqToX(f, w);
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
    ctx.fillText(`${f >= 1000 ? f/1000 + 'k' : f}Hz`, x + 4, h - 6);
  });

  // dB Grid Lines (-60, -40, -20, 0)
  const dBLines = [-60, -40, -20, 0];
  dBLines.forEach(db => {
    const y = dbToY(db, h);
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
    ctx.fillText(`${db}dB`, 6, y - 4);
  });

  const freqs = eqData.frequencies;
  const peaks = eqData.peaks_db;
  const avgs = eqData.magnitudes_db;

  // Draw Peak Hold Line
  ctx.strokeStyle = '#475569';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  for (let i = 0; i < freqs.length; i++) {
    const x = freqToX(freqs[i], w);
    const y = dbToY(peaks[i], h);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Draw Average Spectrum Curve with Cyan Glow Fill
  const gradient = ctx.createLinearGradient(0, 0, 0, h);
  gradient.addColorStop(0, 'rgba(0, 229, 255, 0.35)');
  gradient.addColorStop(1, 'rgba(0, 229, 255, 0.0)');

  ctx.beginPath();
  for (let i = 0; i < freqs.length; i++) {
    const x = freqToX(freqs[i], w);
    const y = dbToY(avgs[i], h);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }

  // Close path for fill
  ctx.lineTo(w, h);
  ctx.lineTo(0, h);
  ctx.closePath();
  ctx.fillStyle = gradient;
  ctx.fill();

  // Stroke line on top
  ctx.strokeStyle = '#00e5ff';
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  for (let i = 0; i < freqs.length; i++) {
    const x = freqToX(freqs[i], w);
    const y = dbToY(avgs[i], h);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

// 2. Tonal Balance Curve
function renderTonalBalance(tbData) {
  const dpr = window.devicePixelRatio || 1;
  const rect = tonalCanvas.getBoundingClientRect();
  tonalCanvas.width = rect.width * dpr;
  tonalCanvas.height = rect.height * dpr;

  const ctx = tonalCanvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const w = rect.width;
  const h = rect.height;

  ctx.fillStyle = '#12141a';
  ctx.fillRect(0, 0, w, h);

  // Background bands fill
  tbData.bands.forEach(b => {
    const x1 = freqToX(b.freq_range[0], w);
    const x2 = freqToX(b.freq_range[1], w);
    ctx.fillStyle = b.color_hex + '22'; // 13% opacity
    ctx.fillRect(x1, 0, x2 - x1, h);

    // Separator line
    ctx.strokeStyle = '#272d3b';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x2, 0);
    ctx.lineTo(x2, h);
    ctx.stroke();
  });

  // Draw continuous spectrum fill under curve
  const curveF = tbData.curve_freqs;
  const curveL = tbData.curve_levels;

  tbData.bands.forEach(b => {
    const fMin = b.freq_range[0];
    const fMax = b.freq_range[1];

    ctx.beginPath();
    let started = false;
    let firstX = 0, lastX = 0;

    for (let i = 0; i < curveF.length; i++) {
      if (curveF[i] >= fMin && curveF[i] <= fMax) {
        const x = freqToX(curveF[i], w);
        const y = dbToY(curveL[i], h);
        if (!started) {
          ctx.moveTo(x, y);
          firstX = x;
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
        lastX = x;
      }
    }

    if (started) {
      ctx.lineTo(lastX, h);
      ctx.lineTo(firstX, h);
      ctx.closePath();
      ctx.fillStyle = b.color_hex + '66'; // 40% opacity
      ctx.fill();
    }
  });

  // Top white smooth curve
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2.0;
  ctx.beginPath();
  for (let i = 0; i < curveF.length; i++) {
    const x = freqToX(curveF[i], w);
    const y = dbToY(curveL[i], h);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

// 3. Spectrogram Heatmap
function renderSpectrogram(specData, forensics) {
  const dpr = window.devicePixelRatio || 1;
  const rect = specCanvas.getBoundingClientRect();
  specCanvas.width = rect.width * dpr;
  specCanvas.height = rect.height * dpr;

  const ctx = specCanvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const w = rect.width;
  const h = rect.height;

  const matrix = specData.matrix_db;
  const numFreqs = matrix.length;
  const numTimes = matrix[0].length;

  // Off-screen canvas for pixel drawing
  const offCanvas = document.createElement('canvas');
  offCanvas.width = numTimes;
  offCanvas.height = numFreqs;
  const offCtx = offCanvas.getContext('2d');
  const imgData = offCtx.createImageData(numTimes, numFreqs);
  const data = imgData.data;

  // Map dB to Inferno colormap
  for (let f = 0; f < numFreqs; f++) {
    const row = matrix[numFreqs - 1 - f]; // invert y so 0Hz is at bottom
    for (let t = 0; t < numTimes; t++) {
      const db = row[t];
      // Normalize -80 dB to 0 dB -> [0.0, 1.0]
      const norm = Math.max(0, Math.min(1, (db + 80) / 80));
      const [r, g, b] = getInfernoColor(norm);

      const pIdx = (f * numTimes + t) * 4;
      data[pIdx] = r;
      data[pIdx + 1] = g;
      data[pIdx + 2] = b;
      data[pIdx + 3] = 255;
    }
  }

  offCtx.putImageData(imgData, 0, 0);

  // Draw stretched to canvas with smooth filtering
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(offCanvas, 0, 0, w, h);

  // Overlay AI Cutoff Line if detected
  if (forensics && forensics.cutoff_detected && forensics.estimated_cutoff_hz) {
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

    // Cutoff Label
    ctx.fillStyle = '#00ffcc';
    ctx.font = 'bold 11px JetBrains Mono';
    ctx.fillText(`AI Brickwall Cutoff: ${forensics.estimated_cutoff_hz.toFixed(0)} Hz`, 12, cutoffY - 6);
  }

  // Draw Frequency Guides (5k, 10k, 15k, 20k)
  const maxHz = specData.frequencies[specData.frequencies.length - 1];
  [5000, 10000, 15000, 20000].forEach(hz => {
    if (hz < maxHz) {
      const y = h - (hz / maxHz) * h;
      ctx.fillStyle = 'rgba(255, 255, 255, 0.45)';
      ctx.font = '10px JetBrains Mono';
      ctx.fillText(`${hz/1000}kHz`, w - 44, y - 4);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w - 50, y);
      ctx.stroke();
    }
  });
}

// Helpers for Coordinates
function freqToX(freq, width) {
  const minF = Math.log10(20);
  const maxF = Math.log10(20000);
  const clamped = Math.max(20, Math.min(20000, freq));
  return ((Math.log10(clamped) - minF) / (maxF - minF)) * width;
}

function dbToY(db, height) {
  const minDb = -80.0;
  const maxDb = 0.0;
  const clamped = Math.max(minDb, Math.min(maxDb, db));
  return height - ((clamped - minDb) / (maxDb - minDb)) * height;
}

// High Quality Inferno Colormap
function getInfernoColor(t) {
  // t is 0.0 to 1.0
  const c0 = [0, 0, 4];
  const c1 = [40, 11, 84];
  const c2 = [101, 21, 110];
  const c3 = [159, 42, 99];
  const c4 = [212, 72, 66];
  const c5 = [245, 125, 21];
  const c6 = [250, 193, 39];
  const c7 = [252, 255, 164];

  const palette = [c0, c1, c2, c3, c4, c5, c6, c7];
  const idx = t * (palette.length - 1);
  const i = Math.floor(idx);
  const f = idx - i;

  if (i >= palette.length - 1) return palette[palette.length - 1];
  const p1 = palette[i];
  const p2 = palette[i + 1];

  return [
    Math.round(p1[0] + (p2[0] - p1[0]) * f),
    Math.round(p1[1] + (p2[1] - p1[1]) * f),
    Math.round(p1[2] + (p2[2] - p1[2]) * f)
  ];
}

// Playback Logic
function togglePlay() {
  if (audioElement.paused) {
    audioElement.play();
    playIcon.innerHTML = '<rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect>';
  } else {
    audioElement.pause();
    playIcon.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"></polygon>';
  }
}

function onPlaybackEnded() {
  playIcon.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"></polygon>';
  playheadCursor.style.left = '0%';
}

function updatePlaybackCursor() {
  if (!currentDuration) return;
  const cur = audioElement.currentTime;
  timeDisplay.innerText = `${formatTime(cur)} / ${formatTime(currentDuration)}`;
  const pct = (cur / currentDuration) * 100;
  playheadCursor.style.left = `${pct}%`;
}

function handleSpecClick(e) {
  if (!currentDuration) return;
  const rect = specCanvas.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const pct = clickX / rect.width;
  audioElement.currentTime = pct * currentDuration;
}

function formatTime(secs) {
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${m}:${s < 10 ? '0' : ''}${s}`;
}

// Export high-res PNG for Thesis / Skripsi
async function exportPlot() {
  if (!currentFileId) return;
  showLoading('Rendering high-resolution 200 DPI publication figure...');
  try {
    const res = await fetch(`/api/export-plot/${currentFileId}`, { method: 'POST' });
    if (!res.ok) throw new Error('Export failed.');
    
    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `forensic_visualizer_${trackTitle.innerText}.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  } catch (err) {
    alert('Error exporting figure: ' + err.message);
  } finally {
    hideLoading();
  }
}

// Auto init on load
window.addEventListener('DOMContentLoaded', init);
