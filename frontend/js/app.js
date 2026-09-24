let currentFileId = null;
let currentDuration = 0;
let analysisData = null;
let currentView = 'spectrogram';
let trimStartSec = 0;
let trimEndSec = 0;

// 3D Waterfall Camera State
let camYaw = -0.68;    // Matches sketch angle
let camPitch = 0.42;   // Matches sketch pitch
let camZoom = 1.0;
let isDragging3D = false;
let dragStartX = 0, dragStartY = 0;

// 360° Drag Inertia State
let dragVelX = 0, dragVelY = 0;      // velocity (rad/frame) for momentum
let lastDragDx = 0, lastDragDy = 0;  // last frame delta for velocity capture
let inertiaRafId = null;             // requestAnimationFrame id for inertia loop

// Touch State (pinch-to-zoom)
let touch1 = null, touch2 = null;
let pinchStartDist = 0, pinchStartZoom = 1.0;

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

// Averaging Window Controls
const averagingSlider = document.getElementById('averagingSlider');
const averagingValBadge = document.getElementById('averagingValBadge');
const presetChips = document.querySelectorAll('.preset-chip');

// Real-Time Audio & Temporal Smoothing State
let audioCtx = null;
let mediaSourceNode = null;
let analyserNode = null;
let liveRawFreqData = null;      // Float32Array from AnalyserNode
let liveMagnitudesDb = null;     // Float32Array (256 log bins)
let livePeakHoldDb = null;       // Float32Array (256 log bins)
let liveBandPercentages = [0, 0, 0, 0, 0];
let lastFrameTime = performance.now();
let averagingMs = 400;           // Default 400ms (standard Ozone Tonal Balance)
let isLiveRendering = false;
let animFrameId = null;

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
const waterfallHintTag = document.getElementById('waterfallHintTag');
const waterfallControlsGroup = document.getElementById('waterfallControlsGroup');

const btnPresetSketch = document.getElementById('btnPresetSketch');
const btnPresetIso = document.getElementById('btnPresetIso');
const btnPresetFront = document.getElementById('btnPresetFront');

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
const mainWaterfallCanvas = document.getElementById('mainWaterfallCanvas');
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
  audioElement.addEventListener('play', () => {
    initWebAudio();
    if (audioCtx && audioCtx.state === 'suspended') {
      audioCtx.resume();
    }
    updatePlayIcons(true);
    startLiveRenderLoop();
  });
  audioElement.addEventListener('pause', () => {
    updatePlayIcons(false);
    stopLiveRenderLoop();
  });

  progressBarBg.addEventListener('click', seekAudio);
  mainSpecCanvas.addEventListener('click', handleSpecCanvasClick);

  // Live Averaging Slider & Presets
  if (averagingSlider) {
    averagingSlider.addEventListener('input', (e) => {
      averagingMs = parseInt(e.target.value, 10);
      if (averagingValBadge) averagingValBadge.innerText = `${averagingMs} ms`;
      presetChips.forEach(chip => {
        chip.classList.toggle('active', parseInt(chip.dataset.avg, 10) === averagingMs);
      });
      if (!isLiveRendering && analysisData) redrawCurrentView();
    });
  }

  presetChips.forEach(chip => {
    chip.addEventListener('click', () => {
      averagingMs = parseInt(chip.dataset.avg, 10);
      if (averagingSlider) averagingSlider.value = averagingMs;
      if (averagingValBadge) averagingValBadge.innerText = `${averagingMs} ms`;
      presetChips.forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      if (!isLiveRendering && analysisData) redrawCurrentView();
    });
  });

  // ── 3D Waterfall: Full 360° Drag with Inertia ──────────────────────────────

  /** Stop any ongoing inertia animation */
  function stopInertia() {
    if (inertiaRafId !== null) {
      cancelAnimationFrame(inertiaRafId);
      inertiaRafId = null;
    }
  }

  /** Kick off momentum after drag-release */
  function startInertia() {
    stopInertia();
    const FRICTION = 0.88;       // decay per frame (lower = stops faster)
    const MIN_VEL = 0.0002;      // stop threshold

    function step() {
      dragVelX *= FRICTION;
      dragVelY *= FRICTION;

      if (Math.abs(dragVelX) < MIN_VEL && Math.abs(dragVelY) < MIN_VEL) {
        inertiaRafId = null;
        return;
      }

      // Full 360° — yaw wraps freely, pitch wraps full circle
      camYaw  += dragVelX;
      camPitch += dragVelY;
      // Keep pitch in (-PI, PI) so it never accumulates to infinity
      camPitch = ((camPitch + Math.PI) % (2 * Math.PI)) - Math.PI;

      if (currentView === 'waterfall') redrawCurrentView();
      inertiaRafId = requestAnimationFrame(step);
    }

    inertiaRafId = requestAnimationFrame(step);
  }

  // Mouse Down — begin drag, stop any ongoing inertia
  mainWaterfallCanvas.addEventListener('mousedown', (e) => {
    if (e.button !== 0) return;          // left button only
    stopInertia();
    isDragging3D = true;
    dragStartX = e.clientX;
    dragStartY = e.clientY;
    dragVelX = 0; dragVelY = 0;
    lastDragDx = 0; lastDragDy = 0;
    mainWaterfallCanvas.style.cursor = 'grabbing';
  });

  // Mouse Move — rotate on every pixel moved
  window.addEventListener('mousemove', (e) => {
    if (!isDragging3D) return;
    const dx = e.clientX - dragStartX;
    const dy = e.clientY - dragStartY;
    dragStartX = e.clientX;
    dragStartY = e.clientY;

    // Full 360° — no clamping
    camYaw   += dx * 0.008;
    camPitch += dy * 0.008;
    camPitch  = ((camPitch + Math.PI) % (2 * Math.PI)) - Math.PI;

    // Capture velocity for inertia
    lastDragDx = dx * 0.008;
    lastDragDy = dy * 0.008;

    if (currentView === 'waterfall') redrawCurrentView();
  });

  // Mouse Up — release and launch inertia
  window.addEventListener('mouseup', (e) => {
    if (!isDragging3D) return;
    isDragging3D = false;
    mainWaterfallCanvas.style.cursor = 'grab';
    // Seed velocity from last frame's delta
    dragVelX = lastDragDx;
    dragVelY = lastDragDy;
    if (currentView === 'waterfall') startInertia();
  });

  // Restore cursor when mouse leaves the canvas during non-drag
  mainWaterfallCanvas.addEventListener('mouseenter', () => {
    if (!isDragging3D) mainWaterfallCanvas.style.cursor = 'grab';
  });
  mainWaterfallCanvas.addEventListener('mouseleave', () => {
    if (!isDragging3D) mainWaterfallCanvas.style.cursor = '';
  });

  // Scroll to Zoom
  mainWaterfallCanvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    camZoom = Math.max(0.5, Math.min(2.5, camZoom - e.deltaY * 0.0012));
    if (currentView === 'waterfall') redrawCurrentView();
  }, { passive: false });

  // ── Touch Support: drag-to-rotate + pinch-to-zoom ──────────────────────────
  mainWaterfallCanvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    stopInertia();
    isDragging3D = false;
    dragVelX = 0; dragVelY = 0;

    if (e.touches.length === 1) {
      isDragging3D = true;
      dragStartX = e.touches[0].clientX;
      dragStartY = e.touches[0].clientY;
      lastDragDx = 0; lastDragDy = 0;
      touch1 = e.touches[0];
      touch2 = null;
    } else if (e.touches.length === 2) {
      isDragging3D = false;
      touch1 = e.touches[0];
      touch2 = e.touches[1];
      pinchStartDist = Math.hypot(
        touch2.clientX - touch1.clientX,
        touch2.clientY - touch1.clientY
      );
      pinchStartZoom = camZoom;
    }
  }, { passive: false });

  mainWaterfallCanvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (e.touches.length === 1 && isDragging3D) {
      const dx = e.touches[0].clientX - dragStartX;
      const dy = e.touches[0].clientY - dragStartY;
      dragStartX = e.touches[0].clientX;
      dragStartY = e.touches[0].clientY;

      camYaw   += dx * 0.008;
      camPitch += dy * 0.008;
      camPitch  = ((camPitch + Math.PI) % (2 * Math.PI)) - Math.PI;
      lastDragDx = dx * 0.008;
      lastDragDy = dy * 0.008;

      if (currentView === 'waterfall') redrawCurrentView();
    } else if (e.touches.length === 2) {
      const dist = Math.hypot(
        e.touches[1].clientX - e.touches[0].clientX,
        e.touches[1].clientY - e.touches[0].clientY
      );
      if (pinchStartDist > 0) {
        camZoom = Math.max(0.5, Math.min(2.5, pinchStartZoom * (dist / pinchStartDist)));
        if (currentView === 'waterfall') redrawCurrentView();
      }
    }
  }, { passive: false });

  mainWaterfallCanvas.addEventListener('touchend', (e) => {
    e.preventDefault();
    if (isDragging3D) {
      isDragging3D = false;
      dragVelX = lastDragDx;
      dragVelY = lastDragDy;
      if (currentView === 'waterfall') startInertia();
    }
    touch1 = null; touch2 = null;
  }, { passive: false });

  mainWaterfallCanvas.addEventListener('click', handleWaterfallCanvasClick);

  // 3D Camera Presets
  btnPresetSketch.addEventListener('click', () => {
    camYaw = -0.68; camPitch = 0.42; camZoom = 1.0;
    setCameraPresetActive(btnPresetSketch);
    if (currentView === 'waterfall') redrawCurrentView();
  });
  btnPresetIso.addEventListener('click', () => {
    camYaw = -0.78; camPitch = 0.60; camZoom = 1.0;
    setCameraPresetActive(btnPresetIso);
    if (currentView === 'waterfall') redrawCurrentView();
  });
  btnPresetFront.addEventListener('click', () => {
    camYaw = 0.0; camPitch = 0.05; camZoom = 1.0;
    setCameraPresetActive(btnPresetFront);
    if (currentView === 'waterfall') redrawCurrentView();
  });

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

function setCameraPresetActive(btn) {
  [btnPresetSketch, btnPresetIso, btnPresetFront].forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
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
  showLoading(`Analyzing "${file.name}"... Computing 3D Waterfall & AI Forensics...`);
  try {
    let res;
    // Attempt 1: Standard multipart/form-data
    try {
      const formData = new FormData();
      formData.append('file', file);

      res = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });
    } catch (netErr) {
      // Attempt 2: Fallback to direct raw binary stream with custom filename header
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

  // Reset live buffers for clean visualization
  liveMagnitudesDb = null;
  livePeakHoldDb = null;
  if (isLiveRendering) stopLiveRenderLoop();

  // Set audio source
  audioElement.crossOrigin = 'anonymous';
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

// View Switcher (Spectrogram, 3D Waterfall, Equalizer, Tonal, Stacked)
function switchView(view) {
  currentView = view;
  document.querySelectorAll('.screen-tab').forEach(t => {
    t.classList.toggle('active', t.dataset.view === view);
  });

  mainSpecCanvas.style.display = (view === 'spectrogram') ? 'block' : 'none';
  mainWaterfallCanvas.style.display = (view === 'waterfall') ? 'block' : 'none';
  mainEqCanvas.style.display = view === 'equalizer' ? 'block' : 'none';
  mainTonalCanvas.style.display = view === 'tonal' ? 'block' : 'none';
  stackedVisualizersCard.style.display = view === 'stacked' ? 'flex' : 'none';

  // 3D Controls Visibility
  if (view === 'waterfall') {
    waterfallHintTag.style.display = 'block';
    waterfallControlsGroup.style.display = 'flex';
    playheadCursor.style.display = 'none';
  } else {
    waterfallHintTag.style.display = 'none';
    waterfallControlsGroup.style.display = 'none';
    playheadCursor.style.display = (view === 'spectrogram') ? 'block' : 'none';
  }

  redrawCurrentView();
}

function redrawCurrentView() {
  if (!analysisData) return;
  const palette = colorPaletteSelect.value;
  const showCutoff = aiCutoffCheck.checked;

  const isPlaying = audioElement && !audioElement.paused;
  updateLiveAudioData(16.6);

  if (currentView === 'spectrogram') {
    renderSpectrogramCanvas(mainSpecCanvas, analysisData.spectrogram, analysisData.forensics, palette, showCutoff);
  }
  if (currentView === 'waterfall') {
    renderWaterfall3DCanvas(mainWaterfallCanvas, analysisData.waterfall_3d);
  }
  if (currentView === 'equalizer') {
    renderEqualizerCanvas(mainEqCanvas, analysisData.fft_spectrum, isPlaying);
  }
  if (currentView === 'tonal') {
    renderTonalCanvas(mainTonalCanvas, analysisData.tonal_balance, isPlaying);
  }
  if (currentView === 'stacked') {
    renderSpectrogramCanvas(mainSpecCanvas, analysisData.spectrogram, analysisData.forensics, palette, showCutoff);
    renderEqualizerCanvas(stackedEqCanvas, analysisData.fft_spectrum, isPlaying);
    renderTonalCanvas(stackedTonalCanvas, analysisData.tonal_balance, isPlaying);
  }
}

// ==========================================
// 1. 3D SPECTRAL WATERFALL RENDERER
// (Matching the exact sketch from 3d visual.png)
// ==========================================
function renderWaterfall3DCanvas(canvas, waterfallData) {
  if (!waterfallData || !waterfallData.slices || waterfallData.slices.length === 0) return;

  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const w = rect.width;
  const h = rect.height;

  // Background
  ctx.fillStyle = '#080a0f';
  ctx.fillRect(0, 0, w, h);

  const slices = waterfallData.slices;
  const numSlices = slices.length;
  const curTime = audioElement ? audioElement.currentTime : 0;

  // Find active slice closest to current playback time
  let activeSliceIdx = 0;
  let minDiff = 999999;
  for (let s = 0; s < numSlices; s++) {
    const diff = Math.abs(slices[s].timestamp_sec - curTime);
    if (diff < minDiff) {
      minDiff = diff;
      activeSliceIdx = s;
    }
  }

  // 3D Projection Helper
  function project(x, y, z) {
    // x: 0 to 1 (Freq 20Hz -> 20kHz)
    // y: 0 to 1 (dB -80 -> 0)
    // z: 0 to 1 (Time 0s -> duration)
    const cx = (x - 0.45) * 380 * camZoom;
    const cy = (y - 0.15) * 220 * camZoom;
    const cz = (z - 0.5) * 340 * camZoom;

    // Yaw rotation (around Y axis)
    const cosY = Math.cos(camYaw);
    const sinY = Math.sin(camYaw);
    const x1 = cx * cosY + cz * sinY;
    const z1 = -cx * sinY + cz * cosY;

    // Pitch rotation (around X axis)
    const cosP = Math.cos(camPitch);
    const sinP = Math.sin(camPitch);
    const y1 = cy * cosP - z1 * sinP;
    const z2 = cy * sinP + z1 * cosP;

    // Perspective transformation
    const dist = 600;
    const fov = dist / (dist + z2);

    const sx = w * 0.52 + x1 * fov;
    const sy = h * 0.68 - y1 * fov;

    return { sx, sy, depth: z2 };
  }

  // Draw 3D Axes matching the sketch in 3d visual.png:
  // Origin: (0, 0, 0)
  const o = project(0, 0, 0);
  const axY = project(0, 1.05, 0);   // volume (y) going straight up
  const axX = project(1.05, 0, 0);   // freq (x) going towards bottom right
  const axZ = project(0, 0, 1.05);   // t (z) going into the depth

  ctx.strokeStyle = '#334155';
  ctx.lineWidth = 1.5;

  // Y axis line
  ctx.beginPath();
  ctx.moveTo(o.sx, o.sy);
  ctx.lineTo(axY.sx, axY.sy);
  ctx.stroke();

  // X axis line
  ctx.beginPath();
  ctx.moveTo(o.sx, o.sy);
  ctx.lineTo(axX.sx, axX.sy);
  ctx.stroke();

  // Z axis line
  ctx.beginPath();
  ctx.moveTo(o.sx, o.sy);
  ctx.lineTo(axZ.sx, axZ.sy);
  ctx.stroke();

  // Axis Labels matching the handwriting in 3d visual.png
  ctx.font = 'bold 12px JetBrains Mono';
  ctx.fillStyle = '#f8fafc';
  ctx.fillText('volume (y)', axY.sx - 36, axY.sy - 8);
  ctx.fillText('freq (x)', axX.sx + 8, axX.sy + 14);
  ctx.fillText('t (z)', axZ.sx + 8, axZ.sy - 4);

  const isPlaying = audioElement && !audioElement.paused;

  // Ground plane outline (Time x Frequency plane)
  const floorO = project(0, 0, 0);
  const floorX = project(1, 0, 0);
  const floorXZ = project(1, 0, 1);
  const floorZ = project(0, 0, 1);

  ctx.strokeStyle = isPlaying ? 'rgba(255, 255, 255, 0.08)' : 'rgba(255, 255, 255, 0.04)';
  ctx.lineWidth = 1.0;
  ctx.beginPath();
  ctx.moveTo(floorO.sx, floorO.sy);
  ctx.lineTo(floorX.sx, floorX.sy);
  ctx.lineTo(floorXZ.sx, floorXZ.sy);
  ctx.lineTo(floorZ.sx, floorZ.sy);
  ctx.closePath();
  ctx.stroke();

  // If playing, draw a travel guide line on the ground plane along the current slice's baseline
  if (isPlaying) {
    const curZ = activeSliceIdx / Math.max(1, numSlices - 1);
    const lineStart = project(0, 0, curZ);
    const lineEnd = project(1, 0, curZ);

    ctx.strokeStyle = 'rgba(0, 229, 255, 0.35)';
    ctx.lineWidth = 1.2;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(lineStart.sx, lineStart.sy);
    ctx.lineTo(lineEnd.sx, lineEnd.sy);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Depth-sort slices (back to front painter's algorithm)
  const sortedIndices = Array.from({ length: numSlices }, (_, i) => i);
  sortedIndices.sort((a, b) => {
    const za = project(0.5, 0, a / (numSlices - 1)).depth;
    const zb = project(0.5, 0, b / (numSlices - 1)).depth;
    return zb - za; // Furthest first
  });

  // When playing: show ONLY the frame that is currently playing.
  // When viewing without playing: keep the original view with all slices visible.
  const indicesToRender = isPlaying ? [activeSliceIdx] : sortedIndices;

  // Render Slices
  indicesToRender.forEach(sIdx => {
    const slice = slices[sIdx];
    const zNorm = sIdx / Math.max(1, numSlices - 1);
    const mags = slice.magnitudes_db;
    const numBins = mags.length;
    const isActive = (sIdx === activeSliceIdx);

    // Compute screen points for this slice
    const pts = [];
    let peakPt = null;
    let maxDb = -999;

    for (let b = 0; b < numBins; b++) {
      const xNorm = b / Math.max(1, numBins - 1);
      const db = mags[b];
      // Normalize -80dB to 0dB -> [0.0, 1.0]
      const yNorm = Math.max(0, Math.min(1, (db + 80) / 80));
      const pt = project(xNorm, yNorm, zNorm);
      pts.push(pt);

      if (db > maxDb) {
        maxDb = db;
        peakPt = pt;
      }
    }

    const baselineStart = project(0, 0, zNorm);
    const baselineEnd = project(1, 0, zNorm);

    // Fill under the ribbon
    ctx.beginPath();
    ctx.moveTo(baselineStart.sx, baselineStart.sy);
    for (let i = 0; i < pts.length; i++) {
      ctx.lineTo(pts[i].sx, pts[i].sy);
    }
    ctx.lineTo(baselineEnd.sx, baselineEnd.sy);
    ctx.closePath();

    const isDense = numSlices > 100;
    const fillAlpha = isPlaying ? '55' : (numSlices > 300 ? '12' : (numSlices > 100 ? '1c' : '28'));

    if (isActive || isPlaying) {
      ctx.fillStyle = slice.color_hex + '55';
    } else {
      ctx.fillStyle = slice.color_hex + fillAlpha;
    }
    ctx.fill();

    // Baseline separator line
    ctx.strokeStyle = (isActive || isPlaying) ? 'rgba(0, 229, 255, 0.9)' : (isDense ? 'rgba(255, 255, 255, 0.03)' : 'rgba(255, 255, 255, 0.08)');
    ctx.lineWidth = (isActive || isPlaying) ? 1.8 : (isDense ? 0.8 : 1);
    ctx.beginPath();
    ctx.moveTo(baselineStart.sx, baselineStart.sy);
    ctx.lineTo(baselineEnd.sx, baselineEnd.sy);
    ctx.stroke();

    // Top Curve Stroke
    ctx.beginPath();
    for (let i = 0; i < pts.length; i++) {
      if (i === 0) ctx.moveTo(pts[i].sx, pts[i].sy);
      else ctx.lineTo(pts[i].sx, pts[i].sy);
    }

    if (isActive || isPlaying) {
      // Glowing highlight for currently playing/scrubbed slice
      ctx.save();
      ctx.shadowColor = slice.color_hex || '#00e5ff';
      ctx.shadowBlur = 14;
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 3.4;
      ctx.stroke();
      ctx.restore();

      // Draw active time marker
      if (peakPt) {
        ctx.fillStyle = slice.color_hex || '#00e5ff';
        ctx.beginPath();
        ctx.arc(peakPt.sx, peakPt.sy, 5.0, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 11px JetBrains Mono';
        ctx.fillText(`▶ ${slice.formatted_time}`, peakPt.sx + 8, peakPt.sy - 8);
      }
    } else {
      ctx.strokeStyle = slice.color_hex;
      ctx.lineWidth = isDense ? 1.1 : 1.8;
      ctx.stroke();
    }
  });

  // Status HUD indicator
  ctx.font = '10px JetBrains Mono, monospace';
  if (isPlaying) {
    ctx.fillStyle = '#00ffcc';
    ctx.fillText(`● LIVE 3D FRAME: ${slices[activeSliceIdx].formatted_time} (Slice ${activeSliceIdx + 1}/${numSlices})`, w - 320, 24);
  } else {
    ctx.fillStyle = '#94a3b8';
    ctx.fillText(`● 3D WATERFALL (${numSlices} Slices - All Visible)`, w - 240, 24);
  }
}

// 3D Waterfall Click-To-Seek
function handleWaterfallCanvasClick(e) {
  if (!analysisData || !analysisData.waterfall_3d || !analysisData.waterfall_3d.slices) return;
  const rect = mainWaterfallCanvas.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const clickY = e.clientY - rect.top;
  const w = rect.width;
  const h = rect.height;

  const slices = analysisData.waterfall_3d.slices;
  let bestIdx = 0;
  let bestDist = 999999;

  slices.forEach((s, idx) => {
    const zNorm = idx / Math.max(1, slices.length - 1);
    // Project midpoint of slice baseline
    const cx = (0.5 - 0.45) * 380 * camZoom;
    const cy = (0.0 - 0.15) * 220 * camZoom;
    const cz = (zNorm - 0.5) * 340 * camZoom;

    const cosY = Math.cos(camYaw);
    const sinY = Math.sin(camYaw);
    const x1 = cx * cosY + cz * sinY;
    const z1 = -cx * sinY + cz * cosY;

    const cosP = Math.cos(camPitch);
    const sinP = Math.sin(camPitch);
    const y1 = cy * cosP - z1 * sinP;
    const z2 = cy * sinP + z1 * cosP;

    const dist = 600;
    const fov = dist / (dist + z2);

    const sx = w * 0.52 + x1 * fov;
    const sy = h * 0.68 - y1 * fov;

    const d = Math.hypot(clickX - sx, clickY - sy);
    if (d < bestDist) {
      bestDist = d;
      bestIdx = idx;
    }
  });

  if (bestDist < 80) {
    safeSeek(slices[bestIdx].timestamp_sec);
  }
}

// 2. Render Spectrogram Canvas (Wave Candy style)
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

// 7 Parametric EQ 2 Bands (Matching FL Studio Fruity Parametric EQ 2 visualizer)
const PEQ2_BANDS = [
  { id: 1, name: 'SUB', freq: 35, range: [20, 60], color: '#ef4444' },
  { id: 2, name: 'BASS', freq: 110, range: [60, 250], color: '#f97316' },
  { id: 3, name: 'LOW MID', freq: 380, range: [250, 600], color: '#eab308' },
  { id: 4, name: 'MID', freq: 1100, range: [600, 2400], color: '#22c55e' },
  { id: 5, name: 'HIGH MID', freq: 3400, range: [2400, 6000], color: '#06b6d4' },
  { id: 6, name: 'HIGH', freq: 7600, range: [6000, 11000], color: '#3b82f6' },
  { id: 7, name: 'TREBLE', freq: 14500, range: [11000, 20000], color: '#a855f7' }
];

const BAND_DEFINITIONS = [
  { name: 'Sub', range: [20, 60], color: '#ef4444' },
  { name: 'Bass', range: [60, 250], color: '#f97316' },
  { name: 'Low Mid', range: [250, 2000], color: '#22c55e' },
  { name: 'High Mid', range: [2000, 8000], color: '#06b6d4' },
  { name: 'Treble / Air', range: [8000, 20000], color: '#a855f7' }
];

// 3. Render Equalizer Canvas (FL Studio Fruity Parametric EQ 2 Real-Time Visualizer)
function renderEqualizerCanvas(canvas, eqData, isLive = false) {
  if (!canvas || !eqData) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const w = rect.width;
  const h = rect.height;

  // Background - Sleek Studio Console
  ctx.fillStyle = '#080b11';
  ctx.fillRect(0, 0, w, h);

  const topHeaderH = 30;
  const bottomAxisH = 22;
  const rightAxisW = 44;
  const contentW = Math.max(100, w - rightAxisW);
  const contentH = Math.max(50, h - topHeaderH - bottomAxisH);
  const topY = topHeaderH;
  const bottomY = topY + contentH;

  // dB to Y mapping tailored for Parametric EQ 2 (+12dB down to -80dB)
  function peqDbToY(db) {
    const minDb = -80.0;
    const maxDb = 12.0;
    const clamped = Math.max(minDb, Math.min(maxDb, db));
    return topY + (1.0 - (clamped - minDb) / (maxDb - minDb)) * contentH;
  }

  // 1. Top 7 Band Header Buttons / Pills
  const pillW = contentW / 7;
  PEQ2_BANDS.forEach((b, idx) => {
    const px = idx * pillW;
    const pw = pillW - 2;

    ctx.fillStyle = 'rgba(18, 24, 38, 0.9)';
    ctx.fillRect(px, 3, pw, topHeaderH - 6);

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;
    ctx.strokeRect(px, 3, pw, topHeaderH - 6);

    // Colored top accent
    ctx.fillStyle = b.color;
    ctx.fillRect(px, 3, pw, 3);

    // Pill badge & name
    ctx.beginPath();
    ctx.arc(px + 10, 15, 6, 0, Math.PI * 2);
    ctx.fillStyle = b.color;
    ctx.fill();

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 8px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`${b.id}`, px + 10, 15);

    ctx.textAlign = 'left';
    ctx.font = 'bold 9px JetBrains Mono, monospace';
    ctx.fillStyle = '#cbd5e1';
    ctx.fillText(b.name, px + 20, 16);
  });
  ctx.textAlign = 'start';
  ctx.textBaseline = 'alphabetic';

  // 2. Frequency Grid Lines & Labels
  const gridFreqs = [20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000, 10000, 20000];
  const majorFreqs = [20, 50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000];

  gridFreqs.forEach(f => {
    const x = freqToX(f, contentW);
    const isMajor = [100, 1000, 10000].includes(f);
    ctx.strokeStyle = isMajor ? 'rgba(255, 255, 255, 0.12)' : 'rgba(255, 255, 255, 0.04)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, topY);
    ctx.lineTo(x, bottomY);
    ctx.stroke();
  });

  majorFreqs.forEach(f => {
    const x = freqToX(f, contentW);
    ctx.fillStyle = '#64748b';
    ctx.font = '9px JetBrains Mono, monospace';
    const txt = f >= 1000 ? `${f / 1000}k` : `${f}`;
    ctx.fillText(txt, x - 6, h - 6);
  });

  // 3. dB Grid Lines & Scale Labels
  const dbLines = [12, 6, 0, -6, -12, -18, -36, -60];
  dbLines.forEach(db => {
    const y = peqDbToY(db);
    ctx.strokeStyle = (db === 0) ? 'rgba(255, 255, 255, 0.22)' : 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = (db === 0) ? 1.2 : 1;
    if (db !== 0) ctx.setLineDash([4, 4]); else ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(contentW, y);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = (db === 0) ? '#cbd5e1' : '#64748b';
    ctx.font = '9px JetBrains Mono, monospace';
    ctx.fillText(`${db > 0 ? '+' : ''}${db}dB`, contentW + 6, y + 3);
  });

  const freqs = eqData.frequencies;
  const globalAvgs = eqData.magnitudes_db;

  const avgs = (liveMagnitudesDb && liveMagnitudesDb.length === freqs.length) ? liveMagnitudesDb : globalAvgs;
  const peaks = (livePeakHoldDb && livePeakHoldDb.length === freqs.length) ? livePeakHoldDb : eqData.peaks_db;

  function getDbAtFreq(targetHz, arr) {
    if (targetHz <= freqs[0]) return arr[0];
    if (targetHz >= freqs[freqs.length - 1]) return arr[freqs.length - 1];
    const logMin = Math.log10(freqs[0]);
    const logMax = Math.log10(freqs[freqs.length - 1]);
    const logT = Math.log10(targetHz);
    const pos = ((logT - logMin) / (logMax - logMin)) * (freqs.length - 1);
    const i0 = Math.floor(pos);
    const i1 = Math.min(freqs.length - 1, i0 + 1);
    const frac = pos - i0;
    return arr[i0] * (1 - frac) + arr[i1] * frac;
  }

  // 4. REAL-TIME VERTICAL SPECTRUM BARS (FL Studio Parametric EQ 2 Style)
  const barSpacing = 3;
  const numBars = Math.floor(contentW / barSpacing);
  const barW = Math.max(1.5, barSpacing - 1);

  for (let i = 0; i < numBars; i++) {
    const x = i * barSpacing;
    const norm = i / (numBars - 1);
    const logF = Math.log10(20) + norm * (Math.log10(20000) - Math.log10(20));
    const f = Math.pow(10, logF);

    const db = getDbAtFreq(f, avgs);
    const peakDb = getDbAtFreq(f, peaks);

    const barY = peqDbToY(db);
    const peakY = peqDbToY(peakDb);
    const barHeight = bottomY - barY;

    if (barHeight > 1) {
      // Flame Gradient (Crimson -> Red -> Orange -> Amber -> Yellow)
      const grad = ctx.createLinearGradient(0, bottomY, 0, barY);
      grad.addColorStop(0.0, '#5a0204');
      grad.addColorStop(0.35, '#dc2626');
      grad.addColorStop(0.70, '#ea580c');
      grad.addColorStop(0.92, '#f59e0b');
      grad.addColorStop(1.0, '#fef08a');

      ctx.fillStyle = grad;
      ctx.fillRect(x, barY, barW, barHeight);

      // Peak Hold Cap
      if (peakY < barY - 1) {
        ctx.fillStyle = 'rgba(254, 240, 138, 0.75)';
        ctx.fillRect(x, peakY, barW, 1.5);
      }
    }
  }

  // 5. Smooth Spectrum Envelope Curve & Translucent Fill
  ctx.beginPath();
  let firstPt = true;
  for (let i = 0; i < freqs.length; i++) {
    const x = freqToX(freqs[i], contentW);
    const y = peqDbToY(avgs[i]);
    if (firstPt) { ctx.moveTo(x, y); firstPt = false; }
    else { ctx.lineTo(x, y); }
  }
  ctx.lineTo(contentW, bottomY);
  ctx.lineTo(0, bottomY);
  ctx.closePath();

  const curveGrad = ctx.createLinearGradient(0, topY, 0, bottomY);
  curveGrad.addColorStop(0.0, 'rgba(249, 115, 22, 0.22)');
  curveGrad.addColorStop(0.5, 'rgba(220, 38, 38, 0.08)');
  curveGrad.addColorStop(1.0, 'rgba(0, 0, 0, 0.0)');
  ctx.fillStyle = curveGrad;
  ctx.fill();

  // Glow line
  ctx.beginPath();
  for (let i = 0; i < freqs.length; i++) {
    const x = freqToX(freqs[i], contentW);
    const y = peqDbToY(avgs[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1.8;
  ctx.shadowColor = 'rgba(254, 240, 138, 0.6)';
  ctx.shadowBlur = 8;
  ctx.stroke();
  ctx.shadowBlur = 0;

  // 6. Numbered Band Tokens (Circles 1 to 7 riding dynamically along the curve)
  PEQ2_BANDS.forEach(b => {
    const bx = freqToX(b.freq, contentW);
    const bDb = getDbAtFreq(b.freq, avgs);
    const by = peqDbToY(bDb);

    ctx.save();
    ctx.shadowColor = b.color;
    ctx.shadowBlur = 12;

    ctx.beginPath();
    ctx.arc(bx, by, 9, 0, Math.PI * 2);
    ctx.fillStyle = '#0c111a';
    ctx.fill();

    ctx.lineWidth = 2.0;
    ctx.strokeStyle = b.color;
    ctx.stroke();
    ctx.restore();

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 9px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`${b.id}`, bx, by);

    ctx.font = '8px JetBrains Mono, monospace';
    ctx.fillStyle = b.color;
    const fLabel = b.freq >= 1000 ? `${(b.freq / 1000).toFixed(1)}k` : `${b.freq}Hz`;
    ctx.fillText(fLabel, bx, by + 14);
  });
  ctx.textAlign = 'start';
  ctx.textBaseline = 'alphabetic';

  // 7. Status Badge
  ctx.font = '10px JetBrains Mono, monospace';
  if (isLive) {
    ctx.fillStyle = '#00ffcc';
    ctx.fillText(`● LIVE PARAMETRIC EQ 2 (${averagingMs}ms avg)`, contentW - 225, topY + 18);
  } else {
    ctx.fillStyle = '#94a3b8';
    ctx.fillText(`● PLAYHEAD REAL-TIME SPECTRUM`, contentW - 200, topY + 18);
  }
}

// 4. Render Tonal Balance Canvas (Live Real-Time or Playhead Slice)
function renderTonalCanvas(canvas, tbData, isLive = false) {
  if (!canvas || !tbData) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const w = rect.width;
  const h = rect.height;

  // Background
  ctx.fillStyle = '#080a0f';
  ctx.fillRect(0, 0, w, h);

  const curveF = (analysisData && analysisData.fft_spectrum)
    ? analysisData.fft_spectrum.frequencies
    : tbData.curve_freqs;

  const curveL = (liveMagnitudesDb && liveMagnitudesDb.length === curveF.length)
    ? liveMagnitudesDb
    : tbData.curve_levels;

  const topY = 32;
  const bottomY = h - 22;
  const contentH = bottomY - topY;

  function tonalDbToY(db) {
    const minDb = -80.0;
    const maxDb = 0.0;
    const clamped = Math.max(minDb, Math.min(maxDb, db));
    return topY + (1.0 - (clamped - minDb) / (maxDb - minDb)) * contentH;
  }

  // Frequency grid
  [60, 250, 1000, 2000, 8000].forEach(f => {
    const x = freqToX(f, w);
    ctx.strokeStyle = '#151c28';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, topY);
    ctx.lineTo(x, bottomY);
    ctx.stroke();

    ctx.fillStyle = '#64748b';
    ctx.font = '10px JetBrains Mono';
    ctx.fillText(`${f >= 1000 ? f / 1000 + 'k' : f}Hz`, x + 4, h - 6);
  });

  // dB grid
  [-12, -24, -36, -48, -60].forEach(db => {
    const y = tonalDbToY(db);
    ctx.strokeStyle = '#151c28';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();

    ctx.fillStyle = '#475569';
    ctx.font = '9px JetBrains Mono';
    ctx.fillText(`${db}dB`, w - 38, y - 3);
  });

  // Target Reference Corridor (Mastering target zone)
  ctx.fillStyle = 'rgba(255, 255, 255, 0.035)';
  ctx.beginPath();
  for (let i = 0; i < curveF.length; i++) {
    const x = freqToX(curveF[i], w);
    const targetDb = -18 - Math.log10(curveF[i] / 20) * 8;
    const yTop = tonalDbToY(targetDb + 4);
    if (i === 0) ctx.moveTo(x, yTop); else ctx.lineTo(x, yTop);
  }
  for (let i = curveF.length - 1; i >= 0; i--) {
    const x = freqToX(curveF[i], w);
    const targetDb = -18 - Math.log10(curveF[i] / 20) * 8;
    const yBot = tonalDbToY(targetDb - 6);
    ctx.lineTo(x, yBot);
  }
  ctx.closePath();
  ctx.fill();

  // Draw 5 multi-band fills
  BAND_DEFINITIONS.forEach(b => {
    const fMin = b.range[0];
    const fMax = b.range[1];
    ctx.beginPath();
    let started = false, firstX = 0, lastX = 0;

    for (let i = 0; i < curveF.length; i++) {
      if (curveF[i] >= fMin && curveF[i] <= fMax) {
        const x = freqToX(curveF[i], w);
        const y = tonalDbToY(curveL[i]);
        if (!started) { ctx.moveTo(x, y); firstX = x; started = true; }
        else { ctx.lineTo(x, y); }
        lastX = x;
      }
    }
    if (started) {
      ctx.lineTo(lastX, bottomY);
      ctx.lineTo(firstX, bottomY);
      ctx.closePath();
      ctx.fillStyle = b.color + '44';
      ctx.fill();
    }
  });

  // White smooth curve
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2.4;
  ctx.shadowColor = 'rgba(255, 255, 255, 0.4)';
  ctx.shadowBlur = 6;
  ctx.beginPath();
  for (let i = 0; i < curveF.length; i++) {
    const x = freqToX(curveF[i], w);
    const y = tonalDbToY(curveL[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Live HUD: Multi-band energy percentage badges
  const hudY = 18;
  let hudX = 14;
  ctx.font = '10px JetBrains Mono';

  BAND_DEFINITIONS.forEach((b, idx) => {
    const pct = isLive
      ? (liveBandPercentages[idx] || 0)
      : (tbData.bands && tbData.bands[idx] && tbData.bands[idx].energy_percent !== undefined
          ? tbData.bands[idx].energy_percent
          : (tbData.bands && tbData.bands[idx] && tbData.bands[idx].energy_pct !== undefined
              ? tbData.bands[idx].energy_pct
              : 0));

    ctx.fillStyle = b.color;
    ctx.fillRect(hudX, hudY - 9, 8, 8);
    ctx.fillStyle = '#e2e8f0';
    const text = `${b.name}: ${Number(pct).toFixed(1)}%`;
    ctx.fillText(text, hudX + 12, hudY - 2);
    hudX += ctx.measureText(text).width + 20;
  });

  // Status badge on top right
  if (isLive) {
    ctx.fillStyle = '#00ffcc';
    ctx.fillText(`● LIVE TONAL (${averagingMs}ms avg)`, w - 180, hudY - 2);
  } else {
    ctx.fillStyle = '#94a3b8';
    ctx.fillText(`● PLAYHEAD TONAL BALANCE`, w - 170, hudY - 2);
  }
}

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
    return [Math.round(255 * Math.pow(t, 0.7)), Math.round(180 * Math.pow(t, 1.5)), Math.round(255 * (1 - t * 0.5))];
  }
  if (palette === 'magma') {
    return [Math.round(255 * t), Math.round(120 * Math.pow(t, 2)), Math.round(200 * Math.pow(1-t, 2))];
  }
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

// ============================================================
// REAL-TIME AUDIO PROCESSING & TEMPORAL SMOOTHING ENGINE
// ============================================================
function initWebAudio() {
  if (audioCtx) {
    if (audioCtx.state === 'suspended') {
      audioCtx.resume();
    }
    return;
  }
  try {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) return;
    audioCtx = new AudioContextClass();
    analyserNode = audioCtx.createAnalyser();
    analyserNode.fftSize = 2048; // 1024 frequency bins
    analyserNode.smoothingTimeConstant = 0.0; // Temporal smoothing handled dynamically via averagingMs!

    if (!mediaSourceNode) {
      mediaSourceNode = audioCtx.createMediaElementSource(audioElement);
      mediaSourceNode.connect(analyserNode);
      analyserNode.connect(audioCtx.destination);
    }

    liveRawFreqData = new Float32Array(analyserNode.frequencyBinCount);
  } catch (err) {
    console.warn('Web Audio initialization note:', err);
  }
}

function updateLiveAudioData(dtMs) {
  if (!analysisData || !analysisData.fft_spectrum) return;

  const freqs = analysisData.fft_spectrum.frequencies;
  const numBins = freqs.length; // 256 log bins

  if (!liveMagnitudesDb || liveMagnitudesDb.length !== numBins) {
    liveMagnitudesDb = new Float32Array(analysisData.fft_spectrum.magnitudes_db);
    livePeakHoldDb = new Float32Array(analysisData.fft_spectrum.peaks_db);
  }

  const rawDbs = new Float32Array(numBins);
  let hasRealTimeSignal = false;

  // 1. Try real-time hardware signal from AnalyserNode
  if (analyserNode && liveRawFreqData && audioElement && !audioElement.paused) {
    analyserNode.getFloatFrequencyData(liveRawFreqData);

    let maxAmp = -150;
    for (let i = 0; i < liveRawFreqData.length; i++) {
      if (liveRawFreqData[i] > maxAmp) maxAmp = liveRawFreqData[i];
    }

    if (maxAmp > -120) {
      hasRealTimeSignal = true;
      const sr = (audioCtx && audioCtx.sampleRate) || 44100;
      const binWidth = (sr / 2) / liveRawFreqData.length;

      for (let i = 0; i < numBins; i++) {
        const targetF = freqs[i];
        const binIdx = targetF / binWidth;
        const b0 = Math.floor(binIdx);
        const b1 = Math.min(liveRawFreqData.length - 1, b0 + 1);
        const frac = binIdx - b0;
        const dbVal = liveRawFreqData[b0] * (1 - frac) + liveRawFreqData[b1] * frac;
        rawDbs[i] = Math.max(-90, Math.min(6, dbVal));
      }
    }
  }

  // 2. High-accuracy fallback: Interpolate precalculated Waterfall or Spectrogram slice at playhead
  if (!hasRealTimeSignal) {
    const curTime = audioElement ? audioElement.currentTime : 0;
    
    if (analysisData.waterfall_3d && analysisData.waterfall_3d.slices && analysisData.waterfall_3d.slices.length > 0) {
      const slices = analysisData.waterfall_3d.slices;
      let sIdx = 0;
      let minDiff = 999999;
      for (let s = 0; s < slices.length; s++) {
        const diff = Math.abs(slices[s].timestamp_sec - curTime);
        if (diff < minDiff) {
          minDiff = diff;
          sIdx = s;
        }
      }
      const sliceMags = slices[sIdx].magnitudes_db;
      for (let i = 0; i < numBins; i++) {
        const ratio = i / Math.max(1, numBins - 1);
        const wfIdx = ratio * (sliceMags.length - 1);
        const w0 = Math.floor(wfIdx);
        const w1 = Math.min(sliceMags.length - 1, w0 + 1);
        const frac = wfIdx - w0;
        rawDbs[i] = sliceMags[w0] * (1 - frac) + sliceMags[w1] * frac;
      }
    } else if (analysisData.spectrogram) {
      const spec = analysisData.spectrogram;
      const times = spec.time_points || [];
      const matrix = spec.matrix_db;

      let tIdx = 0;
      let minDiff = 999999;
      for (let t = 0; t < times.length; t++) {
        const diff = Math.abs(times[t] - curTime);
        if (diff < minDiff) {
          minDiff = diff;
          tIdx = t;
        }
      }

      const specFreqs = spec.frequencies;
      for (let i = 0; i < numBins; i++) {
        const targetF = freqs[i];
        let fIdx = (targetF / specFreqs[specFreqs.length - 1]) * (specFreqs.length - 1);
        fIdx = Math.max(0, Math.min(specFreqs.length - 1, fIdx));
        const f0 = Math.floor(fIdx);
        const f1 = Math.min(specFreqs.length - 1, f0 + 1);
        const frac = fIdx - f0;
        const v0 = matrix[f0] ? matrix[f0][tIdx] : -80;
        const v1 = matrix[f1] ? matrix[f1][tIdx] : -80;
        rawDbs[i] = Math.max(-90, Math.min(6, v0 * (1 - frac) + v1 * frac));
      }
    } else {
      for (let i = 0; i < numBins; i++) {
        rawDbs[i] = analysisData.fft_spectrum.magnitudes_db[i];
      }
    }
  }

  // Temporal Smoothing with user configurable averagingMs (3ms - 400ms)
  const tau = Math.max(3, averagingMs);
  const alpha = Math.exp(-dtMs / tau);
  const oneMinusAlpha = 1.0 - alpha;
  const peakDecay = 0.05 * (dtMs / 16.6);

  for (let i = 0; i < numBins; i++) {
    liveMagnitudesDb[i] = alpha * liveMagnitudesDb[i] + oneMinusAlpha * rawDbs[i];
    if (liveMagnitudesDb[i] > livePeakHoldDb[i]) {
      livePeakHoldDb[i] = liveMagnitudesDb[i];
    } else {
      livePeakHoldDb[i] = Math.max(-90, livePeakHoldDb[i] - peakDecay);
    }
  }

  computeLiveTonalBands(freqs, liveMagnitudesDb);
}

function computeLiveTonalBands(freqs, magnitudes) {
  let totalEnergy = 1e-12;
  const bandEnergies = [0, 0, 0, 0, 0];

  for (let i = 0; i < freqs.length; i++) {
    const f = freqs[i];
    const power = Math.pow(10, magnitudes[i] / 10);
    totalEnergy += power;

    for (let b = 0; b < BAND_DEFINITIONS.length; b++) {
      if (f >= BAND_DEFINITIONS[b].range[0] && f < BAND_DEFINITIONS[b].range[1]) {
        bandEnergies[b] += power;
        break;
      }
    }
  }

  for (let b = 0; b < BAND_DEFINITIONS.length; b++) {
    liveBandPercentages[b] = (bandEnergies[b] / totalEnergy) * 100;
  }
}

// 60 FPS Real-Time Render Loop
function startLiveRenderLoop() {
  if (isLiveRendering) return;
  isLiveRendering = true;
  lastFrameTime = performance.now();

  function loop(now) {
    if (!isLiveRendering) return;
    const dtMs = Math.max(1, Math.min(100, now - lastFrameTime));
    lastFrameTime = now;

    updateLiveAudioData(dtMs);

    // Update playhead, scrubber and timecodes at 60 FPS
    const curTime = audioElement ? audioElement.currentTime : 0;
    if (currentDuration > 0) {
      const pct = (curTime / currentDuration) * 100;
      progressBarFill.style.width = `${pct}%`;
      playheadCursor.style.left = `${pct}%`;
      timecodeDisplay.innerText = `${formatTimecode(curTime)} / ${formatTimecode(currentDuration)}`;
    }

    // Render active views in real-time
    if (currentView === 'equalizer') {
      renderEqualizerCanvas(mainEqCanvas, analysisData.fft_spectrum, true);
    } else if (currentView === 'tonal') {
      renderTonalCanvas(mainTonalCanvas, analysisData.tonal_balance, true);
    } else if (currentView === 'waterfall') {
      renderWaterfall3DCanvas(mainWaterfallCanvas, analysisData.waterfall_3d);
    } else if (currentView === 'stacked') {
      renderEqualizerCanvas(stackedEqCanvas, analysisData.fft_spectrum, true);
      renderTonalCanvas(stackedTonalCanvas, analysisData.tonal_balance, true);
    }

    animFrameId = requestAnimationFrame(loop);
  }

  animFrameId = requestAnimationFrame(loop);
}

function stopLiveRenderLoop() {
  isLiveRendering = false;
  if (animFrameId) {
    cancelAnimationFrame(animFrameId);
    animFrameId = null;
  }
  redrawCurrentView();
}

// Playback Logic
function togglePlayback() {
  initWebAudio();
  if (audioCtx && audioCtx.state === 'suspended') {
    audioCtx.resume();
  }

  if (audioElement.paused) {
    audioElement.play().catch(err => console.warn('Play interrupted:', err));
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

  if (audioElement.paused && analysisData) {
    updateLiveAudioData(16.6);
    if (currentView === 'waterfall' && analysisData.waterfall_3d) {
      renderWaterfall3DCanvas(mainWaterfallCanvas, analysisData.waterfall_3d);
    } else if (currentView === 'equalizer') {
      renderEqualizerCanvas(mainEqCanvas, analysisData.fft_spectrum, false);
    } else if (currentView === 'tonal') {
      renderTonalCanvas(mainTonalCanvas, analysisData.tonal_balance, false);
    } else if (currentView === 'stacked') {
      renderEqualizerCanvas(stackedEqCanvas, analysisData.fft_spectrum, false);
      renderTonalCanvas(stackedTonalCanvas, analysisData.tonal_balance, false);
    }
  }
}

function onEnded() {
  stopLiveRenderLoop();
  updatePlayIcons(false);
  progressBarFill.style.width = '0%';
  playheadCursor.style.left = '0%';
  if (currentView === 'waterfall' && analysisData && analysisData.waterfall_3d) {
    renderWaterfall3DCanvas(mainWaterfallCanvas, analysisData.waterfall_3d);
  }
}

// Safe Seek implementation - ensures seeking never resets to 0
function safeSeek(targetTime) {
  if (!currentDuration) return;
  const clampedTime = Math.max(0, Math.min(currentDuration - 0.02, targetTime));

  try {
    audioElement.currentTime = clampedTime;
  } catch (err) {
    console.warn('Seek error:', err);
  }

  // Instant UI feedback
  const pct = (clampedTime / currentDuration) * 100;
  progressBarFill.style.width = `${pct}%`;
  playheadCursor.style.left = `${pct}%`;
  timecodeDisplay.innerText = `${formatTimecode(clampedTime)} / ${formatTimecode(currentDuration)}`;

  // If paused, update visualizer slice and curves at the sought point
  if (audioElement.paused && analysisData) {
    updateLiveAudioData(16.6);
    if (currentView === 'waterfall' && analysisData.waterfall_3d) {
      renderWaterfall3DCanvas(mainWaterfallCanvas, analysisData.waterfall_3d);
    } else if (currentView === 'equalizer') {
      renderEqualizerCanvas(mainEqCanvas, analysisData.fft_spectrum, true);
    } else if (currentView === 'tonal') {
      renderTonalCanvas(mainTonalCanvas, analysisData.tonal_balance, true);
    } else if (currentView === 'stacked') {
      renderEqualizerCanvas(stackedEqCanvas, analysisData.fft_spectrum, true);
      renderTonalCanvas(stackedTonalCanvas, analysisData.tonal_balance, true);
    }
  }
}

function seekAudio(e) {
  if (!currentDuration) return;
  const rect = progressBarBg.getBoundingClientRect();
  const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  safeSeek(pct * currentDuration);
}

function handleSpecCanvasClick(e) {
  if (!currentDuration) return;
  const rect = mainSpecCanvas.getBoundingClientRect();
  const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  safeSeek(pct * currentDuration);
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
