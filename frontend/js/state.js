// Central State & DOM Element Registry

export const state = {
  currentFileId: null,
  currentDuration: 0,
  analysisData: null,
  currentView: 'spectrogram',
  trimStartSec: 0,
  trimEndSec: 0,

  // 3D Waterfall Camera State
  camYaw: -0.68,    // Matches sketch angle
  camPitch: 0.42,   // Matches sketch pitch
  camZoom: 1.0,
  isDragging3D: false,
  dragStartX: 0,
  dragStartY: 0,

  // 360° Drag Inertia State
  dragVelX: 0,
  dragVelY: 0,
  lastDragDx: 0,
  lastDragDy: 0,
  inertiaRafId: null,

  // Touch State (pinch-to-zoom)
  touch1: null,
  touch2: null,
  pinchStartDist: 0,
  pinchStartZoom: 1.0,

  // Spectrogram Moving View State
  specViewMode: 'scroll', // 'scroll' (Wave Candy) or 'full' (Edison)
  specWindowSec: 6.0,     // Time window (3s, 6s, 12s)
  specOffscreenCanvas: null,
  specOffscreenKey: null,

  // High-Performance 3D Waterfall Scratch Buffers & Opacity
  _wfScreenX: new Float32Array(1024),
  _wfScreenY: new Float32Array(1024),
  wfInactiveAlpha: 0.15,  // Inactive frames transparent by default (15%)

  // Real-Time Audio & AnalyserNode State
  audioCtx: null,
  mediaSourceNode: null,
  analyserNode: null,
  liveRawFreqData: null,
  liveMagnitudesDb: null,
  livePeakHoldDb: null,
  liveBandPercentages: [0, 0, 0, 0, 0],
  lastFrameTime: performance.now(),
  averagingMs: 400,       // Default 400ms (Ozone standard)
  isLiveRendering: false,
  animFrameId: null,

  // Stem Splitting State
  currentStemsData: null,
  isSplittingStems: false
};

// Cached DOM Elements
export const elements = {};

export function initDOMElements() {
  // Drop & Screen
  elements.mediaDropZone = document.getElementById('mediaDropZone');
  elements.dropPrompt = document.getElementById('dropPrompt');
  elements.mediaScreen = document.getElementById('mediaScreen');
  elements.audioFileInput = document.getElementById('audioFileInput');
  elements.browseBtn = document.getElementById('browseBtn');
  elements.fabBtn = document.getElementById('fabBtn');
  elements.loadDemoBtn = document.getElementById('loadDemoBtn');
  elements.localPathInput = document.getElementById('localPathInput');
  elements.analyzePathBtn = document.getElementById('analyzePathBtn');
  elements.exportTopBtn = document.getElementById('exportTopBtn');
  elements.primaryActionBtn = document.getElementById('primaryActionBtn');

  // Stem Splitter Elements
  elements.stemSplitterCard = document.getElementById('stemSplitterCard');
  elements.splitStemsBtn = document.getElementById('splitStemsBtn');
  elements.sideSplitStemsBtn = document.getElementById('sideSplitStemsBtn');
  elements.quickStemGroup = document.getElementById('quickStemGroup');
  elements.stemProcessStatus = document.getElementById('stemProcessStatus');
  elements.stemResultsContainer = document.getElementById('stemResultsContainer');
  elements.stemList = document.getElementById('stemList');
  elements.stemStatusTitle = document.getElementById('stemStatusTitle');
  elements.stemStatusSub = document.getElementById('stemStatusSub');

  // Averaging Controls
  elements.averagingSlider = document.getElementById('averagingSlider');
  elements.averagingValBadge = document.getElementById('averagingValBadge');
  elements.presetChips = document.querySelectorAll('.preset-chip');

  // Spectrogram Toolbar
  elements.specToolbarTag = document.getElementById('specToolbarTag');
  elements.btnSpecScroll = document.getElementById('btnSpecScroll');
  elements.btnSpecFull = document.getElementById('btnSpecFull');
  elements.zoomChips = document.querySelectorAll('.btn-zoom-chip');

  // Player Controls
  elements.audioElement = document.getElementById('audioElement');
  elements.bigPlayBtn = document.getElementById('bigPlayBtn');
  elements.bigPlayIcon = document.getElementById('bigPlayIcon');
  elements.playOverlay = document.getElementById('playOverlay');
  elements.miniPlayBtn = document.getElementById('miniPlayBtn');
  elements.miniPlayIcon = document.getElementById('miniPlayIcon');
  elements.progressBarBg = document.getElementById('progressBarBg');
  elements.progressBarFill = document.getElementById('progressBarFill');
  elements.trimRangeHighlight = document.getElementById('trimRangeHighlight');
  elements.timecodeDisplay = document.getElementById('timecodeDisplay');
  elements.playheadCursor = document.getElementById('playheadCursor');
  elements.canvasCutoffBadge = document.getElementById('canvasCutoffBadge');
  elements.cutoffValText = document.getElementById('cutoffValText');
  elements.waterfallHintTag = document.getElementById('waterfallHintTag');
  elements.waterfallControlsGroup = document.getElementById('waterfallControlsGroup');
  elements.wfOpacitySlider = document.getElementById('wfOpacitySlider');
  elements.wfOpacityBadge = document.getElementById('wfOpacityBadge');

  // Camera Presets
  elements.btnPresetSketch = document.getElementById('btnPresetSketch');
  elements.btnPresetIso = document.getElementById('btnPresetIso');
  elements.btnPresetFront = document.getElementById('btnPresetFront');

  // Position Buttons & Stats
  elements.useTrimStartBtn = document.getElementById('useTrimStartBtn');
  elements.useTrimEndBtn = document.getElementById('useTrimEndBtn');
  elements.resetTrimBtn = document.getElementById('resetTrimBtn');
  elements.trackStats = document.getElementById('trackStats');
  elements.statRate = document.getElementById('statRate');
  elements.statChannels = document.getElementById('statChannels');
  elements.statDuration = document.getElementById('statDuration');

  // Options Sidebar
  elements.trimStartInput = document.getElementById('trimStartInput');
  elements.trimEndInput = document.getElementById('trimEndInput');
  elements.copyStartBtn = document.getElementById('copyStartBtn');
  elements.copyEndBtn = document.getElementById('copyEndBtn');
  elements.fftSizeSelect = document.getElementById('fftSizeSelect');
  elements.colorPaletteSelect = document.getElementById('colorPaletteSelect');
  elements.aiCutoffCheck = document.getElementById('aiCutoffCheck');

  // Forensics Summary
  elements.forensicsSummaryBox = document.getElementById('forensicsSummaryBox');
  elements.cutoffBadge = document.getElementById('cutoffBadge');
  elements.valCutoff = document.getElementById('valCutoff');
  elements.valRolloff = document.getElementById('valRolloff');
  elements.valCentroid = document.getElementById('valCentroid');
  elements.valAirPower = document.getElementById('valAirPower');
  elements.forensicNotes = document.getElementById('forensicNotes');

  // Canvases
  elements.mainSpecCanvas = document.getElementById('mainSpecCanvas');
  elements.mainWaterfallCanvas = document.getElementById('mainWaterfallCanvas');
  elements.mainEqCanvas = document.getElementById('mainEqCanvas');
  elements.mainTonalCanvas = document.getElementById('mainTonalCanvas');
  elements.stackedVisualizersCard = document.getElementById('stackedVisualizersCard');
  elements.stackedEqCanvas = document.getElementById('stackedEqCanvas');
  elements.stackedTonalCanvas = document.getElementById('stackedTonalCanvas');

  elements.loadingOverlay = document.getElementById('loadingOverlay');
  elements.loadingText = document.getElementById('loadingText');
}

// Initial populate if DOM ready
if (typeof document !== 'undefined' && document.readyState !== 'loading') {
  initDOMElements();
}

// Utilities
export function hexToRgba(hex, alpha) {
  if (!hex || hex[0] !== '#') return `rgba(139, 92, 246, ${alpha})`;
  const r = parseInt(hex.slice(1, 3), 16) || 139;
  const g = parseInt(hex.slice(3, 5), 16) || 92;
  const b = parseInt(hex.slice(5, 7), 16) || 246;
  return `rgba(${r}, ${g}, ${b}, ${Number(alpha).toFixed(2)})`;
}

export function formatTimecode(totalSec) {
  if (isNaN(totalSec) || totalSec < 0) totalSec = 0;
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = Math.floor(totalSec % 60);
  const ms = Math.floor((totalSec % 1) * 100);
  const pad = (n) => String(n).padStart(2, '0');
  return `${pad(h)} : ${pad(m)} : ${pad(s)} . ${pad(ms)}`;
}

export function formatShortTime(sec) {
  if (isNaN(sec) || sec < 0) sec = 0;
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  const ms = Math.floor((sec % 1) * 100);
  const pad = (n) => String(n).padStart(2, '0');
  return `${pad(m)}:${pad(s)}.${pad(ms)}`;
}

export function freqToX(freq, w) {
  const minF = 20;
  const maxF = 20000;
  const clamped = Math.max(minF, Math.min(maxF, freq));
  return ((Math.log10(clamped) - Math.log10(minF)) / (Math.log10(maxF) - Math.log10(minF))) * w;
}

export function dbToY(db, h) {
  const minDb = -80;
  const maxDb = 5;
  const clamped = Math.max(minDb, Math.min(maxDb, db));
  return h - ((clamped - minDb) / (maxDb - minDb)) * h;
}

export function getColorForPalette(t, palette) {
  t = Math.max(0, Math.min(1, t));
  if (palette === 'inferno') {
    if (t < 0.25) return [Math.round(t * 4 * 80), 0, Math.round(t * 4 * 120)];
    if (t < 0.5) return [Math.round(80 + (t - 0.25) * 4 * 140), Math.round((t - 0.25) * 4 * 50), 120];
    if (t < 0.75) return [Math.round(220 + (t - 0.5) * 4 * 35), Math.round(50 + (t - 0.5) * 4 * 130), Math.round(120 - (t - 0.5) * 4 * 100)];
    return [255, Math.round(180 + (t - 0.75) * 4 * 75), Math.round(20 + (t - 0.75) * 4 * 180)];
  }
  if (palette === 'magma') {
    return [Math.round(t * 255), Math.round(Math.pow(t, 2) * 180), Math.round(Math.pow(t, 0.5) * 220)];
  }
  if (palette === 'plasma') {
    return [Math.round(Math.sin(t * Math.PI) * 230), Math.round(t * 180), Math.round((1 - t) * 240 + t * 50)];
  }
  // Default Cyan Neon
  return [Math.round(t * 30), Math.round(t * 229), Math.round(t * 255)];
}

export function showLoading(text) {
  if (elements.loadingText) elements.loadingText.innerText = text || 'Processing audio...';
  if (elements.loadingOverlay) elements.loadingOverlay.style.display = 'flex';
}

export function hideLoading() {
  if (elements.loadingOverlay) elements.loadingOverlay.style.display = 'none';
}
