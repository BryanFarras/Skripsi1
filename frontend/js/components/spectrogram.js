// 2D Spectrogram Renderer (Wave Candy Scrolling & Edison Full Track Modes)
import { state, elements, getColorForPalette, formatShortTime } from '../state.js';
import { safeSeek } from '../audioPlayer.js';

// Caching offscreen colormapped spectrogram for 60 FPS hardware-accelerated rendering
export function getOrCreateSpecOffscreen(specData, palette) {
  const cacheKey = `${state.currentFileId}_${palette}`;
  if (state.specOffscreenCanvas && state.specOffscreenKey === cacheKey) {
    return state.specOffscreenCanvas;
  }

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
  state.specOffscreenCanvas = off;
  state.specOffscreenKey = cacheKey;
  return off;
}

// Render Spectrogram Canvas with DAW Playhead Line
export function renderSpectrogramCanvas(canvas, specData, forensics, palette, showCutoff, curTimeParam) {
  if (!canvas || !specData) return;
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

  const curTime = (curTimeParam !== undefined) ? curTimeParam : (elements.audioElement ? elements.audioElement.currentTime : 0);
  const offCanvas = getOrCreateSpecOffscreen(specData, palette);
  const offW = offCanvas.width;
  const offH = offCanvas.height;

  ctx.imageSmoothingEnabled = true;

  let playheadX = 0;

  if (state.specViewMode === 'scroll' && state.currentDuration > 0) {
    const winSec = Math.min(state.currentDuration, state.specWindowSec);
    const anchorRatio = 0.35; // Playhead anchored at 35% from left

    let tStart = curTime - anchorRatio * winSec;
    if (tStart < 0) tStart = 0;
    if (tStart > state.currentDuration - winSec) tStart = Math.max(0, state.currentDuration - winSec);

    const sx = (tStart / state.currentDuration) * offW;
    const sW = (winSec / state.currentDuration) * offW;

    ctx.drawImage(offCanvas, sx, 0, sW, offH, 0, 0, w, h);
    playheadX = ((curTime - tStart) / winSec) * w;
  } else {
    // Full Track Overview
    ctx.drawImage(offCanvas, 0, 0, offW, offH, 0, 0, w, h);
    playheadX = state.currentDuration > 0 ? (curTime / state.currentDuration) * w : 0;
  }

  // Draw AI Cutoff line
  if (showCutoff && forensics && forensics.cutoff_detected && forensics.estimated_cutoff_hz) {
    const maxHz = specData.frequencies[specData.frequencies.length - 1];
    const cutoffY = h - (forensics.estimated_cutoff_hz / maxHz) * h;

    ctx.strokeStyle = '#00ffcc';
    ctx.lineWidth = 1.8;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(0, cutoffY);
    ctx.lineTo(w, cutoffY);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Frequency tick labels on left side
  const maxHz = specData.frequencies[specData.frequencies.length - 1];
  [20000, 15000, 10000, 5000, 1000].forEach(hz => {
    if (hz < maxHz) {
      const y = h - (hz / maxHz) * h;
      ctx.fillStyle = 'rgba(255, 255, 255, 0.45)';
      ctx.font = '9px JetBrains Mono';
      ctx.fillText(`${hz >= 1000 ? hz / 1000 + 'kHz' : hz + 'Hz'}`, 6, y - 3);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(42, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }
  });

  // DAW Playhead line
  if (playheadX >= 0 && playheadX <= w) {
    // Column slice highlight
    ctx.fillStyle = 'rgba(0, 255, 204, 0.15)';
    ctx.fillRect(playheadX - 3, 0, 6, h);

    // Glowing stroke
    ctx.save();
    ctx.shadowColor = '#00ffcc';
    ctx.shadowBlur = 10;
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2.0;
    ctx.beginPath();
    ctx.moveTo(playheadX, 0);
    ctx.lineTo(playheadX, h);
    ctx.stroke();
    ctx.restore();

    // Top triangle marker
    const markerSize = 7;
    ctx.fillStyle = '#00ffcc';
    ctx.shadowColor = '#00ffcc';
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.moveTo(playheadX - markerSize, 0);
    ctx.lineTo(playheadX + markerSize, 0);
    ctx.lineTo(playheadX, markerSize * 1.5);
    ctx.closePath();
    ctx.fill();

    // Bottom triangle marker
    ctx.beginPath();
    ctx.moveTo(playheadX - markerSize, h);
    ctx.lineTo(playheadX + markerSize, h);
    ctx.lineTo(playheadX, h - markerSize * 1.5);
    ctx.closePath();
    ctx.fill();
    ctx.shadowBlur = 0;

    // Timecode badge
    const timeText = formatShortTime(curTime);
    ctx.font = 'bold 10px JetBrains Mono';
    const textW = ctx.measureText(timeText).width;
    const badgeW = textW + 12;
    const badgeH = 18;
    let badgeX = playheadX + 8;
    if (badgeX + badgeW > w - 10) badgeX = playheadX - badgeW - 8;
    const badgeY = 16;

    ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
    ctx.strokeStyle = 'rgba(0, 255, 204, 0.6)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(badgeX, badgeY, badgeW, badgeH, 4);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#ffffff';
    ctx.fillText(timeText, badgeX + 6, badgeY + 13);
  }

  // Sync DOM playhead cursor position
  if (elements.playheadCursor) {
    elements.playheadCursor.style.left = `${(playheadX / w) * 100}%`;
    elements.playheadCursor.style.display = 'block';
  }
}

// Spectrogram Click & Drag Seek
export function handleSpecCanvasClick(e) {
  if (!state.currentDuration) return;
  const rect = elements.mainSpecCanvas.getBoundingClientRect();
  const clickX = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
  const clickPct = clickX / rect.width;

  let targetTime = 0;
  if (state.specViewMode === 'scroll') {
    const winSec = Math.min(state.currentDuration, state.specWindowSec);
    const anchorRatio = 0.35;
    const curTime = elements.audioElement ? elements.audioElement.currentTime : 0;
    let tStart = curTime - anchorRatio * winSec;
    if (tStart < 0) tStart = 0;
    if (tStart > state.currentDuration - winSec) tStart = Math.max(0, state.currentDuration - winSec);
    targetTime = tStart + clickPct * winSec;
  } else {
    targetTime = clickPct * state.currentDuration;
  }

  safeSeek(targetTime);
}

// Setup Spectrogram Toolbar Handlers
export function setupSpectrogramControls(onRedraw) {
  const specZoomGroup = document.getElementById('specZoomGroup');

  if (elements.btnSpecScroll && elements.btnSpecFull) {
    elements.btnSpecScroll.addEventListener('click', () => {
      state.specViewMode = 'scroll';
      elements.btnSpecScroll.classList.add('active');
      elements.btnSpecFull.classList.remove('active');
      if (specZoomGroup) specZoomGroup.style.display = 'flex';
      if (onRedraw) onRedraw();
    });

    elements.btnSpecFull.addEventListener('click', () => {
      state.specViewMode = 'full';
      elements.btnSpecFull.classList.add('active');
      elements.btnSpecScroll.classList.remove('active');
      if (specZoomGroup) specZoomGroup.style.display = 'none';
      if (onRedraw) onRedraw();
    });
  }

  elements.zoomChips.forEach(chip => {
    chip.addEventListener('click', (e) => {
      elements.zoomChips.forEach(c => c.classList.remove('active'));
      e.target.classList.add('active');
      state.specWindowSec = parseFloat(e.target.dataset.win) || 6.0;
      if (onRedraw) onRedraw();
    });
  });

  if (elements.mainSpecCanvas) {
    let isScrubbingSpec = false;
    elements.mainSpecCanvas.addEventListener('mousedown', (e) => {
      if (e.button !== 0) return;
      isScrubbingSpec = true;
      handleSpecCanvasClick(e);
    });
    window.addEventListener('mousemove', (e) => {
      if (isScrubbingSpec) handleSpecCanvasClick(e);
    });
    window.addEventListener('mouseup', () => {
      isScrubbingSpec = false;
    });
    elements.mainSpecCanvas.addEventListener('click', handleSpecCanvasClick);
  }
}
