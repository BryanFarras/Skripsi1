// Stem Separation Component for AI Music Forensic Visualizer
import { state, elements } from '../state.js';
import { handleAnalysisLoaded } from '../uploader.js';

let activeStemAudio = null;
let activeStemKey = null;
let animFrameId = null;
let isDraggingWave = false;
let draggingKey = null;

const stemControllers = new Map();

export function initStemSplitter() {
  if (elements.splitStemsBtn) {
    elements.splitStemsBtn.addEventListener('click', handleSplitStemsClick);
  }
  if (elements.sideSplitStemsBtn) {
    elements.sideSplitStemsBtn.addEventListener('click', handleSplitStemsClick);
  }

  // Handle global window mouse events for drag scrubbing
  window.addEventListener('mouseup', () => {
    isDraggingWave = false;
    draggingKey = null;
  });

  window.addEventListener('mousemove', (e) => {
    if (isDraggingWave && draggingKey) {
      handleWaveSeek(draggingKey, e, false);
    }
  });

  // Re-render waveforms on window resize so canvas resolution stays sharp
  window.addEventListener('resize', debounce(() => {
    stemControllers.forEach((ctrl) => {
      const progress = (activeStemKey === ctrl.key && activeStemAudio && activeStemAudio.duration)
        ? (activeStemAudio.currentTime / activeStemAudio.duration)
        : ctrl.lastProgress || 0;
      drawWaveform(ctrl.canvas, ctrl.peaks, progress, ctrl.conf);
    });
  }, 100));
}

export function onAudioLoadedForStems() {
  if (elements.stemSplitterCard) {
    elements.stemSplitterCard.style.display = 'block';
  }
  if (elements.quickStemGroup) {
    elements.quickStemGroup.style.display = 'block';
  }
  if (elements.splitStemsBtn) {
    elements.splitStemsBtn.disabled = false;
    elements.splitStemsBtn.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242"></path>
        <path d="M12 12v9"></path>
        <path d="m8 17 4 4 4-4"></path>
      </svg>
      Split Audio Stems
    `;
  }
  if (elements.stemResultsContainer) {
    elements.stemResultsContainer.style.display = 'none';
  }
  if (elements.stemProcessStatus) {
    elements.stemProcessStatus.style.display = 'none';
  }
  stopStemAudioPreview();
  stemControllers.clear();
}

async function handleSplitStemsClick() {
  if (!state.currentFileId) {
    alert('Please load an audio file or sample first.');
    return;
  }

  if (state.isSplittingStems) return;
  state.isSplittingStems = true;

  if (elements.splitStemsBtn) {
    elements.splitStemsBtn.disabled = true;
    elements.splitStemsBtn.innerHTML = `
      <span class="spinner-inline"></span>
      Splitting Audio Stems...
    `;
  }
  if (elements.sideSplitStemsBtn) {
    elements.sideSplitStemsBtn.disabled = true;
    elements.sideSplitStemsBtn.innerText = 'Splitting Stems in Progress...';
  }

  if (elements.stemProcessStatus) {
    elements.stemProcessStatus.style.display = 'flex';
  }
  if (elements.stemResultsContainer) {
    elements.stemResultsContainer.style.display = 'none';
  }

  try {
    const payload = { file_id: state.currentFileId };
    if (elements.localPathInput && elements.localPathInput.value.trim()) {
      payload.file_path = elements.localPathInput.value.trim();
    }

    const res = await fetch('/api/split-stems', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Failed to split stems' }));
      throw new Error(err.detail || 'Failed to split audio stems');
    }

    const data = await res.json();
    state.currentStemsData = data;
    renderStemResults(data);

  } catch (err) {
    console.error('Stem split error:', err);
    alert('Error during stem separation: ' + err.message);
  } finally {
    state.isSplittingStems = false;
    if (elements.stemProcessStatus) {
      elements.stemProcessStatus.style.display = 'none';
    }
    if (elements.splitStemsBtn) {
      elements.splitStemsBtn.disabled = false;
      elements.splitStemsBtn.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="20 6 9 17 4 12"></polyline>
        </svg>
        Re-split Audio Stems
      `;
    }
    if (elements.sideSplitStemsBtn) {
      elements.sideSplitStemsBtn.disabled = false;
      elements.sideSplitStemsBtn.innerText = 'Re-split Audio Stems';
    }
  }
}

const STEM_CONFIGS = {
  mixture: {
    label: 'Full Mix',
    desc: 'Unseparated original audio track',
    fillColor: '#10b981',
    bgBarColor: 'rgba(16, 185, 129, 0.28)'
  },
  vocals: {
    label: 'Vocals',
    desc: 'Isolated singing voice & speech (Forensic #1 for pitch & vocoder phase)',
    fillColor: '#a855f7',
    bgBarColor: 'rgba(168, 85, 247, 0.28)'
  },
  drums: {
    label: 'Drums',
    desc: 'Percussive transients, kick, snare, hi-hats',
    fillColor: '#f97316',
    bgBarColor: 'rgba(249, 115, 22, 0.28)'
  },
  bass: {
    label: 'Bass',
    desc: 'Sub-bass frequencies & basslines',
    fillColor: '#06b6d4',
    bgBarColor: 'rgba(6, 182, 212, 0.28)'
  },
  other: {
    label: 'Other',
    desc: 'Synths, guitars, keys, accompaniment, & reverb tail',
    fillColor: '#3b82f6',
    bgBarColor: 'rgba(59, 130, 246, 0.28)'
  }
};

export function renderStemResults(data) {
  if (!elements.stemList || !elements.stemResultsContainer) return;

  stopStemAudioPreview();
  stemControllers.clear();
  elements.stemList.innerHTML = '';

  const stems = data.stems || {};
  const stemOrder = ['mixture', 'vocals', 'drums', 'bass', 'other'];
  const trackDuration = data.duration_seconds || state.currentDuration || 0;

  stemOrder.forEach(key => {
    const item = stems[key];
    if (!item) return;

    const conf = STEM_CONFIGS[key] || {
      label: key.toUpperCase(),
      desc: 'Stem component',
      fillColor: '#6366f1',
      bgBarColor: 'rgba(99, 102, 241, 0.28)'
    };

    const duration = item.duration || trackDuration || 0;
    const initialPeaks = (item.peaks && item.peaks.length > 0) ? item.peaks : generateFallbackPeaks(140);

    const card = document.createElement('div');
    card.className = 'stem-item-card';
    card.dataset.stemKey = key;

    card.innerHTML = `
      <div class="stem-card-top">
        <div class="stem-item-left">
          <div class="stem-info">
            <div class="stem-name-row">
              <span class="stem-name">${conf.label}</span>
              <span class="stem-size">${item.size_formatted}</span>
            </div>
            <p class="stem-desc">${conf.desc}</p>
          </div>
        </div>
        <div class="stem-item-actions">
          <button class="btn-stem-preview" data-stem="${key}" title="Play or pause audio preview">
            <svg class="preview-play-icon" width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
              <polygon points="5 3 19 12 5 21 5 3"></polygon>
            </svg>
            <span class="preview-label">Preview</span>
          </button>

          <button class="btn-stem-inspect" data-token="${data.stem_token}" data-stem="${key}" title="Load this isolated stem into 3D Waterfall & Spectrogram">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"></path>
              <circle cx="12" cy="12" r="3"></circle>
            </svg>
            Inspect
          </button>

          <a href="${item.stream_url}" download="${item.filename}" class="btn-stem-download" title="Download WAV stem file">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="7 10 12 15 17 10"></polyline>
              <line x1="12" y1="15" x2="12" y2="3"></line>
            </svg>
          </a>
        </div>
      </div>

      <div class="stem-wave-wrap" data-stem="${key}">
        <div class="stem-wave-canvas-box" title="Click or drag to seek">
          <canvas class="stem-wave-canvas"></canvas>
          <div class="stem-wave-playhead" style="left: 0%;"></div>
          <div class="stem-wave-hover-line" style="display: none; left: 0%;">
            <span class="stem-wave-hover-time">00:00.0</span>
          </div>
        </div>
        <div class="stem-wave-time-readout">
          <span class="stem-current-time">00:00.0</span> / <span class="stem-total-time">${formatMinSec(duration)}</span>
        </div>
      </div>
    `;

    // DOM references for this stem controller
    const canvas = card.querySelector('.stem-wave-canvas');
    const boxEl = card.querySelector('.stem-wave-canvas-box');
    const playheadEl = card.querySelector('.stem-wave-playhead');
    const hoverLineEl = card.querySelector('.stem-wave-hover-line');
    const hoverTimeEl = card.querySelector('.stem-wave-hover-time');
    const currentTimeEl = card.querySelector('.stem-current-time');
    const previewBtn = card.querySelector('.btn-stem-preview');
    const inspectBtn = card.querySelector('.btn-stem-inspect');

    const ctrl = {
      key,
      conf,
      url: item.stream_url,
      duration,
      peaks: initialPeaks,
      lastProgress: 0,
      cardEl: card,
      canvas,
      boxEl,
      playheadEl,
      hoverLineEl,
      hoverTimeEl,
      currentTimeEl,
      previewBtn
    };

    stemControllers.set(key, ctrl);

    // Initial render of waveform
    requestAnimationFrame(() => {
      drawWaveform(canvas, ctrl.peaks, 0, conf);
    });

    // If peaks were fallback envelope, asynchronously decode exact audio in background
    if (!item.peaks || item.peaks.length === 0) {
      extractAudioPeaksClient(item.stream_url).then(extracted => {
        if (extracted && extracted.length > 0) {
          ctrl.peaks = extracted;
          const prog = (activeStemKey === key && activeStemAudio && activeStemAudio.duration)
            ? (activeStemAudio.currentTime / activeStemAudio.duration)
            : ctrl.lastProgress || 0;
          drawWaveform(canvas, extracted, prog, conf);
        }
      });
    }

    // Click & Drag seeking on waveform
    boxEl.addEventListener('click', (e) => {
      e.stopPropagation();
      handleWaveSeek(key, e, true);
    });

    boxEl.addEventListener('mousedown', (e) => {
      e.stopPropagation();
      isDraggingWave = true;
      draggingKey = key;
      handleWaveSeek(key, e, true);
    });

    // Hover timestamp guide line
    boxEl.addEventListener('mousemove', (e) => {
      const rect = boxEl.getBoundingClientRect();
      const clientX = (e.clientX !== undefined) ? e.clientX : (e.pageX || 0);
      const mouseX = Math.max(0, Math.min(rect.width, (e.offsetX !== undefined && e.offsetX >= 0 && e.offsetX <= rect.width) ? e.offsetX : (clientX - rect.left)));
      const pct = rect.width > 0 ? (mouseX / rect.width) : 0;
      const hoverSec = pct * (ctrl.duration || (activeStemAudio && activeStemAudio.duration) || 0);

      hoverLineEl.style.display = 'block';
      hoverLineEl.style.left = `${(pct * 100).toFixed(2)}%`;
      hoverTimeEl.innerText = formatMinSec(hoverSec);
    });

    boxEl.addEventListener('mouseleave', () => {
      hoverLineEl.style.display = 'none';
    });

    // Preview Button Click
    previewBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      toggleStemAudio(key, item.stream_url);
    });

    // Inspect in Visualizer Button Click
    inspectBtn.addEventListener('click', async (e) => {
      e.stopPropagation();
      await loadStemToVisualizer(data.stem_token, key, inspectBtn);
    });

    elements.stemList.appendChild(card);
  });

  elements.stemResultsContainer.style.display = 'block';
  elements.stemResultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function handleWaveSeek(key, e, isInitialClick = false) {
  const ctrl = stemControllers.get(key);
  if (!ctrl) return;

  const rect = ctrl.boxEl.getBoundingClientRect();
  const clientX = (e.clientX !== undefined) ? e.clientX : (e.pageX || 0);
  const clickX = Math.max(0, Math.min(rect.width, (e.offsetX !== undefined && e.offsetX >= 0 && e.offsetX <= rect.width) ? e.offsetX : (clientX - rect.left)));
  const pct = rect.width > 0 ? (clickX / rect.width) : 0;
  const dur = ctrl.duration || (activeStemAudio && activeStemAudio.duration) || state.currentDuration || 0;
  const targetTime = pct * dur;

  // Immediately update UI so playhead and timestamp update instantly
  updateStemPlayhead(key, pct, targetTime);

  if (activeStemKey === key && activeStemAudio) {
    try {
      activeStemAudio.currentTime = targetTime;
    } catch (err) {
      console.warn('Seek error:', err);
    }
    if (activeStemAudio.paused && isInitialClick) {
      activeStemAudio.play().then(() => {
        updatePlayButtonState(key, true);
        startPlaybackLoop();
      }).catch(console.warn);
    }
  } else {
    // Switch to this stem and start playback from targetTime
    startStemPlayback(key, ctrl.url, targetTime);
  }
}

function toggleStemAudio(key, url) {
  const ctrl = stemControllers.get(key);
  if (!ctrl) return;

  // If this stem is already active
  if (activeStemKey === key && activeStemAudio) {
    if (!activeStemAudio.paused) {
      activeStemAudio.pause();
      updatePlayButtonState(key, false);
      if (animFrameId) {
        cancelAnimationFrame(animFrameId);
        animFrameId = null;
      }
    } else {
      activeStemAudio.play().catch(console.warn);
      updatePlayButtonState(key, true);
      startPlaybackLoop();
    }
    return;
  }

  // Otherwise start playback from current playhead position or 0
  const curPct = ctrl.lastProgress || 0;
  const startTime = curPct * (ctrl.duration || 0);
  startStemPlayback(key, url, startTime);
}

function startStemPlayback(key, url, startTime = 0) {
  // Stop previously active stem
  stopStemAudioPreview();

  // Pause main app player to prevent simultaneous audio
  if (elements.audioElement && !elements.audioElement.paused) {
    elements.audioElement.pause();
  }

  const ctrl = stemControllers.get(key);
  if (!ctrl) return;

  activeStemKey = key;
  activeStemAudio = new Audio(url);
  activeStemAudio.preload = 'auto';

  ctrl.cardEl.classList.add('is-active-stem');
  updatePlayButtonState(key, true);

  if (startTime > 0) {
    const initialPct = (ctrl.duration > 0) ? (startTime / ctrl.duration) : 0;
    updateStemPlayhead(key, initialPct, startTime);
  }

  activeStemAudio.addEventListener('loadedmetadata', () => {
    if (activeStemAudio && activeStemAudio.duration) {
      ctrl.duration = activeStemAudio.duration;
      const totalEl = ctrl.cardEl.querySelector('.stem-total-time');
      if (totalEl) totalEl.innerText = formatMinSec(activeStemAudio.duration);
    }
    if (startTime > 0) {
      activeStemAudio.currentTime = Math.min(startTime, activeStemAudio.duration || startTime);
    }
  });

  activeStemAudio.addEventListener('ended', () => {
    updatePlayButtonState(key, false);
    updateStemPlayhead(key, 0, 0);
    ctrl.cardEl.classList.remove('is-active-stem');
    if (animFrameId) {
      cancelAnimationFrame(animFrameId);
      animFrameId = null;
    }
    activeStemAudio = null;
    activeStemKey = null;
  });

  activeStemAudio.play().then(() => {
    if (startTime > 0 && activeStemAudio) {
      activeStemAudio.currentTime = startTime;
    }
    startPlaybackLoop();
  }).catch(err => {
    console.warn('Stem audio preview failed:', err);
    stopStemAudioPreview();
  });
}

function startPlaybackLoop() {
  if (animFrameId) cancelAnimationFrame(animFrameId);

  function loop() {
    if (!activeStemAudio || activeStemAudio.paused || !activeStemKey) {
      return;
    }
    const ctrl = stemControllers.get(activeStemKey);
    if (ctrl) {
      const cur = activeStemAudio.currentTime;
      const dur = activeStemAudio.duration || ctrl.duration || 1;
      const pct = Math.max(0, Math.min(1, cur / dur));
      updateStemPlayhead(activeStemKey, pct, cur);
    }
    animFrameId = requestAnimationFrame(loop);
  }

  animFrameId = requestAnimationFrame(loop);
}

function updateStemPlayhead(key, progress, currentTime) {
  const ctrl = stemControllers.get(key);
  if (!ctrl) return;

  ctrl.lastProgress = progress;
  ctrl.playheadEl.style.left = `${(progress * 100).toFixed(2)}%`;
  ctrl.currentTimeEl.innerText = formatMinSec(currentTime);
  drawWaveform(ctrl.canvas, ctrl.peaks, progress, ctrl.conf);
}

function updatePlayButtonState(key, isPlaying) {
  const ctrl = stemControllers.get(key);
  if (!ctrl || !ctrl.previewBtn) return;

  const label = ctrl.previewBtn.querySelector('.preview-label');
  const icon = ctrl.previewBtn.querySelector('svg');

  if (isPlaying) {
    ctrl.previewBtn.classList.add('playing');
    if (label) label.innerText = 'Pause';
    if (icon) {
      icon.innerHTML = '<rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect>';
    }
  } else {
    ctrl.previewBtn.classList.remove('playing');
    if (label) label.innerText = 'Preview';
    if (icon) {
      icon.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"></polygon>';
    }
  }
}

export function stopStemAudioPreview() {
  if (animFrameId) {
    cancelAnimationFrame(animFrameId);
    animFrameId = null;
  }
  if (activeStemAudio) {
    activeStemAudio.pause();
    activeStemAudio.src = '';
    activeStemAudio = null;
  }
  if (activeStemKey) {
    updatePlayButtonState(activeStemKey, false);
    const ctrl = stemControllers.get(activeStemKey);
    if (ctrl) {
      ctrl.cardEl.classList.remove('is-active-stem');
    }
    activeStemKey = null;
  }
}

function drawWaveform(canvas, peaks, progress, conf) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  if (rect.width === 0) return;

  const targetW = Math.round(rect.width * dpr);
  const targetH = Math.round(rect.height * dpr);
  if (canvas.width !== targetW || canvas.height !== targetH) {
    canvas.width = targetW;
    canvas.height = targetH;
  }

  const cssW = rect.width;
  const cssH = rect.height;

  ctx.save();
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, cssW, cssH);

  const midY = cssH / 2;

  // Subtle central baseline
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, midY);
  ctx.lineTo(cssW, midY);
  ctx.stroke();

  const data = (peaks && peaks.length > 0) ? peaks : generateFallbackPeaks(120);
  const barCount = data.length;
  const barWidth = Math.max(2, Math.floor(cssW / barCount) - 1);
  const gap = (cssW - (barCount * barWidth)) / Math.max(1, barCount - 1);
  const step = barWidth + gap;

  for (let i = 0; i < barCount; i++) {
    const x = i * step;
    const norm = Math.max(0.06, Math.min(1.0, data[i]));
    const barH = Math.max(2, norm * (cssH - 6));
    const y = midY - barH / 2;
    const barFraction = (x + barWidth / 2) / cssW;

    if (barFraction <= progress) {
      ctx.fillStyle = conf.fillColor || '#6366f1';
    } else {
      ctx.fillStyle = conf.bgBarColor || 'rgba(148, 163, 184, 0.28)';
    }

    const r = Math.min(barWidth / 2, 2);
    ctx.beginPath();
    ctx.roundRect(x, y, barWidth, barH, r);
    ctx.fill();
  }

  ctx.restore();
}

function generateFallbackPeaks(count = 140) {
  const peaks = [];
  for (let i = 0; i < count; i++) {
    const t = i / count;
    const env = 0.18 + 0.45 * Math.sin(t * Math.PI) * (0.6 + 0.4 * Math.sin(t * 32) * Math.cos(t * 14));
    peaks.push(Math.max(0.08, Math.min(1.0, Math.abs(env))));
  }
  return peaks;
}

async function extractAudioPeaksClient(url) {
  try {
    const res = await fetch(url);
    const ab = await res.arrayBuffer();
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const buf = await ctx.decodeAudioData(ab);
    const chan = buf.getChannelData(0);
    const count = 160;
    const step = Math.floor(chan.length / count);
    const p = [];
    for (let i = 0; i < count; i++) {
      let mx = 0;
      const s = i * step;
      const e = Math.min(s + step, chan.length);
      for (let j = s; j < e; j += 4) {
        const val = Math.abs(chan[j]);
        if (val > mx) mx = val;
      }
      p.push(mx);
    }
    const maxVal = Math.max(...p) || 1;
    return p.map(v => Number((v / maxVal).toFixed(3)));
  } catch (err) {
    return null;
  }
}

async function loadStemToVisualizer(stemToken, stemName, btn) {
  const origText = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner-inline"></span> Loading...`;

  try {
    const res = await fetch('/api/load-stem', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ stem_token: stemToken, stem_name: stemName })
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Failed to load stem' }));
      throw new Error(err.detail || 'Failed to load stem');
    }

    const data = await res.json();
    stopStemAudioPreview();
    handleAnalysisLoaded(data);

    btn.innerHTML = `
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polyline points="20 6 9 17 4 12"></polyline>
      </svg>
      Loaded
    `;
    setTimeout(() => {
      btn.disabled = false;
      btn.innerHTML = origText;
    }, 2000);

  } catch (err) {
    alert('Error loading stem to visualizer: ' + err.message);
    btn.disabled = false;
    btn.innerHTML = origText;
  }
}

function formatMinSec(sec) {
  if (!sec || isNaN(sec) || sec < 0) sec = 0;
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  const ms = Math.floor((sec % 1) * 10);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${ms}`;
}

function debounce(fn, wait) {
  let timer;
  return function(...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), wait);
  };
}
