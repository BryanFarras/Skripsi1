// Audio Player, Web Audio AnalyserNode & Real-Time Smoothing Engine
import { state, elements, formatTimecode } from './state.js';
import { BAND_DEFINITIONS } from './components/tonalBalance.js';
import { renderSpectrogramCanvas } from './components/spectrogram.js';
import { renderWaterfall3DCanvas } from './components/waterfall3d.js';
import { renderEqualizerCanvas } from './components/equalizer.js';
import { renderTonalCanvas } from './components/tonalBalance.js';

// Web Audio API & AnalyserNode Initialization
export function initWebAudio() {
  if (state.audioCtx) {
    if (state.audioCtx.state === 'suspended') {
      state.audioCtx.resume();
    }
    return;
  }
  try {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) return;
    state.audioCtx = new AudioContextClass();
    state.analyserNode = state.audioCtx.createAnalyser();
    state.analyserNode.fftSize = 4096; // 2048 high-resolution frequency bins
    state.analyserNode.smoothingTimeConstant = 0.0; // Temporal smoothing handled dynamically via averagingMs!

    if (!state.mediaSourceNode && elements.audioElement) {
      state.mediaSourceNode = state.audioCtx.createMediaElementSource(elements.audioElement);
      state.mediaSourceNode.connect(state.analyserNode);
      state.analyserNode.connect(state.audioCtx.destination);
    }

    state.liveRawFreqData = new Float32Array(state.analyserNode.frequencyBinCount);
  } catch (err) {
    console.warn('Web Audio initialization note:', err);
  }
}

// Live Audio Smoothing & Waterfall/Spectrogram Interpolation Fallback
export function updateLiveAudioData(dtMs) {
  if (!state.analysisData || !state.analysisData.fft_spectrum) return;

  const freqs = state.analysisData.fft_spectrum.frequencies;
  const numBins = freqs.length;

  if (!state.liveMagnitudesDb || state.liveMagnitudesDb.length !== numBins) {
    state.liveMagnitudesDb = new Float32Array(state.analysisData.fft_spectrum.magnitudes_db);
    state.livePeakHoldDb = new Float32Array(state.analysisData.fft_spectrum.peaks_db);
  }

  const rawDbs = new Float32Array(numBins);
  let hasRealTimeSignal = false;

  // 1. Try real-time hardware signal from AnalyserNode
  if (state.analyserNode && state.liveRawFreqData && elements.audioElement && !elements.audioElement.paused) {
    state.analyserNode.getFloatFrequencyData(state.liveRawFreqData);

    let maxAmp = -150;
    for (let i = 0; i < state.liveRawFreqData.length; i++) {
      if (state.liveRawFreqData[i] > maxAmp) maxAmp = state.liveRawFreqData[i];
    }

    if (maxAmp > -120) {
      hasRealTimeSignal = true;
      const sr = (state.audioCtx && state.audioCtx.sampleRate) || 44100;
      const binWidth = (sr / 2) / state.liveRawFreqData.length;

      for (let i = 0; i < numBins; i++) {
        const targetF = freqs[i];
        const binIdx = targetF / binWidth;
        const b0 = Math.floor(binIdx);
        const b1 = Math.min(state.liveRawFreqData.length - 1, b0 + 1);
        const frac = binIdx - b0;
        const dbVal = state.liveRawFreqData[b0] * (1 - frac) + state.liveRawFreqData[b1] * frac;
        rawDbs[i] = Math.max(-90, Math.min(6, dbVal));
      }
    }
  }

  // 2. High-accuracy fallback: Interpolate precalculated Waterfall or Spectrogram slice at playhead
  if (!hasRealTimeSignal) {
    const curTime = elements.audioElement ? elements.audioElement.currentTime : 0;
    
    if (state.analysisData.waterfall_3d && state.analysisData.waterfall_3d.slices && state.analysisData.waterfall_3d.slices.length > 0) {
      const slices = state.analysisData.waterfall_3d.slices;
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
    } else if (state.analysisData.spectrogram) {
      const spec = state.analysisData.spectrogram;
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
        rawDbs[i] = state.analysisData.fft_spectrum.magnitudes_db[i];
      }
    }
  }

  // Temporal Smoothing with user configurable averagingMs (3ms - 400ms)
  const tau = Math.max(3, state.averagingMs);
  const alpha = Math.exp(-dtMs / tau);
  const oneMinusAlpha = 1.0 - alpha;
  const peakDecay = 0.05 * (dtMs / 16.6);

  for (let i = 0; i < numBins; i++) {
    state.liveMagnitudesDb[i] = alpha * state.liveMagnitudesDb[i] + oneMinusAlpha * rawDbs[i];
    if (state.liveMagnitudesDb[i] > state.livePeakHoldDb[i]) {
      state.livePeakHoldDb[i] = state.liveMagnitudesDb[i];
    } else {
      state.livePeakHoldDb[i] = Math.max(-90, state.livePeakHoldDb[i] - peakDecay);
    }
  }

  computeLiveTonalBands(freqs, state.liveMagnitudesDb);
}

export function computeLiveTonalBands(freqs, magnitudes) {
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
    state.liveBandPercentages[b] = (bandEnergies[b] / totalEnergy) * 100;
  }
}

// 60 FPS Real-Time Render Loop
export function startLiveRenderLoop() {
  if (state.isLiveRendering) return;
  state.isLiveRendering = true;
  state.lastFrameTime = performance.now();

  function loop(now) {
    if (!state.isLiveRendering) return;
    const dtMs = Math.max(1, Math.min(100, now - state.lastFrameTime));
    state.lastFrameTime = now;

    updateLiveAudioData(dtMs);

    // Update playhead, scrubber and timecodes at 60 FPS
    const curTime = elements.audioElement ? elements.audioElement.currentTime : 0;
    if (state.currentDuration > 0) {
      const pct = (curTime / state.currentDuration) * 100;
      if (elements.progressBarFill) elements.progressBarFill.style.width = `${pct}%`;
      if (elements.playheadCursor) elements.playheadCursor.style.left = `${pct}%`;
      if (elements.timecodeDisplay) elements.timecodeDisplay.innerText = `${formatTimecode(curTime)} / ${formatTimecode(state.currentDuration)}`;
    }

    // Render active views in real-time
    const palette = elements.colorPaletteSelect ? elements.colorPaletteSelect.value : 'inferno';
    const showCutoff = elements.aiCutoffCheck ? elements.aiCutoffCheck.checked : true;

    if (state.currentView === 'spectrogram' && state.analysisData) {
      renderSpectrogramCanvas(elements.mainSpecCanvas, state.analysisData.spectrogram, state.analysisData.forensics, palette, showCutoff, curTime);
    } else if (state.currentView === 'equalizer' && state.analysisData) {
      renderEqualizerCanvas(elements.mainEqCanvas, state.analysisData.fft_spectrum, true);
    } else if (state.currentView === 'tonal' && state.analysisData) {
      renderTonalCanvas(elements.mainTonalCanvas, state.analysisData.tonal_balance, true);
    } else if (state.currentView === 'waterfall' && state.analysisData) {
      renderWaterfall3DCanvas(elements.mainWaterfallCanvas, state.analysisData.waterfall_3d);
    } else if (state.currentView === 'stacked' && state.analysisData) {
      renderSpectrogramCanvas(elements.mainSpecCanvas, state.analysisData.spectrogram, state.analysisData.forensics, palette, showCutoff, curTime);
      renderEqualizerCanvas(elements.stackedEqCanvas, state.analysisData.fft_spectrum, true);
      renderTonalCanvas(elements.stackedTonalCanvas, state.analysisData.tonal_balance, true);
    }

    state.animFrameId = requestAnimationFrame(loop);
  }

  state.animFrameId = requestAnimationFrame(loop);
}

export function stopLiveRenderLoop() {
  state.isLiveRendering = false;
  if (state.animFrameId) {
    cancelAnimationFrame(state.animFrameId);
    state.animFrameId = null;
  }
}

// Playback Logic
export function togglePlayback() {
  initWebAudio();
  if (state.audioCtx && state.audioCtx.state === 'suspended') {
    state.audioCtx.resume();
  }

  if (!elements.audioElement) return;

  if (elements.audioElement.paused) {
    elements.audioElement.play().catch(err => console.warn('Play interrupted:', err));
  } else {
    elements.audioElement.pause();
  }
}

export function updatePlayIcons(isPlaying) {
  if (isPlaying) {
    if (elements.bigPlayIcon) elements.bigPlayIcon.innerHTML = '<rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect>';
    if (elements.miniPlayIcon) elements.miniPlayIcon.innerHTML = '<rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect>';
    if (elements.playOverlay) elements.playOverlay.classList.add('hidden');
  } else {
    if (elements.bigPlayIcon) elements.bigPlayIcon.innerHTML = '<polygon points="6 3 20 12 6 21 6 3"></polygon>';
    if (elements.miniPlayIcon) elements.miniPlayIcon.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"></polygon>';
    if (elements.playOverlay) elements.playOverlay.classList.remove('hidden');
  }
}

export function onTimeUpdate() {
  if (!state.currentDuration || !elements.audioElement) return;
  const cur = elements.audioElement.currentTime;
  const pct = (cur / state.currentDuration) * 100;
  if (elements.progressBarFill) elements.progressBarFill.style.width = `${pct}%`;
  if (elements.playheadCursor) elements.playheadCursor.style.left = `${pct}%`;
  if (elements.timecodeDisplay) elements.timecodeDisplay.innerText = `${formatTimecode(cur)} / ${formatTimecode(state.currentDuration)}`;

  if (elements.audioElement.paused && state.analysisData) {
    updateLiveAudioData(16.6);
    const palette = elements.colorPaletteSelect ? elements.colorPaletteSelect.value : 'inferno';
    const showCutoff = elements.aiCutoffCheck ? elements.aiCutoffCheck.checked : true;

    if (state.currentView === 'spectrogram') {
      renderSpectrogramCanvas(elements.mainSpecCanvas, state.analysisData.spectrogram, state.analysisData.forensics, palette, showCutoff, cur);
    } else if (state.currentView === 'waterfall' && state.analysisData.waterfall_3d) {
      renderWaterfall3DCanvas(elements.mainWaterfallCanvas, state.analysisData.waterfall_3d);
    } else if (state.currentView === 'equalizer') {
      renderEqualizerCanvas(elements.mainEqCanvas, state.analysisData.fft_spectrum, false);
    } else if (state.currentView === 'tonal') {
      renderTonalCanvas(elements.mainTonalCanvas, state.analysisData.tonal_balance, false);
    } else if (state.currentView === 'stacked') {
      renderSpectrogramCanvas(elements.mainSpecCanvas, state.analysisData.spectrogram, state.analysisData.forensics, palette, showCutoff, cur);
      renderEqualizerCanvas(elements.stackedEqCanvas, state.analysisData.fft_spectrum, false);
      renderTonalCanvas(elements.stackedTonalCanvas, state.analysisData.tonal_balance, false);
    }
  }
}

export function onEnded() {
  stopLiveRenderLoop();
  updatePlayIcons(false);
  if (elements.progressBarFill) elements.progressBarFill.style.width = '0%';
  if (elements.playheadCursor) elements.playheadCursor.style.left = '0%';
  if (state.analysisData) {
    const palette = elements.colorPaletteSelect ? elements.colorPaletteSelect.value : 'inferno';
    const showCutoff = elements.aiCutoffCheck ? elements.aiCutoffCheck.checked : true;
    if (state.currentView === 'spectrogram') {
      renderSpectrogramCanvas(elements.mainSpecCanvas, state.analysisData.spectrogram, state.analysisData.forensics, palette, showCutoff, 0);
    } else if (state.currentView === 'waterfall' && state.analysisData.waterfall_3d) {
      renderWaterfall3DCanvas(elements.mainWaterfallCanvas, state.analysisData.waterfall_3d);
    }
  }
}

// Safe Seek
export function safeSeek(targetTime) {
  if (!state.currentDuration || !elements.audioElement) return;
  const clampedTime = Math.max(0, Math.min(state.currentDuration - 0.02, targetTime));

  try {
    elements.audioElement.currentTime = clampedTime;
  } catch (err) {
    console.warn('Seek error:', err);
  }

  const pct = (clampedTime / state.currentDuration) * 100;
  if (elements.progressBarFill) elements.progressBarFill.style.width = `${pct}%`;
  if (elements.playheadCursor) elements.playheadCursor.style.left = `${pct}%`;
  if (elements.timecodeDisplay) elements.timecodeDisplay.innerText = `${formatTimecode(clampedTime)} / ${formatTimecode(state.currentDuration)}`;

  if (elements.audioElement.paused && state.analysisData) {
    updateLiveAudioData(16.6);
    const palette = elements.colorPaletteSelect ? elements.colorPaletteSelect.value : 'inferno';
    const showCutoff = elements.aiCutoffCheck ? elements.aiCutoffCheck.checked : true;

    if (state.currentView === 'spectrogram') {
      renderSpectrogramCanvas(elements.mainSpecCanvas, state.analysisData.spectrogram, state.analysisData.forensics, palette, showCutoff, clampedTime);
    } else if (state.currentView === 'waterfall' && state.analysisData.waterfall_3d) {
      renderWaterfall3DCanvas(elements.mainWaterfallCanvas, state.analysisData.waterfall_3d);
    } else if (state.currentView === 'equalizer') {
      renderEqualizerCanvas(elements.mainEqCanvas, state.analysisData.fft_spectrum, true);
    } else if (state.currentView === 'tonal') {
      renderTonalCanvas(elements.mainTonalCanvas, state.analysisData.tonal_balance, true);
    } else if (state.currentView === 'stacked') {
      renderSpectrogramCanvas(elements.mainSpecCanvas, state.analysisData.spectrogram, state.analysisData.forensics, palette, showCutoff, clampedTime);
      renderEqualizerCanvas(elements.stackedEqCanvas, state.analysisData.fft_spectrum, true);
      renderTonalCanvas(elements.stackedTonalCanvas, state.analysisData.tonal_balance, true);
    }
  }
}

export function seekAudio(e) {
  if (!state.currentDuration || !elements.progressBarBg) return;
  const rect = elements.progressBarBg.getBoundingClientRect();
  const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  safeSeek(pct * state.currentDuration);
}

// Setup Player Listeners
export function setupAudioPlayer() {
  if (elements.bigPlayBtn) elements.bigPlayBtn.addEventListener('click', togglePlayback);
  if (elements.miniPlayBtn) elements.miniPlayBtn.addEventListener('click', togglePlayback);
  if (elements.progressBarBg) elements.progressBarBg.addEventListener('click', seekAudio);

  if (elements.audioElement) {
    elements.audioElement.addEventListener('play', () => {
      initWebAudio();
      updatePlayIcons(true);
      startLiveRenderLoop();
    });

    elements.audioElement.addEventListener('pause', () => {
      updatePlayIcons(false);
      stopLiveRenderLoop();
    });

    elements.audioElement.addEventListener('timeupdate', onTimeUpdate);
    elements.audioElement.addEventListener('ended', onEnded);
  }
}
