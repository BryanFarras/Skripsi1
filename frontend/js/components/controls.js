// Visualizer Navigation & Options Controls
import { state, elements, formatTimecode } from '../state.js';
import { renderSpectrogramCanvas } from './spectrogram.js';
import { renderWaterfall3DCanvas } from './waterfall3d.js';
import { renderEqualizerCanvas } from './equalizer.js';
import { renderTonalCanvas } from './tonalBalance.js';
import { updateLiveAudioData } from '../audioPlayer.js';

// View Switcher (Spectrogram, 3D Waterfall, Equalizer, Tonal, Stacked)
export function switchView(view) {
  state.currentView = view;
  document.querySelectorAll('.screen-tab').forEach(t => {
    t.classList.toggle('active', t.dataset.view === view);
  });

  elements.mainSpecCanvas.style.display = (view === 'spectrogram') ? 'block' : 'none';
  elements.mainWaterfallCanvas.style.display = (view === 'waterfall') ? 'block' : 'none';
  elements.mainEqCanvas.style.display = view === 'equalizer' ? 'block' : 'none';
  elements.mainTonalCanvas.style.display = view === 'tonal' ? 'block' : 'none';
  elements.stackedVisualizersCard.style.display = view === 'stacked' ? 'flex' : 'none';

  // Spectrogram Toolbar Visibility
  if (elements.specToolbarTag) {
    elements.specToolbarTag.style.display = (view === 'spectrogram') ? 'flex' : 'none';
  }

  // 3D Controls Visibility
  if (view === 'waterfall') {
    elements.waterfallHintTag.style.display = 'block';
    elements.waterfallControlsGroup.style.display = 'flex';
    elements.playheadCursor.style.display = 'none';
  } else {
    elements.waterfallHintTag.style.display = 'none';
    elements.waterfallControlsGroup.style.display = 'none';
    elements.playheadCursor.style.display = (view === 'spectrogram') ? 'block' : 'none';
  }

  redrawCurrentView();
}

export function redrawCurrentView() {
  if (!state.analysisData) return;
  const palette = elements.colorPaletteSelect ? elements.colorPaletteSelect.value : 'inferno';
  const showCutoff = elements.aiCutoffCheck ? elements.aiCutoffCheck.checked : true;
  const curTime = elements.audioElement ? elements.audioElement.currentTime : 0;
  const isPlaying = elements.audioElement && !elements.audioElement.paused;

  updateLiveAudioData(16.6);

  if (state.currentView === 'spectrogram') {
    renderSpectrogramCanvas(elements.mainSpecCanvas, state.analysisData.spectrogram, state.analysisData.forensics, palette, showCutoff, curTime);
  }
  if (state.currentView === 'waterfall') {
    renderWaterfall3DCanvas(elements.mainWaterfallCanvas, state.analysisData.waterfall_3d);
  }
  if (state.currentView === 'equalizer') {
    renderEqualizerCanvas(elements.mainEqCanvas, state.analysisData.fft_spectrum, isPlaying);
  }
  if (state.currentView === 'tonal') {
    renderTonalCanvas(elements.mainTonalCanvas, state.analysisData.tonal_balance, isPlaying);
  }
  if (state.currentView === 'stacked') {
    renderSpectrogramCanvas(elements.mainSpecCanvas, state.analysisData.spectrogram, state.analysisData.forensics, palette, showCutoff, curTime);
    renderEqualizerCanvas(elements.stackedEqCanvas, state.analysisData.fft_spectrum, isPlaying);
    renderTonalCanvas(elements.stackedTonalCanvas, state.analysisData.tonal_balance, isPlaying);
  }
}

// Trim Position Handlers
export function setTrimStartFromCurrent() {
  if (!elements.audioElement) return;
  state.trimStartSec = elements.audioElement.currentTime;
  if (elements.trimStartInput) elements.trimStartInput.value = formatTimecode(state.trimStartSec);
  updateTrimHighlight();
}

export function setTrimEndFromCurrent() {
  if (!elements.audioElement) return;
  state.trimEndSec = elements.audioElement.currentTime;
  if (elements.trimEndInput) elements.trimEndInput.value = formatTimecode(state.trimEndSec);
  updateTrimHighlight();
}

export function resetTrim() {
  state.trimStartSec = 0;
  state.trimEndSec = state.currentDuration;
  if (elements.trimStartInput) elements.trimStartInput.value = formatTimecode(0);
  if (elements.trimEndInput) elements.trimEndInput.value = formatTimecode(state.currentDuration);
  updateTrimHighlight();
}

export function updateTrimHighlight() {
  if (!state.currentDuration || !elements.trimRangeHighlight) return;
  const sPct = (state.trimStartSec / state.currentDuration) * 100;
  const ePct = (state.trimEndSec / state.currentDuration) * 100;
  elements.trimRangeHighlight.style.display = 'block';
  elements.trimRangeHighlight.style.left = `${sPct}%`;
  elements.trimRangeHighlight.style.width = `${Math.max(0, ePct - sPct)}%`;
}

// Setup Tab Buttons, Trim, Averaging, Transparency, Palette & Cutoff Controls
export function setupControls() {
  // Screen Tabs
  document.querySelectorAll('.screen-tab').forEach(tab => {
    tab.addEventListener('click', (e) => {
      const view = e.target.closest('.screen-tab').dataset.view;
      switchView(view);
    });
  });

  // Averaging Window Slider & Preset Chips
  if (elements.averagingSlider) {
    elements.averagingSlider.addEventListener('input', (e) => {
      state.averagingMs = parseInt(e.target.value, 10);
      if (elements.averagingValBadge) elements.averagingValBadge.innerText = `${state.averagingMs} ms`;
      elements.presetChips.forEach(c => {
        c.classList.toggle('active', parseInt(c.dataset.avg, 10) === state.averagingMs);
      });
      if (!state.isLiveRendering && state.analysisData) redrawCurrentView();
    });
  }

  elements.presetChips.forEach(chip => {
    chip.addEventListener('click', () => {
      state.averagingMs = parseInt(chip.dataset.avg, 10);
      if (elements.averagingSlider) elements.averagingSlider.value = state.averagingMs;
      if (elements.averagingValBadge) elements.averagingValBadge.innerText = `${state.averagingMs} ms`;
      elements.presetChips.forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      if (!state.isLiveRendering && state.analysisData) redrawCurrentView();
    });
  });

  // 3D Waterfall Inactive Transparency Slider
  if (elements.wfOpacitySlider) {
    elements.wfOpacitySlider.addEventListener('input', (e) => {
      const val = parseInt(e.target.value, 10);
      state.wfInactiveAlpha = val / 100.0;
      if (elements.wfOpacityBadge) {
        elements.wfOpacityBadge.innerText = val === 0 ? 'Invisible (0%)' : (val <= 20 ? `Transparent (${val}%)` : `Semi-Opaque (${val}%)`);
      }
      if (state.currentView === 'waterfall' && state.analysisData) {
        renderWaterfall3DCanvas(elements.mainWaterfallCanvas, state.analysisData.waterfall_3d);
      }
    });
  }

  // Trim Controls
  if (elements.useTrimStartBtn) elements.useTrimStartBtn.addEventListener('click', setTrimStartFromCurrent);
  if (elements.useTrimEndBtn) elements.useTrimEndBtn.addEventListener('click', setTrimEndFromCurrent);
  if (elements.copyStartBtn) elements.copyStartBtn.addEventListener('click', setTrimStartFromCurrent);
  if (elements.copyEndBtn) elements.copyEndBtn.addEventListener('click', setTrimEndFromCurrent);
  if (elements.resetTrimBtn) elements.resetTrimBtn.addEventListener('click', resetTrim);

  // Colormap & AI Cutoff Select
  if (elements.colorPaletteSelect) {
    elements.colorPaletteSelect.addEventListener('change', () => redrawCurrentView());
  }
  if (elements.aiCutoffCheck) {
    elements.aiCutoffCheck.addEventListener('change', () => redrawCurrentView());
  }

  window.addEventListener('resize', () => redrawCurrentView());
}
