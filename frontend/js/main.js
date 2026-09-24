// AI Music Detector Forensic Visualizer - Main Modular Entry Point
import { state, elements, initDOMElements } from './state.js';
import { setupUploader } from './uploader.js';
import { setupAudioPlayer } from './audioPlayer.js';
import { setupControls, redrawCurrentView } from './components/controls.js';
import { setupSpectrogramControls } from './components/spectrogram.js';
import { setupWaterfallOrbitControls } from './components/waterfall3d.js';
import { exportPlot } from './components/forensics.js';

export function init() {
  // Ensure all DOM elements are indexed
  initDOMElements();

  // Setup component event listeners and handlers
  setupUploader();
  setupAudioPlayer();
  setupControls();
  setupSpectrogramControls(redrawCurrentView);
  setupWaterfallOrbitControls(elements.mainWaterfallCanvas, redrawCurrentView);

  // Bind Export Skripsi Plot buttons
  if (elements.exportTopBtn) {
    elements.exportTopBtn.addEventListener('click', exportPlot);
  }
  if (elements.primaryActionBtn) {
    elements.primaryActionBtn.addEventListener('click', exportPlot);
  }

  console.log('AI Music Forensic Visualizer: Modules initialized successfully.');
}

// Bootstrap on DOM ready
if (document.readyState === 'loading') {
  window.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
