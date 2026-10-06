// ==========================================================================
// AI Music Forensic Detector — Main Application Orchestrator
// Modular ES Architecture coordinating all decoupled components
// ==========================================================================

import { setupNavbar } from './components/xaiNavbar.js';
import { WaveformAudioPlayer } from './components/xaiWaveformPlayer.js';
import { MetadataInspector } from './components/xaiMetadata.js';
import { ForensicVerdictManager } from './components/xaiVerdict.js';
import { SignificantFeaturesManager } from './components/xaiSignificantFeatures.js';
import { AudioUploader } from './components/xaiUploader.js';

// Global Reactive State
const state = {
  activeFileId: null,
  activeFilename: '',
  analysisData: null,
  mlDetector: null,
  xai: null,
  isPlaying: false,
  currentTime: 0,
  duration: 0,
  activeFilter: 'all', // 'all', 'ai', 'human'
  activeLang: 'id',   // 'id', 'en'
  waveformPeaks: []
};

// UI Overlay Helpers
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingText = document.getElementById('loadingText');
const loadingSub = document.getElementById('loadingSub');

function showLoading(msg, sub = 'Extracting 110-D DSP features & running TreeSHAP...') {
  if (loadingText) loadingText.innerText = msg;
  if (loadingSub) loadingSub.innerText = sub;
  if (loadingOverlay) loadingOverlay.style.display = 'flex';
}

function hideLoading() {
  if (loadingOverlay) loadingOverlay.style.display = 'none';
}

// -----------------------------------------------------------------------------
// Component Instantiations & Lifecycle
// -----------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  // 1. Navbar & Editorial Modals
  setupNavbar();

  // 2. Audio Waveform Player (Waves seekbar visualizer)
  const player = new WaveformAudioPlayer(state);

  // 3. Significant Features Manager (TreeSHAP attributions & filters)
  const featuresManager = new SignificantFeaturesManager(state);

  // 4. Forensic Verdict, Narrative & Timeline
  const verdictManager = new ForensicVerdictManager(state, (seekTimestamp) => {
    player.seekTo(seekTimestamp);
  });

  // 5. Metadata Inspector (Audio signal profile & exports)
  let uploader;
  const metadataInspector = new MetadataInspector(
    state,
    (localPath) => uploader?.analyzeLocalPath(localPath),
    showLoading,
    hideLoading
  );

  // 6. Audio Uploader (Drag & Drop, File Picker, Sample Loader)
  uploader = new AudioUploader(
    (analysisData, rawBuffer) => {
      handleAnalysisSuccess(analysisData, rawBuffer);
    },
    showLoading,
    hideLoading
  );

  // Handler on successful audio analysis
  function handleAnalysisSuccess(data, rawBuffer = null) {
    state.analysisData = data;
    state.activeFileId = data.metadata.file_id;
    state.activeFilename = data.metadata.original_filename;
    state.duration = data.metadata.duration_seconds;
    state.mlDetector = data.ml_detector;
    state.xai = data.ml_detector ? data.ml_detector.xai : null;

    const totalSlices = data.ml_detector ? data.ml_detector.total_slices_analyzed : 1;

    // Load track and render waveform in player
    player.loadTrack(
      state.activeFileId,
      state.activeFilename,
      state.duration,
      data.metadata.sample_rate,
      totalSlices,
      rawBuffer,
      data.metadata
    );

    // Pass 5s slice timeline to player for red/blue segment coloring and seeking
    if (data.ml_detector && data.ml_detector.timeline) {
      player.setTimeline(data.ml_detector.timeline);
    }

    // Update metadata profile
    metadataInspector.render(data.metadata, totalSlices);

    // Update verdict, synthesis narrative and timeline
    verdictManager.renderAll(data.ml_detector, state.xai);

    // Update significant features list
    featuresManager.render(state.xai);
  }

  // 7. Auto-load initial demo track on first visit
  uploader.analyzeLocalPath('backend/uploads/02f6189c-c03.mp3');

  console.log('AI Music Detector — TreeSHAP Modular Architecture Initialized.');
});
