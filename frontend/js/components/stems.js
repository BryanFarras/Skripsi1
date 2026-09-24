// Stem Separation Component for AI Music Detector
import { state, elements } from '../state.js';
import { handleAnalysisLoaded } from '../uploader.js';

let activeStemAudio = null;
let activePlayingBtn = null;

export function initStemSplitter() {
  if (elements.splitStemsBtn) {
    elements.splitStemsBtn.addEventListener('click', handleSplitStemsClick);
  }
  if (elements.sideSplitStemsBtn) {
    elements.sideSplitStemsBtn.addEventListener('click', handleSplitStemsClick);
  }
}

export function onAudioLoadedForStems() {
  // Show stem splitting card & sidebar trigger when an audio file is active
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
  // Clear previous stem results if loading new song
  if (elements.stemResultsContainer) {
    elements.stemResultsContainer.style.display = 'none';
  }
  if (elements.stemProcessStatus) {
    elements.stemProcessStatus.style.display = 'none';
  }
  stopStemAudioPreview();
}

async function handleSplitStemsClick() {
  if (!state.currentFileId) {
    alert('Please load an audio file or sample first.');
    return;
  }

  if (state.isSplittingStems) return;
  state.isSplittingStems = true;

  // Update UI to processing state
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
      elements.sideSplitStemsBtn.innerText = '🎛️ Re-split Audio Stems';
    }
  }
}

const STEM_CONFIGS = {
  mixture: {
    label: 'Full Mix',
    desc: 'Unseparated original audio track',
    icon: '🎵',
    badgeClass: 'badge-mix'
  },
  vocals: {
    label: 'Vocals',
    desc: 'Isolated singing voice & speech (Forensic #1 for pitch & vocoder phase)',
    icon: '🎙️',
    badgeClass: 'badge-voc'
  },
  drums: {
    label: 'Drums',
    desc: 'Percussive transients, kick, snare, hi-hats',
    icon: '🥁',
    badgeClass: 'badge-drm'
  },
  bass: {
    label: 'Bass',
    desc: 'Sub-bass frequencies & basslines',
    icon: '🎸',
    badgeClass: 'badge-bass'
  },
  other: {
    label: 'Other',
    desc: 'Synths, guitars, keys, accompaniment, & reverb tail',
    icon: '🎹',
    badgeClass: 'badge-oth'
  }
};

export function renderStemResults(data) {
  if (!elements.stemList || !elements.stemResultsContainer) return;

  elements.stemList.innerHTML = '';
  const stems = data.stems || {};
  const stemOrder = ['mixture', 'vocals', 'drums', 'bass', 'other'];

  stemOrder.forEach(key => {
    const item = stems[key];
    if (!item) return;

    const conf = STEM_CONFIGS[key] || {
      label: key.toUpperCase(),
      desc: 'Stem component',
      icon: '🔊',
      badgeClass: 'badge-mix'
    };

    const card = document.createElement('div');
    card.className = 'stem-item-card';
    card.dataset.stemKey = key;

    card.innerHTML = `
      <div class="stem-item-left">
        <div class="stem-badge-icon ${conf.badgeClass}">${conf.icon}</div>
        <div class="stem-info">
          <div class="stem-name-row">
            <span class="stem-name">${conf.label}</span>
            <span class="stem-size">${item.size_formatted}</span>
          </div>
          <p class="stem-desc">${conf.desc}</p>
        </div>
      </div>
      <div class="stem-item-actions">
        <!-- Preview Play Button -->
        <button class="btn-stem-preview" data-url="${item.stream_url}" title="Quick listen to isolated stem">
          <svg class="preview-play-icon" width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
            <polygon points="5 3 19 12 5 21 5 3"></polygon>
          </svg>
          <span class="preview-label">Preview</span>
        </button>

        <!-- Inspect in Visualizer Button -->
        <button class="btn-stem-inspect" data-token="${data.stem_token}" data-stem="${key}" title="Load this isolated stem into 3D Waterfall & Spectrogram">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"></path>
            <circle cx="12" cy="12" r="3"></circle>
          </svg>
          Inspect in Visualizer
        </button>

        <!-- Download WAV -->
        <a href="${item.stream_url}" download="${item.filename}" class="btn-stem-download" title="Download WAV stem file">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
            <polyline points="7 10 12 15 17 10"></polyline>
            <line x1="12" y1="15" x2="12" y2="3"></line>
          </svg>
        </a>
      </div>
    `;

    // Hook up preview button
    const previewBtn = card.querySelector('.btn-stem-preview');
    if (previewBtn) {
      previewBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        toggleStemAudioPreview(item.stream_url, previewBtn);
      });
    }

    // Hook up inspect in visualizer button
    const inspectBtn = card.querySelector('.btn-stem-inspect');
    if (inspectBtn) {
      inspectBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        await loadStemToVisualizer(data.stem_token, key, inspectBtn);
      });
    }

    elements.stemList.appendChild(card);
  });

  elements.stemResultsContainer.style.display = 'block';
  // Scroll down smoothly to show results
  elements.stemResultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function stopStemAudioPreview() {
  if (activeStemAudio) {
    activeStemAudio.pause();
    activeStemAudio.src = '';
    activeStemAudio = null;
  }
  if (activePlayingBtn) {
    activePlayingBtn.classList.remove('playing');
    const label = activePlayingBtn.querySelector('.preview-label');
    if (label) label.innerText = 'Preview';
    const icon = activePlayingBtn.querySelector('svg');
    if (icon) {
      icon.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"></polygon>';
    }
    activePlayingBtn = null;
  }
}

function toggleStemAudioPreview(url, btn) {
  // If clicking on already playing stem, stop it
  if (activePlayingBtn === btn && activeStemAudio && !activeStemAudio.paused) {
    stopStemAudioPreview();
    return;
  }

  // Stop any other preview
  stopStemAudioPreview();

  // Also pause main app player if playing to avoid audio clash
  if (elements.audioElement && !elements.audioElement.paused) {
    elements.audioElement.pause();
  }

  activeStemAudio = new Audio(url);
  activePlayingBtn = btn;
  btn.classList.add('playing');

  const label = btn.querySelector('.preview-label');
  if (label) label.innerText = 'Pause';
  const icon = btn.querySelector('svg');
  if (icon) {
    icon.innerHTML = '<rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect>';
  }

  activeStemAudio.play().catch(err => {
    console.warn('Stem audio preview failed:', err);
    stopStemAudioPreview();
  });

  activeStemAudio.addEventListener('ended', () => {
    stopStemAudioPreview();
  });
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

    // Highlight button success
    btn.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polyline points="20 6 9 17 4 12"></polyline>
      </svg>
      Inspecting Active
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
