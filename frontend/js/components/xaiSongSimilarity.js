// ==========================================================================
// Component: XAI Song Similarity Search & Internet Audio Matching
// Based on Similarity Song Finder, DISCO.ac, and Shazam Landmark Retrieval
// Retrieves real matching songs from the web with artwork, previews, & explanations
// ==========================================================================

export class SongSimilarityManager {
  constructor(state, onAnalyzeTrack = null) {
    this.state = state;
    this.onAnalyzeTrack = onAnalyzeTrack;
    this.currentPreviewUrl = null;
    this.previewAudio = new Audio();
    this.currentData = null;
    this.currentMatches = [];
    this.activeFilter = 'all';

    this.dom = {
      similarityCard: document.getElementById('similarityCard'),
      simMatchesList: document.getElementById('simMatchesList'),
      simSearchInput: document.getElementById('simSearchInput'),
      simSearchBtn: document.getElementById('simSearchBtn'),
      simAcousticTag: document.getElementById('simAcousticTag'),
      simStatusCue: document.getElementById('simStatusCue'),
      simDescriptorsBar: document.getElementById('simDescriptorsBar'),
      simValTempo: document.getElementById('simValTempo'),
      simValKey: document.getElementById('simValKey'),
      simValVocal: document.getElementById('simValVocal'),
      simValPeaks: document.getElementById('simValPeaks'),
      simFilterTabs: document.getElementById('simFilterTabs')
    };

    this.init();
  }

  init() {
    // 1. Search button and input Enter key
    this.dom.simSearchBtn?.addEventListener('click', (e) => {
      e.stopPropagation();
      this.performSearch();
    });

    this.dom.simSearchInput?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        this.performSearch();
      }
    });

    // 2. Filter tabs click handling
    this.dom.simFilterTabs?.querySelectorAll('.sim-filter-btn').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        this.dom.simFilterTabs.querySelectorAll('.sim-filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.activeFilter = btn.dataset.filter || 'all';
        this.filterAndRenderMatches();
      });
    });

    // 3. Audio preview events
    this.previewAudio.addEventListener('ended', () => {
      this.resetPreviewIcons();
    });
    this.previewAudio.addEventListener('pause', () => {
      this.resetPreviewIcons();
    });
  }

  resetPreviewIcons() {
    this.currentPreviewUrl = null;
    const btns = this.dom.simMatchesList?.querySelectorAll('.btn-track-preview');
    btns?.forEach((btn) => {
      btn.innerHTML = '<polygon points="5 3 19 12 5 21 5 3" fill="currentColor"></polygon>';
      btn.classList.remove('playing');
    });
  }

  togglePreview(previewUrl, btnEl) {
    if (!previewUrl) {
      alert('Audio preview not available for this track.');
      return;
    }

    if (this.currentPreviewUrl === previewUrl && !this.previewAudio.paused) {
      this.previewAudio.pause();
      this.resetPreviewIcons();
    } else {
      this.previewAudio.src = previewUrl;
      this.previewAudio.play().then(() => {
        this.resetPreviewIcons();
        this.currentPreviewUrl = previewUrl;
        btnEl.innerHTML = '<rect x="6" y="4" width="4" height="16" fill="currentColor"></rect><rect x="14" y="4" width="4" height="16" fill="currentColor"></rect>';
        btnEl.classList.add('playing');
      }).catch(err => {
        console.warn('Audio preview play error:', err);
      });
    }
  }

  async performSearch(customQuery = null) {
    const query = customQuery !== null 
      ? customQuery 
      : (this.dom.simSearchInput?.value.trim() || '');

    if (this.dom.simStatusCue) {
      this.dom.simStatusCue.innerText = 'Searching internet audio database...';
    }

    try {
      const fileId = this.state.activeFileId || '';
      const params = new URLSearchParams();
      if (fileId) params.append('file_id', fileId);
      if (query) params.append('query', query);

      const res = await fetch(`/api/similar-songs?${params.toString()}`);
      if (!res.ok) throw new Error('Search failed');

      const data = await res.json();
      this.render(data);
    } catch (err) {
      console.warn('Similarity search error:', err);
      if (this.dom.simStatusCue) {
        this.dom.simStatusCue.innerText = 'Internet match retrieval standby.';
      }
    }
  }

  render(similarityData = null) {
    if (!similarityData || !this.dom.simMatchesList) return;

    this.currentData = similarityData;
    this.currentMatches = similarityData.matches || [];

    const profile = similarityData.profile || {};

    // 1. Update live acoustic descriptor pills (DISCO & Shazam features)
    if (this.dom.simValTempo && profile.tempo_bpm) {
      this.dom.simValTempo.innerText = `${profile.tempo_bpm} BPM`;
    }
    if (this.dom.simValKey && profile.musical_key) {
      this.dom.simValKey.innerText = profile.musical_key;
    }
    if (this.dom.simValVocal && profile.vocal_presence) {
      // Shorten string e.g. "Vocal-Forward (65%)" or "Instrumental (15%)"
      this.dom.simValVocal.innerText = profile.vocal_presence;
    }
    if (this.dom.simValPeaks && profile.constellation_density) {
      this.dom.simValPeaks.innerText = profile.constellation_density;
    }

    // 2. Update acoustic profile tag in card header
    if (this.dom.simAcousticTag) {
      this.dom.simAcousticTag.innerText = `${profile.vibe || 'Acoustic Match'} • ${profile.frequency_rolloff || 'Rolloff'}`;
    }

    // 3. Update status cue line
    if (this.dom.simStatusCue) {
      const q = similarityData.query || profile.search_term || 'Seed Audio';
      this.dom.simStatusCue.innerText = `Internet matches for: "${q}"`;
    }

    // 4. Render matching track cards with active filter applied
    this.filterAndRenderMatches();
  }

  filterAndRenderMatches() {
    if (!this.dom.simMatchesList) return;

    let filtered = this.currentMatches;
    if (this.activeFilter === 'twin') {
      filtered = this.currentMatches.filter(m => m.match_type === 'Acoustic Twin');
    } else if (this.activeFilter === 'tempo') {
      filtered = this.currentMatches.filter(m => m.match_type === 'Same Tempo');
    } else if (this.activeFilter === 'key') {
      filtered = this.currentMatches.filter(m => m.match_type === 'Same Key');
    } else if (this.activeFilter === 'vibe') {
      filtered = this.currentMatches.filter(m => m.match_type === 'Vibe Sibling');
    }

    if (!filtered || filtered.length === 0) {
      const filterLabel = {
        all: 'audio matches',
        twin: 'Acoustic Twins',
        tempo: 'Same Tempo matches',
        key: 'Same Key matches',
        vibe: 'Vibe Siblings'
      }[this.activeFilter] || 'tracks';

      this.dom.simMatchesList.innerHTML = `
        <div class="sim-empty-state">
          <span>No ${filterLabel} found for this audio query. Switch to "All Matches" to view all acoustic recommendations.</span>
        </div>
      `;
      return;
    }

    let html = '';
    filtered.forEach((m) => {
      const art = m.artwork || 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="60" height="60" fill="%23003049"><rect width="100%" height="100%" fill="%23FDF0D5"/><text x="50%" y="55%" font-size="20" text-anchor="middle" fill="%23003049">♫</text></svg>';
      const score = m.similarity_score || 85;

      // Color coding for match type badge
      let badgeClass = 'badge-broader';
      if (m.match_type === 'Acoustic Twin') badgeClass = 'badge-twin';
      else if (m.match_type === 'Same Tempo') badgeClass = 'badge-tempo';
      else if (m.match_type === 'Same Key') badgeClass = 'badge-key';
      else if (m.match_type === 'Vibe Sibling') badgeClass = 'badge-vibe';

      html += `
        <div class="sim-track-card">
          <div class="sim-track-art-wrap">
            <img src="${art}" alt="${m.title}" class="sim-track-artwork" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'50\\' height=\\'50\\' fill=\\'%23003049\\'><rect width=\\'100%\\' height=\\'100%\\' fill=\\'%23FAF2DF\\'/><text x=\\'50%\\' y=\\'60%\\' font-size=\\'22\\' text-anchor=\\'middle\\' fill=\\'%23003049\\'>♫</text></svg>'" />
            ${m.preview_url ? `
              <button type="button" class="btn-track-preview" data-url="${m.preview_url}" title="Listen to 30s Audio Preview">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                  <polygon points="5 3 19 12 5 21 5 3"></polygon>
                </svg>
              </button>
            ` : ''}
          </div>

          <div class="sim-track-info">
            <div class="sim-track-top-row">
              <span class="sim-track-title" title="${m.title}">${m.title}</span>
              <div class="sim-badges-group">
                <span class="sim-type-badge ${badgeClass}">${m.match_type || 'Match'}</span>
                <span class="sim-match-badge">${score}% Match</span>
              </div>
            </div>

            <div class="sim-track-artist-row">
              <span class="sim-track-artist">${m.artist}</span>
              <span class="sim-meta-dot">•</span>
              <span class="sim-track-genre">${m.genre || 'Electronic'}</span>
              ${m.bpm ? `<span class="sim-meta-dot">•</span><span class="sim-track-spec">${m.bpm} BPM</span>` : ''}
              ${m.key ? `<span class="sim-meta-dot">•</span><span class="sim-track-spec">${m.key}</span>` : ''}
            </div>

            <p class="sim-why-text" title="${m.why_similar}">
              <strong class="sim-why-label">Why similar:</strong> ${m.why_similar}
            </p>
          </div>

          <div class="sim-track-actions-group">
            ${m.preview_url ? `
              <button type="button" class="btn-analyze-similar" data-url="${m.preview_url}" title="Directly run full AI forensic detection on this track">
                Analyze ⚡
              </button>
            ` : ''}
            ${m.external_url ? `
              <a href="${m.external_url}" target="_blank" rel="noopener noreferrer" class="btn-sim-external" title="Open recording online">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                  <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
                  <polyline points="15 3 21 3 21 9"></polyline>
                  <line x1="10" y1="14" x2="21" y2="3"></line>
                </svg>
              </a>
            ` : ''}
          </div>
        </div>
      `;
    });

    this.dom.simMatchesList.innerHTML = html;

    // Attach preview play listeners
    this.dom.simMatchesList.querySelectorAll('.btn-track-preview').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const url = btn.getAttribute('data-url');
        this.togglePreview(url, btn);
      });
    });

    // Attach analyze track listeners
    this.dom.simMatchesList.querySelectorAll('.btn-analyze-similar').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const url = btn.getAttribute('data-url');
        if (url && this.onAnalyzeTrack) {
          this.onAnalyzeTrack(url);
        }
      });
    });
  }
}
