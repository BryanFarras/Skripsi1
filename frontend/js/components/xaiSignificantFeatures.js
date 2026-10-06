// ==========================================================================
// Component: XAI TreeSHAP Significant Features
// Handles: Attributions Toolbar, Filter Chips, Interactive Feature Item Cards
// ==========================================================================

export class SignificantFeaturesManager {
  constructor(state) {
    this.state = state;
    this.dom = {
      featureList: document.getElementById('featureList'),
      filterAll: document.getElementById('filterAll'),
      filterAi: document.getElementById('filterAi'),
      filterHuman: document.getElementById('filterHuman')
    };

    this.init();
  }

  init() {
    this.dom.filterAll?.addEventListener('click', () => {
      this.setActiveFilter('all', this.dom.filterAll);
    });

    this.dom.filterAi?.addEventListener('click', () => {
      this.setActiveFilter('ai', this.dom.filterAi);
    });

    this.dom.filterHuman?.addEventListener('click', () => {
      this.setActiveFilter('human', this.dom.filterHuman);
    });
  }

  setActiveFilter(filter, activeChip) {
    this.state.activeFilter = filter;
    [this.dom.filterAll, this.dom.filterAi, this.dom.filterHuman].forEach(c => c?.classList.remove('active'));
    activeChip?.classList.add('active');
    this.render();
  }

  render(xaiData = null) {
    if (xaiData) {
      this.state.xai = xaiData;
    }

    if (!this.state.xai || !this.dom.featureList) return;

    const topPos = this.state.xai.top_positive_features || [];
    const topNeg = this.state.xai.top_negative_features || [];

    let items = [];
    if (this.state.activeFilter === 'ai') {
      items = topPos;
    } else if (this.state.activeFilter === 'human') {
      items = topNeg;
    } else {
      // Interleaved sorted by absolute magnitude
      items = [...topPos, ...topNeg].sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value));
    }

    if (items.length === 0) {
      this.dom.featureList.innerHTML = '<div style="color: var(--c-blue-steel); padding: 16px; text-align: center;">No features match the selected filter.</div>';
      return;
    }

    const maxAbsShap = Math.max(...items.map(f => Math.abs(f.shap_value)), 0.05);

    let html = '';
    items.forEach((item, idx) => {
      const isAi = item.shap_value >= 0;
      const absVal = Math.abs(item.shap_value);
      const barPct = Math.min(100, (absVal / maxAbsShap) * 100);
      const sign = isAi ? '+' : '';
      const pillClass = isAi ? 'pill-ai' : 'pill-human';
      const barClass = isAi ? 'fill-ai' : 'fill-human';
      const cueClass = isAi ? '' : 'cue-human';

      html += `
        <div class="feature-item-row" data-feature="${item.feature}">
          <div class="feature-row-top">
            <div class="feature-identity">
              <span class="feature-rank">#${idx + 1}</span>
              <span class="feature-title-name">${item.title}</span>
              <span class="feature-code-tag">${item.feature}</span>
            </div>
            <div class="feature-metrics-badge">
              <span class="feature-raw-val">Raw: ${item.raw_value.toFixed(2)} ${item.unit || ''}</span>
              <span class="shap-pill ${pillClass}">${sign}${item.shap_value.toFixed(4)}</span>
            </div>
          </div>

          <div class="feature-bar-track">
            <div class="feature-bar-fill ${barClass}" style="width: ${barPct.toFixed(1)}%;"></div>
          </div>

          <div class="feature-sub-boxes">
            <div class="feature-box box-cue ${cueClass}">
              <strong>Diagnostic Cue:</strong> ${item.forensic_cue}
            </div>
            <div class="feature-box box-meaning">
              <strong>Physical Meaning:</strong> ${item.explanation}
            </div>
          </div>
        </div>
      `;
    });

    this.dom.featureList.innerHTML = html;
  }
}
