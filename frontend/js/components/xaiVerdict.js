// ==========================================================================
// Component: XAI Forensic Verdict, Narrative & Timeline
// Handles: Hero Classification Verdict, Probability Meter, Thesis Synthesis, 5s Timeline
// ==========================================================================

export class ForensicVerdictManager {
  constructor(state, onSeekAudio) {
    this.state = state;
    this.onSeekAudio = onSeekAudio;

    this.dom = {
      verdictCard: document.getElementById('verdictCard'),
      verdictTitle: document.getElementById('verdictTitle'),
      probAiText: document.getElementById('probAiText'),
      probHumanText: document.getElementById('probHumanText'),
      meterAi: document.getElementById('meterAi'),
      meterHuman: document.getElementById('meterHuman'),
      statConfidence: document.getElementById('statConfidence'),
      statFlaggedSlices: document.getElementById('statFlaggedSlices'),
      statBaseValue: document.getElementById('statBaseValue'),
      statTotalFeatures: document.getElementById('statTotalFeatures'),
      synthesisText: document.getElementById('synthesisText'),
      langBtnId: document.getElementById('langBtnId'),
      langBtnEn: document.getElementById('langBtnEn'),
      copySynthesisBtn: document.getElementById('copySynthesisBtn'),
      timelineChart: document.getElementById('timelineChart')
    };

    this.init();
  }

  init() {
    this.setupLanguageToggle();
    this.setupCopyButton();
  }

  setupLanguageToggle() {
    this.dom.langBtnId?.addEventListener('click', () => {
      this.state.activeLang = 'id';
      this.dom.langBtnId.classList.add('active');
      this.dom.langBtnEn?.classList.remove('active');
      this.renderSynthesis();
    });

    this.dom.langBtnEn?.addEventListener('click', () => {
      this.state.activeLang = 'en';
      this.dom.langBtnEn.classList.add('active');
      this.dom.langBtnId?.classList.remove('active');
      this.renderSynthesis();
    });
  }

  setupCopyButton() {
    this.dom.copySynthesisBtn?.addEventListener('click', () => {
      if (!this.state.xai) return;
      const text = this.state.activeLang === 'id' 
        ? this.state.xai.forensic_synthesis_id 
        : this.state.xai.forensic_synthesis_en;

      navigator.clipboard.writeText(text).then(() => {
        const orig = this.dom.copySynthesisBtn.innerHTML;
        this.dom.copySynthesisBtn.innerHTML = '✓ Copied to Clipboard!';
        setTimeout(() => {
          this.dom.copySynthesisBtn.innerHTML = orig;
        }, 2000);
      });
    });
  }

  renderAll(mlDetector, xaiData) {
    this.state.mlDetector = mlDetector;
    this.state.xai = xaiData;

    this.renderVerdict();
    this.renderSynthesis();
    this.renderTimeline();
  }

  renderVerdict() {
    const ml = this.state.mlDetector;
    if (!ml) return;

    const isSpoof = ml.prediction.includes('SPOOF');
    const aiProb = ml.overall_ai_probability;
    const humanProb = ml.overall_human_probability;

    if (this.dom.verdictCard) {
      this.dom.verdictCard.className = `xai-panel-card verdict-panel ${isSpoof ? 'verdict-spoof' : 'verdict-bonafide'}`;
    }

    if (this.dom.verdictTitle) {
      this.dom.verdictTitle.innerHTML = isSpoof
        ? '<span class="verdict-dot" style="color: var(--c-ai-red);">●</span> SPOOF (AI-GENERATED)'
        : '<span class="verdict-dot" style="color: var(--c-human-teal);">●</span> BONA-FIDE (HUMAN)';
    }

    if (this.dom.probAiText) this.dom.probAiText.innerText = `${(aiProb * 100).toFixed(1)}%`;
    if (this.dom.probHumanText) this.dom.probHumanText.innerText = `${(humanProb * 100).toFixed(1)}%`;

    if (this.dom.meterAi) this.dom.meterAi.style.width = `${(aiProb * 100).toFixed(1)}%`;
    if (this.dom.meterHuman) this.dom.meterHuman.style.width = `${(humanProb * 100).toFixed(1)}%`;

    if (this.dom.statConfidence) this.dom.statConfidence.innerText = `${ml.confidence_percent.toFixed(1)}%`;
    if (this.dom.statFlaggedSlices) {
      this.dom.statFlaggedSlices.innerText = `${(ml.flagged_slices_ratio * 100).toFixed(0)}% (${ml.total_slices_analyzed} slices)`;
    }
    if (this.dom.statBaseValue && this.state.xai) {
      this.dom.statBaseValue.innerText = `ψ₀ = ${this.state.xai.base_value.toFixed(2)}`;
    }
  }

  renderSynthesis() {
    if (!this.state.xai || !this.dom.synthesisText) return;
    const text = this.state.activeLang === 'id' 
      ? this.state.xai.forensic_synthesis_id 
      : this.state.xai.forensic_synthesis_en;

    const formatted = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    this.dom.synthesisText.innerHTML = formatted;
  }

  renderTimeline() {
    if (!this.state.mlDetector || !this.dom.timelineChart) return;
    const timeline = this.state.mlDetector.timeline || [];
    if (timeline.length === 0) return;

    let html = '';
    timeline.forEach((slice) => {
      const isAi = slice.is_ai_flagged;
      const barHeightPct = Math.max(10, slice.ai_prob * 100);
      const sliceClass = isAi ? 'slice-ai' : 'slice-human';

      html += `
        <div class="xai-slice-col" data-start="${slice.start_sec}" title="Slice #${slice.slice_idx + 1}: ${slice.start_sec}s - ${(slice.ai_prob * 100).toFixed(1)}% AI">
          <span class="xai-slice-prob">${(slice.ai_prob * 100).toFixed(0)}%</span>
          <div class="xai-slice-bar-wrap">
            <div class="xai-slice-bar ${sliceClass}" style="height: ${barHeightPct}%;"></div>
          </div>
          <span class="xai-slice-time">${slice.start_sec}s</span>
        </div>
      `;
    });

    this.dom.timelineChart.innerHTML = html;

    this.dom.timelineChart.querySelectorAll('.xai-slice-col').forEach(col => {
      col.addEventListener('click', () => {
        const st = parseFloat(col.getAttribute('data-start') || '0');
        if (this.onSeekAudio) {
          this.onSeekAudio(st);
        }
      });
    });
  }
}
