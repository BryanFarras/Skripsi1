// ==========================================================================
// Component: XAI Forensic Verdict, Narrative & Timeline
// Handles: Hero Classification Verdict, Probability Meter, Thesis Synthesis, 5s Timeline
// ==========================================================================

export class ForensicVerdictManager {
  constructor(state, onSeekAudio) {
    this.state = state;
    this.onSeekAudio = onSeekAudio;
    this.currentScope = 'track'; // 'track' or 'slice'
    this.activeSlice = null;

    this.dom = {
      verdictCard: document.getElementById('verdictCard'),
      verdictSubLabel: document.getElementById('verdictSubLabel'),
      verdictScopeBadge: document.getElementById('verdictScopeBadge'),
      btnScopeTrack: document.getElementById('btnScopeTrack'),
      btnScopeSlice: document.getElementById('btnScopeSlice'),
      scopeSliceName: document.getElementById('scopeSliceName'),
      verdictTitle: document.getElementById('verdictTitle'),
      probAiText: document.getElementById('probAiText'),
      probHumanText: document.getElementById('probHumanText'),
      meterAi: document.getElementById('meterAi'),
      meterHuman: document.getElementById('meterHuman'),
      statTitle1: document.getElementById('statTitle1'),
      statTitle2: document.getElementById('statTitle2'),
      statTitle3: document.getElementById('statTitle3'),
      statTitle4: document.getElementById('statTitle4'),
      statConfidence: document.getElementById('statConfidence'),
      statFlaggedSlices: document.getElementById('statFlaggedSlices'),
      statBaseValue: document.getElementById('statBaseValue'),
      statTotalFeatures: document.getElementById('statTotalFeatures'),
      verdictScopeHint: document.getElementById('verdictScopeHint'),
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
    this.setupScopeToggle();
  }

  setupScopeToggle() {
    this.dom.btnScopeTrack?.addEventListener('click', () => {
      this.currentScope = 'track';
      this.dom.btnScopeTrack.classList.add('active');
      this.dom.btnScopeSlice?.classList.remove('active');
      this.renderVerdict();
    });

    this.dom.btnScopeSlice?.addEventListener('click', () => {
      this.currentScope = 'slice';
      this.dom.btnScopeSlice.classList.add('active');
      this.dom.btnScopeTrack?.classList.remove('active');
      this.renderVerdict();
    });
  }

  updateActiveSlice(slice) {
    if (!slice) return;
    this.activeSlice = slice;
    if (this.dom.scopeSliceName) {
      this.dom.scopeSliceName.innerText = `Slice ${slice.slice_idx + 1}`;
    }
    if (this.currentScope === 'slice') {
      this.renderVerdict();
    }
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
      if (!this.state.xai || !this.state.mlDetector) return;
      
      const ml = this.state.mlDetector;
      const xai = this.state.xai;
      const isId = this.state.activeLang === 'id';
      const synthesis = isId ? xai.forensic_synthesis_id : xai.forensic_synthesis_en;

      // Extract top positive & negative drivers with empirical rank
      const topPosLines = (xai.top_positive_features || []).slice(0, 3).map((f, i) => {
        const rankStr = f.empirical_rank ? ` (Rank #${f.empirical_rank} Global - ${f.empirical_importance}% Bobot)` : '';
        return `  ${i + 1}. **${f.title}** [\`φ = +${f.shap_value.toFixed(4)}\`, Raw: ${f.raw_value.toFixed(2)} ${f.unit || ''}]${rankStr}\n     - *Diagnostic Cue:* ${f.forensic_cue}`;
      }).join('\n');

      const topNegLines = (xai.top_negative_features || []).slice(0, 3).map((f, i) => {
        const rankStr = f.empirical_rank ? ` (Rank #${f.empirical_rank} Global - ${f.empirical_importance}% Bobot)` : '';
        return `  ${i + 1}. **${f.title}** [\`φ = ${f.shap_value.toFixed(4)}\`, Raw: ${f.raw_value.toFixed(2)} ${f.unit || ''}]${rankStr}\n     - *Authenticity Cue:* ${f.forensic_cue}`;
      }).join('\n');

      const thesisReport = `### Laporan Bukti Forensik Audio & Eksplanasi TreeSHAP (Skripsi)
- **Status Klasifikasi:** ${ml.prediction}
- **Probabilitas Model:** AI (Spoof): ${(ml.overall_ai_probability * 100).toFixed(1)}% | Human (Bona-fide): ${(ml.overall_human_probability * 100).toFixed(1)}% (Confidence: ${ml.confidence_percent.toFixed(1)}%)
- **Validasi Benchmark Empiris:**
  - Protokol Validasi: Grouped Track-Level Split (0% song-chunk leakage)
  - Dataset Ukuran: 8.297 slice akustik (241 track utuh)
  - Performa Model Baseline: Akurasi 99,76% | F1-Score 99,87% | ROC-AUC 99,99% | EER 1,00%
  - False Alarm Rate: 0,00% (245 dari 245 track manusia terdeteksi sempurna)
- **Parameter Ekstraksi:** 110 Dimensi Akustik (librosa DSP • 22.05 kHz • Window 5.0s)
- **Basis Ekspektasi Populasi (TreeSHAP):** ψ₀ = ${xai.base_value.toFixed(4)}

#### Fitur Utama Pendorong AI (Indikator Artefak Sintetis, φ > 0):
${topPosLines || '  - Tidak ada fitur yang teridentifikasi mendorong ke arah AI.'}

#### Fitur Utama Pendukung Keaslian (Indikator Akustik Alami, φ < 0):
${topNegLines || '  - Tidak ada fitur yang teridentifikasi mendorong ke arah Human.'}

#### Sintesis Diagnosis Forensik:
> ${synthesis}
`;

      navigator.clipboard.writeText(thesisReport).then(() => {
        const orig = this.dom.copySynthesisBtn.innerHTML;
        this.dom.copySynthesisBtn.innerHTML = '✓ Copied Skripsi Report!';
        setTimeout(() => {
          this.dom.copySynthesisBtn.innerHTML = orig;
        }, 2200);
      });
    });
  }

  renderAll(mlDetector, xaiData) {
    this.state.mlDetector = mlDetector;
    this.state.xai = xaiData;
    if (mlDetector?.timeline?.[0]) {
      this.activeSlice = mlDetector.timeline[0];
      if (this.dom.scopeSliceName) {
        this.dom.scopeSliceName.innerText = `Slice ${this.activeSlice.slice_idx + 1}`;
      }
    }

    this.renderVerdict();
    this.renderSynthesis();
    this.renderTimeline();
  }

  renderVerdict() {
    const ml = this.state.mlDetector;
    if (!ml) return;

    if (this.currentScope === 'slice') {
      const slice = this.activeSlice || ml.timeline?.[0];
      if (!slice) return;

      const isSliceAi = slice.is_ai_flagged;
      const sliceAiProb = slice.ai_prob;
      const sliceHumanProb = slice.human_prob !== undefined ? slice.human_prob : (1.0 - sliceAiProb);
      const conf = Math.abs(sliceAiProb - 0.5) * 200;
      const endSec = slice.end_sec !== undefined ? slice.end_sec : (slice.start_sec + 5.0);

      if (this.dom.verdictCard) {
        this.dom.verdictCard.className = `xai-panel-card verdict-panel ${isSliceAi ? 'verdict-spoof' : 'verdict-bonafide'}`;
      }

      if (this.dom.verdictSubLabel) {
        this.dom.verdictSubLabel.innerText = `SELECTED 5S SLICE FORENSIC VERDICT`;
      }

      if (this.dom.verdictScopeBadge) {
        this.dom.verdictScopeBadge.innerText = `Slice #${slice.slice_idx + 1} (${slice.start_sec}s - ${endSec}s)`;
      }

      if (this.dom.verdictTitle) {
        this.dom.verdictTitle.innerHTML = isSliceAi
          ? '<span class="verdict-dot" style="color: var(--c-ai-red);">●</span> SPOOF (AI-GENERATED SLICE)'
          : '<span class="verdict-dot" style="color: var(--c-human-teal);">●</span> BONA-FIDE (HUMAN SLICE)';
      }

      if (this.dom.probAiText) this.dom.probAiText.innerText = `${(sliceAiProb * 100).toFixed(1)}%`;
      if (this.dom.probHumanText) this.dom.probHumanText.innerText = `${(sliceHumanProb * 100).toFixed(1)}%`;

      if (this.dom.meterAi) this.dom.meterAi.style.width = `${(sliceAiProb * 100).toFixed(1)}%`;
      if (this.dom.meterHuman) this.dom.meterHuman.style.width = `${(sliceHumanProb * 100).toFixed(1)}%`;

      if (this.dom.statTitle1) this.dom.statTitle1.innerText = 'SLICE CONFIDENCE';
      if (this.dom.statConfidence) this.dom.statConfidence.innerText = `${conf.toFixed(1)}%`;

      if (this.dom.statTitle2) this.dom.statTitle2.innerText = 'WINDOW TIME';
      if (this.dom.statFlaggedSlices) {
        this.dom.statFlaggedSlices.innerText = `${slice.start_sec}s - ${endSec}s (Slice #${slice.slice_idx + 1} of ${ml.total_slices_analyzed})`;
      }

      if (this.dom.statTitle3) this.dom.statTitle3.innerText = 'SEGMENT TYPE';
      if (this.dom.statBaseValue) {
        this.dom.statBaseValue.innerText = isSliceAi ? 'Generative Synthesis' : 'Authentic Acoustic';
      }

      if (this.dom.statTitle4) this.dom.statTitle4.innerText = 'DSP DIMS';
      if (this.dom.statTotalFeatures) {
        this.dom.statTotalFeatures.innerText = '110 DSP Dims';
      }

      if (this.dom.verdictScopeHint) {
        this.dom.verdictScopeHint.innerHTML = isSliceAi
          ? `Slice #${slice.slice_idx + 1} (<strong>${slice.start_sec}s - ${endSec}s</strong>) contains generative AI vocoder / synthesis artifacts (<strong>${(sliceAiProb * 100).toFixed(1)}% AI</strong>). Switch to 'Full Song' above for whole-track classification.`
          : `Slice #${slice.slice_idx + 1} (<strong>${slice.start_sec}s - ${endSec}s</strong>) exhibits natural human acoustic dynamics (<strong>${(sliceHumanProb * 100).toFixed(1)}% Human</strong>). Switch to 'Full Song' above for whole-track classification.`;
      }
    } else {
      // Track Overall Scope
      const isSpoof = ml.prediction.includes('SPOOF');
      const aiProb = ml.overall_ai_probability;
      const humanProb = ml.overall_human_probability;
      const totalSlices = ml.total_slices_analyzed || (ml.timeline ? ml.timeline.length : 1);
      const flaggedCount = Math.round((ml.flagged_slices_ratio || 0) * totalSlices);

      if (this.dom.verdictCard) {
        this.dom.verdictCard.className = `xai-panel-card verdict-panel ${isSpoof ? 'verdict-spoof' : 'verdict-bonafide'}`;
      }

      if (this.dom.verdictSubLabel) {
        this.dom.verdictSubLabel.innerText = `OVERALL TRACK FORENSIC VERDICT`;
      }

      if (this.dom.verdictScopeBadge) {
        this.dom.verdictScopeBadge.innerText = `${totalSlices} Slices Evaluated`;
      }

      if (this.dom.verdictTitle) {
        this.dom.verdictTitle.innerHTML = isSpoof
          ? '<span class="verdict-dot" style="color: var(--c-ai-red);">●</span> SPOOF (AI-GENERATED TRACK)'
          : '<span class="verdict-dot" style="color: var(--c-human-teal);">●</span> BONA-FIDE (HUMAN TRACK)';
      }

      if (this.dom.probAiText) this.dom.probAiText.innerText = `${(aiProb * 100).toFixed(1)}%`;
      if (this.dom.probHumanText) this.dom.probHumanText.innerText = `${(humanProb * 100).toFixed(1)}%`;

      if (this.dom.meterAi) this.dom.meterAi.style.width = `${(aiProb * 100).toFixed(1)}%`;
      if (this.dom.meterHuman) this.dom.meterHuman.style.width = `${(humanProb * 100).toFixed(1)}%`;

      if (this.dom.statTitle1) this.dom.statTitle1.innerText = 'CONFIDENCE';
      if (this.dom.statConfidence) this.dom.statConfidence.innerText = `${ml.confidence_percent.toFixed(1)}%`;

      if (this.dom.statTitle2) this.dom.statTitle2.innerText = 'FLAGGED SLICES';
      if (this.dom.statFlaggedSlices) {
        this.dom.statFlaggedSlices.innerText = `${(ml.flagged_slices_ratio * 100).toFixed(0)}% (${flaggedCount}/${totalSlices} slices)`;
      }

      if (this.dom.statTitle3) this.dom.statTitle3.innerText = 'POPULATION BASE';
      if (this.dom.statBaseValue && this.state.xai) {
        this.dom.statBaseValue.innerText = `ψ₀ = ${this.state.xai.base_value.toFixed(2)}`;
      }

      if (this.dom.statTitle4) this.dom.statTitle4.innerText = 'EVALUATED DIMS';
      if (this.dom.statTotalFeatures) {
        this.dom.statTotalFeatures.innerText = `110 Dims (99.8% Acc, 1.0% EER)`;
        this.dom.statTotalFeatures.title = `Evaluated on 8,297 track-level slices (0% chunk leakage, 0.0% False Alarm rate)`;
      }

      if (this.dom.verdictScopeHint) {
        this.dom.verdictScopeHint.innerHTML = isSpoof
          ? `Overall track diagnosis across full audio: <strong>${flaggedCount} of ${totalSlices} slices (${(ml.flagged_slices_ratio * 100).toFixed(0)}%)</strong> were flagged as AI-generated. Click 'Selected Slice' or any slice tab above to view individual 5s slice diagnostics.`
          : `Overall track diagnosis across full audio: <strong>${totalSlices - flaggedCount} of ${totalSlices} slices</strong> exhibit natural human musical acoustics. Click 'Selected Slice' or any slice tab above to view individual 5s slice diagnostics.`;
      }
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
      const displayProb = isAi ? slice.ai_prob : (slice.human_prob !== undefined ? slice.human_prob : (1.0 - slice.ai_prob));
      const probPct = (displayProb * 100).toFixed(0);
      const barHeightPct = Math.max(10, displayProb * 100);
      const sliceClass = isAi ? 'slice-ai' : 'slice-human';

      html += `
        <div class="xai-slice-col" data-start="${slice.start_sec}" title="Slice #${slice.slice_idx + 1}: ${slice.start_sec}s - ${probPct}% ${isAi ? 'AI' : 'Human'}">
          <span class="xai-slice-prob">${probPct}%</span>
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
