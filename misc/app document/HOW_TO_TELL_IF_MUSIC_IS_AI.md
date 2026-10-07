# FORENSIC COMPENDIUM: HOW TO TELL IF MUSIC IS AI-GENERATED
## Theoretical Foundations, Modern Generative Pipelines, and Multi-Domain Detection Methodologies

---

### EXECUTIVE ABSTRACT & SCOPE
With the emergence of commercial audio generative systems (Suno v3/v4, Udio 130k/v1.5) and state-of-the-art open architectures (ACE-Step 1.5, Stable Audio Open, MusicGen), synthetic music generation has transitioned from primitive symbolic MIDI stitching to end-to-end continuous acoustic latent diffusion. Today's AI models generate multi-instrumental, vocalized, and mastered stereo tracks directly from text prompts. 

However, despite reaching human-level perceptual plausibility on brief listening, **all synthetic audio leaves indelible mathematical, physical, and compositional artifacts**. This document provides an exhaustive, engineering-grade technical analysis of:
1. **The Generative Engine**: How modern AI music is generated (analyzing the exact architectural pipeline of **ACE-Step 1.5**, Suno, and Udio).
2. **The Forensic Signatures**: Every physical acoustic, digital signal processing (DSP), temporal, harmonic, and linguistic artifact that exposes AI music.
3. **The Diagnostic Toolkit**: Visual spectrogram analysis, algorithmic metrics, and machine learning detection frameworks (CQT-ResNet, AASIST, WavLM, CLAD).

---

# PART I: HOW MODERN AI MUSIC IS GENERATED

To detect synthetic music with forensic accuracy, an engineer must first understand the physics and machine learning mechanics of the synthesis pipeline. Modern generative music does not sample real WAV files; it reconstructs continuous audio latents via probabilistic diffusion.

```
+---------------------------------------------------------------------------------------+
|                                MODERN GENERATIVE PIPELINE                             |
|                                                                                       |
|   User Prompt / Lyrics / Style Tags                                                   |
|             |                                                                         |
|             v                                                                         |
|   +-------------------------------------------------------------------------------+   |
|   | STAGE 1: Semantic & Structural Planner (Autoregressive LM @ 5Hz)              |   |
|   | - Tokenizes lyrics & estimates syllabic durations (phoneme timestamping)      |   |
|   | - Generates musical metadata: BPM, musical key, time signature, structure     |   |
|   | - Produces discrete semantic conditioning codes at low frame rate (~5Hz)      |   |
|   +-------------------------------------------------------------------------------+   |
|             |                                                                         |
|             | Conditioning Vectors (Text Embeddings + 5Hz Codes + Lyric Cross-Attn)   |
|             v                                                                         |
|   +-------------------------------------------------------------------------------+   |
|   | STAGE 2: Acoustic Latent Diffusion Engine (Diffusion Transformer - DiT)       |   |
|   | - Operates on continuous latent space compressed by an Audio VAE              |   |
|   | - Iterative Denoising: Gaussian Noise z_T ---> Denoised Latent z_0            |   |
|   | - Schedulers: Rectified Flow / Turbo (8 steps) vs SFT / DDIM (50 steps + CFG) |   |
|   +-------------------------------------------------------------------------------+   |
|             |                                                                         |
|             | Continuous Acoustic Latents z_0                                         |
|             v                                                                         |
|   +-------------------------------------------------------------------------------+   |
|   | STAGE 3: Audio Reconstruction & Neural Vocoding (Continuous Audio VAE)        |   |
|   | - Decoder (e.g., AutoencoderOobleck / HiFi-GAN / Snake ConvNets)              |   |
|   | - Transposed convolutions upsample latents (e.g. 2048x) to 44.1kHz/48kHz PCM  |   |
|   | - Multi-Band STFT loss reconstruction yields final stereo waveform            |   |
|   +-------------------------------------------------------------------------------+   |
+---------------------------------------------------------------------------------------+
```

---

### 1.1 Architectural Deep-Dive: The ACE-Step 1.5 Pipeline
The open-source **ACE-Step 1.5** architecture represents the current pinnacle of open-weights music generation. An inspection of `acestep/handler.py`, `llm_inference.py`, and `acestep_v15_pipeline.py` reveals the modern dual-stage paradigm:

#### A. Stage 1: The 5Hz Language Model Planner (`LLMHandler`)
* **Role**: High-level semantic planning and alignment.
* **Model Scale**: Pre-trained autoregressive transformers (0.6B, 1.7B, or 4B parameters based on Qwen architectures).
* **Frame Rate**: Operates at an ultra-low frame rate of **5 Hz** (one token represents 200 ms of musical time).
* **Tasks**:
  1. *Lyric Scoring & Timestamping*: Maps raw input lyric text into discrete phonetic tokens, assigning millisecond-level start and end timestamps (`LyricTimestampMixin`).
  2. *Metadata Constrained Generation*: Utilizes `MetadataConstrainedLogitsProcessor` to enforce consistent musical keys (e.g., C minor, F# Major), tempos (BPM ranges 60-180), and arrangement tags (Intro, Verse, Chorus, Bridge, Outro).
  3. *Semantic Code Output*: Emits discrete audio semantic codes that act as cross-attention anchors for the subsequent acoustic diffusion model.

#### B. Stage 2: The Diffusion Transformer (`AceStepHandler` / DiT)
* **Role**: Fine-grained acoustic synthesis in latent space.
* **Latent Compression Space**: Instead of computing diffusion over raw 44.1 kHz PCM audio ($44,100 \text{ samples/sec} \times 2 \text{ channels} = 88,200 \text{ values/sec}$, which is computationally intractable), diffusion occurs within a compressed continuous latent space.
* **Conditioning Mechanisms**:
  * Cross-attention over text prompt embeddings (via pre-trained text encoders like T5 or specialized audio-text models).
  * Cross-attention over the 5Hz semantic audio codes generated by the LM.
  * Adaptive Layer Normalization (AdaLN) conditioned on diffusion timestep $t$, style tags, and Classifier-Free Guidance (CFG) scale.
* **Denoising Dynamics**:
  * **Turbo Models**: Employ **Rectified Flow Matching** and **Shift Distillation** to generate 3-minute stereo music in just **8 steps**.
  * **SFT Models**: Standard diffusion formulations requiring **50 steps** with CFG scales between $3.0$ and $7.0$ for maximum harmonic fidelity.

#### C. Stage 3: The Neural Autoencoder / Vocoder (`AutoencoderOobleck`)
* **Role**: Reconstructing the raw continuous time-domain audio waveform $x \in \mathbb{R}^{2 \times T}$ from latent $z_0 \in \mathbb{R}^{C \times (T/H)}$.
* **Architecture**: Employs Stability AI's `AutoencoderOobleck` (or HiFi-GAN/BigVGAN-style neural vocoders).
* **Upsampling**: A sequence of 1D transposed convolutional layers with residual dilated blocks, upsampling the latent representation by temporal factors (e.g., $8 \times 8 \times 4 \times 4 = 1024$ or $2048$).
* **Activation Functions**: Snake activations ($\alpha + \frac{1}{\beta}\sin^2(\beta x)$) or PReLU designed for periodic signal synthesis.

---

### 1.2 Suno & Udio: Closed-Source Variations
1. **Suno (v3 / v3.5 / v4)**:
   * Uses an end-to-end latent diffusion pipeline coupled with proprietary acoustic codebooks.
   * Employs heavy post-processing and mastering filters built into the inference graph (multiband compression, stereo widening).
   * Generates up to $44.1\text{ kHz}$ or $48\text{ kHz}$, but early versions (v1-v2) strictly truncated audio at $16\text{ kHz}$ or $18\text{ kHz}$.
2. **Udio (130k / v1.5)**:
   * Generates music in contiguous 32-second chunks via masked audio inpainting diffusion.
   * Exhibits distinct vocoder timbre: exceptionally clean vocal formants but susceptible to boundary discontinuities and subtle phase smearing during track extensions.

---

# PART II: HOW TO TELL IF MUSIC IS AI OR NOT (THE FORENSIC BLUEPRINT)

When determining whether an audio file is human-composed/studio-recorded or machine-generated, forensic analysts and machine learning systems evaluate **five core diagnostic dimensions**:

```
+-----------------------------------------------------------------------------------+
|                        5-DIMENSIONAL FORENSIC TAXONOMY                            |
+-----------------------------------------------------------------------------------+
|  1. VISUAL & SPECTRAL DSP FORENSICS                                               |
|     - Brickwall cutoffs (16kHz / 18kHz / 20kHz)                                   |
|     - Vocoder checkerboard & transposed convolution horizontal striping           |
|     - High-Frequency (HF) Energy Collapse / Air Power Ratio                       |
|     - Spectral Flux smearing and transient softening                              |
+-----------------------------------------------------------------------------------+
|  2. TEMPORAL & RHYTHMIC ANOMALIES                                                 |
|     - The "Drifting / Wandering Beat" phenomenon (Dr. Neal Krawetz discovery)     |
|     - Micro-timing absence vs. unanchored tempo drift                             |
|     - Lack of true transient rise times (diffused drum impacts)                   |
+-----------------------------------------------------------------------------------+
|  3. STEREO FIELD & PHASE FORENSICS                                                |
|     - Phase Incoherence: "watery", "comb-filtered", "swirling" cymbal decay       |
|     - Mono collapse degradation (destructive phase cancellation when summed)      |
|     - Phantom stereo width (decorrelated noise instead of distinct mic placement) |
+-----------------------------------------------------------------------------------+
|  4. VOCAL TIMBRE & LINGUISTIC ARTIFACTS                                           |
|     - Formant morphing & uncanny micro-pitch quantization                         |
|     - Phantom breaths, mid-word gasps, and missing vocal cord closures            |
|     - Cliché LLM lyrical tropes & unnatural syllable stress patterns              |
+-----------------------------------------------------------------------------------+
|  5. HARMONIC, COMPOSITIONAL & ARRANGEMENT TELLS                                   |
|     - Voice leading violations & gradual microtonal pitch drift                   |
|     - Cross-talk / Spectral Bleeding (reverb tail modulates instrument stems)     |
|     - Dynamic flatlining (lack of true expressive arrangement arc)                |
+-----------------------------------------------------------------------------------+
```

---

### 2.1 DIMENSION 1: Visual & Spectral DSP Forensics
Spectral analysis using a Short-Time Fourier Transform (STFT) with a linear frequency axis ($0 \text{ Hz} - 24 \text{ kHz}$) is the single most definitive forensic tool.

#### A. The "Brickwall" Frequency Cutoff
* **The Mechanism**: To conserve VRAM and training compute, audio VAEs are often trained on audio downsampled to $32\text{ kHz}$ (Nyquist $= 16\text{ kHz}$), $36\text{ kHz}$ (Nyquist $= 18\text{ kHz}$), or $44.1\text{ kHz}$ with bandlimiting. When exported to a standard $44.1\text{ kHz}$ or $48\text{ kHz}$ WAV/MP3 container, zero-padding or upsampling interpolation occurs.
* **Visual Signature**: On a linear STFT spectrogram, natural studio recordings display continuous, decaying energy up to $22.05\text{ kHz}$ (cymbals, snare snap, room air). AI audio exhibits a **sharp, unnatural horizontal cutoff line** (e.g. exactly at $15.8\text{ kHz}$, $16\text{ kHz}$, $17.5\text{ kHz}$, or $19.2\text{ kHz}$) with near-zero energy (black/dark void) above it.
* **Mathematical Diagnostic**:
  $$\text{Cutoff Threshold} = \min_{f > 14\text{kHz}} \left( \frac{\int_{f}^{f_s/2} |X(f)|^2 df}{\int_{0}^{f} |X(f)|^2 df} < 10^{-4} \right)$$

#### B. Vocoder Checkerboard & High-Frequency Banding Artifacts
* **The Mechanism**: Neural vocoders (such as HiFi-GAN and BigVGAN) rely on 1D transposed convolutions for temporal upsampling. When kernel size is not perfectly divisible by stride, **uneven kernel overlap** occurs, injecting periodic temporal-frequency checkerboard artifacts.
* **Visual Signature**: Zooming into the $12\text{ kHz} - 20\text{ kHz}$ region reveals faint, repeating horizontal grid lines or unnatural equidistant harmonic ridges that do not correlate with any physical musical instrument's overtones.

#### C. Spectral Flatness & "Air Power" Ratio
* **The Mechanism**: Natural acoustic instruments produce exponentially decaying harmonic overtones that blend into natural thermal room noise. In contrast, diffusion models either over-smooth the high frequencies (yielding an abnormally low spectral centroid) or generate unstructured white noise latents in the top band.
* **Algorithmic Indicator**:
  $$\text{Air Power Ratio} = \frac{\sum_{k=f_{16\text{kHz}}}^{f_{22\text{kHz}}} |S(k)|^2}{\sum_{k=f_{20\text{Hz}}}^{f_{22\text{kHz}}} |S(k)|^2}$$
  * Human Studio Master: $\text{Air Power Ratio} \approx 0.03 - 0.08$
  * AI Generated (Bandlimited): $\text{Air Power Ratio} < 0.005$
  * AI Generated (Noisy Vocoder): High spectral flatness ($>0.7$) in high bands with absent harmonic structure.

---

### 2.2 DIMENSION 2: Temporal & Rhythmic Anomalies (The "Drifting Beat" Tell)
One of the most profound discoveries in AI audio forensics—documented by digital forensics expert **Dr. Neal Krawetz (HackerFactor)**—is the divergence between human timing, DAW quantization, and generative AI tempo tracking.

#### A. The Three Rhythmic Paradigms
1. **Live Human Musicians**: Possess micro-tempo variations ($\pm 2\text{ to } 5\text{ BPM}$) driven by human emotion and motor control. Tempo speeds up during choruses and slows slightly during expressive verses.
2. **Modern DAW / Electronic Music**: Human producers align drums to a strict grid (Sequencer Clock / MIDI Quantization). The BPM is exact to 3 decimal places (e.g., $124.000\text{ BPM}$), perfectly lock-stepped across the entire track.
3. **Generative AI (Suno / Udio / ACE-Step)**:
   * AI has **no internal metronome or MIDI clock**. It generates rhythm purely based on statistical token prediction and cross-attention correlations.
   * **The Artifact**: The BPM does not stay fixed (like a DAW), nor does it change intentionally (like a human). Instead, it exhibits a **slow, wandering drift**. A track that begins at $120.4\text{ BPM}$ might drift to $122.1\text{ BPM}$ at 45 seconds, drop to $118.9\text{ BPM}$ at 1:30, and accelerate to $123.5\text{ BPM}$ at 2:15.
   * *Diagnostic Test*: Align the song against a rigid DAW click track. While modern pop/rock/EDM stays locked to the grid, AI tracks will drift out of sync within 8 to 16 bars.

#### B. Transient Smearing (Diffused Drum Attacks)
* **The Physics**: A physical drumstick hitting a snare or a beater hitting a kick creates a near-instantaneous pressure discontinuity: a rise time of $< 1\text{ ms}$ with broad spectral dispersion.
* **The AI Artifact**: Because latent diffusion denoises smooth continuous distributions, high-velocity percussive transients are slightly smeared across multiple latent frames.
* **Auditory Perception**: Kick drums sound "soft" or "pillowy", lacking sub-bass punch and click transient separation. Snares sound as if they were recorded through a heavy compression limiter with instantaneous attack, swallowing the drumstick crack.

---

### 2.3 DIMENSION 3: Stereo Field & Phase Forensics
A human audio mixing engineer positions instruments in a 2D soundstage using pan pots (level panning), Haas delays, and stereo microphone arrays (XY, ORTF, Mid-Side). Each instrument maintains a coherent acoustic relationship to the stereo field.

#### A. Phase Incoherence & "Watery" / "Phasery" Timbre
* **The Mechanism**: Diffusion models independently predict left and right latent channels or predict a mid/side representation with imperfect cross-channel phase correlation.
* **The Auditory Tell**: High-frequency sustained elements—specifically **cymbals, acoustic guitar strums, and vocal reverb tails**—exhibit an unnatural swirling, underwater, or "phaser-like" flange.
* **The Sum-to-Mono Destruction Test**:
  $$\text{Mono Sum} = \frac{\text{Left}(t) + \text{Right}(t)}{2}$$
  * *Test*: Sum the track to mono.
  * In genuine human recordings, the vocal, bass, and kick remain punchy and centered, with only stereo reverbs receding slightly.
  * In AI tracks, the high frequencies often suffer severe **destructive comb-filtering phase cancellation**, making cymbals, vocals, and synth leads hollow, thin, or nearly silent.

#### B. Decorrelated Phantom Stereo Width
* While human stereo width is achieved through distinct acoustic sources (e.g. double-tracked rhythm guitars panned 100% Left and 100% Right), AI generates stereo width by adding uncorrelated diffuse noise between channels, resulting in a diffuse, "unfocused" stereo image where individual instruments cannot be pinpointed in physical space.

---

### 2.4 DIMENSION 4: Vocal Timbre & Linguistic Forensics
Vocals are the most revealing element for human listeners and algorithmic detectors alike.

#### A. The Vocal "Uncanny Valley"
* **Formant Transitions**: Human vocal tracts change shape continuously via physical articulators (tongue, lips, velum, pharynx). AI vocals often exhibit abrupt "formant stepping" or synthetic gliding between pitch targets without realistic vocal tract resonance shifts.
* **Absence of Vocal Fry & True Vocal Cord Closures**: In physical singing, low notes and sustained phrases feature micro-irregularities (jitter, shimmer, and vocal fry). AI synthesizers smooth these out, producing an unnaturally polished, metallic, or robotic sheen.
* **Phantom Breaths**: Human singers inhale before vocal phrases. AI models often generate breath sounds at acoustically impossible moments:
  * Inhaling in the middle of a multi-syllable word.
  * Generating a loud inhale without a subsequent vocal line.
  * Singing continuous 20-second vocal passages without a single breath intake.

#### B. Lyric Hallucinations and Linguistic Clichés
* **Lyrical Tropes**: AI music models (conditioned on LLMs like GPT-4 or Claude for lyrics) rely heavily on statistically over-represented poetic tropes:
  * *Overused keywords*: "shadows and echoes", "neon lights", "whispers in the dark", "ignite the flame", "tapestry of time", "dancing in the rain".
  * *Rhyme Schemes*: Predictable AABB or ABAB schemes with simplistic rhymes (light/night, fire/desire, heart/start).
* **Phonetic Hallucinations (The Outro Breakdown)**:
  * As generative diffusion models reach the end of an audio clip or handle fade-outs, the semantic alignment breaks down.
  * *The Tell*: Vocals frequently dissolve into pseudo-language, phonetic babble, dropped word endings (e.g., dropping the "g" in "-ing" words unnaturally), or suddenly switch language accents midway through a verse.

---

### 2.5 DIMENSION 5: Harmonic & Compositional Tells
1. **Cross-Talk & Stem Entanglement (Spectral Bleeding)**:
   * In a real studio multi-track, instruments are isolated on separate tracks before mastering.
   * In end-to-end AI music (ACE-Step, Suno), all instruments and voices are generated on a **single unified diffusion latent**.
   * *The Tell*: When the vocal gets loud, the background cymbal or acoustic guitar subtly modulates or warps. The vocal reverb tail literally "bleeds" into the snare drum tone. If you attempt stem separation (using Demucs or Spleeter), the isolated stems show heavy acoustic cross-contamination and spectral tearing.
2. **Harmonic Voice Leading & Microtonal Drift**:
   * AI generators often create convincing chords, but voice-leading between chords can be musically nonsensical: parallel fifths, unresolved suspended fourths, and chord progressions that slowly drift out of standard 440 Hz concert pitch tuning over the span of 2 minutes.
3. **Dynamic Flatlining (Absence of Macro-Dynamics)**:
   * AI tracks frequently maintain a uniform, hyper-compressed RMS level from the first second to the last. There is no true acoustic quietude: even in an "acoustic solo" section, a persistent floor of synthetic room ambience and diffuse latent noise persists.

---

# PART III: FORENSIC DETECTION WORKFLOW & ML ARCHITECTURES

```
+-----------------------------------------------------------------------------------------------+
|                             FORENSIC DECISION MATRIX / PIPELINE                               |
+-----------------------------------------------------------------------------------------------+
|                                                                                               |
|   INPUT AUDIO (WAV / FLAC / MP3)                                                              |
|        |                                                                                      |
|        +---> [1. Visual / DSP Inspection]                                                     |
|        |       - High-Frequency Cutoff at 16kHz/18kHz? -------> YES: 95% Probability AI       |
|        |       - Vocoder Horizontal Banding Lines? -----------> YES: 90% Probability AI       |
|        |       - Severe Mono Phase Cancellation? -------------> YES: 85% Probability AI       |
|        |                                                                                      |
|        +---> [2. Temporal / Grid Alignment]                                                   |
|        |       - Wandering BPM Drift (e.g. 120.2 -> 122.4)? --> YES: Strong Indicator AI      |
|        |       - Transient Softening (<1ms rise absent)? -----> YES: Characteristic Diffusion |
|        |                                                                                      |
|        +---> [3. Machine Learning Classifiers]                                                |
|                |                                                                              |
|                +---> CQT / Mel-Spectrogram + ResNet Classifier                                |
|                +---> Raw Waveform: AASIST / RawNet2 (Graph Attention)                         |
|                +---> Self-Supervised Audio Representations: WavLM / CLAD                      |
|                |                                                                              |
|                v                                                                              |
|         ENSEMBLE VERDICT: Human Authentic vs. AI-Generated Synthetic                          |
|                                                                                               |
+-----------------------------------------------------------------------------------------------+
```

### 3.1 Feature Extraction for Algorithmic Detection
In academic research (ASVspoof, IEEE, ArXiv 2508.11694), handcrafted DSP features and deep acoustic representations are fed into neural classifiers:

| Feature Domain | Representation | What It Exposes in AI Music |
| :--- | :--- | :--- |
| **Spectral** | Constant-Q Transform (CQT) | Logarithmic musical pitch resolution exposing microtonal tuning drift and harmonic smearing. |
| **Spectral** | Linear STFT Spectrogram | Unnatural brickwall cutoffs (16/18kHz) and vocoder transposed convolution checkerboards. |
| **Cepstral** | Linear Frequency Cepstral Coefficients (LFCC) | High-frequency filterbank distribution; significantly superior to MFCC in detecting high-frequency synthetic artifacts. |
| **Temporal** | Transient Energy Envelope (RMS Rise Time) | Quantifies the absence of sharp acoustic impulse attacks in drums and percussive hits. |
| **Phase** | Phase Coherence / Inter-Channel Phase Difference (IPD) | Detects decorrelated stereo noise and phase smearing across left/right channels. |

---

### 3.2 State-of-the-Art Deep Learning Detection Models
1. **CQT-ResNet / EfficientNet Classifiers**:
   * High-resolution Constant-Q Transforms capture both low-frequency pitch harmonics and high-frequency noise bands. A deep 2D CNN (ResNet-34 or EfficientNet-B4) trained on authentic vs. synthetic datasets achieves $>98\%$ detection accuracy on uncompressed audio.
2. **AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention)**:
   * Treats spectro-temporal feature maps as graph nodes, using heterogeneous graph attention to model long-range temporal dependencies and spectral anomalies across distant frequency bins simultaneously.
3. **CLAD & WavLM (Self-Supervised Audio Foundation Models)**:
   * Foundation models pre-trained on tens of thousands of hours of speech and music (WavLM Large) possess rich internal representations of natural acoustic physics.
   * Fine-tuning WavLM on human vs. AI music tasks exposes subtle latent inconsistencies that evade traditional spectrogram inspection.

---

### 3.3 Practical 5-Step Human Auditory Checklist
When you don't have access to audio visualizer tools, apply this quick 5-step listening test:

1. **The Cymbal Test**: Put on studio headphones and focus entirely on the hi-hats, crash cymbals, or ride cymbals. Do they sound like physical bronze metal vibrating in air, or do they sound like a "whispering aerosol can" or an underwater "phaser pedal"? *(If aerosol/underwater $\rightarrow$ AI).*
2. **The Solo Vocal Reverb Test**: Listen to the vocal during a transition. When the vocal stops, does the reverb tail suddenly stop or warp into a guitar chord? Does the vocal have phantom gasps for air between syllables? *(If yes $\rightarrow$ AI).*
3. **The Click Track Test**: Tap your foot to the beat. Does the song stay locked like an electronic pop track, or does your foot feel like it has to constantly adjust its speed slightly every 15 seconds without any artistic reason? *(If wandering $\rightarrow$ AI).*
4. **The Lyric Cliché Test**: Does the singer sound passionate while singing generic phrases like *"Through the shadows and the light, our echoes ignite the night"* with no specific narrative details? *(If yes $\rightarrow$ AI).*
5. **The Drum Punch Test**: Does the kick drum hit you in the chest with a clean punch, or does it sound muffled, pillowy, and glued to the bassline? *(If pillowy and glued $\rightarrow$ AI).*

---

### SUMMARY FOR THESIS & ENGINEERING IMPLEMENTATION
In Bryan's Skripsi project and the accompanying forensic visualizer (`MY-AIDETECTOR`):
* The **3D Waterfall Spectrogram** exposes frequency cutoffs and periodic vocoder stripes in real-time.
* The **Parametric EQ 2** tool isolates frequency bands (e.g., soloing $>16\text{ kHz}$) to audibly demonstrate the collapse of natural air power.
* The combination of **CQT spectral feature modeling** and **temporal envelope analysis** provides the mathematical framework to classify authentic human art from generative diffusion latents with near-zero false-positive rates.
