# Audio Machine Learning: Foundations, Architectures, and End-to-End Engineering Guide

> **Authoritative Handbook for Audio Deepfake Detection, AI-Generated Music Forensics, and Acoustic Pattern Recognition**  
> _Synthesized from academic literature, international speech benchmarks (ASVspoof, In-the-Wild), and digital signal processing standards._

---

## Table of Contents

1. [Foundational Mathematics & Digital Signal Processing (DSP)](#1-foundational-mathematics--digital-signal-processing-dsp)
2. [Audio Representation & Feature Engineering Taxonomy](#2-audio-representation--feature-engineering-taxonomy)
3. [Deep Learning Architectures for Audio](#3-deep-learning-architectures-for-audio)
4. [End-to-End Implementation Blueprint: From Raw Audio to Trained Model](#4-end-to-end-implementation-blueprint-from-raw-audio-to-trained-model)
5. [Evaluation Protocols, Biometric Metrics & Forensic Analysis](#5-evaluation-protocols-biometric-metrics--forensic-analysis)
6. [Robustness, Manipulation Attacks & Countermeasures](#6-robustness-manipulation-attacks--countermeasures)
7. [Production Architecture, MLOps & Real-Time Deployment](#7-production-architecture-mlops--real-time-deployment)
8. [Production-Ready PyTorch Implementation](#8-production-ready-pytorch-implementation)

---

## 1. Foundational Mathematics & Digital Signal Processing (DSP)

Audio machine learning bridges continuous acoustic physics and discrete computational linear algebra. Unlike images where spatial pixel coordinates are uniform, audio signals are one-dimensional continuous temporal pressure waves whose critical informational structures exist simultaneously across **time**, **frequency**, and **phase**.

```
Analog Pressure Wave (Continuous)
           │
           ▼  [Sampling: fs ≥ 2·fmax] (Nyquist-Shannon)
           ▼  [Quantization: Bit Depth (16/24/32-bit float)]
Discrete Waveform: x[n]  (Time Domain, 1D Array)
           │
     ┌─────┴────────────────────────────────┐
     ▼                                      ▼
[Direct Raw Waveform Models]      [Time-Frequency Analysis (DSP)]
(SincNet, RawNet2, AASIST)        ├── Short-Time Fourier Transform (STFT)
                                  ├── Mel-Spectrogram (Psychoacoustic)
                                  ├── Constant-Q Transform (CQT, Musical)
                                  └── Cepstral Coefficients (MFCC, LFCC, CQCC)
```

### 1.1. Digitalization of Sound

1. **Sampling Theorem (Nyquist-Shannon)**:
   An analog bandlimited continuous signal $x(t)$ with maximum frequency $f_{\max}$ can be uniquely determined without loss of information if sampled at frequency $f_s$:
   $$f_s \ge 2 f_{\max}$$
   If $f_s < 2 f_{\max}$, high-frequency spectral components fold into lower frequencies, creating irreversible **aliasing distortion**.
   - Speech standard: $f_s = 16\,000\text{ Hz}$ ($f_{\max} = 8\text{ kHz}$, covering human vocal formants).
   - High-fidelity music standard: $f_s = 44\,100\text{ Hz}$ or $48\,000\text{ Hz}$ ($f_{\max} = 22.05\text{ kHz}$, covering human hearing limits).
2. **Quantization & Bit Depth**:
   Continuous amplitude $x(t) \in [-V, +V]$ is discretized into $2^B$ levels where $B$ is bit depth:
   - 16-bit integer: $65\,536$ levels, theoretical Dynamic Range $\approx 96.3\text{ dB}$.
   - 32-bit floating point: Dynamic range $> 1500\text{ dB}$, eliminating clipping risks in preprocessing.
   - Signal-to-Quantization-Noise Ratio:
     $$\text{SQNR} \approx 6.02 B + 1.76\text{ dB}$$

### 1.2. The Fourier Transform & Frequency Decomposition

The continuous audio wave is decomposed into constituent sinusoidal basis functions via the Fourier Transform:
$$X(f) = \int_{-\infty}^{\infty} x(t) e^{-j 2 \pi f t} \, dt$$

In discrete time:
$$X[k] = \sum_{n=0}^{N-1} x[n] e^{-j \frac{2\pi}{N} k n}, \quad k = 0, 1, \dots, N-1$$
Where $X[k]$ represents the complex frequency bin with magnitude $|X[k]|$ and phase $\angle X[k]$.

### 1.3. Short-Time Fourier Transform (STFT) & The Spectrogram

Real audio is **non-stationary**; its frequency content changes continuously over time. The STFT partitions the signal into short, quasi-stationary frames using an overlapping sliding window $w[n]$:
$$X(m, k) = \sum_{n=0}^{N-1} x[n + mH] \cdot w[n] \cdot e^{-j \frac{2\pi}{N} k n}$$
Where:

- $m$: Frame index (temporal coordinate).
- $k$: Frequency bin index.
- $N$: Window length (FFT size, e.g., $1024$ or $2048$ samples).
- $H$: Hop length (frame shift, typically $N/4$ or $N/2$, e.g., $256$ or $512$ samples).
- $w[n]$: Tapering window function (e.g., Hann or Hamming window) to minimize spectral leakage at boundary discontinuities:
  $$w_{\text{Hann}}[n] = 0.5 \left(1 - \cos\left(\frac{2\pi n}{N-1}\right)\right)$$

**The Power Spectrogram & Decibel Conversion**:
The power spectrum represents signal power across time and frequency:
$$P(m, k) = |X(m, k)|^2$$
Human hearing perceives loudness logarithmically. Therefore, raw linear power is mapped to decibels:
$$S_{\text{dB}}(m, k) = 10 \log_{10} \left( \frac{P(m, k)}{\max(P) + \epsilon} \right)$$

```
Frequency (kHz)
   ▲
20 ┤    .·:*:·.         ·:·*·:·.        (High-frequency harmonics & breath)
10 ┤  :***:::***:     :***:::***:
 5 ┤ :*****:*****:   :*****:*****:      (Vocal formants / Lead instruments)
 1 ┤:*************: :*************:
 0 └───────────────────────────────► Time (seconds)
```

---

## 2. Audio Representation & Feature Engineering Taxonomy

Choosing or designing an audio feature representation is the most critical decision in building an audio machine learning pipeline. Audio features fall into four major classes:

| Feature Class             | Representations                        | Best Suited For                                  | Key Advantages                                                      | Key Limitations                                                     |
| ------------------------- | -------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------- | ------------------------------------------------------------------- |
| **Linear Spectral**       | STFT, Linear Spectrogram               | AI Cutoff Detection, Vocoder Artifacts           | Preserves raw frequency resolution up to Nyquist                    | High dimensionality; linear scale does not match human hearing      |
| **Psychoacoustic**        | Mel-Spectrogram, MFCC                  | Speech Recognition, General Audio Classification | Mirrors human cochlear filter frequency perception                  | Compresses high frequencies; loses fine vocal pitch micro-harmonics |
| **Musical / Harmonic**    | Constant-Q Transform (CQT), Chromagram | Music Detection, Chord/Instrument Forensics      | Logarithmic frequency resolution, constant octave bins              | Computational overhead; requires variable temporal windows          |
| **Self-Supervised (SSL)** | WavLM, Wav2Vec 2.0, HuBERT, MERT       | Deepfake Detection, Speaker Verification         | High-level contextual semantics, generalizable embeddings           | Computationally heavy; acts as a complex black box                  |
| **End-to-End Raw**        | Raw Waveform ($x[n]$) + SincNet        | Anti-spoofing (AASIST, RawNet2)                  | Retains raw phase information and time-frequency cross-interactions | Requires specialized filter architectures and large datasets        |

---

### 2.1. Mel-Scale & Mel-Frequency Cepstral Coefficients (MFCC)

Human perception of frequency is approximately linear below $1\text{ kHz}$ and logarithmic above $1\text{ kHz}$. The **Mel Scale** formalizes this perceptual response:
$$m = 2595 \log_{10}\left(1 + \frac{f}{700}\right)$$

```
Frequency (Hz)        Mel Filterbank (Overlapping Triangular Filters)
  0 ───┼───┼───┼─────┼───────┼─────────┼───────────┼─────────────► 20 kHz
      F1  F2  F3    F4      F5        F6          F7            F_M
      (Dense spacing in Bass)          (Wide spacing in Treble)
```

1. **Mel-Spectrogram Generation**:
   Multiply the linear power spectrogram by a filterbank of $M$ overlapping triangular bandpass filters ($M \in [40, 128]$):
   $$S_{\text{Mel}}(m, b) = \sum_{k=0}^{N/2} P(m, k) \cdot H_b[k]$$
2. **Discrete Cosine Transform (DCT) for MFCCs**:
   Apply DCT-II to log Mel energies to decorrelate filterbank outputs and compress energy into $C$ cepstral coefficients ($C \in [13, 20]$):
   $$c_n = \sum_{b=0}^{M-1} \log(S_{\text{Mel}}[b]) \cos\left[ \frac{\pi n}{M} \left(b + \frac{1}{2}\right) \right]$$
3. **Dynamic Features ($\Delta$ and $\Delta\Delta$)**:
   $$\Delta c_n(t) = \frac{\sum_{\tau=1}^D \tau (c_n(t+\tau) - c_n(t-\tau))}{2 \sum_{\tau=1}^D \tau^2}$$
   - Static MFCC: Spectral envelope shape.
   - Delta ($\Delta$): First-order velocity of spectral transitions.
   - Delta-Delta ($\Delta\Delta$): Second-order acceleration of vocal tract changes.

---

### 2.2. Constant-Q Transform (CQT) for Musical Audio

While STFT uses a fixed window length $N$ across all frequencies, the **Constant-Q Transform** dynamically adjusts the window length $N_k$ inversely proportional to frequency $f_k$:
$$Q = \frac{f_k}{\Delta f_k} = \text{constant}$$
$$N_k = Q \cdot \frac{f_s}{f_k}$$

- **At low frequencies (e.g., $40\text{ Hz}$)**: $N_k$ is long $\rightarrow$ **high frequency resolution** (separates distinct musical bass notes).
- **At high frequencies (e.g., $10\text{ kHz}$)**: $N_k$ is short $\rightarrow$ **high temporal resolution** (captures fast percussive transients).
- **Bins per Octave ($B$)**: Geometrically spaced frequencies:
  $$f_k = f_0 \cdot 2^{k / B}$$
  With $B=12$ or $B=24$, every bin corresponds exactly to a semitone or quarter-tone in Western music temperament.

---

### 2.3. Explainable Forensic Acoustic Descriptors

In digital forensics and academic research (e.g., Elsevier/IEEE forensic standards), statistical summaries of physical audio descriptors provide transparent explainability:

1. **Spectral Centroid**: The "center of gravity" of the spectrum (perceived brightness):
   $$\mu_{\text{Centroid}} = \frac{\sum_{k} f_k |X[k]|}{\sum_{k} |X[k]|}$$
2. **Spectral Bandwidth / Spread**: The second central moment around the centroid:
   $$\mu_{\text{Bandwidth}} = \sqrt{\frac{\sum_{k} (f_k - \mu_{\text{Centroid}})^2 |X[k]|}{\sum_{k} |X[k]|}}$$
3. **Spectral Rolloff ($R_{85}, R_{95}$)**: The frequency below which $85\%$ or $95\%$ of total spectral energy is contained:
   $$\sum_{k=0}^{K_{\text{rolloff}}} |X[k]|^2 = 0.85 \sum_{k=0}^{N/2} |X[k]|^2$$
   _Crucial forensic indicator:_ AI audio synthesizers often exhibit brickwall cutoff drops at $16\text{ kHz}$ or $18\text{ kHz}$ where $R_{95}$ collapses.
4. **Spectral Flatness (Wiener Entropy)**: Ratio of geometric mean to arithmetic mean:
   $$\text{SF} = \frac{\exp\left(\frac{1}{K}\sum_{k=1}^K \ln |X[k]|^2\right)}{\frac{1}{K}\sum_{k=1}^K |X[k]|^2}$$
   Measures whether sound is tonal/harmonic ($\text{SF} \to 0$) or noise-like ($\text{SF} \to 1$).
5. **Zero-Crossing Rate (ZCR)**:
   $$\text{ZCR} = \frac{1}{2(N-1)} \sum_{n=1}^{N-1} |\text{sgn}(x[n]) - \text{sgn}(x[n-1])|$$

---

## 3. Deep Learning Architectures for Audio

```
                               AUDIO MODEL TAXONOMY
                                        │
      ┌──────────────────┬──────────────┴────────────────┬──────────────────┐
      ▼                  ▼                               ▼                  ▼
[2D Vision CNNs]   [Raw 1D End-to-End]             [Graph Neural Nets] [Self-Supervised & Foundation]
• Input: Spectrogram• Input: x[n] (waveform)       • Input: Spectro-  • Input: Raw audio / tokens
• ResNet-18/34/50  • SincNet                      Temporal Graph     • Wav2Vec 2.0, HuBERT
• EfficientNet-B0  • RawNet2                      • AASIST (HS-GAL)  • WavLM, MERT
• MobileNetV3      • Res-TSSDNet                  • GAT Mechanisms   • Contrastive (CLAD / MoCo)
```

---

### 3.1. The Computer Vision Paradigm (2D CNNs on Spectrograms / CQT)

When audio is mapped to an image $S \in \mathbb{R}^{C \times F \times T}$ (Channels, Frequency bins, Time frames):

- **Convolution in Audio**:
  - A 2D kernel $K \in \mathbb{R}^{k_F \times k_T}$ extracts local spectro-temporal patterns (e.g., harmonic stacks, pitch glides, percussive transients).
- **Residual Networks (ResNet)**:
  Deep networks suffer from vanishing/exploding gradients. ResNet introduces identity shortcut connections:
  $$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$
  Allowing gradients to backpropagate unimpeded across 50+ layers.
- **Why it spots AI audio artifacts**: Neural vocoders (HiFi-GAN, WaveGlow) and diffusion generators leave periodic horizontal checkerboard patterns, blurring along high-frequency bins, and vertical discontinuity lines that 2D CNN filters detect easily.

---

### 3.2. Raw Waveform End-to-End Networks & SincNet

Instead of unconstrained learned convolutional filters that may learn irrelevant noise, **SincNet** constrains first-layer 1D convolutional filters to bandpass filters defined by parameterized sinc functions:
$$g[n, f_1, f_2] = 2f_2 \text{sinc}(2\pi f_2 n) - 2f_1 \text{sinc}(2\pi f_1 n)$$
$$\text{sinc}(x) = \frac{\sin(x)}{x}$$
Where only two parameters—the low cutoff $f_1$ and high cutoff $f_2$—are learned via gradient descent for each filter. This preserves interpretability, drastically reduces parameters, and protects raw phase information.

---

### 3.3. AASIST: Integrated Spectro-Temporal Graph Attention Networks

AASIST is the premier academic benchmark architecture for logical access spoofing detection:

```
Raw Waveform x[n]
       │
       ▼
[SincNet Conv Front-End] ──► Feature Map F ∈ R^(C × F × T)
       │
       ├──► Max-pooling along Time ──► Spectral Nodes G_s ∈ R^(F × D)
       └──► Max-pooling along Freq ──► Temporal Nodes G_t ∈ R^(T × D)
                                               │
                                               ▼
                              [Heterogeneous Graph Construction]
                              G_st = Stack(G_s, G_t)
                                               │
                                               ▼
                              [HS-GAL: Graph Attention Layers]
                              (Edge weighting between Time & Freq)
                                               │
                                               ▼
                              [Readout Layer & Softmax Classifier]
                              ──► [P(Bona Fide), P(Spoof)]
```

- **Core Insight**: Spoofing artifacts do not appear exclusively in the time domain or the frequency domain; they appear as **cross-domain inconsistencies** (e.g., an unnaturally steady pitch that does not match natural temporal vocal dynamics).
- **Graph Attention (GAT)** assigns dynamic attention weights $e_{ij}$ to edges connecting temporal node $i$ and spectral node $j$:
  $$\alpha_{ij} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^\top [\mathbf{W}\mathbf{h}_i \,\|\, \mathbf{W}\mathbf{h}_j]\right)\right)}{\sum_{k \in \mathcal{N}_i} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^\top [\mathbf{W}\mathbf{h}_i \,\|\, \mathbf{W}\mathbf{h}_k]\right)\right)}$$

---

### 3.4. Contrastive Learning & Representation Clustering (CLAD)

Traditional supervised learning models minimize Cross-Entropy, fitting decision boundaries directly to training examples. When evaluated on unseen out-of-domain attacks or simple volume/noise manipulations, supervised models fail catastrophically.

**CLAD (Contrastive Learning-based Audio Deepfake Detector)** solves this with a two-phase architecture:

1. **Momentum Contrast (MoCo) Pretraining**:
   Maintains a dynamic queue of negative samples ($K=6144$) updated via momentum encoder $\theta_k \leftarrow \mu \theta_k + (1-\mu)\theta_q$.
   $$\mathcal{L}_{\text{CL}} = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(q_i \cdot k_i^+ / \tau)}{\sum_{j=1}^K \exp(q_i \cdot k_j / \tau)}$$
2. **Length Loss for Compact One-Class Real Clustering**:
   Because bona fide human audio shares organic acoustic physics, while deepfakes vary wildly across different generative tools, **Length Loss** forces real samples to cluster tightly near the coordinate origin while pushing deepfake embeddings outward:
   $$\mathcal{L}_{\text{len}} = \frac{1}{N}\sum_{i=1}^N \left[ y_i \cdot w \cdot \|q_i\|_2 + (1 - y_i) \cdot \max(0, \text{margin} - \|q_i\|_2) \right]$$
   - Combined Pretraining Objective:
     $$\mathcal{L}_{\text{pretrain}} = \mathcal{L}_{\text{CL}} + \lambda \mathcal{L}_{\text{len}}$$
3. **Downstream Fine-Tuning**: Freeze or fine-tune encoder with a single linear classification head.

---

## 4. End-to-End Implementation Blueprint: From Raw Audio to Trained Model

Building a production-grade Audio ML system requires adherence to a strict 6-phase engineering lifecycle:

```
[Phase 1: Dataset Hygiene] ──► [Phase 2: Standardization] ──► [Phase 3: Augmentation Engine]
         │                              │                               │
         ▼                              ▼                               ▼
• Balanced Classes (50/50)      • Resample to 16/44.1kHz        • AWGN (SNR 0-20dB)
• Speaker/Track Disjoint        • Downmix to Mono               • Fading & Vol Control
• Fixed Seed Splitting          • Peak / LUFS Normalization     • Pitch / Time Shifts
                                • Fixed Duration (Pad/Trunc)
         ┌──────────────────────────────────────────────────────────────┘
         ▼
[Phase 4: Feature / Model Engine] ──► [Phase 5: Optimization] ──► [Phase 6: Biometric Eval]
• DSP (CQT/Mel) vs Raw Waveform       • AdamW (lr=1e-4, wd=1e-4) • EER (FAR = FRR)
• Backbone (ResNet / AASIST)          • Cosine Annealing LR      • ROC-AUC & DET Curves
• Metric / Classification Head        • Early Stopping (Val EER) • Confusion Matrix
```

---

### Phase 1: Dataset Curation & Partitioning Rules

1. **Class Balance**: Equal distribution of positive ($y=1$, Bona fide) and negative ($y=0$, Spoof/AI) samples.
2. **Transcript Matching**: In speech/vocal experiments, both real and synthetic audio must pronounce identical transcripts to prevent the model from learning lexical/textual bias.
3. **Strict Disjoint Partitioning (Crucial against Data Leakage)**:
   - **Speaker-Disjoint**: No speaker appearing in `train` can ever appear in `val` or `test`.
   - **Track/Artist-Disjoint**: In music, no song or stem from the same artist/session can span splits.
   - Standard Split Ratios: $70\%$ Train, $15\%$ Validation, $15\%$ Test (or $60/20/20$).

---

### Phase 2: Signal Standardization Pipeline

1. **Resampling ($f_s = 16\,000\text{ Hz}$ or $44\,100\text{ Hz}$)**:
   Polyphase filtering or Kaiser-windowed sinc interpolation (`torchaudio.transforms.Resample`).
2. **Channel Downmixing**:
   $$x_{\text{mono}}[n] = \frac{x_L[n] + x_R[n]}{2}$$
3. **Peak Amplitude Normalization**:
   $$x_{\text{norm}}[n] = \frac{x[n]}{\max(|x[n]|) + 10^{-7}}$$
   Ensures all audio signals reside in $[-1.0, +1.0]$, preventing loudness-based shortcut learning.
4. **Uniform Duration Alignment (Padding & Truncation)**:
   Neural networks require uniform tensor shapes within each batch. Given target duration $T$ seconds ($N_{\text{target}} = f_s \times T$, e.g., $4\text{ s} \times 16\,000 = 64\,000$ samples):
   $$\text{If } N_x > N_{\text{target}}: \quad x_{\text{aligned}} = x[0 : N_{\text{target}}]$$
   $$\text{If } N_x < N_{\text{target}}: \quad x_{\text{aligned}} = \text{Repeat}(x)[0 : N_{\text{target}}]$$
   _Note:_ Repeat-padding preserves acoustic pitch and timbre harmonics much better than silent zero-padding.

---

### Phase 3: Augmentation & Robustness Injection

To prevent overfitting and simulate real-world transmission:

#### Additive White Gaussian Noise (AWGN) at Target SNR

Given signal $x[n]$ and target $\text{SNR}_{\text{dB}}$:

1. Compute signal power:
   $$P_{\text{signal}} = \frac{1}{N} \sum_{n=0}^{N-1} x[n]^2$$
2. Compute required noise power:
   $$P_{\text{noise}} = \frac{P_{\text{signal}}}{10^{\text{SNR}_{\text{dB}} / 10}}$$
3. Sample Gaussian noise vector:
   $$w[n] \sim \mathcal{N}\left(0, \sqrt{P_{\text{noise}}}\right)$$
4. Add noise to audio:
   $$y[n] = x[n] + w[n]$$

#### Additional Manipulations

- **Volume Scaling**: $x[n] \leftarrow \alpha \cdot x[n]$ with $\alpha \in [0.1, 0.9]$.
- **Smooth Fading**: Multiply start/end by linear or half-sinusoidal envelope.
- **Time Stretching**: Phase vocoder stretch without pitch shift (factor $0.9 - 1.1$).

---

### Phase 4: Loss Functions & Training Regimes

1. **Cross-Entropy Loss (with Class Weighting)**:
   $$\mathcal{L}_{\text{CE}} = - \frac{1}{N}\sum_{i=1}^N \left[ w_1 y_i \log(\hat{p}_i) + w_0 (1 - y_i) \log(1 - \hat{p}_i) \right]$$
2. **Optimizer**: AdamW with $\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $\lambda = 10^{-4}$.
3. **Learning Rate Schedule**:
   - Initial learning rate $\eta_0 = 10^{-4}$.
   - Cosine Annealing:
     $$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})\left(1 + \cos\left(\frac{t}{T_{\max}}\pi\right)\right)$$
4. **Early Stopping & Checkpoint Strategy**:
   - Track **Validation Equal Error Rate (EER)**, not training loss!
   - Save checkpoint whenever Validation EER reaches a new minimum.
   - Stop training if Validation EER does not improve for $P = 10$ consecutive epochs.

---

## 5. Evaluation Protocols, Biometric Metrics & Forensic Analysis

Audio deepfake and biometrics systems cannot rely on accuracy alone due to threshold sensitivity and class vulnerabilities.

```
                         ACTUAL TRUTH
                     Bona Fide (Real)       Spoof (Fake)
                 ┌──────────────────────┬──────────────────────┐
Predicted Real   │  True Positive (TP)  │ False Positive (FP)  │
                 │  (Correctly verified)│ (CRITICAL SECURITY   │
                 │                      │  BREACH: False Accept)│
                 ├──────────────────────┼──────────────────────┤
Predicted Fake   │ False Negative (FN)  │  True Negative (TN)  │
                 │ (False Rejection of  │ (Correctly blocked)  │
                 │  genuine user)       │                      │
                 └──────────────────────┴──────────────────────┘
```

### 5.1. Academic Metric Definitions

1. **False Acceptance Rate (FAR)** (Spoof accepted as genuine):
   $$\text{FAR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$
2. **False Rejection Rate (FRR)** (Genuine rejected as spoof):
   $$\text{FRR} = \frac{\text{FN}}{\text{FN} + \text{TP}}$$
3. **Equal Error Rate (EER)**:
   The operating point where the threshold $\theta^*$ balances false accepts and false rejects:
   $$\text{EER} = \text{FAR}(\theta^*) = \text{FRR}(\theta^*), \quad \theta^* = \arg\min_\theta |\text{FAR}(\theta) - \text{FRR}(\theta)|$$
   _Lower EER indicates a superior classifier regardless of arbitrary decision thresholds._
4. **Receiver Operating Characteristic (ROC) & AUC**:
   Plot of $\text{True Positive Rate} = 1 - \text{FRR}$ against $\text{False Positive Rate} = \text{FAR}$ across all thresholds $\theta \in [0, 1]$.
   $$\text{AUC} = \int_0^1 \text{TPR}(\text{FPR}) \, d(\text{FPR})$$
5. **Detection Error Tradeoff (DET) Curve**:
   Plots $\text{FRR}$ versus $\text{FAR}$ on a normal deviate scale ($y = \Phi^{-1}(\text{FRR})$ vs $x = \Phi^{-1}(\text{FAR})$), where ideal systems appear as straight lines toward the lower-left origin.

---

## 6. Robustness, Manipulation Attacks & Countermeasures

Recent findings (CLAD 2026, Procedia Computer Science) demonstrate that audio deepfake models trained solely on clean data collapse when exposed to simple, imperceptible audio manipulations:

| Manipulation Attack       | Mechanism                                   | Vulnerability Mechanism in Standard Baselines                            | Effective Countermeasure                     |
| ------------------------- | ------------------------------------------- | ------------------------------------------------------------------------ | -------------------------------------------- |
| **Volume Attenuation**    | Multiply audio by $0.1$                     | Alters amplitude variance; shifts neural activation ranges               | Peak normalization + Length Loss             |
| **Half-Sine Fading**      | Attenuate start/end by $50\%$               | Destroys natural boundary transients; introduces artificial window ramps | Contrastive pretraining (MoCo)               |
| **White Noise (AWGN)**    | Inject Gaussian noise at $15\text{ dB}$     | High-frequency noise masks vocoder synthesis artifacts                   | Multi-level AWGN Fine-Tuning ($0\text{ dB}$) |
| **Time Stretching**       | WSOLA / Phase vocoder stretch ($0.9\times$) | Introduces synthetic phase smearing that confuses detectors              | Multi-rate data augmentation                 |
| **Lossy Codec (MP3/AAC)** | Psychoacoustic masking compression          | Deletes frequency bins above cutoff ($16\text{ kHz}$)                    | Hybrid Handcrafted + SSL fusion              |

> **Key Rule from Research**: Fine-tuning with extreme noise (e.g., $-5\text{ dB}$) ruins model clean accuracy. Fine-tuning at **$\text{SNR} = 0\text{ dB}$ (Signal Power = Noise Power)** provides the optimal Pareto frontier: achieves $\text{EER} < 2.5\%$ under extreme noise without sacrificing $100\%$ accuracy on clean audio.

---

## 7. Production Architecture, MLOps & Real-Time Deployment

A notebook model is not an application. Deploying an Audio ML system requires low-latency, cross-platform architecture:

```
[Client / Audio Upload] ──► [FastAPI REST / WebSocket Gateway]
                                    │
                                    ▼
                         [Audio Ingestion & Preprocessing]
                         ├── librosa / torchaudio resample (16kHz)
                         ├── Mono conversion & Peak Normalization
                         └── On-the-fly STFT / CQT Tensor Extraction
                                    │
                                    ▼
                         [High-Throughput Inference Engine]
                         ├── ONNX Runtime / TensorRT (GPU-accelerated)
                         └── Quantized FP16 / INT8 Execution
                                    │
                                    ▼
                         [Forensics & Visualization Service]
                         ├── Binary Prediction: Real vs AI-Generated
                         ├── Brickwall Cutoff Detection (Hz)
                         ├── Spectral Centroid, Rolloff, Air Power
                         └── 3D Waterfall / Spectrogram Payload
                                    │
                                    ▼
                         [User Interface / DAW Dashboard]
```

### Critical Optimization Strategies

1. **Model Export to ONNX**:
   Converting PyTorch models to ONNX allows execution with `onnxruntime-gpu`, reducing inference latency from $\sim 120\text{ ms}$ to $< 10\text{ ms}$ per 3-second utterance.
2. **Chunking & Strided Inference for Long Tracks**:
   Do not process a 4-minute song in a single forward pass (causes VRAM out-of-memory). Slide a $4.0\text{ s}$ window with $2.0\text{ s}$ hop size, compute chunk-level predictions, and aggregate via median or trimmed mean:
   $$\hat{P}_{\text{track}} = \text{median}\left(\{\hat{p}_1, \hat{p}_2, \dots, \hat{p}_K\}\right)$$
3. **Continuous Integration & Data Drift Monitoring**:
   AI music generators (Suno, Udio, ElevenLabs) evolve rapidly. Log out-of-distribution audio with low prediction confidence to a human-in-the-loop (HITL) queue for active learning and model retraining.

---

## 8. Production-Ready PyTorch Implementation

Below is a self-contained, modular Python script illustrating the complete pipeline: dataset definition with AWGN noise injection, 2D CQT/Mel-Spectrogram extraction, ResNet classifier, training loop, and exact EER calculation.

```python
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
from sklearn.metrics import roc_curve, auc

# =====================================================================
# 1. AUDIO PREPROCESSING & ON-THE-FLY DATASET
# =====================================================================
class AudioMLDataset(Dataset):
    """
    Standardized Audio Dataset with on-the-fly resampling, duration normalization,
    and calibrated Additive White Gaussian Noise (AWGN) augmentation.
    """
    def __init__(self, file_paths, labels, target_sr=16000, duration_sec=4.0, snr_db=None):
        self.file_paths = file_paths
        self.labels = labels
        self.target_sr = target_sr
        self.target_samples = int(target_sr * duration_sec)
        self.snr_db = snr_db

        # Audio to Mel-Spectrogram transform (CQT or Mel)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            n_mels=80,
            f_min=20.0,
            f_max=target_sr // 2
        )
        self.amplitude_to_db = T.AmplitudeToDB(top_db=80.0)

    def __len__(self):
        return len(self.file_paths)

    def _inject_awgn(self, waveform, target_snr):
        """Incorporate AWGN based on signal power calculation"""
        signal_power = torch.mean(waveform ** 2)
        if signal_power <= 0:
            return waveform
        noise_power = signal_power / (10.0 ** (target_snr / 10.0))
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        return waveform + noise

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]

        # 1. Load Audio
        waveform, sr = torchaudio.load(path)

        # 2. Downmix to Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 3. Resample
        if sr != self.target_sr:
            resampler = T.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)

        # 4. Peak Normalization
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val

        # 5. Fixed Duration Alignment (Repeat-Padding or Truncation)
        num_samples = waveform.shape[-1]
        if num_samples < self.target_samples:
            repeats = math.ceil(self.target_samples / num_samples)
            waveform = waveform.repeat(1, repeats)[:, :self.target_samples]
        elif num_samples > self.target_samples:
            waveform = waveform[:, :self.target_samples]

        # 6. AWGN Noise Injection (if enabled)
        if self.snr_db is not None:
            waveform = self._inject_awgn(waveform, self.snr_db)

        # 7. Convert to 2D Log Mel-Spectrogram (Shape: [1, n_mels, time_frames])
        mel_spec = self.mel_transform(waveform)
        log_mel_spec = self.amplitude_to_db(mel_spec)

        return log_mel_spec, torch.tensor(label, dtype=torch.long)

# =====================================================================
# 2. AUDIO CLASSIFICATION BACKBONE (ResNet-style Block)
# =====================================================================
class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class AudioClassifier(nn.Module):
    """
    2D Deep Residual Spectrogram Classifier with Global Average Pooling
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.stage1 = ResidualBlock2D(32, 64, stride=2)
        self.stage2 = ResidualBlock2D(64, 128, stride=2)
        self.stage3 = ResidualBlock2D(128, 256, stride=2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        feat = torch.flatten(x, 1)
        logits = self.fc(feat)
        return logits, feat

# =====================================================================
# 3. METRIC CALCULATION: EQUAL ERROR RATE (EER)
# =====================================================================
def compute_eer(bonafide_scores, spoof_scores):
    """
    Computes Equal Error Rate (EER) where FAR equals FRR.
    bonafide_scores: Scores for genuine samples (higher = more genuine)
    spoof_scores: Scores for fake/spoof samples
    """
    labels = np.array([1] * len(bonafide_scores) + [0] * len(spoof_scores))
    scores = np.concatenate([bonafide_scores, spoof_scores])

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    # EER is point where |fpr - fnr| is minimized
    diff = np.abs(fpr - fnr)
    idx = np.argmin(diff)
    eer = (fpr[idx] + fnr[idx]) / 2.0
    threshold = thresholds[idx]

    return eer * 100.0, threshold

# =====================================================================
# 4. TRAINING & EVALUATION HARNESS
# =====================================================================
def train_audio_model(model, train_loader, val_loader, epochs=30, device="cuda"):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_eer = float("inf")
    best_model_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for specs, labels in train_loader:
            specs, labels = specs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(specs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation Loop
        model.eval()
        bona_scores, spoof_scores = [], []
        with torch.no_grad():
            for specs, labels in val_loader:
                specs = specs.to(device)
                logits, _ = model(specs)
                # Score = probability of bona fide class (index 1)
                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                labels_np = labels.numpy()

                for p, l in zip(probs, labels_np):
                    if l == 1:
                        bona_scores.append(p)
                    else:
                        spoof_scores.append(p)

        val_eer, _ = compute_eer(bona_scores, spoof_scores)
        print(f"Epoch [{epoch:02d}/{epochs}] Loss: {total_loss/len(train_loader):.4f} | Val EER: {val_eer:.2f}%")

        # Early Stopping / Best Checkpoint Save
        if val_eer < best_val_eer:
            best_val_eer = val_eer
            best_model_state = model.state_dict().copy()
            print(f"[*] Best model saved with Val EER: {best_val_eer:.2f}%")

    return best_model_state
```

---

## 9. Synthesis & Recommended Next Steps for Bryan's Skripsi

1. **Title Selection**:
   - _Option A (Direct Academic Adaptation)_: _"Pengembangan Teknologi Deteksi Audio Deepfake Berbasis Arsitektur Deep Learning terhadap Gangguan Additive White Gaussian Noise"_
   - _Option B (Music-Forensic Focused)_: _"Analisis Ketahanan Ekstraksi Fitur Spektro-Temporal dan Deep Learning dalam Mendeteksi Musik AI (AI-Generated Music) pada Kondisi Noisy"_
2. **Dataset Architecture**:
   - Utilize **MUSDB18** (`mixture.wav`, `vocals.wav`, `drums.wav`, `bass.wav`) as ground-truth real human music.
   - Utilize modern AI music generators (Suno, Udio) as the spoof dataset.
   - Employ $16\text{ kHz}$ or $44.1\text{ kHz}$ standardized sampling with mono downmixing.
3. **Acoustic Artifacts to Document in Chapter 4 (Hasil dan Analisis)**:
   - Visualizing the brickwall high-frequency cutoff ($\le 16\text{ kHz}$) using the forensic visualizer app.
   - Comparing EER across clean audio versus $\text{SNR} = [20, 15, 10, 5, 0, -5]\text{ dB}$.
   - Evaluating confusion matrices and DET curves comparing ResNet/CQT against AASIST/RawNet2.
