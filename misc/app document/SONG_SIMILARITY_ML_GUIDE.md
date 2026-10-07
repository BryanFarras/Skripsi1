# Machine Learning Guide: Acoustic Song Similarity & Explainable Audio Retrieval (MIR)

---

## 1. Executive Summary & Problem Formulation

When building a **Song Similarity System** (similar to [DISCO.ac](https://disco.ac) or [similarsongfinder.io](https://similarsongfinder.io)), a common misconception is that a machine can _"listen to and scan the raw internet in real-time"_.

In practice, the internet does not expose an open, unindexed raw audio search API. Audio processing of unindexed waveforms is computationally intensive. Real-world systems solve this using **Content-Based Music Information Retrieval (CBMIR)** and **Audio Metric Learning**:

```
[Reference Music Corpus / Database]
               │
               ▼
   [Audio Feature Extraction] ──► Pre-computes D-dimensional vectors
               │
               ▼
   [Vector Index (FAISS / Scikit-Learn)]  (Offline Precomputed Database)
                               ▲
                               │ Nearest Neighbor Search (Cosine / L2)
                               │ (Sub-10ms query time)
[User Uploaded Track] ─────────┴────────────────────────────────────────
               │
               ▼
   [Extract D-dim Vector] ──► Returns Top-K Matches + Subspace Explanations
```

---

## 2. Core Methodological Distinction (Crucial for Skripsi)

In audio research and academic literature, you must distinguish between two fundamentally different tasks:

| Dimension           | Audio Fingerprinting (Shazam, Chromaprint)                                                               | Acoustic / Semantic Similarity (DISCO.ac, Spotify)                                                   |
| :------------------ | :------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------- |
| **Goal**            | Identifies the **exact identical recording** even through background noise, phone mics, or low bitrates. | Identifies **different songs that share the same vibe, timbre, chords, tempo, or production style**. |
| **Representation**  | Exact spectrogram peak constellations, hashes, or landmark hashes.                                       | Dense continuous vectors (110-D DSP features or deep neural embeddings).                             |
| **Distance Metric** | Hamming distance, bitwise exact matching.                                                                | Cosine similarity, Euclidean distance ($L_2$), or Triplet Loss margin.                               |
| **Application**     | Copyright tracking, radio broadcast monitoring.                                                          | Music recommendation, reference track search, soundalike discovery.                                  |

---

## 3. The 3 Architectural Approaches for Your Project

### Approach A: 110-D DSP Acoustic Feature Vector Space (Skripsi Native & Explainable)

_Since your thesis pipeline already extracts 110-dimensional DSP features, this approach requires **no extra neural training** while offering **100% explainability**._

#### 1. Feature Subspace Decomposition

Normalize your 110-D features ($z$-score standardization with `StandardScaler`) and group them into 4 perceptual subspaces:

1. **Timbral Texture Subspace ($\mathbf{f}_{\text{timbre}}$)**:
   - MFCCs 1–20 (mean + standard deviation).
   - Spectral Centroid, Spectral Rolloff (85% & 95%), Spectral Flux.
   - _Captures instrument timbre, brightness, and audio production warmth._
2. **Harmonic & Tonal Subspace ($\mathbf{f}_{\text{harmonics}}$)**:
   - Chroma CENS (12 pitch classes).
   - Tonnetz (harmonic network: fifths, major thirds, minor thirds).
   - _Captures musical key, chord progression, and tonal color._
3. **Rhythmic Cadence Subspace ($\mathbf{f}_{\text{rhythm}}$)**:
   - Estimated BPM / Tempo.
   - Onset strength envelope and pulse clarity.
   - _Captures groove, percussion speed, and danceability._
4. **Tonal Balance & Dynamics Subspace ($\mathbf{f}_{\text{balance}}$)**:
   - 5-Band energy distribution: Sub-bass ($<60$ Hz), Bass ($60$–$250$ Hz), Mids ($250$–$2$ kHz), High-mids ($2$–$6$ kHz), Brilliance ($>6$ kHz).
   - RMS energy and crest factor (dynamic compression).

#### 2. Similarity Computation

Given query track $\mathbf{q}$ and reference track $\mathbf{r}$:

$$\text{Sim}_{\text{total}}(\mathbf{q}, \mathbf{r}) = w_1 \cdot S(\mathbf{q}_{\text{timbre}}, \mathbf{r}_{\text{timbre}}) + w_2 \cdot S(\mathbf{q}_{\text{harmonics}}, \mathbf{r}_{\text{harmonics}}) + w_3 \cdot S(\mathbf{q}_{\text{rhythm}}, \mathbf{r}_{\text{rhythm}}) + w_4 \cdot S(\mathbf{q}_{\text{balance}}, \mathbf{r}_{\text{balance}})$$

Where $S(\mathbf{u}, \mathbf{v})$ is the Cosine Similarity:
$$S(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$

Default recommended weights: $w_1 = 0.35$ (Timbre), $w_2 = 0.25$ (Harmonics), $w_3 = 0.20$ (Rhythm), $w_4 = 0.20$ (Tonal Balance).

---

### Approach B: Deep Metric Learning (Using Pre-Trained Neural Embeddings)

If you want to leverage deep neural networks:

#### 1. Pre-trained Neural Audio Models

Instead of training from scratch on millions of audio files, utilize pre-trained models that map audio directly into dense semantic representations:

- **CLAP (Contrastive Language-Audio Pretraining)** by Microsoft / LAION: Maps audio into the same multimodal embedding space as text.
- **Essentia / MusiCNN**: Pre-trained on Discogs and the Million Song Dataset (MSD) for genre and mood embeddings.
- **OpenL3**: Self-supervised deep audio representations trained on AudioSet.

#### 2. How to Train a Custom Metric Learning Model (Triplet Loss)

If you wish to train your own CNN (e.g., ResNet-18 or ConvNeXt) for music similarity:

1. **Prepare Triplet Batches**:
   - **Anchor ($A$)**: Original track.
   - **Positive ($P$)**: Track of the same artist, same sub-genre, or pitch/time-stretched variation of $A$.
   - **Negative ($N$)**: Track of a contrasting genre or distinct acoustic structure.
2. **Loss Function (Triplet Margin Loss)**:
   $$\mathcal{L}(A, P, N) = \max\Big(0,\; \mathcal{D}(f(A), f(P)) - \mathcal{D}(f(A), f(N)) + \alpha\Big)$$
   - $\mathcal{D}(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_2^2$ (Squared Euclidean Distance).
   - $\alpha$: Margin hyperparameter (typically $0.2 \le \alpha \le 0.4$).
3. **Training Objective**:
   The network adjusts its convolutional weights so that distance to the positive track is smaller than the distance to the negative track by at least margin $\alpha$.

---

## 4. How to Generate the Explainable "Why Similar" Attribution

One of the key contributions for an Explainable AI (XAI) thesis is providing an interpretable rationale for each recommendation.

### Attribution Algorithm

1. Compute the sub-scores:
   - $S_{\text{timbre}}$
   - $S_{\text{harmonics}}$
   - $S_{\text{rhythm}}$
   - $S_{\text{balance}}$
2. Find the dominant matching subspace:
   $$\text{Subspace}^* = \arg\max \left( S_{\text{timbre}}, S_{\text{harmonics}}, S_{\text{rhythm}}, S_{\text{balance}} \right)$$
3. Map the dominant subspace to a human-readable forensic explanation:

| Dominant Factor   | Condition                       | Generated Explanation                                                                                      |
| :---------------- | :------------------------------ | :--------------------------------------------------------------------------------------------------------- |
| **Timbre**        | $S_{\text{timbre}} \ge 0.88$    | _"Shares high-frequency rolloff ceiling (~6.3 kHz) and synthetic spectral saturation."_                    |
| **Harmonics**     | $S_{\text{harmonics}} \ge 0.85$ | _"Shares identical chromatic key distribution and circular chord progression (Tonnetz distance $<0.15$)."_ |
| **Rhythm**        | $S_{\text{rhythm}} \ge 0.90$    | _"Identical four-on-the-floor percussive tempo (~120 BPM) and synchronized transient onset decay."_        |
| **Tonal Balance** | $S_{\text{balance}} \ge 0.88$   | _"Matches sub-bass vs low-mid energy distribution (>60% energy concentrated in low frequencies)."_         |

---

## 5. Practical Implementation Blueprint

Here is the concrete implementation architecture for integrating local index retrieval into your project:

### Step 1: Offline Catalog Indexer (`scripts/index_reference_catalog.py`)

```python
import os
import glob
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Import your existing DSP extractor
from backend.app.services.dsp_service import extract_full_110d_features

CATALOG_DIR = "data/reference_catalog"
INDEX_OUTPUT = "backend/models/catalog_index.pkl"

def build_catalog_index():
    audio_files = glob.glob(os.path.join(CATALOG_DIR, "*.*"))
    feature_matrix = []
    metadata = []

    print(f"Indexing {len(audio_files)} reference songs...")
    for path in audio_files:
        try:
            feats = extract_full_110d_features(path)
            feature_matrix.append(feats)
            metadata.append({
                "path": path,
                "title": os.path.splitext(os.path.basename(path))[0],
                "artist": "Reference Catalog"
            })
        except Exception as e:
            print(f"Error processing {path}: {e}")

    X = np.array(feature_matrix)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Nearest Neighbors with Cosine Distance
    nn_model = NearestNeighbors(n_neighbors=5, metric="cosine")
    nn_model.fit(X_scaled)

    # Save scaler, index, and metadata
    bundle = {
        "scaler": scaler,
        "index": nn_model,
        "metadata": metadata,
        "features": X_scaled
    }
    with open(INDEX_OUTPUT, "wb") as f:
        pickle.dump(bundle, f)
    print("Catalog indexing complete!")

if __name__ == "__main__":
    build_catalog_index()
```

### Step 2: Querying from the Backend Service (`similarity_service.py`)

```python
import pickle
import numpy as np

INDEX_PATH = "backend/models/catalog_index.pkl"

class AcousticSimilarityEngine:
    def __init__(self):
        try:
            with open(INDEX_PATH, "rb") as f:
                self.bundle = pickle.load(f)
            self.scaler = self.bundle["scaler"]
            self.index = self.bundle["index"]
            self.metadata = self.bundle["metadata"]
            self.loaded = True
        except FileNotFoundError:
            self.loaded = False

    def query(self, query_110d_vector, top_k=5):
        if not self.loaded:
            return []

        # 1. Scale query vector using fitted scaler
        q_scaled = self.scaler.transform([query_110d_vector])

        # 2. Query k-nearest neighbors
        distances, indices = self.index.kneighbors(q_scaled, n_neighbors=top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            match_meta = self.metadata[idx]
            similarity_pct = int(round((1.0 - dist) * 100))

            # Compute explanation
            why = self.explain_similarity(q_scaled[0], self.bundle["features"][idx])

            results.append({
                "title": match_meta["title"],
                "artist": match_meta["artist"],
                "similarity_score": max(50, min(99, similarity_pct)),
                "why_similar": why
            })
        return results

    def explain_similarity(self, q, r):
        # Slice vectors into subspaces (indices corresponding to 110-D extraction)
        timbre_sim = np.dot(q[:40], r[:40]) / (np.linalg.norm(q[:40]) * np.linalg.norm(r[:40]) + 1e-9)
        chroma_sim = np.dot(q[40:70], r[40:70]) / (np.linalg.norm(q[40:70]) * np.linalg.norm(r[40:70]) + 1e-9)
        energy_sim = np.dot(q[70:], r[70:]) / (np.linalg.norm(q[70:]) * np.linalg.norm(r[70:]) + 1e-9)

        if timbre_sim >= chroma_sim and timbre_sim >= energy_sim:
            return "Shares spectral centroid brightness and high-frequency rolloff characteristics."
        elif chroma_sim >= energy_sim:
            return "Shares harmonic pitch chroma distributions and tonal chord intervals."
        else:
            return "Shares dynamics, sub-bass weight, and percussive transient energy profile."
```

---

## 6. Datasets for Training & Indexing

To build a reference catalog for your research:

1. **Free Music Archive (FMA)**: [FMA GitHub](https://github.com/mdeff/fma) — 106,574 tracks across 161 genres with full metadata and raw audio files (Creative Commons).
2. **MUSDB18**: Multi-track dataset with vocals, bass, drums, and other stems for isolated stem similarity.
3. **GTZAN Genre Collection**: 1,000 30-second audio excerpts across 10 genres (classic benchmark).
4. **Million Song Dataset (MSD)**: Large collection of audio features and metadata for commercial-scale benchmark comparisons.

---

## 7. Summary for Skripsi Writing

In Chapter 3 (Metodologi Penelitian) or Chapter 4 (Hasil dan Pembahasan) of your thesis, you can frame this feature as:

> _"Implementasi Sistem Temu Kembali Informasi Musik (Music Information Retrieval) Berbasis Ruang Vektor Fitur Akustik 110-Dimensi dan Penjelasan Sub-Ruang Frekuensi Terdistribusi (Explainable Audio Retrieval)."_

This shows that your system not only detects synthetic vs. authentic music using TreeSHAP, but also contextualizes where the audio sits in the acoustic landscape of human music recordings.
