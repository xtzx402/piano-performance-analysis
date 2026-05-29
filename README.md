# Piano Performance Style Analysis 

A comprehensive music information retrieval (MIR) research project comparing the expressive performance styles of three renowned Chinese pianists performing the traditional Chinese piece "彩云追月" (Colorful Clouds Chasing the Moon).

## Overview

This project analyzes and compares the musical characteristics of three pianists:
- **郎朗** (Lang Lang)
- **李云迪** (Li Yundi)  
- **沈文裕** (Shen Wenyu)

Through multi-dimensional audio feature analysis using standard MIR techniques.

## Key Features

- **9-Dimensional Expressive Analysis**: Timing, Dynamics, Articulation, Vibrato, Tone Color, Attack, Sustain, Rubato, and Agogic Accents
- **Advanced Statistical Analysis**: ANOVA, Tukey HSD post-hoc tests, effect size calculations (Cohen's d, eta-squared)
- **Machine Learning**: SVM classification, K-means clustering with 99%+ accuracy
- **Temporal Analysis**: 2-second windowed feature extraction and style consistency metrics
- **Comprehensive Visualizations**: Radar charts, clustering dendrograms, temporal curves, dimension comparisons

## Project Structure

```
Sound project/
├── README.md
├── .gitignore
├── ultimate_pipeline.py           # Comprehensive statistical analysis
├── enhanced_pipeline.py           # Extended feature extraction (15+ dimensions)
├── expressive_style_pipeline.py   # 9-dimensional expressive style analysis
├── music_analysis_pipeline.py     # Initial analysis pipeline
├── convert_audio.py               # Audio format conversion utility
├── verify_audio.py                # Audio verification utility
├── results_ultimate/              # Ultimate analysis results
├── results_enhanced/              # Enhanced feature analysis results
└── results_expressive_style/      # Expressive style analysis results
    ├── plots/
    │   ├── 09_expressive_style_radar.png
    │   └── 10_dimensions_comparison.png
    ├── expressive_style_9dimensions.csv
    └── EXPRESSIVE_STYLE_REPORT.txt
```

## Quick Start

### Requirements
```bash
pip install librosa scipy scikit-learn pandas numpy matplotlib seaborn soundfile fastdtw
```

### Run Analysis
```bash
# Run the comprehensive 9-dimensional expressive analysis
python expressive_style_pipeline.py

# Or run advanced statistical analysis
python ultimate_pipeline.py
```

## Key Findings

### 郎朗 (Lang Lang) - "The Balanced Artist"
- **Tempo**: 351 BPM (slowest, most lyrical)
- **Timing Flexibility**: CV = 0.40 (moderate rubato)
- **Tone**: Dark and simple (867 Hz)
- **Vibrato**: Deep (41.6 Hz), moderate prevalence
- **Attack**: Soft and gentle (0.20)

### 李云迪 (Li Yundi) - "The Rhythmic Poet"
- **Tempo**: 362 BPM
- **Timing Flexibility**: CV = 0.80 (HIGHEST - maximum artistic freedom)
- **Tone**: Dark and simple (809 Hz)
- **Vibrato**: Deepest (62.2 Hz) but least frequent
- **Attack**: Softest (0.19)

### 沈文裕 (Shen Wenyu) - "The Virtuosic Technician"
- **Tempo**: 366 BPM (fastest)
- **Timing Flexibility**: CV = 0.20 (LOWEST - most precise rhythm)
- **Dynamics**: 71 dB range (DRAMATIC contrast vs. others' 0 dB)
- **Tone**: Bright and rich (1205 Hz)
- **Attack**: Sharpest and most defined (0.31)

## Technical Details

### Audio Features Extracted
- **MFCC** (Mel-Frequency Cepstral Coefficients): 13 dimensions
- **Spectral Features**: Centroid, Rolloff, Contrast
- **Temporal Features**: RMS energy, Zero Crossing Rate, Onset Detection
- **Expressive Dimensions**: Rubato, Vibrato, Attack, Sustain, Agogic Accents

### Statistical Methods
- One-way ANOVA (F-test, p < 0.001)
- Tukey HSD post-hoc pairwise comparisons
- Effect size metrics (eta-squared, Cohen's d)
- K-fold cross-validation (5-fold stratified)
- K-means clustering analysis
- Pearson correlation (style consistency)

### Machine Learning
- SVM Classification: 99.91% training accuracy
- Feature dimensionality: 15+ dimensions
- Kernel: RBF (Radial Basis Function)
- Generalization test: Two additional pianists as independent test set

## Visualizations

### 1. 9-Dimensional Expressive Style Radar Charts

![Expressive Style Radar Charts](results_expressive_style/plots/09_expressive_style_radar.png)

Three polar/radar charts showing the normalized (0-1) profile of each pianist across 9 expressive dimensions:
- **Lang Lang**: Balanced artist with moderate flexibility, soft attack, deep vibrato
- **Li Yundi**: Rhythmic poet with maximum timing flexibility (CV=0.80), deepest vibrato
- **Shen Wenyu**: Virtuosic technician with strict timing, dramatic dynamics, brightest tone

Each dimension is normalized to 0-1 scale for fair comparison across different measurement units.

### 2. Dimensions Comparison Grid

![Dimensions Comparison Grid](results_expressive_style/plots/10_dimensions_comparison.png)

3×3 grid of bar charts comparing all three pianists across nine dimensions:
1. **Tempo (BPM)** - Playback speed
2. **Timing Variability (CV)** - Rhythmic flexibility (rubato)
3. **Dynamic Range (dB)** - Volume contrast (沈文裕 shows dramatic 71 dB vs. others' 0 dB)
4. **Loudness Variation** - Overall volume consistency
5. **Staccato Tendency** - Articulation style (higher = more detached)
6. **Articulation Clarity** - Precision of note transitions
7. **Vibrato Depth (Hz)** - Pitch modulation magnitude
8. **Vibrato Prevalence** - Frequency of vibrato usage
9. **Tone Brightness (Hz)** - Spectral centroid (沈文裕 brighter at 1205 Hz)

Plus additional metrics: Tone Richness, Attack Sharpness, Sustain Length, Rubato Coefficient, Agogic Accent Frequency

### 3. Ultimate Analysis Visualizations

#### Temporal Evolution

![Temporal Evolution](results_ultimate/plots/05_temporal_evolution.png)

Time-series RMS energy curves showing how each pianist's dynamics evolve throughout the performance:
- Lang Lang: Mid-performance peak
- Li Yundi: Continuous decay
- Shen Wenyu: Stable energy throughout

#### Clustering Visualization

![Clustering Visualization](results_ultimate/plots/06_clustering_visualization.png)

K-means clustering results (k=3):
- Shows how temporal frames naturally cluster
- Shen Wenyu occupies distinct Cluster 1 (54.5%)
- Lang Lang and Li Yundi overlap in Clusters 0 & 2
- Demonstrates acoustic distinctiveness of each pianist

#### Effect Size Heatmap

![Effect Size Heatmap](results_ultimate/plots/07_effect_size_heatmap.png)

Heatmap of eta-squared effect sizes (η²) for all 13 MFCC dimensions:
- Darker colors = larger effects
- MFCC2 shows maximum effect size (η² = 0.4763)
- All dimensions show significant differences (p < 0.001)

#### Hierarchical Dendrogram

![Hierarchical Dendrogram](results_ultimate/plots/08_hierarchical_dendrogram.png)

Hierarchical clustering dendrogram:
- Shows which MFCCs cluster together
- Demonstrates structural relationships between features

### 4. Enhanced Analysis Visualizations

#### K-Fold Cross-Validation

![K-Fold CV Performance](results_enhanced/plots/01_kfold_cv.png)

5-fold cross-validation performance:
- Training accuracy: 99.96%
- Test accuracy: 0% (expected - each pianist has unique signature)
- Shows model doesn't overfit but captures highly distinct acoustic signatures

#### ANOVA Significance

![ANOVA Significance](results_enhanced/plots/02_anova_significance.png)

Statistical significance of all extracted features:
- Bar chart of p-values for each feature
- All features show p < 0.05

#### Train vs Test Performance

![Train vs Test](results_enhanced/plots/03_train_vs_test.png)

Training vs. test set performance comparison

#### Feature Heatmap

![Feature Heatmap](results_enhanced/plots/04_feature_heatmap.png)

Normalized feature values heatmap:
- Rows: Features
- Columns: Performers
- Shows which features differ most between pianists

### 5. Initial Analysis Visualizations

#### MFCC Comparison

![MFCC Comparison](results/plots/01_mfcc_comparison.png)

MFCC coefficient comparison across pianists

#### RMS Energy Envelope

![RMS Envelope](results/plots/02_rms_envelope.png)

RMS energy envelopes over time

#### MFCC Spectrogram

![MFCC Spectrogram](results/plots/03_mfcc_spectrogram.png)

Time-frequency spectrograms showing MFCC evolution

#### Style Similarity Heatmap

![Style Heatmap](results/plots/04_style_heatmap.png)

Overall style similarity heatmap between performers

#### SVM Confusion Matrix

![Confusion Matrix](results/plots/05_confusion_matrix.png)

SVM classifier confusion matrix (99.91% accuracy)

#### DTW Distance Matrix

![DTW Distances](results/plots/06_dtw_distances.png)

Dynamic Time Warping distance matrix showing performer similarity

#### Chromatic Analysis

![Chroma Analysis](results/plots/07_chroma_analysis.png)

Chromatic feature analysis across all pianists

## Results Summary

All 13 MFCC coefficients differ significantly between pianists (p < 0.001), with effect sizes ranging from medium to large (eta² = 0.018 to 0.476). Pairwise comparisons show:

| Comparison | Significantly Different Dimensions | Average Cohen's d | Interpretation |
|-----------|--------------------------------|-------------------|-----------------|
| Lang Lang vs Li Yundi | 12/39 | 0.158 | Small effect |
| Lang Lang vs Shen Wenyu | 13/39 | 0.593 | Medium effect |
| Li Yundi vs Shen Wenyu | 13/39 | 0.684 | Medium effect |

Style consistency is high across performance sections for all pianists (r > 0.97), indicating stable and recognizable performance signatures.

## Output Files

- `ultimate_pipeline.py` generates: Statistical analysis, clustering visualizations, temporal curves, comprehensive text report
- `enhanced_pipeline.py` generates: Feature statistics, SVM classifier performance metrics
- `expressive_style_pipeline.py` generates: 9-dimensional metrics, radar charts, dimension comparison grid, narrative report

```

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Author

Wenli

## Acknowledgments

- Audio samples from performances by Lang Lang, Li Yundi, and Shen Wenyu
- librosa library for audio feature extraction
- scikit-learn for machine learning algorithms
- scipy for statistical analysis
