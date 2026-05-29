# Piano Performance Style Analysis - ICASSP 2026

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

## Citation

If you use this research, please cite:

```
@inproceedings{piano_style_2026,
  title={Expressive Performance Style Analysis: Comparing Three Masters of Chinese Piano Performance},
  author={Your Name},
  booktitle={ICASSP 2026},
  year={2026}
}
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
