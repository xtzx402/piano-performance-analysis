# Quantifying Piano Performance Style with MIR
## A Computational Analysis of 《彩云追月》 (Colorful Clouds Chasing the Moon)

This project applies music information retrieval (MIR) techniques to **quantify and explain performer-specific expressive strategies** in a shared piano repertoire. Nine recordings of Wang Jianzhong's 1975 arrangement of 彩云追月 — performed by pianists from professional stages, recording studios, and home environments — are analysed across multiple temporal scales: global pace, section-level dynamics and rubato, note-level agogic deviation, and per-onset attack sharpness.

The central research question is: **do acoustic measurements corroborate listener perception of expressive character, and if so, through which mechanisms?**

### Performers (N = 9)

| Performer | Recording Type | Duration | BPM | ForteRatio |
|-----------|---------------|----------|-----|-----------|
| 郎朗 Lang Lang | Live stage | 127.4 s | **135** | 10.4% |
| 李云迪 Li Yundi | Studio | 187.2 s | 92 | **4.0%** (lowest) |
| 沈文裕 Shen Wenyu | Home | 201.9 s | 85 | 10.4% |
| 殷承宗 Yin Chengzong | Live | 169.5 s | 101 | **27.9%** (highest) |
| Niemczuk | Studio | 215.8 s | **74** (slowest) | 7.5% |
| HIEW Tzejia | Studio | 213.6 s | 75 | 6.1% |
| Jasmine Wong | Studio | 183.6 s | 94 | 9.0% |
| 陈洁 Chen Jie | Unknown | 179.2 s | 96 | 8.7% |
| Live (Anon) | Live | 221.9 s | 75 | 4.6% |

*BPM = median over the piece via DTW-aligned sliding window (score baseline = 114 BPM)*

---

### Methodological Contributions

Three aspects of the pipeline address challenges specific to cross-recording performance comparison:

1. **Device-independent dynamics**: A two-step noise-floor subtraction and peak-normalisation removes recording-gain and microphone-noise differences before any amplitude comparison, enabling valid dynamics comparison across live, studio, and home recordings.

2. **MIDI-grounded score alignment**: Rather than audio-to-audio DTW (which fails across different recording environments), performer onsets are aligned to MIDI score onsets on a shared time axis via fastdtw on 1D onset sequences. This yields per-note agogic deviation in milliseconds without timbre-matching assumptions.

3. **Multi-scale analysis framework**: Features are extracted at four nested scales — global pace ratio, per-section medians, per-note alignment, and per-onset attack slope — making it possible to distinguish macro-level interpretation choices from micro-level expressive gestures.

---

## Results: Core Findings (validated across N = 9)

### Finding 1 — The Lang Lang Effect: Speed, Not Volume

Lang Lang plays at **135 BPM** — 18% faster than the written score (114 BPM) and the fastest of all nine performers. His note density is the highest.

But his sustained loudness (ForteRatio 10.4%) is mid-range, and his measured attack sharpness (0.113) is the **lowest of the nine**. His "powerful" impression is generated almost entirely by **tempo and note density**, not by actually hitting the keys harder or louder. Even with 9 performers spanning a wide range of styles, this counterintuitive gap is clear.

| Performer | BPM | ForteRatio | Attack Sharpness |
|-----------|-----|-----------|-----------------|
| Lang Lang | **135** | 10.4% | 0.113 † (lowest) |
| Yin Chengzong | 101 | **27.9%** | 0.177 |
| Jasmine Wong | 94 | 9.0% | 0.200 |
| Chen Jie | 96 | 8.7% | 0.200 |
| Li Yundi | 92 | 4.0% | 0.225 |
| Shen Wenyu | 85 | 10.4% | 0.203 |
| HIEW Tzejia | 75 | 6.1% | **0.251** (highest) |
| Live (Anon) | 75 | 4.6% | 0.188 |
| Niemczuk | 74 | 7.5% | 0.186 |
| Score baseline | 114 | — | — |

*† Lang Lang's attack sharpness is underestimated due to live hall reverb smearing onsets.*

---

### Finding 2 — The Li Yundi Paradox: Precision Without Volume

Li Yundi has the **second-highest attack sharpness** (0.225) and the **fastest note decay rate** among the main performers — meaning his individual notes are the most clearly defined and shortest-lasting. He is the most articulate, most precisely controlled pianist in terms of touch.

Yet his sustained loudness is the **lowest of all nine** (ForteRatio 4.0%).

**Clarity ≠ Loudness.** Li Yundi deploys precise, well-articulated touches in service of a pianissimo dynamic palette. His "lyrical" quality is built from exact, controlled softness — not vague legato blurring.

This paradox is only visible when attack sharpness and sustained dynamics are measured separately. Notably, HIEW Tzejia shows a similar pattern at larger scale (highest attack 0.251, mid-range ForteRatio 6.1%), suggesting this attack–dynamics decoupling strategy is not unique to one performer.

---

### Finding 3 — The Shen Wenyu Paradox: Hidden Dynamic Range

Shen Wenyu's home recording has one of the lower absolute amplitudes. A naive amplitude comparison would underestimate his intensity.

After **noise-subtracted peak-normalisation**:
- His cadenza (华彩) ForteRatio reaches **28.8%** — 7.4× Li Yundi's 3.9%
- His overall ForteRatio (10.4%) matches Lang Lang despite the quieter recording environment

The dynamic contrast that defines his "intense" impression is real, but **invisible without device-independent measurement**.

The expanded dataset reveals an even more striking case: **殷承宗 Yin Chengzong** (co-arranger of this piece with Wang Jianzhong) achieves a 华彩 ForteRatio of **56.2%** — the most extreme forte–piano contrast of all nine recordings. His mean decay rate (−0.083 RMS/s) is also the fastest, indicating the most percussive, least-pedalled articulation style overall.

---

### Finding 4 — Macro Stability, Micro Freedom (Universal Pattern)

Despite widely different styles, **all nine performers share one structural property**: tempo is consistent within each performer's reading across sections. No performer uses section-level tempo changes as an expressive tool. Instead, all musical expression occurs at the **note level**: per-note agogic deviations, concentrated especially in the coda (尾声).

Tempo clusters clearly into three groups:
- **Fast**: Lang Lang (~135 BPM)
- **Medium**: Yin Chengzong (~101), Chen Jie (~96), Jasmine Wong (~94), Li Yundi (~92)
- **Slow**: Shen Wenyu (~85), HIEW Tzejia (~81), Niemczuk (~80), Live (Anon) (~78)

Notably, **Niemczuk** (the only non-Chinese interpreter in the dataset) plays at the slowest global pace (74 BPM global average) — suggesting a different expressive framework, though conclusions from a single recording are speculative.

The **cadenza (华彩)** is the section of maximum divergence across all dimensions — ForteRatio spans from 3.9% (Li Yundi) to 56.2% (Yin Chengzong), making it the strongest single discriminant of style across all nine performers.

---

### Discussion: Answering the Research Question

> *Do acoustic measurements corroborate listener perception of expressive character, and if so, through which mechanisms?*

The expanded dataset of nine performers provides stronger support for the finding that perceived character maps reliably to measurable acoustic structure — but the mechanism is often counterintuitive.

**Lang Lang's "energetic" impression** is confirmed acoustically, but its source is tempo (135 BPM) and note density rather than dynamic intensity. His ForteRatio is mid-range and his attack sharpness is the lowest of the nine (partly confounded by hall reverb). Perceived energy is a rate effect, not an amplitude effect.

**Li Yundi's "lyrical" impression** is confirmed, but not through legato blurring. He is among the most precisely articulate performers: second-highest attack sharpness and fastest decay. His lyricism arises from applying sharp, well-defined touches at the softest dynamic level — precision in service of quietness. The dataset now shows a similar pattern in HIEW Tzejia, suggesting this strategy is systematically available.

**Shen Wenyu's "intense" impression** is acoustically grounded, but only observable through device-independent normalisation. The expanded dataset reveals that **Yin Chengzong** represents an even more extreme dynamic performer — a finding invisible from raw amplitude alone.

These results suggest that **listener perception reliably tracks performer intent at the level of expressive strategy**, even when the specific acoustic mechanism differs from naive expectation. The finding generalises beyond the original three performers: across nine recordings spanning live, studio, and home environments, the same decoupling between raw amplitude and normalised dynamic contrast is consistently visible.

---

## Performer Profiles

**郎朗 Lang Lang — The Velocity Architect**
- Fastest tempo (135 BPM, 18% above score); note density drives perceived energy
- ForteRatio 10.4%; attack sharpness lowest (0.113, partly reverb)
- "Energetic" impression is a tempo effect, not an amplitude effect

**李云迪 Li Yundi — The Articulate Whisperer**
- Second-sharpest attack (0.225) and fastest decay — most precisely defined note boundaries
- Softest sustained dynamics of all nine (ForteRatio 4.0%); highest rhythmic freedom
- Coda (尾声): near-silent with extreme temporal freedom (agogic std ≈ 470 ms)

**沈文裕 Shen Wenyu — The Dynamic Extremist (Home)**
- Most extreme forte–piano contrast among studio/home recordings; cadenza ForteRatio = 28.8%
- Home recording is quiet in absolute terms — paradox invisible without normalisation
- Slowest tempo in the original trio (85 BPM)

**殷承宗 Yin Chengzong — The Co-Arranger's Authority**
- Highest ForteRatio of all nine (27.9% global, 56.2% in cadenza)
- Fastest note decay rate (−0.083 RMS/s) — most percussive, least-pedalled articulation
- Plays at 101 BPM — close to the written score tempo (114 BPM)
- As co-arranger, likely plays closest to the intended character of the piece

**Niemczuk — The Cross-Cultural Interpreter**
- Slowest global tempo (74 BPM, 35% below score baseline)
- Mid-range dynamics and attack — metrically spacious, not extreme in any dimension
- Only non-Chinese interpreter; suggests different expressive pacing framework

**HIEW Tzejia — The Precise Articulator**
- Highest attack sharpness of all nine (0.251) — most percussive onset character
- Mid-range ForteRatio (6.1%) — high attack without high sustained loudness
- Replicates the attack/dynamics decoupling pattern of Li Yundi at a different scale

**Jasmine Wong — The Balanced Interpreter**
- Well-rounded profile: mid-range BPM (94), ForteRatio (9.0%), attack (0.200)
- No single extreme dimension; balanced expression across all metrics

**陈洁 Chen Jie — The Quiet Articulator**
- Lowest raw peak RMS of all nine recordings
- Moderate timing flexibility (CV 0.224); attack 0.200; low ForteRatio (8.7%)

**Live (Anon) — The Slow Expressionist**
- Slowest of the live recordings (75 BPM); highest rhythmic flexibility (CV 0.255)
- ForteRatio 4.6% — soft dynamics maintained throughout

---

## Methodology

### Score Baseline

All analyses are anchored to the Wang Jianzhong MIDI (`cai-yun-zhui-yue-ren-guang-qu-wang-jian-zhong-gai-bian.mid`) at 114 BPM as mechanical ground truth.

### Device-Independent Dynamics

Two-step debiasing applied before any cross-recording amplitude comparison:
1. **Noise floor subtraction**: 5th-percentile RMS frame subtracted from every frame
2. **Peak normalisation**: residual scaled to max = 1.0

Removes microphone noise bias and recording gain differences. What remains is relative dynamic effort within each recording.

### Score Alignment (Note-Level)

936 unique MIDI onset times are extracted from the score. Performer onsets are detected via `librosa.peak_pick`. Both sequences are normalised to the same time axis (dividing by the pace ratio), then aligned with **fastdtw** on 1D onset time sequences (radius = 60). This computes per-note agogic deviation in milliseconds without audio-to-audio DTW, which fails across different recording environments.

### Attack Sharpness

Peak-normalised onset strength at each detected onset. Captures how suddenly each note's amplitude rises (percussive vs smooth attack). Note: live hall reverb attenuates measured onset sharpness for live recordings (Lang Lang, Yin Chengzong, Live Anon).

### Local BPM

Sliding window of 30 unique score onsets. For each window: `local_BPM = 114 × (delta_score / delta_perf)`. Mapped to score time for cross-performer alignment.

---

## Audio Preprocessing

All audio normalised to 22050 Hz mono. Leading silence trimmed where present (threshold: 1% of peak RMS). See `performers.json` for full metadata.

| Pianist | Recording Type | Duration | Preprocessing |
|---------|---------------|----------|--------------|
| 郎朗 Lang Lang | Live stage | 127.4 s | Sample rate normalised |
| 李云迪 Li Yundi | Studio | 187.2 s | — |
| 沈文裕 Shen Wenyu | Home | 201.9 s | 7.4 s pre-music noise trimmed |
| 殷承宗 Yin Chengzong | Live | 169.5 s | — |
| Niemczuk | Studio | 215.8 s | 7.5 s leading silence trimmed |
| HIEW Tzejia | Studio | 213.6 s | 3.4 s leading silence trimmed |
| Jasmine Wong | Studio | 183.6 s | 1.4 s leading silence trimmed |
| 陈洁 Chen Jie | Unknown | 179.2 s | — |
| Live (Anon) | Live | 221.9 s | — |

---

## Project Structure

```
Sound project/
├── ANALYSIS SCRIPTS
├── score_vs_performers.py       # Global pace, dynamics, rubato vs score
├── section_analysis.py          # Per-section features (A段/B段/华彩/尾声)
├── note_alignment.py            # Note-level agogic deviation via DTW
├── tempo_analysis.py            # Local BPM curves from aligned onsets
├── attack_analysis.py           # Attack sharpness, decay rate, fingerprint
├── dynamics_paradox.py          # Device-independent dynamics visualisation
├── ultimate_pipeline.py         # MFCC statistical classification
├── enhanced_pipeline.py         # Extended feature extraction
├── expressive_style_pipeline.py # 9-dimensional expressive analysis
│
├── DATASET MANAGEMENT
├── performers.json              # Central performer registry (name, file, color, metadata)
├── config.py                    # Shared constants loaded by all analysis scripts
├── add_performer.py             # CLI helper: preprocess + register new recordings
│
├── SCORE REFERENCE
├── synthesize_reference.py
├── cai-yun-zhui-yue-ren-guang-qu-...mid   # Real MIDI from OMR
│
├── AUDIO
├── normalized_audio/            # Preprocessed WAVs (git-ignored)
│
└── results_ultimate/
    ├── plots/
    │   ├── 12_score_vs_performers_curves.png
    │   ├── 13_score_deviation_bars.png
    │   ├── 14_section_comparison.png      # Section ForteRatio grouped bar
    │   ├── 15_section_profiles.png        # Per-performer section bar grid
    │   ├── 16_section_rubato_dynamics.png # Timing CV + ForteRatio line plots
    │   ├── 17_agogic_deviation.png        # Per-note timing deviation curves
    │   ├── 18_agogic_overlay.png          # All performers overlaid
    │   ├── 19_agogic_boxplot.png          # Box plots per section
    │   ├── 20_dynamics_paradox.png        # Raw vs normalised RMS
    │   ├── 21_local_bpm_curves.png        # Local BPM overlay
    │   ├── 22_local_bpm_subplots.png      # BPM per performer
    │   ├── 23_section_bpm_bars.png        # Section median BPM bars
    │   ├── 24_attack_sharpness_curves.png # Attack sharpness over time
    │   ├── 25_liyundi_paradox.png         # Attack × dynamics bubble chart
    │   ├── 26_decay_rate_bars.png         # Note decay rate per section
    │   └── 27_performer_fingerprint.png   # 5-dim radar fingerprint
    ├── note_alignment.csv
    ├── section_analysis.csv
    ├── attack_analysis.csv
    └── score_vs_performers.csv
```

---

## Key Visualisations

### Performer Fingerprint
![Fingerprint](results_ultimate/plots/27_performer_fingerprint.png)

Five-dimensional normalised radar: Speed, Dynamics, Attack Sharpness, Rubato, Sustain. The nine performers occupy distinctly different regions of the feature space.

### Li Yundi Paradox
![Li Yundi Paradox](results_ultimate/plots/25_liyundi_paradox.png)

Bubble chart: X = attack sharpness, Y = ForteRatio, bubble size = tempo. Li Yundi sits at upper-left (sharp attack, soft dynamics). Lang Lang has the largest bubble (tempo as main differentiator). HIEW Tzejia has the rightmost position (highest attack).

### Dynamics Paradox
![Dynamics Paradox](results_ultimate/plots/20_dynamics_paradox.png)

Raw vs device-independent RMS. Shen Wenyu is among the quietest in absolute level but shows high dynamic contrast after normalisation. Yin Chengzong shows the most extreme cadenza ForteRatio (56.2%) of all nine.

### Note-Level Agogic Deviation
![Agogic Deviation](results_ultimate/plots/17_agogic_deviation.png)

Per-note timing deviation from expected beat. The lyrical interlude (B段) is the most metronomic section across all performers; Li Yundi's coda (尾声) has extreme temporal freedom.

### Local BPM
![Local BPM](results_ultimate/plots/21_local_bpm_curves.png)

All nine performers maintain consistent tempo within their own reading throughout the piece. Expression is at the note level, not the architectural level.

### Section Comparison
![Section Comparison](results_ultimate/plots/14_section_comparison.png)

The cadenza (华彩) is the widest point of divergence across all metrics and all performers.

---

## Tech Stack

| Library | Version | Role |
|---------|---------|------|
| [librosa](https://librosa.org) | ≥ 0.10 | Audio loading, onset detection, MFCC, RMS, ZCR, onset strength |
| [fastdtw](https://github.com/slaypni/fastdtw) | ≥ 0.3 | Approximate DTW for score-to-performer onset alignment |
| [mido](https://mido.readthedocs.io) | ≥ 1.3 | MIDI parsing — tempo map and note event extraction |
| [numpy](https://numpy.org) | ≥ 1.24 | Numerical computation throughout |
| [scipy](https://scipy.org) | ≥ 1.11 | `linregress` for decay slope; `uniform_filter1d` for smoothing |
| [pandas](https://pandas.pydata.org) | ≥ 2.0 | Feature tables and CSV export |
| [scikit-learn](https://scikit-learn.org) | ≥ 1.3 | SVM classification, K-means clustering, ANOVA, Tukey HSD |
| [matplotlib](https://matplotlib.org) | ≥ 3.7 | All visualisations (line plots, box plots, radar charts, scatter) |
| [soundfile](https://pysoundfile.readthedocs.io) | ≥ 0.12 | WAV writing for preprocessed audio |

Python ≥ 3.10 recommended.

## Quick Start

```bash
pip install -r requirements.txt
python synthesize_reference.py     # build score WAV (once)

# Add new performers
python add_performer.py <audio_file> "<Name>" --type "Studio" --notes "..."

# Run full analysis pipeline
python score_vs_performers.py      # global comparison
python section_analysis.py         # section-by-section
python note_alignment.py           # note-level DTW  (run before tempo_analysis)
python tempo_analysis.py           # local BPM
python attack_analysis.py          # attack + fingerprint
python dynamics_paradox.py         # dynamics paradox
```

---

## Limitations

- Single piece per performer; findings characterise this recording, not the performers' full artistic identity
- Recording conditions differ (live / studio / home); debiasing mitigates but cannot fully eliminate environmental effects — live hall reverb attenuates measured attack sharpness for Lang Lang, Yin Chengzong, and Live (Anon)
- Acoustic features capture *what* happens but not *why*; differences may reflect technique, musical philosophy, instrument, score edition, or recording chain
- Cross-cultural comparisons (Niemczuk vs. Chinese-trained pianists) are speculative from a single recording each

---

## License

MIT License — Wenli

## Acknowledgments

Performances by Lang Lang, Li Yundi, Shen Wenyu, Yin Chengzong, Niemczuk, HIEW Tzejia, Jasmine Wong, Chen Jie, and anonymous performers · Wang Jianzhong and 殷承宗 Yin Chengzong piano arrangement (1975)
librosa · scikit-learn · scipy · fastdtw · mido
