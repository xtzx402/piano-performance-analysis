#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Score vs Performers Comparison
Baseline: mechanical MIDI score (Wang Jianzhong arrangement, 114 BPM, CV=0)
Performers: normalized WAVs (onset-aligned, same SR, same RMS)
Metrics: pace deviation, spectral brightness, timing flexibility (rubato)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

base       = Path(__file__).parent
plots_path = base / 'results_ultimate' / 'plots'
plots_path.mkdir(parents=True, exist_ok=True)

# ── Score baseline (from MIDI metadata — no onset detection needed) ─────────
SCORE_BPM      = 114.0    # set_tempo from MIDI file
SCORE_DURATION = 150.5    # synthesized WAV duration (seconds)
SCORE_CV       = 0.0      # mechanical = zero timing variability by definition
SCORE_COLOR    = '#2ECC71'

# Synthesized WAV used only for spectral centroid (tone colour of the arrangement)
SCORE_WAV = base / 'reference_score.wav'

from config import PERFORMERS as _PERFORMERS
PERFORMERS = {path: (name, color) for name, (path, color) in _PERFORMERS.items()}
HOP = 512

print("=" * 70)
print("Score vs Performers: Deviation from Score Baseline")
print("=" * 70)
print(f"\nScore baseline  : {SCORE_DURATION}s  |  {SCORE_BPM} BPM  |  Timing CV = {SCORE_CV}  (mechanical)")

# ── Load audio ──────────────────────────────────────────────────────────────
print("\nLoading normalized performer files...")
performers_audio = {}
for path, (name, color) in PERFORMERS.items():
    y, sr = librosa.load(str(path), sr=22050)
    performers_audio[name] = {'y': y, 'sr': sr, 'color': color,
                               'duration': librosa.get_duration(y=y, sr=sr)}
    print(f"  {name:<12}: {performers_audio[name]['duration']:.1f}s")

# Score WAV for centroid only
y_score, sr = librosa.load(str(SCORE_WAV), sr=22050)
score_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y_score, sr=sr, hop_length=HOP)[0]))

# ── Feature extraction ───────────────────────────────────────────────────────
def noise_subtracted_peak_normalised_rms(y, hop=HOP):
    """
    Two-step debiasing for cross-recording amplitude comparison:
      1. Subtract noise floor (5th-percentile RMS) — removes mic/room noise
      2. Normalise to peak = 1.0 — removes recording gain differences
    What remains reflects the performer's own dynamic choices.
    """
    rms        = librosa.feature.rms(y=y, hop_length=hop)[0]
    noise_floor = np.percentile(rms, 5)            # quietest 5% = noise estimate
    rms_clean  = np.maximum(rms - noise_floor, 0)  # subtract and clamp
    peak       = np.max(rms_clean)
    rms_norm   = rms_clean / (peak + 1e-10)        # scale so max = 1.0
    return rms_norm, noise_floor, peak

def extract_features(y, sr, hop=HOP):
    rms_raw = librosa.feature.rms(y=y, hop_length=hop)[0]
    times   = librosa.times_like(rms_raw, sr=sr, hop_length=hop)
    zcr     = librosa.feature.zero_crossing_rate(y=y, hop_length=hop)[0]

    # Noise-subtracted, peak-normalised RMS for cross-recording dynamics
    rms_norm, noise_floor, signal_peak = noise_subtracted_peak_normalised_rms(y, hop)

    onset_env    = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    onset_frames = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3,
                                          pre_avg=3, post_avg=3, delta=0.1, wait=10)
    onset_times  = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop)
    ioi = np.diff(onset_times)
    ioi = ioi[ioi > 0.05]

    duration = librosa.get_duration(y=y, sr=sr)

    # Dynamic features on clean, normalised signal (comparable across recordings)
    mean_intensity     = float(np.mean(rms_norm))          # avg effort (0-1)
    high_intens_ratio  = float(np.mean(rms_norm > 0.6))    # % time in forte region
    dynamic_range_cv   = float(np.std(rms_norm) / (np.mean(rms_norm) + 1e-10))  # variation

    # ZCR corrected for noise floor contribution
    # Estimate noise ZCR from quietest frames
    quiet_mask = rms_raw < np.percentile(rms_raw, 10)
    noise_zcr  = float(np.mean(zcr[quiet_mask])) if quiet_mask.any() else 0
    zcr_clean  = np.maximum(zcr - noise_zcr, 0)

    return {
        'times': times, 'rms': rms_norm, 'zcr': zcr_clean,
        'onset_times': onset_times, 'ioi': ioi,
        'noise_floor':      float(noise_floor),
        'signal_peak':      float(signal_peak),
        'mean_intensity':   mean_intensity,        # avg effort level (device-independent)
        'high_intens_ratio': high_intens_ratio,    # % time playing forte
        'dynamic_range_cv': dynamic_range_cv,      # variation in intensity
        'timing_cv':        float(np.std(ioi) / np.mean(ioi)) if len(ioi) > 1 else 0,
        'onset_density':    len(onset_times) / duration,
        'avg_zcr':          float(np.mean(zcr_clean)),
    }

print("\nExtracting performer features...")
perf_feat = {name: extract_features(d['y'], d['sr'])
             for name, d in performers_audio.items()}

# ── Summary table ────────────────────────────────────────────────────────────
# Pre-normalization RMS (original loudness) — from raw files
raw_files = {
    'Lang Lang':  base / 'langlang_caiyun.wav',
    'Li Yundi':   base / 'liyundi_caiyun.wav',
    'Shen Wenyu': base / 'shenwenyu_caiyun.wav',
}
print("\nReading original loudness from raw files...")
original_rms = {}
for name, path in raw_files.items():
    y_raw, sr_raw = librosa.load(str(path), sr=None)
    original_rms[name] = float(np.sqrt(np.mean(y_raw**2)))
    print(f"  {name:<12}: original RMS = {original_rms[name]:.4f}")

print("\n" + "=" * 70)
print("Deviation from Score Baseline  (dynamics: noise-subtracted, peak-normalised)")
print("=" * 70)
print(f"\n  {'':14} {'Duration':>9} {'Pace':>8} {'MeanIntens':>11} {'ForteRatio':>11} {'DynCV':>7} {'ZCR':>8} {'Density':>9} {'TimingCV':>10}")
print("  " + "-" * 96)

rows = []
for name, feat in perf_feat.items():
    dur      = performers_audio[name]['duration']
    pace_pct = (dur / SCORE_DURATION - 1) * 100
    mi       = feat['mean_intensity']
    hir      = feat['high_intens_ratio']
    dcv      = feat['dynamic_range_cv']
    zcr      = feat['avg_zcr']
    dens     = feat['onset_density']
    cv       = feat['timing_cv']
    nf       = feat['noise_floor']
    print(f"  {name:<14} {dur:>7.1f}s {pace_pct:>+7.1f}%  {mi:>11.3f}  {hir:>10.1%}  {dcv:>6.3f}  {zcr:>7.4f}  {dens:>7.2f}/s  {cv:>9.4f}  [noise={nf:.4f}]")
    rows.append({'Performer': name, 'Duration (s)': dur,
                 'Pace vs Score (%)': round(pace_pct, 1),
                 'Mean Intensity (0-1)': round(mi, 3),
                 'Forte Ratio (%)': round(hir * 100, 1),
                 'Dynamic Range CV': round(dcv, 3),
                 'Avg ZCR (clean)': round(zcr, 4),
                 'Onset Density (/s)': round(dens, 2),
                 'Timing CV': round(cv, 4)})

print(f"  {'Score (ref)':<14} {SCORE_DURATION:>7.1f}s {'±0.0%':>8}  {'—':>11}  {'—':>10}  {'—':>6}  {'—':>7}  {'—':>7}  {SCORE_CV:>9.4f}")

df = pd.DataFrame(rows)
df.to_csv(base / 'results_ultimate' / 'score_vs_performers.csv', index=False, encoding='utf-8-sig')

# ── Plot 1: Time-series curves (normalised 0-100%) ───────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
fig.suptitle('Score vs Performers: Feature Evolution\n'
             '(Time axis normalised to 0–100% of each performance)',
             fontsize=13, fontweight='bold')

# Score spectral centroid curve
sc_times  = librosa.times_like(librosa.feature.spectral_centroid(y=y_score, sr=sr, hop_length=HOP)[0],
                                sr=sr, hop_length=HOP)
sc_cent   = librosa.feature.spectral_centroid(y=y_score, sr=sr, hop_length=HOP)[0]
sc_rms    = librosa.feature.rms(y=y_score, hop_length=HOP)[0]
sc_pct    = sc_times / SCORE_DURATION * 100

# RMS energy shape
ax = axes[0]
ax.plot(sc_pct, sc_rms, color=SCORE_COLOR, lw=2.5, linestyle='--',
        alpha=0.9, label='Score (mechanical, normalised)')
for name, feat in perf_feat.items():
    dur = performers_audio[name]['duration']
    pct = feat['times'] / dur * 100
    ax.plot(pct, feat['rms'], color=performers_audio[name]['color'],
            lw=1.5, alpha=0.8, label=name)
ax.set_ylabel('RMS Energy (normalised)', fontsize=10)
ax.set_title('Dynamic Shape — how loudness evolves relative to score', fontsize=11)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.25)

# ZCR (articulation proxy)
ax = axes[1]
sc_zcr = librosa.feature.zero_crossing_rate(y=y_score, hop_length=HOP)[0]
ax.plot(sc_pct[:len(sc_zcr)], sc_zcr, color=SCORE_COLOR, lw=2.5, linestyle='--',
        alpha=0.9, label='Score (mechanical)')
for name, feat in perf_feat.items():
    dur = performers_audio[name]['duration']
    pct = feat['times'] / dur * 100
    min_len = min(len(pct), len(feat['zcr']))
    ax.plot(pct[:min_len], feat['zcr'][:min_len],
            color=performers_audio[name]['color'], lw=1.5, alpha=0.8, label=name)
ax.set_xlabel('Performance Progress (%)', fontsize=10)
ax.set_ylabel('Zero Crossing Rate', fontsize=10)
ax.set_title('Articulation (ZCR) — staccato vs legato relative to score', fontsize=11)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.25)

plt.tight_layout()
out1 = plots_path / '12_score_vs_performers_curves.png'
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Saved: {out1.name}")

# ── Plot 2: Deviation bar charts (2x3 grid) ──────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Quantified Deviation from Score Baseline\n'
             f'(Score: {SCORE_DURATION}s, {SCORE_BPM} BPM, Timing CV = 0)',
             fontsize=13, fontweight='bold')

names  = list(perf_feat.keys())
colors = [performers_audio[n]['color'] for n in names]

def bar_plot(ax, vals, title, ylabel, ref_val=None, ref_label=None, fmt='.2f', ref_color='black'):
    bars = ax.bar(names, vals, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
    if ref_val is not None:
        ax.axhline(ref_val, color=ref_color, lw=1.8, linestyle='--',
                   label=ref_label or f'Score = {ref_val}')
        ax.legend(fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.25, axis='y')
    ax.set_xticklabels(names, fontsize=9)
    for bar, val in zip(bars, vals):
        label = f'{val:{fmt}}'
        yoff  = (max(vals) - min(vals)) * 0.03
        ax.text(bar.get_x() + bar.get_width()/2,
                val + yoff, label, ha='center', fontsize=10, fontweight='bold')

# Row 0
bar_plot(axes[0,0],
         [(performers_audio[n]['duration'] / SCORE_DURATION - 1)*100 for n in names],
         f'Pace vs Score\n(+% = slower than {SCORE_DURATION}s baseline)',
         'Duration deviation (%)', ref_val=0, ref_label='Score baseline', fmt='+.1f')

bar_plot(axes[0,1],
         [perf_feat[n]['mean_intensity'] for n in names],
         'Mean Intensity\n(noise-subtracted, peak-normalised)\nhigher = plays with more sustained effort',
         'Mean intensity (0–1)', fmt='.3f')

bar_plot(axes[0,2],
         [perf_feat[n]['high_intens_ratio'] * 100 for n in names],
         'Forte Ratio\n(% time above 60% of own peak)\nhigher = more time at high force',
         '% time in forte', fmt='.1f')

# Row 1
bar_plot(axes[1,0],
         [perf_feat[n]['dynamic_range_cv'] for n in names],
         'Dynamic Range (CV)\n(std/mean of clean RMS)\nhigher = larger loud/soft swings',
         'Intensity CV', fmt='.3f')

bar_plot(axes[1,1],
         [perf_feat[n]['onset_density'] for n in names],
         'Onset Density\n(notes per second)',
         'Onsets / second', fmt='.2f')

bar_plot(axes[1,2],
         [perf_feat[n]['timing_cv'] for n in names],
         'Timing Flexibility / Rubato\n(IOI CV vs 0 for mechanical score)',
         'IOI Coefficient of Variation',
         ref_val=SCORE_CV, ref_label='Score CV = 0 (mechanical)', fmt='.3f')

plt.tight_layout()
out2 = plots_path / '13_score_deviation_bars.png'
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {out2.name}")

# ── Print key findings ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Key Findings: How Each Performer Interprets the Score")
print("=" * 70)

for name, feat in perf_feat.items():
    dur      = performers_audio[name]['duration']
    pace_pct = (dur / SCORE_DURATION - 1) * 100

    mi   = feat['mean_intensity']
    hir  = feat['high_intens_ratio']
    dcv  = feat['dynamic_range_cv']
    zcr  = feat['avg_zcr']
    dens = feat['onset_density']
    cv   = feat['timing_cv']
    nf   = feat['noise_floor']

    print(f"\n{name}:  [noise floor = {nf:.4f}]")
    print(f"  Pace          : {pace_pct:+.1f}% vs score  "
          f"({'slower — lyrical' if pace_pct > 15 else 'slightly slower' if pace_pct > 0 else 'faster — driven'})")
    print(f"  Mean intensity: {mi:.3f}  "
          f"({'high sustained effort' if mi > 0.35 else 'moderate' if mi > 0.25 else 'restrained'})")
    print(f"  Forte ratio   : {hir:.1%} of time above 60% peak  "
          f"({'frequently loud' if hir > 0.3 else 'occasional fortes' if hir > 0.15 else 'mostly soft'})")
    print(f"  Dynamic range : CV = {dcv:.3f}  "
          f"({'wide strong/soft contrasts' if dcv > 0.6 else 'moderate' if dcv > 0.4 else 'even/controlled'})")
    print(f"  Note density  : {dens:.2f} onsets/s  "
          f"({'dense' if dens > 5 else 'moderate' if dens > 3 else 'spacious'})")
    print(f"  Rubato        : CV = {cv:.3f}  "
          f"({'high' if cv > 0.25 else 'moderate' if cv > 0.15 else 'low'} timing freedom)")

print("\n" + "=" * 70)
print("Done. Results saved to results_ultimate/")
print("=" * 70)
