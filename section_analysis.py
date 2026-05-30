#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section-by-Section Analysis of 彩云追月
Sections derived from MIDI structure (largest gap at 76.84s, density peak at 90s):
  A段  (Theme A)  : 0    – 60.8s
  B段  (Lyrical)  : 60.8 – 76.8s
  华彩 (Cadenza)  : 76.8 – 130.0s
  尾声 (Coda)     : 130.0– 151.6s
Each performer's section boundaries are scaled linearly by their duration ratio.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

base = Path(__file__).parent
out  = base / 'results_ultimate' / 'plots'
out.mkdir(parents=True, exist_ok=True)

# ── Score section boundaries (seconds at 114 BPM) ───────────────────────────
SCORE_DURATION = 151.6
SECTIONS = [
    ('A段\n主题',   0.0,   60.8),
    ('B段\n抒情',  60.8,   76.8),
    ('华彩\n技巧',  76.8,  130.0),
    ('尾声',       130.0,  151.6),
]
SECTION_NAMES = [s[0] for s in SECTIONS]

# ── Performers ───────────────────────────────────────────────────────────────
PERFORMERS = {
    'Lang Lang':  (base / 'normalized_audio' / 'normalized_langlang_caiyun.wav',  '#E74C3C'),
    'Li Yundi':   (base / 'normalized_audio' / 'normalized_liyundi_caiyun.wav',   '#F39C12'),
    'Shen Wenyu': (base / 'normalized_audio' / 'normalized_shenwenyu_caiyun.wav', '#3498DB'),
}

HOP = 512
SR  = 22050

# ── Feature extraction helpers ───────────────────────────────────────────────
def noise_floor(y):
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    return float(np.percentile(rms, 5))

def section_features(y, sr, start_sec, end_sec):
    """Extract features for a slice of audio."""
    s = int(start_sec * sr)
    e = int(end_sec   * sr)
    if e <= s or e > len(y):
        return None
    seg = y[s:e]

    # RMS – noise subtracted + peak-normalised (device-independent dynamics)
    rms_raw = librosa.feature.rms(y=seg, hop_length=HOP)[0]
    nf      = np.percentile(rms_raw, 5)
    rms_c   = np.maximum(rms_raw - nf, 0)
    peak    = np.max(rms_c) + 1e-10
    rms_n   = rms_c / peak

    mean_intensity  = float(np.mean(rms_n))
    forte_ratio     = float(np.mean(rms_n > 0.6))
    dynamic_range_cv = float(np.std(rms_n) / (np.mean(rms_n) + 1e-10))

    # Onset / timing
    onset_env    = librosa.onset.onset_strength(y=seg, sr=sr, hop_length=HOP)
    onset_frames = librosa.util.peak_pick(onset_env,
                                          pre_max=3, post_max=3,
                                          pre_avg=3, post_avg=3,
                                          delta=0.1, wait=10)
    onset_times  = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP)
    duration     = end_sec - start_sec
    ioi          = np.diff(onset_times)
    ioi          = ioi[ioi > 0.05]
    timing_cv    = float(np.std(ioi) / np.mean(ioi)) if len(ioi) > 1 else 0.0
    onset_density = len(onset_times) / duration if duration > 0 else 0.0

    # ZCR (noise-corrected)
    zcr      = librosa.feature.zero_crossing_rate(y=seg, hop_length=HOP)[0]
    rms_q    = rms_raw < np.percentile(rms_raw, 10)
    noise_zcr = float(np.mean(zcr[rms_q])) if rms_q.any() else 0.0
    avg_zcr  = float(np.mean(np.maximum(zcr - noise_zcr, 0)))

    return {
        'mean_intensity':   mean_intensity,
        'forte_ratio':      forte_ratio,
        'dynamic_range_cv': dynamic_range_cv,
        'timing_cv':        timing_cv,
        'onset_density':    onset_density,
        'avg_zcr':          avg_zcr,
        'duration':         duration,
    }

# ── Load and analyse ─────────────────────────────────────────────────────────
print("=" * 65)
print("Section-by-Section Analysis / 分段分析")
print("=" * 65)

results = {}
for name, (path, color) in PERFORMERS.items():
    print(f"\nLoading {name}...")
    y, sr = librosa.load(str(path), sr=SR)
    perf_dur = librosa.get_duration(y=y, sr=sr)
    ratio    = perf_dur / SCORE_DURATION          # pace ratio vs score
    print(f"  Duration: {perf_dur:.1f}s  |  Pace ratio: {ratio:.3f}  "
          f"({'faster' if ratio < 1 else 'slower'} than score)")

    sections = {}
    for (sec_name, sc_start, sc_end) in SECTIONS:
        p_start = sc_start * ratio
        p_end   = sc_end   * ratio
        feat    = section_features(y, sr, p_start, p_end)
        if feat is None:
            continue
        feat['score_start'] = sc_start
        feat['score_end']   = sc_end
        feat['perf_start']  = p_start
        feat['perf_end']    = p_end
        feat['pace_ratio']  = ratio            # local = global for linear scaling
        sections[sec_name]  = feat

        label = sec_name.replace('\n', ' ')
        print(f"  [{label}]  "
              f"{p_start:.1f}-{p_end:.1f}s  "
              f"MeanIntens={feat['mean_intensity']:.3f}  "
              f"ForteRatio={feat['forte_ratio']:.1%}  "
              f"TimingCV={feat['timing_cv']:.3f}  "
              f"Density={feat['onset_density']:.2f}/s")

    results[name] = {'sections': sections, 'color': color,
                     'duration': perf_dur, 'ratio': ratio}

# ── Build summary table ──────────────────────────────────────────────────────
rows = []
for name, data in results.items():
    for sec_name, feat in data['sections'].items():
        rows.append({
            'Performer': name,
            'Section':   sec_name.replace('\n', ' '),
            'Perf_Start': round(feat['perf_start'], 1),
            'Perf_End':   round(feat['perf_end'],   1),
            'MeanIntens': round(feat['mean_intensity'],   3),
            'ForteRatio': round(feat['forte_ratio'],      3),
            'DynCV':      round(feat['dynamic_range_cv'], 3),
            'TimingCV':   round(feat['timing_cv'],        3),
            'Density':    round(feat['onset_density'],    2),
            'ZCR':        round(feat['avg_zcr'],          4),
        })

df = pd.DataFrame(rows)
csv_path = base / 'results_ultimate' / 'section_analysis.csv'
df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\nSaved: {csv_path}")

# ── Visualisation ─────────────────────────────────────────────────────────────
METRICS = [
    ('mean_intensity',   '动态强度\nMean Intensity',       False),
    ('forte_ratio',      '强奏比例\nForte Ratio (>60%)',   True),
    ('timing_cv',        'Rubato (IOI CV)',                False),
    ('onset_density',    '音符密度\nOnset Density (/s)',   False),
]
PERF_NAMES  = list(results.keys())
COLORS      = [results[n]['color'] for n in PERF_NAMES]
N_SECTIONS  = len(SECTIONS)
N_METRICS   = len(METRICS)

# ── Plot 1: grouped bar chart per metric per section ─────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.flatten()

x = np.arange(N_SECTIONS)
width = 0.25

for mi, (ax, (metric_key, metric_label, as_pct)) in enumerate(zip(axes, METRICS)):
    for pi, (name, color) in enumerate(zip(PERF_NAMES, COLORS)):
        vals = []
        for sec_name, _, _ in SECTIONS:
            feat = results[name]['sections'].get(sec_name, {})
            vals.append(feat.get(metric_key, 0))
        offset = (pi - 1) * width
        bars = ax.bar(x + offset, vals, width, label=name,
                      color=color, alpha=0.85, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([s[0].replace('\n', ' ') for s in SECTIONS], fontsize=10)
    ax.set_ylabel(metric_label, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, axis='y')
    if as_pct:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))

fig.suptitle('分段特征比较 / Section-by-Section Feature Comparison\n'
             '《彩云追月》 — Lang Lang · Li Yundi · Shen Wenyu',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
p1 = out / '14_section_comparison.png'
fig.savefig(p1, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {p1}")

# ── Plot 2: per-performer section profiles (radar-like bar grids) ─────────────
fig = plt.figure(figsize=(15, 10))
gs  = gridspec.GridSpec(len(PERF_NAMES), N_SECTIONS,
                        hspace=0.45, wspace=0.35)

# Normalise each metric to 0-1 across all performers × all sections for comparison
norm_vals = {m: [] for m, _, _ in METRICS}
for name in PERF_NAMES:
    for sec_name, _, _ in SECTIONS:
        feat = results[name]['sections'].get(sec_name, {})
        for mk, _, _ in METRICS:
            norm_vals[mk].append(feat.get(mk, 0))
norm_min = {mk: min(v) for mk, v in norm_vals.items()}
norm_max = {mk: max(v) + 1e-10 for mk, v in norm_vals.items()}

for ri, name in enumerate(PERF_NAMES):
    for ci, (sec_name, sc_start, sc_end) in enumerate(SECTIONS):
        ax  = fig.add_subplot(gs[ri, ci])
        feat = results[name]['sections'].get(sec_name, {})
        metric_labels_short = ['Intens', 'Forte', 'Rubato', 'Density']
        vals_norm = [(feat.get(mk, 0) - norm_min[mk]) / (norm_max[mk] - norm_min[mk])
                     for mk, _, _ in METRICS]
        color = results[name]['color']
        ax.barh(metric_labels_short, vals_norm, color=color, alpha=0.75)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Normalised', fontsize=7)
        title = sec_name.replace('\n', ' ')
        if ri == 0:
            ax.set_title(title, fontsize=10, fontweight='bold')
        if ci == 0:
            ax.set_ylabel(name, fontsize=10, fontweight='bold', color=color)
        ax.tick_params(axis='both', labelsize=7)
        ax.grid(True, alpha=0.2, axis='x')

fig.suptitle('各演奏家×各段落 归一化特征分布\n'
             'Normalised Feature Profile per Performer per Section',
             fontsize=12, fontweight='bold')
p2 = out / '15_section_profiles.png'
fig.savefig(p2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {p2}")

# ── Plot 3: Timing CV and Forte Ratio side-by-side across sections ───────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

sec_labels = [s[0].replace('\n', ' ') for s in SECTIONS]

for name, color in zip(PERF_NAMES, COLORS):
    cvs    = [results[name]['sections'].get(s[0], {}).get('timing_cv', 0)    for s in SECTIONS]
    fortes = [results[name]['sections'].get(s[0], {}).get('forte_ratio', 0)  for s in SECTIONS]
    ax1.plot(sec_labels, cvs,    marker='o', color=color, linewidth=2.5,
             markersize=8, label=name)
    ax2.plot(sec_labels, fortes, marker='s', color=color, linewidth=2.5,
             markersize=8, label=name)

ax1.set_title('Rubato (IOI Timing CV) per Section / 各段落节奏自由度', fontsize=11, fontweight='bold')
ax1.set_ylabel('Timing CV  (higher = more flexible / 越高越自由)')
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.set_title('Forte Ratio per Section / 各段落强奏比例', fontsize=11, fontweight='bold')
ax2.set_ylabel('Forte Ratio  (higher = louder / 越高越大声)')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
ax2.legend(); ax2.grid(True, alpha=0.3)

fig.suptitle('三位演奏家在四个段落中的 Rubato 与动态变化\n'
             'Rubato & Dynamics Across Sections — 《彩云追月》',
             fontsize=12, fontweight='bold')
plt.tight_layout()
p3 = out / '16_section_rubato_dynamics.png'
fig.savefig(p3, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {p3}")

# ── Console summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("KEY FINDINGS / 关键发现")
print("=" * 65)

for sec_name, sc_start, sc_end in SECTIONS:
    label = sec_name.replace('\n', ' ')
    print(f"\n【{label}】  (score {sc_start:.0f}–{sc_end:.0f}s)")
    print(f"  {'':12s}  TimingCV   ForteRatio   Density    MeanIntens")
    for name in PERF_NAMES:
        f = results[name]['sections'].get(sec_name, {})
        if not f:
            continue
        ps, pe = f['perf_start'], f['perf_end']
        print(f"  {name:<12s}  "
              f"{f['timing_cv']:.3f}      "
              f"{f['forte_ratio']:.1%}        "
              f"{f['onset_density']:.2f}/s      "
              f"{f['mean_intensity']:.3f}"
              f"   [{ps:.0f}–{pe:.0f}s]")

print("\n" + "=" * 65)
print(f"Done. Plots: 14_section_comparison, 15_section_profiles, 16_section_rubato_dynamics")
print("=" * 65)
