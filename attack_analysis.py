#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attack Sharpness & Decay Rate Analysis
---------------------------------------
For each detected note onset:
  attack_sharpness = peak-normalised onset strength at that frame
                     (high → percussive/staccato, low → smooth/legato)
  decay_rate       = slope of RMS in the 200 ms after the onset
                     (steep negative → short note/less pedal,
                      shallow negative → long note/more pedal)

Together these explain the Lang Lang paradox:
  high attack sharpness + moderate mean RMS = "powerful" perception
  without sustained forte dynamics.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import uniform_filter1d
from scipy.stats import linregress
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

base = Path(__file__).parent
out  = base / 'results_ultimate' / 'plots'
out.mkdir(parents=True, exist_ok=True)

from config import PERFORMERS, SECTIONS, SEC_EN, SCORE_DURATION, SR, HOP, SCORE_BPM

DECAY_MS     = 200          # ms to look ahead for decay slope
DECAY_FRAMES = int(DECAY_MS / 1000 * SR / HOP)   # ≈ 8 frames

# ── Feature extraction ────────────────────────────────────────────────────────
print("=" * 65)
print("Attack Sharpness & Decay Rate Analysis / 起音锐度与衰减分析")
print("=" * 65)

all_data = {}

for name, (path, color) in PERFORMERS.items():
    print(f"\nProcessing {name}...")
    y, sr  = librosa.load(str(path), sr=SR)
    dur    = librosa.get_duration(y=y, sr=SR)
    ratio  = dur / SCORE_DURATION        # pace ratio

    # Onset strength envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)

    # Peak-normalise onset strength (device-independent)
    nf_env    = np.percentile(onset_env, 5)
    env_c     = np.maximum(onset_env - nf_env, 0)
    env_peak  = np.max(env_c) + 1e-10
    env_n     = env_c / env_peak          # 0–1 normalised attack sharpness

    # Detect onsets
    onset_frames = librosa.util.peak_pick(onset_env,
                                          pre_max=3, post_max=3,
                                          pre_avg=3, post_avg=3,
                                          delta=0.15, wait=8)
    onset_times  = librosa.frames_to_time(onset_frames, sr=SR, hop_length=HOP)

    # IOI CV (inter-onset-interval coefficient of variation) — rubato measure
    ioi = np.diff(onset_times)
    ioi_cv = float(ioi.std() / ioi.mean()) if len(ioi) > 1 else 0.0

    # RMS envelope (for decay computation)
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]

    # Per-onset: attack sharpness + decay rate
    records = []
    for fi, t in zip(onset_frames, onset_times):
        # Attack sharpness
        atk = float(env_n[fi])

        # Decay slope (RMS over next DECAY_FRAMES frames)
        seg = rms[fi : min(fi + DECAY_FRAMES, len(rms))]
        if len(seg) >= 3:
            x_  = np.arange(len(seg)) / (SR / HOP)  # in seconds
            slope, *_ = linregress(x_, seg)
            decay = float(slope)                      # RMS/s; negative = decaying
        else:
            decay = np.nan

        # Map to score time
        t_score = t / ratio

        records.append({
            'performer':    name,
            'perf_time':    t,
            'score_time':   t_score,
            'attack':       atk,
            'decay_rate':   decay,
        })

    df = pd.DataFrame(records)

    # Section label
    def get_sec(t):
        for sn, ss, se, _ in SECTIONS:
            if ss <= t < se:
                return sn
        return '尾声'
    df['section'] = df['score_time'].apply(get_sec)

    # Global peak RMS (for ForteRatio cross-check)
    nf_rms   = np.percentile(rms, 5)
    rms_c    = np.maximum(rms - nf_rms, 0)
    rms_n    = rms_c / (np.max(rms_c) + 1e-10)
    forte_ratio = float(np.mean(rms_n > 0.6))

    all_data[name] = {
        'df':          df,
        'color':       color,
        'ratio':       ratio,
        'global_bpm':  SCORE_BPM / ratio,
        'ioi_cv':      ioi_cv,
        'env_n':       env_n,
        'forte_ratio': forte_ratio,
        'mean_attack': float(df['attack'].mean()),
        'mean_decay':  float(df['decay_rate'].dropna().mean()),
    }

    print(f"  Mean attack sharpness : {df['attack'].mean():.3f}")
    print(f"  Mean decay rate (RMS/s): {df['decay_rate'].dropna().mean():.4f}")
    print(f"  ForteRatio (norm RMS>0.6): {forte_ratio:.1%}")
    for sec_name, *_ in SECTIONS:
        sub = df[df['section'] == sec_name]
        print(f"  [{sec_name:<8}]  attack={sub['attack'].mean():.3f}"
              f"  decay={sub['decay_rate'].dropna().mean():.4f}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
rows = pd.concat([d['df'] for d in all_data.values()], ignore_index=True)
rows.to_csv(base / 'results_ultimate' / 'attack_analysis.csv',
            index=False, encoding='utf-8-sig')

# ── Section shading helper ────────────────────────────────────────────────────
def shade(ax):
    for _, start, end, bg in SECTIONS:
        ax.axvspan(start, end, color=bg, alpha=0.28, zorder=0)
    for _, start, _, _ in SECTIONS[1:]:
        ax.axvline(start, color='gray', linewidth=0.7, linestyle=':', zorder=1)

def label_sections(ax):
    yhi, ylo = ax.get_ylim()
    for sec_name, start, end, _ in SECTIONS:
        ax.text((start + end) / 2, yhi - (yhi - ylo) * 0.04,
                SEC_EN.get(sec_name, sec_name),
                ha='center', va='top', fontsize=7.5,
                color='#444', fontweight='bold')

# ── Plot A: Attack sharpness curves over score time ───────────────────────────
_N = len(all_data)
_NCOLS = 3
_NROWS = (_N + _NCOLS - 1) // _NCOLS
fig, _axes_grid = plt.subplots(_NROWS, _NCOLS, figsize=(15, 4 * _NROWS + 1), sharex=True)
axes = _axes_grid.flatten() if hasattr(_axes_grid, 'flatten') else [_axes_grid]
for _ax in axes[_N:]:
    _ax.set_visible(False)

for ax, (name, d) in zip(axes, all_data.items()):
    df    = d['df']
    color = d['color']
    shade(ax)

    # Scatter (raw, translucent)
    ax.scatter(df['score_time'], df['attack'],
               color=color, alpha=0.15, s=6, zorder=2)

    # Smoothed curve
    sort_idx = df['score_time'].argsort()
    t_sorted = df['score_time'].values[sort_idx]
    a_sorted = df['attack'].values[sort_idx]
    smooth   = uniform_filter1d(a_sorted, size=20)
    ax.plot(t_sorted, smooth, color=color, linewidth=2.0, zorder=3)

    ax.axhline(df['attack'].mean(), color=color, linewidth=1.0,
               linestyle='--', alpha=0.6,
               label=f"mean = {df['attack'].mean():.3f}")
    ax.set_ylabel('Attack Sharpness\n(norm. onset strength)', fontsize=9)
    ax.set_title(f'{name}  (mean attack {d["mean_attack"]:.3f}  |  '
                 f'ForteRatio {d["forte_ratio"]:.1%})',
                 fontsize=11, color=color, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2, axis='y')

for _ax in axes[max(0, _N - _NCOLS):_N]:
    _ax.set_xlabel('Score Time (s)', fontsize=10)
label_sections(axes[0])

fig.suptitle('Attack Sharpness over Score Time / 逐音符起音锐度\n'
             'Higher = more percussive / staccato  |  《彩云追月》',
             fontsize=12, fontweight='bold')
plt.tight_layout()
p_a = out / '24_attack_sharpness_curves.png'
fig.savefig(p_a, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {p_a}")

sec_names = [s[0] for s in SECTIONS]

# Global BPM and Timing CV computed from audio (stored during feature extraction above)
GLOBAL_BPM = {name: d['global_bpm'] for name, d in all_data.items()}
TIMING_CV  = {name: d['ioi_cv']     for name, d in all_data.items()}

# ── Plot B: Li Yundi Paradox — bubble chart (attack × forte, size = BPM) ─────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: per-section attack sharpness box plot
ax = axes[0]
from matplotlib.patches import Patch
positions = np.arange(len(sec_names))
w = 0.22
for i, (name, d) in enumerate(all_data.items()):
    df = d['df']
    data_by_sec = [df[df['section'] == s]['attack'].values for s in sec_names]
    bp = ax.boxplot(data_by_sec,
                    positions=positions + (i - 1) * w,
                    widths=w * 0.85,
                    patch_artist=True,
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker='.', markersize=3, alpha=0.4))
    for patch in bp['boxes']:
        patch.set_facecolor(d['color'])
        patch.set_alpha(0.75)

ax.set_xticks(positions)
ax.set_xticklabels([SEC_EN.get(s, s) for s in sec_names], fontsize=8)
ax.set_ylabel('Attack Sharpness (normalised)', fontsize=10)
ax.set_title('Attack Sharpness per Section / 各段落起音锐度\n'
             'Li Yundi consistently highest across all sections', fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.2, axis='y')
ax.legend([Patch(color=d['color'], alpha=0.75) for d in all_data.values()],
          list(all_data.keys()), fontsize=8, loc='upper right')

# Right: bubble scatter — attack × ForteRatio, bubble size = BPM
ax2 = axes[1]
for name, d in all_data.items():
    bpm   = GLOBAL_BPM[name]
    atk   = d['mean_attack']
    forte = d['forte_ratio'] * 100
    color = d['color']

    # Bubble size proportional to BPM
    bubble = (bpm / 80) ** 2 * 300

    ax2.scatter(atk, forte, s=bubble, color=color, alpha=0.85,
                edgecolors='black', linewidths=1.5, zorder=5)
    ax2.annotate(f'{name.split()[0]}\n({bpm:.0f} BPM)',
                 (atk, forte),
                 textcoords='offset points', xytext=(6, 4),
                 fontsize=8, fontweight='bold', color=color)

# Reverb caveat for Lang Lang (if present)
if 'Lang Lang' in all_data:
    ax2.annotate('* live hall reverb\n  lowers measured attack',
                 xy=(all_data['Lang Lang']['mean_attack'],
                     all_data['Lang Lang']['forte_ratio'] * 100),
                 xytext=(0.13, 2.0),
                 fontsize=7.5, color='#888', style='italic',
                 arrowprops=dict(arrowstyle='->', color='#aaa', lw=0.8))

# Quadrant dividers
xlim = ax2.get_xlim(); ylim = ax2.get_ylim()
xm = np.mean([d['mean_attack'] for d in all_data.values()])
ym = np.mean([d['forte_ratio'] * 100 for d in all_data.values()])
ax2.axvline(xm, color='lightgray', linewidth=0.8, linestyle=':')
ax2.axhline(ym, color='lightgray', linewidth=0.8, linestyle=':')

ax2.text(xlim[1] - 0.002, ylim[1] - 0.3, 'Sharp + Loud',
         ha='right', va='top', fontsize=7.5, color='#aaa', style='italic')
ax2.text(xlim[0] + 0.002, ylim[1] - 0.3, 'Soft + Loud',
         ha='left',  va='top', fontsize=7.5, color='#aaa', style='italic')
ax2.text(xlim[1] - 0.002, ylim[0] + 0.3, 'Sharp + Quiet',
         ha='right', va='bottom', fontsize=7.5, color='#aaa', style='italic')
ax2.text(xlim[0] + 0.002, ylim[0] + 0.3, 'Soft + Quiet',
         ha='left',  va='bottom', fontsize=7.5, color='#aaa', style='italic')

ax2.set_xlabel('Mean Attack Sharpness  (higher = more percussive / 触键越硬)', fontsize=10)
ax2.set_ylabel('ForteRatio %  (higher = more sustained loud / 持续响度越高)', fontsize=10)
ax2.set_title('The Li Yundi Paradox / 李云迪悖论\n'
              'Sharpest attack + fastest decay → yet softest sustained dynamics\n'
              'Bubble size ∝ tempo (BPM)', fontsize=10, fontweight='bold')
ax2.grid(True, alpha=0.2)

fig.suptitle('Attack Sharpness vs Sustained Dynamics / 起音锐度 × 持续响度\n'
             '《彩云追月》 — Three distinct articulation strategies',
             fontsize=11, fontweight='bold')
plt.tight_layout()
p_b = out / '25_liyundi_paradox.png'
fig.savefig(p_b, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {p_b}")

# ── Plot C: Decay rate per section ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))

for i, (name, d) in enumerate(all_data.items()):
    df   = d['df']
    vals = [df[df['section'] == s]['decay_rate'].dropna().mean() for s in sec_names]
    _NR  = len(all_data)
    _bw  = min(0.26, 0.8 / _NR)
    x    = np.arange(len(sec_names))
    ax.bar(x + (i - (_NR - 1) / 2) * _bw, vals, _bw,
           label=name, color=d['color'], alpha=0.82,
           edgecolor='white', linewidth=0.5)

ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xticks(np.arange(len(sec_names)))
ax.set_xticklabels([SEC_EN.get(s, s) for s in sec_names], fontsize=9)
ax.set_ylabel('Mean Decay Rate (RMS/s)\nmore negative = note fades faster / less pedal', fontsize=9)
ax.set_title('Note Decay Rate per Section / 各段落音符衰减速率\n'
             'More negative = note fades faster (less pedal / more staccato)',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2, axis='y')
plt.tight_layout()
p_c = out / '26_decay_rate_bars.png'
fig.savefig(p_c, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {p_c}")

# ── Plot D: Comprehensive Performer Fingerprint — radar chart ─────────────────
# Five dimensions (all oriented: outward = "more" of that quality)
# 1. Speed      : BPM / score_BPM  (higher = faster)
# 2. Dynamics   : ForteRatio       (higher = louder)
# 3. Attack     : mean_attack      (higher = more percussive)
# 4. Rubato     : timing CV        (higher = more flexible)
# 5. Sustain    : 1 / |decay_rate| (higher = notes ring longer)

raw = {
    name: {
        'Speed':   GLOBAL_BPM[name] / 114.0,
        'Dynamics\n(ForteRatio)': d['forte_ratio'],
        'Attack\nSharpness':      d['mean_attack'],
        'Rubato\n(Timing CV)':    TIMING_CV[name],
        'Sustain\n(Decay⁻¹)':    1.0 / abs(d['mean_decay']),
    }
    for name, d in all_data.items()
}

dims = list(raw[next(iter(raw))].keys())
N    = len(dims)

# Normalise each dimension 0–1 across performers
norm = {}
for dim in dims:
    vals = [raw[name][dim] for name in all_data]
    lo, hi = min(vals), max(vals)
    for name in all_data:
        norm.setdefault(name, {})[dim] = (raw[name][dim] - lo) / (hi - lo + 1e-10)

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]   # close the loop

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

for name, d in all_data.items():
    vals = [norm[name][dim] for dim in dims] + [norm[name][dims[0]]]
    ax.plot(angles, vals, color=d['color'], linewidth=2.2, label=name, zorder=3)
    ax.fill(angles, vals, color=d['color'], alpha=0.12, zorder=2)

ax.set_thetagrids(np.degrees(angles[:-1]), dims, fontsize=10)
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=7, color='gray')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=10)

# Note about Lang Lang attack
ax.text(0.5, -0.10, '* Attack Sharpness for Lang Lang underestimated\n'
        '  due to live hall reverb (not a playing-style artefact)',
        transform=ax.transAxes, ha='center', fontsize=7.5,
        color='gray', style='italic')

ax.set_title('Performer Style Fingerprint / 演奏风格指纹\n'
             '5-dimensional normalised profile  |  《彩云追月》',
             fontsize=11, fontweight='bold', pad=25)

p_d = out / '27_performer_fingerprint.png'
fig.savefig(p_d, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {p_d}")

# ── Console summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("KEY FINDINGS / 关键发现")
print("=" * 65)

print(f"\n{'':12s}  {'Attack':>8}  {'Decay(RMS/s)':>13}  {'ForteRatio':>11}")
print("-" * 50)
for name, d in all_data.items():
    print(f"{name:<12}  {d['mean_attack']:>8.3f}  "
          f"{d['mean_decay']:>13.4f}  {d['forte_ratio']:>10.1%}")

print("\n解读 / Interpretation:")
names  = list(all_data.keys())
atks   = [all_data[n]['mean_attack'] for n in names]
decays = [all_data[n]['mean_decay']  for n in names]
forts  = [all_data[n]['forte_ratio'] for n in names]
bpms_list = [all_data[n]['global_bpm'] for n in names]

fastest_atk   = names[np.argmax(atks)]
slowest_decay = names[np.argmin(decays)]   # most negative = fastest decay
highest_forte = names[np.argmax(forts)]
fastest_bpm   = names[np.argmax(bpms_list)]
softest       = names[np.argmin(forts)]

print(f"  最高起音锐度 (最硬触键): {fastest_atk}  ({max(atks):.3f})")
print(f"  最快音符衰减 (最少踏板): {slowest_decay}  ({min(decays):.4f} RMS/s)")
print(f"  最高强音比例 (最大音量): {highest_forte}  ({max(forts):.1%})")
print(f"  最快演奏速度:            {fastest_bpm}  ({max(bpms_list):.0f} BPM)")
print(f"  最轻柔动态:              {softest}  ({min(forts):.1%} ForteRatio)")

if 'Li Yundi' in all_data:
    ly = all_data['Li Yundi']
    print(f"\n  → 李云迪悖论 / Li Yundi Paradox:")
    print(f"    起音锐度={ly['mean_attack']:.3f}  衰减={ly['mean_decay']:.4f} RMS/s"
          f"  →  ForteRatio={ly['forte_ratio']:.1%}")
if 'Lang Lang' in all_data:
    ll = all_data['Lang Lang']
    print(f"\n  → 郎朗 / Lang Lang:")
    print(f"    BPM={ll['global_bpm']:.0f}  attack={ll['mean_attack']:.3f}"
          f"  ForteRatio={ll['forte_ratio']:.1%}")

print("\n" + "=" * 65)
print("Done.")
print("=" * 65)
