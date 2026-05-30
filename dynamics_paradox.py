#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方向C: 沈文裕动态悖论 — 设备无关动态分析
Dynamics Paradox: Shen Wenyu's home recording is quietest in raw amplitude,
yet shows the highest ForteRatio after noise-subtracted peak-normalisation.
This visualises WHY device-independent normalisation matters.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

from config import PERFORMERS, SECTIONS, SEC_EN, SCORE_DURATION, SR, HOP

# ── Load and compute RMS envelopes ────────────────────────────────────────────
print("=" * 65)
print("方向C: 动态悖论分析 / Dynamics Paradox Analysis")
print("=" * 65)

data = {}
for name, (path, color) in PERFORMERS.items():
    print(f"\nLoading {name}...")
    y, sr = librosa.load(str(path), sr=SR)
    dur   = librosa.get_duration(y=y, sr=sr)
    ratio = dur / SCORE_DURATION

    # Raw RMS (frame-by-frame)
    rms_raw   = librosa.feature.rms(y=y, hop_length=HOP)[0]
    times_raw = librosa.frames_to_time(np.arange(len(rms_raw)), sr=SR, hop_length=HOP)
    # Convert performer time → score time
    times_score = times_raw / ratio

    # Noise-subtracted peak-normalised RMS
    nf    = np.percentile(rms_raw, 5)
    rms_c = np.maximum(rms_raw - nf, 0)
    peak  = np.max(rms_c) + 1e-10
    rms_n = rms_c / peak

    # Per-section stats
    section_stats = {}
    for sec_name, sc_start, sc_end, _ in SECTIONS:
        p_start = sc_start * ratio
        p_end   = sc_end   * ratio
        mask    = (times_raw >= p_start) & (times_raw < p_end)
        if not mask.any():
            continue
        raw_seg  = rms_raw[mask]
        norm_seg = rms_n[mask]
        section_stats[sec_name] = {
            'raw_mean':   float(np.mean(raw_seg)),
            'raw_peak':   float(np.max(raw_seg)),
            'norm_mean':  float(np.mean(norm_seg)),
            'forte_ratio': float(np.mean(norm_seg > 0.6)),
            'noise_floor': float(nf),
            'peak_rms':   float(peak),
        }
        print(f"  [{sec_name}]  RawMean={np.mean(raw_seg):.4f}  "
              f"ForteRatio={np.mean(norm_seg > 0.6):.1%}  NoiseFloor={nf:.4f}")

    data[name] = {
        'color':         color,
        'ratio':         ratio,
        'dur':           dur,
        'rms_raw':       rms_raw,
        'rms_norm':      rms_n,
        'times_score':   times_score,
        'noise_floor':   float(nf),
        'peak_rms':      float(peak),
        'section_stats': section_stats,
    }

# ── Figure: 4-panel dynamics paradox ─────────────────────────────────────────
fig = plt.figure(figsize=(16, 14))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

ax_raw   = fig.add_subplot(gs[0, :])   # full-width: raw RMS curves
ax_norm  = fig.add_subplot(gs[1, :])   # full-width: normalised RMS curves
ax_bar_r = fig.add_subplot(gs[2, 0])   # per-section raw peak
ax_bar_f = fig.add_subplot(gs[2, 1])   # per-section forte ratio

# Section shading helper
def shade(ax, max_x=160):
    for sec_name, start, end, bg in SECTIONS:
        ax.axvspan(start, min(end, max_x), color=bg, alpha=0.30, zorder=0)
    for _, start, end, _ in SECTIONS[1:]:
        ax.axvline(start, color='gray', linewidth=0.7, linestyle=':', zorder=1)

# ── Panel 1: Raw RMS (absolute amplitude) ─────────────────────────────────────
shade(ax_raw)
global_raw_max = 0.0
for name, d in data.items():
    # Smooth for readability
    from scipy.ndimage import uniform_filter1d
    smooth = uniform_filter1d(d['rms_raw'], size=30)
    ax_raw.plot(d['times_score'], smooth,
                color=d['color'], linewidth=1.8, label=name, zorder=3)
    global_raw_max = max(global_raw_max, smooth.max())

# Label sections
yhi = ax_raw.get_ylim()[1]
for sec_name, start, end, _ in SECTIONS:
    ax_raw.text((start+end)/2, yhi * 0.92,
                SEC_EN.get(sec_name, sec_name), ha='center', va='top', fontsize=8.5,
                color='#444', fontweight='bold')

ax_raw.set_xlim(0, 156)
ax_raw.set_ylabel('Raw RMS (linear amplitude)', fontsize=10)
ax_raw.set_title('① 原始 RMS 包络（录音设备差异直接体现）\n'
                 'Raw RMS envelopes — Shen Wenyu home recording is quietest in absolute level',
                 fontsize=10.5, fontweight='bold')
ax_raw.legend(fontsize=10, loc='upper right')
ax_raw.grid(True, alpha=0.2, axis='y')
ax_raw.set_xlabel('Score Time (s)', fontsize=9)

# Annotate noise floor arrows
for name, d in data.items():
    ax_raw.axhline(d['noise_floor'], color=d['color'],
                   linewidth=0.7, linestyle='--', alpha=0.5)

# ── Panel 2: Normalised RMS (device-independent) ──────────────────────────────
shade(ax_norm)
for name, d in data.items():
    smooth_n = uniform_filter1d(d['rms_norm'], size=30)
    ax_norm.plot(d['times_score'], smooth_n,
                 color=d['color'], linewidth=1.8, label=name, zorder=3)

ax_norm.axhline(0.6, color='purple', linewidth=1.2, linestyle='--',
                label='forte threshold (0.6)', zorder=4)
ax_norm.set_xlim(0, 156)
ax_norm.set_ylabel('Normalised RMS\n(noise-subtracted, peak-scaled)', fontsize=10)
ax_norm.set_title('② 噪声扣除 + 峰值归一化后（设备无关）\n'
                  'After noise-subtracted peak-normalisation — Shen Wenyu shows highest dynamic contrast',
                  fontsize=10.5, fontweight='bold')
ax_norm.legend(fontsize=10, loc='upper right')
ax_norm.grid(True, alpha=0.2, axis='y')
ax_norm.set_xlabel('Score Time (s)', fontsize=9)

yhi_n = ax_norm.get_ylim()[1]
for sec_name, start, end, _ in SECTIONS:
    ax_norm.text((start+end)/2, yhi_n * 0.92,
                 SEC_EN.get(sec_name, sec_name), ha='center', va='top', fontsize=8.5,
                 color='#444', fontweight='bold')

# ── Panel 3: Per-section raw peak RMS bar chart ───────────────────────────────
sec_names  = [s[0] for s in SECTIONS]
performers = list(data.keys())
colors     = [data[n]['color'] for n in performers]
x          = np.arange(len(sec_names))
_NP        = len(performers)
width      = min(0.26, 0.8 / _NP)

for i, (name, col) in enumerate(zip(performers, colors)):
    vals = [data[name]['section_stats'].get(s, {}).get('raw_peak', 0) for s in sec_names]
    bars = ax_bar_r.bar(x + (i - (_NP - 1) / 2) * width, vals, width,
                        label=name.split()[0], color=col, alpha=0.80,
                        edgecolor='white', linewidth=0.5)

ax_bar_r.set_xticks(x)
ax_bar_r.set_xticklabels([SEC_EN.get(s, s) for s in sec_names], fontsize=8)
ax_bar_r.set_ylabel('Peak RMS (raw)', fontsize=9)
ax_bar_r.set_title('③ Peak RMS per Section (Raw) / 各段峰值 RMS（原始）\nShen Wenyu lowest raw amplitude',
                   fontsize=10, fontweight='bold')
ax_bar_r.legend(fontsize=9)
ax_bar_r.grid(True, alpha=0.2, axis='y')

# ── Panel 4: Per-section ForteRatio after normalisation ───────────────────────
for i, (name, col) in enumerate(zip(performers, colors)):
    vals = [data[name]['section_stats'].get(s, {}).get('forte_ratio', 0) * 100
            for s in sec_names]
    ax_bar_f.bar(x + (i - (_NP - 1) / 2) * width, vals, width,
                 label=name.split()[0], color=col, alpha=0.80,
                 edgecolor='white', linewidth=0.5)

# Highlight 华彩 with a box (dynamic index)
_cadenza_idx = sec_names.index('华彩') if '华彩' in sec_names else 2
ax_bar_f.axvspan(_cadenza_idx - 0.5, _cadenza_idx + 0.5,
                 color='gold', alpha=0.15, zorder=0, label='华彩 (Cadenza) highlight')

ax_bar_f.set_xticks(x)
ax_bar_f.set_xticklabels([SEC_EN.get(s, s) for s in sec_names], fontsize=8)
ax_bar_f.set_ylabel('Forte Ratio (%)\n[norm. RMS > 0.6]', fontsize=9)
ax_bar_f.set_title('④ ForteRatio after Normalisation / 归一化后强音比例\nShen Wenyu highest in 华彩 (Cadenza)',
                   fontsize=10, fontweight='bold')
ax_bar_f.legend(fontsize=9)
ax_bar_f.grid(True, alpha=0.2, axis='y')

fig.suptitle('The Dynamics Paradox / 动态对比悖论：家庭录制 ≠ 弱演奏\n'
             'The Dynamics Paradox — Shen Wenyu\'s home recording is acoustically quietest,\n'
             'yet reveals the most dramatic forte contrasts after device-independent normalisation',
             fontsize=12, fontweight='bold', y=0.98)

p_out = out / '20_dynamics_paradox.png'
fig.savefig(p_out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {p_out}")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("关键数据对比 / Key Numbers")
print("=" * 65)
print(f"\n{'Performer':<12}  {'NoiseFloor':>11}  {'PeakRMS':>9}  "
      f"{'华彩 RawPeak':>12}  {'华彩 ForteRatio':>15}")
print("-" * 65)
for name, d in data.items():
    hua  = d['section_stats'].get('华彩', {})
    print(f"{name:<12}  {d['noise_floor']:>11.5f}  {d['peak_rms']:>9.5f}  "
          f"{hua.get('raw_peak', 0):>12.5f}  {hua.get('forte_ratio', 0):>14.1%}")

print("\n结论 / Conclusion:")
# Dynamically identify the extremes
_all_names = list(data.keys())
_peak_rms  = {n: data[n]['peak_rms'] for n in _all_names}
_cadenza_fr = {n: data[n]['section_stats'].get('华彩', {}).get('forte_ratio', 0) for n in _all_names}
_quietest  = min(_peak_rms, key=_peak_rms.get)
_loudest_cadenza = max(_cadenza_fr, key=_cadenza_fr.get)
_quietest_cadenza = min(_cadenza_fr, key=_cadenza_fr.get)
print(f"  Quietest recording (lowest raw peak RMS):     {_quietest}  ({_peak_rms[_quietest]:.4f})")
print(f"  Highest 华彩 ForteRatio after normalisation:  {_loudest_cadenza}  ({_cadenza_fr[_loudest_cadenza]:.1%})")
print(f"  Lowest  华彩 ForteRatio after normalisation:  {_quietest_cadenza}  ({_cadenza_fr[_quietest_cadenza]:.1%})")
_ratio = _cadenza_fr[_loudest_cadenza] / (_cadenza_fr[_quietest_cadenza] + 1e-10)
print(f"\n  The device-independent paradox: {_loudest_cadenza} has {_ratio:.1f}× the 华彩 ForteRatio")
print(f"  of {_quietest_cadenza}, though raw amplitude comparison may suggest otherwise.")

print("\n" + "=" * 65)
print("Done.")
print("=" * 65)
