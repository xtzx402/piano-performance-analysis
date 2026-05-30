#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tempo / Local BPM Analysis
Uses DTW-aligned onset data from note_alignment.csv to compute
local BPM for each performer via a sliding window over unique score onsets.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

base = Path(__file__).parent
out  = base / 'results_ultimate' / 'plots'
out.mkdir(parents=True, exist_ok=True)

SCORE_BPM = 114.0
WINDOW    = 30   # sliding window (number of unique score onsets)

SECTIONS = [
    ('A段 主题',  0.0,   60.8,  '#AED6F1'),
    ('B段 抒情', 60.8,   76.8,  '#A9DFBF'),
    ('华彩',     76.8,  130.0,  '#F9E79F'),
    ('尾声',    130.0,  151.6,  '#F5CBA7'),
]
SEC_EN = {
    'A段 主题': 'A段 主题\n(Theme A)',
    'B段 抒情': 'B段 抒情\n(Lyrical)',
    '华彩':     '华彩\n(Cadenza)',
    '尾声':     '尾声\n(Coda)',
}

PERFORMERS = {
    'Lang Lang':  '#E74C3C',
    'Li Yundi':   '#F39C12',
    'Shen Wenyu': '#3498DB',
}

# ── Load data ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("Tempo / Local BPM Analysis / 实时演奏速度分析")
print("=" * 65)

df = pd.read_csv(base / 'results_ultimate' / 'note_alignment.csv')

# ── Local BPM computation ─────────────────────────────────────────────────────
def compute_local_bpm(sub, window=WINDOW):
    """
    Deduplicate chord notes → compute local BPM from unique score onsets.
    local_BPM[i] = SCORE_BPM × (delta_score / delta_perf)
    where delta_score and delta_perf span 'window' unique onsets centred at i.
    """
    # One row per unique score onset (chord notes share onset → take first perf_time)
    onsets = (sub.groupby('score_time', sort=True)['perf_time']
                .first().reset_index())
    onsets = onsets.sort_values('score_time').reset_index(drop=True)

    n    = len(onsets)
    half = window // 2
    bpms  = np.full(n, np.nan)
    times = onsets['score_time'].values

    for i in range(half, n - half):
        d_score = times[i + half] - times[i - half]
        d_perf  = onsets.loc[i + half, 'perf_time'] - onsets.loc[i - half, 'perf_time']
        if d_perf > 0.3 and d_score > 0:
            bpms[i] = SCORE_BPM * d_score / d_perf

    # Remove gross outliers (DTW edge artefacts)
    med = np.nanmedian(bpms)
    bpms[np.abs(bpms - med) > 70] = np.nan

    return times, bpms

results = {}
for name, color in PERFORMERS.items():
    sub = df[df['performer'] == name].copy()
    times, bpms = compute_local_bpm(sub)

    # Global BPM from first/last valid onset
    valid = ~np.isnan(bpms)
    glob_bpm = SCORE_BPM * (times[valid][-1] - times[valid][0]) / \
               (df[(df['performer'] == name)].groupby('score_time')['perf_time']
                .first().sort_index().iloc[[0, -1]].diff().iloc[-1])

    results[name] = {'times': times, 'bpms': bpms,
                     'color': color, 'global_bpm': float(glob_bpm)}

    print(f"\n{name}  (global avg: {float(glob_bpm):.1f} BPM)")
    for sec_name, sc_s, sc_e, _ in SECTIONS:
        mask = (times >= sc_s) & (times < sc_e) & valid
        if mask.any():
            med_bpm = np.nanmedian(bpms[mask])
            print(f"  [{sec_name:<8}]  median BPM = {med_bpm:.1f}")

# ── Helper: section shading ───────────────────────────────────────────────────
def shade(ax):
    for sec_name, start, end, bg in SECTIONS:
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

# ── Plot A: Local BPM curves (all performers overlaid) ───────────────────────
fig, ax = plt.subplots(figsize=(15, 6))
shade(ax)
ax.axhline(SCORE_BPM, color='black', linewidth=1.5, linestyle='--',
           label=f'Score baseline ({SCORE_BPM:.0f} BPM)', zorder=2)

for name, res in results.items():
    times = res['times']
    bpms  = res['bpms']
    color = res['color']
    valid = ~np.isnan(bpms)
    if valid.sum() < 5:
        continue
    smooth = uniform_filter1d(bpms[valid], size=12)
    ax.plot(times[valid], smooth,
            color=color, linewidth=2.0, zorder=3,
            label=f"{name}  (avg {res['global_bpm']:.0f} BPM)")
    ax.fill_between(times[valid], smooth, SCORE_BPM,
                    where=(smooth > SCORE_BPM),
                    color=color, alpha=0.08, zorder=1)
    ax.fill_between(times[valid], smooth, SCORE_BPM,
                    where=(smooth < SCORE_BPM),
                    color=color, alpha=0.08, zorder=1)

ax.set_xlim(0, 155)
ax.set_xlabel('Score Time (s)', fontsize=10)
ax.set_ylabel('Local BPM', fontsize=10)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2, axis='y')
label_sections(ax)

fig.suptitle('Local BPM over Score Time / 实时演奏速度曲线\n'
             f'Sliding window = {WINDOW} onsets  |  Score = {SCORE_BPM:.0f} BPM  |  《彩云追月》',
             fontsize=12, fontweight='bold')
plt.tight_layout()
p_a = out / '21_local_bpm_curves.png'
fig.savefig(p_a, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {p_a}")

# ── Plot B: Subplots per performer (BPM + shaded vs baseline) ────────────────
fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)

for ax, (name, res) in zip(axes, results.items()):
    shade(ax)
    times = res['times']
    bpms  = res['bpms']
    color = res['color']
    valid = ~np.isnan(bpms)

    smooth = uniform_filter1d(bpms[valid], size=12)
    ax.plot(times[valid], smooth, color=color, linewidth=2.0, zorder=3)
    ax.axhline(SCORE_BPM, color='black', linewidth=1.0, linestyle='--',
               zorder=2, label=f'Score {SCORE_BPM:.0f} BPM')
    ax.fill_between(times[valid], smooth, SCORE_BPM,
                    where=(smooth > SCORE_BPM),
                    color='#E74C3C', alpha=0.20, label='faster than score')
    ax.fill_between(times[valid], smooth, SCORE_BPM,
                    where=(smooth < SCORE_BPM),
                    color='#3498DB', alpha=0.20, label='slower than score')

    ax.set_ylabel('Local BPM', fontsize=9)
    ax.set_title(f"{name}  (global avg: {res['global_bpm']:.0f} BPM)",
                 fontsize=11, color=color, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2, axis='y')

axes[-1].set_xlabel('Score Time (s)', fontsize=10)
label_sections(axes[0])

fig.suptitle('Local BPM per Performer / 各演奏者实时速度\n'
             'Red fill = faster than score  |  Blue fill = slower than score  |  《彩云追月》',
             fontsize=12, fontweight='bold')
plt.tight_layout()
p_b = out / '22_local_bpm_subplots.png'
fig.savefig(p_b, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {p_b}")

# ── Plot C: Per-section median BPM bar chart ──────────────────────────────────
sec_names = [s[0] for s in SECTIONS]
x     = np.arange(len(sec_names))
width = 0.26

fig, ax = plt.subplots(figsize=(11, 5))

for i, (name, res) in enumerate(results.items()):
    times = res['times']
    bpms  = res['bpms']
    valid = ~np.isnan(bpms)
    vals  = []
    for sec_name, sc_s, sc_e, _ in SECTIONS:
        mask = (times >= sc_s) & (times < sc_e) & valid
        vals.append(float(np.nanmedian(bpms[mask])) if mask.any() else 0.0)

    bars = ax.bar(x + (i - 1) * width, vals, width,
                  label=name, color=res['color'], alpha=0.82,
                  edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f'{val:.0f}', ha='center', va='bottom', fontsize=8)

ax.axhline(SCORE_BPM, color='black', linewidth=1.5, linestyle='--',
           label=f'Score ({SCORE_BPM:.0f} BPM)', zorder=5)
ax.set_xticks(x)
ax.set_xticklabels([SEC_EN.get(s, s) for s in sec_names], fontsize=9)
ax.set_ylabel('Median Local BPM', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2, axis='y')

fig.suptitle('Median BPM per Section / 各段落演奏速度\n'
             'Dashed line = score baseline (114 BPM)  |  《彩云追月》',
             fontsize=12, fontweight='bold')
plt.tight_layout()
p_c = out / '23_section_bpm_bars.png'
fig.savefig(p_c, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {p_c}")

# ── Console summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("KEY FINDINGS / 关键发现")
print("=" * 65)
print(f"\n{'':12s}  {'A段 主题':>10}  {'B段 抒情':>10}  {'华彩':>10}  {'尾声':>10}  {'Global':>10}")
print("-" * 65)
for name, res in results.items():
    times = res['times']
    bpms  = res['bpms']
    valid = ~np.isnan(bpms)
    row = []
    for sec_name, sc_s, sc_e, _ in SECTIONS:
        mask = (times >= sc_s) & (times < sc_e) & valid
        row.append(f"{np.nanmedian(bpms[mask]):.0f}" if mask.any() else "—")
    row.append(f"{res['global_bpm']:.0f}")
    print(f"{name:<12}  {'  '.join(f'{v:>10}' for v in row)}")
print(f"\n{'Score':12}  {'114':>10}  {'114':>10}  {'114':>10}  {'114':>10}  {'114':>10}")

print("\n" + "=" * 65)
print("Done.")
print("=" * 65)
