#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Note-Level Score Alignment Analysis
────────────────────────────────────
1. Extract note onset times from real MIDI (score ground truth)
2. Compute onset-strength envelopes for score WAV and each performer
3. DTW-align envelopes → warp score time axis to performer time axis
4. For each score note: compute agogic deviation (ms) = actual - expected
5. Visualise note-by-note timing deviations, coloured by section
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import mido
import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from pathlib import Path
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.ndimage import uniform_filter1d
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

base = Path(__file__).parent
out  = base / 'results_ultimate' / 'plots'
out.mkdir(parents=True, exist_ok=True)

SR  = 22050
HOP = 512          # frame resolution for onset detection and feature extraction
DTW_RADIUS = 60    # ±60 onset-index positions (handles ornaments / missed detections)

# ── Section boundaries (score seconds) ───────────────────────────────────────
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
    'Lang Lang':  (base / 'normalized_audio' / 'normalized_langlang_caiyun.wav',  '#E74C3C'),
    'Li Yundi':   (base / 'normalized_audio' / 'normalized_liyundi_caiyun.wav',   '#F39C12'),
    'Shen Wenyu': (base / 'normalized_audio' / 'normalized_shenwenyu_caiyun.wav', '#3498DB'),
}

SCORE_WAV  = base / 'reference_score.wav'
SCORE_MIDI = base / 'cai-yun-zhui-yue-ren-guang-qu-wang-jian-zhong-gai-bian.mid'

# ── 1. Extract MIDI note events ───────────────────────────────────────────────
print("=" * 65)
print("Note-Level Score Alignment / 音符级对齐分析")
print("=" * 65)

mid        = mido.MidiFile(str(SCORE_MIDI))
tpb        = mid.ticks_per_beat
tempo_map  = []      # (abs_tick, tempo_us)
abs_tick   = 0
for track in mid.tracks:
    abs_tick = 0
    for msg in track:
        abs_tick += msg.time
        if msg.type == 'set_tempo':
            tempo_map.append((abs_tick, msg.tempo))
tempo_map.sort()

def tick_to_sec(tick):
    elapsed, prev_tick, prev_tempo = 0.0, 0, 526316  # default 114 BPM
    for t, tempo in tempo_map:
        if t >= tick:
            break
        elapsed   += (t - prev_tick) * prev_tempo / (tpb * 1_000_000)
        prev_tick  = t
        prev_tempo = tempo
    elapsed += (tick - prev_tick) * prev_tempo / (tpb * 1_000_000)
    return elapsed

# Collect note_on events from all tracks
midi_notes = []   # (time_sec, pitch, velocity)
for track in mid.tracks:
    abs_tick = 0
    for msg in track:
        abs_tick += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            midi_notes.append((tick_to_sec(abs_tick), msg.note, msg.velocity))

midi_notes.sort(key=lambda x: x[0])
print(f"\nMIDI notes extracted: {len(midi_notes)}")
print(f"Score duration:        {midi_notes[-1][0]:.1f}s")

# Section label per note
def get_section(t):
    for name, start, end, _ in SECTIONS:
        if start <= t < end:
            return name
    return '尾声'

# ── 2. Load score WAV ─────────────────────────────────────────────────────────
# Unique score onset times (chords share one onset time)
score_onset_times = np.array(sorted(set(t for t, p, v in midi_notes)))
score_duration    = float(score_onset_times[-1]) + 1.0
print(f"\nScore unique onset times: {len(score_onset_times)}  |  duration: {score_duration:.1f}s")

# ── 3. DTW align each performer → compute agogic deviations ──────────────────
all_results = {}

for name, (perf_path, color) in PERFORMERS.items():
    print(f"\n{'─'*50}")
    print(f"Aligning: {name}")
    y_perf, _ = librosa.load(str(perf_path), sr=SR)
    perf_dur   = librosa.get_duration(y=y_perf, sr=SR)
    pace_ratio = perf_dur / score_duration
    print(f"  Duration: {perf_dur:.1f}s  |  pace ratio: {pace_ratio:.3f}")

    # Detect performer onsets from audio
    onset_env    = librosa.onset.onset_strength(y=y_perf, sr=SR, hop_length=HOP)
    onset_frames = librosa.util.peak_pick(onset_env,
                                          pre_max=3, post_max=3,
                                          pre_avg=3, post_avg=3,
                                          delta=0.15, wait=8)
    perf_onset_times = librosa.frames_to_time(onset_frames, sr=SR, hop_length=HOP)
    print(f"  Performer onsets detected: {len(perf_onset_times)}")

    # Normalise performer times into score-time units (remove global tempo)
    perf_norm = perf_onset_times / pace_ratio

    # DTW on onset-time sequences (1D, in score seconds)
    # Both sequences now live in the same time coordinate → radius handles only rubato
    n_s, n_p = len(score_onset_times), len(perf_norm)
    print(f"  Running DTW on onset sequences ({n_s} score × {n_p} perf, radius={DTW_RADIUS})...")
    _, path = fastdtw(
        score_onset_times.reshape(-1, 1).astype(np.float32),
        perf_norm.reshape(-1, 1).astype(np.float32),
        dist=euclidean,
        radius=DTW_RADIUS,
    )

    # Build score-onset-index → performer onset time mapping
    path_arr = np.array(path)
    # For each score onset index, collect all matched performer indices
    from collections import defaultdict
    s2p = defaultdict(list)
    for si, pi in path_arr:
        s2p[int(si)].append(int(pi))
    # Take median performer index for each score onset
    s2p_median = {si: int(np.median(pis)) for si, pis in s2p.items()}

    # Map each MIDI note to the nearest score onset index, then to performer time
    # (chord notes share one onset index)
    score_onset_arr = score_onset_times  # shape (K,)

    # Compute agogic deviation for each MIDI note
    records = []
    for note_idx, (t_score, pitch, vel) in enumerate(midi_notes):
        # Find which score onset bucket this note belongs to
        si = int(np.searchsorted(score_onset_arr, t_score, side='left'))
        si = min(si, len(score_onset_arr) - 1)
        # If this note time is closer to previous bucket, use that
        if si > 0 and abs(score_onset_arr[si-1] - t_score) < abs(score_onset_arr[si] - t_score):
            si -= 1

        pi         = s2p_median.get(si, si)
        pi         = min(pi, len(perf_onset_times) - 1)
        t_perf_dtw = perf_onset_times[pi]   # actual performer onset time
        t_expected  = t_score * pace_ratio   # where uniform tempo predicts
        dev_ms      = (t_perf_dtw - t_expected) * 1000   # ms; + = late, - = early

        records.append({
            'note_idx':   note_idx,
            'score_time': t_score,
            'perf_time':  t_perf_dtw,
            'expected':   t_expected,
            'dev_ms':     dev_ms,
            'pitch':      pitch,
            'velocity':   vel,
            'section':    get_section(t_score),
        })

    df = pd.DataFrame(records)
    all_results[name] = {'df': df, 'color': color, 'pace': pace_ratio}

    # Per-section summary
    print(f"\n  Section          Mean±Std dev (ms)   |dev| median")
    for sec_name, *_ in SECTIONS:
        sub = df[df['section'] == sec_name]['dev_ms']
        if len(sub) == 0:
            continue
        print(f"  {sec_name:<16}  {np.mean(sub):+6.0f} ± {np.std(sub):.0f} ms"
              f"    median |dev|={np.median(np.abs(sub)):.0f} ms")

# ── 4. Save CSV ───────────────────────────────────────────────────────────────
rows = []
for name, res in all_results.items():
    df = res['df'].copy()
    df.insert(0, 'performer', name)
    rows.append(df)
combined = pd.concat(rows, ignore_index=True)
csv_path = base / 'results_ultimate' / 'note_alignment.csv'
combined.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\nSaved: {csv_path}")

# ── 5. Visualisation ──────────────────────────────────────────────────────────

# Helper: section background shading
def shade_sections(ax, y_bottom=-9999, y_top=9999):
    for sec_name, start, end, bg in SECTIONS:
        ax.axvspan(start, end, color=bg, alpha=0.35, zorder=0)
    for _, start, end, _ in SECTIONS[1:]:
        ax.axvline(start, color='gray', linewidth=0.8, linestyle=':', zorder=1)

# ── Plot A: Agogic deviation over score time (smoothed) ──────────────────────
fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)

for ax, (name, res) in zip(axes, all_results.items()):
    df     = res['df']
    color  = res['color']
    shade_sections(ax)

    # Raw deviation (translucent dots)
    ax.scatter(df['score_time'], df['dev_ms'],
               color=color, alpha=0.18, s=6, zorder=2)

    # Smoothed curve (window = 15 notes)
    smooth = uniform_filter1d(df['dev_ms'].values, size=15)
    ax.plot(df['score_time'], smooth,
            color=color, linewidth=2.0, zorder=3, label='smoothed')

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', zorder=2)
    ax.set_ylabel('Agogic Dev (ms)\n+ = late  – = early', fontsize=9)
    ax.set_title(f'{name}  (pace {res["pace"]:.2f}×)', fontsize=11,
                 color=color, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')

    # Annotate section labels on top panel
    if name == list(all_results.keys())[0]:
        for sec_name, start, end, _ in SECTIONS:
            mid_x = (start + end) / 2
            ax.text(mid_x, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 200,
                    SEC_EN.get(sec_name, sec_name), ha='center', va='top', fontsize=8,
                    color='#555555', fontweight='bold')

axes[-1].set_xlabel('Score Time (s)', fontsize=10)

# Add section labels after limits are set
for ax, (name, res) in zip(axes, all_results.items()):
    yhi = ax.get_ylim()[1]
    for sec_name, start, end, _ in SECTIONS:
        ax.text((start+end)/2, yhi * 0.90,
                SEC_EN.get(sec_name, sec_name), ha='center', va='top',
                fontsize=7.5, color='#444', fontweight='bold')

fig.suptitle('Agogic Deviation per Note / 逐音符提前·延后量\n'
             'Deviation from expected beat (ms)  |  《彩云追月》',
             fontsize=13, fontweight='bold')
plt.tight_layout()
p_a = out / '17_agogic_deviation.png'
fig.savefig(p_a, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {p_a}")

# ── Plot B: Overlay comparison ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 6))
shade_sections(ax)

for name, res in all_results.items():
    df     = res['df']
    color  = res['color']
    smooth = uniform_filter1d(df['dev_ms'].values, size=20)
    ax.plot(df['score_time'], smooth,
            color=color, linewidth=2.2, label=name, zorder=3)

ax.axhline(0, color='black', linewidth=1.0, linestyle='--', zorder=2)
ax.set_xlabel('Score Time (s)', fontsize=10)
ax.set_ylabel('Agogic Deviation (ms)  [smoothed, window=20 notes]', fontsize=10)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.2, axis='y')

# Section labels
yhi, ylo = ax.get_ylim()
for sec_name, start, end, _ in SECTIONS:
    ax.text((start+end)/2, yhi - (yhi-ylo)*0.06,
            SEC_EN.get(sec_name, sec_name), ha='center', va='top',
            fontsize=8, color='#444', fontweight='bold')

fig.suptitle('Agogic Deviation Overlay / 三位演奏者逐音符时值偏差叠加对比\n'
             'Lang Lang · Li Yundi · Shen Wenyu  |  《彩云追月》',
             fontsize=13, fontweight='bold')
plt.tight_layout()
p_b = out / '18_agogic_overlay.png'
fig.savefig(p_b, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {p_b}")

# ── Plot C: Box plot per section ──────────────────────────────────────────────
sec_names = [s[0] for s in SECTIONS]
fig, axes = plt.subplots(1, len(sec_names), figsize=(14, 5), sharey=True)

for ax, (sec_name, sc_s, sc_e, bg) in zip(axes, SECTIONS):
    ax.set_facecolor(bg + '55')
    data   = []
    labels = []
    colors = []
    for name, res in all_results.items():
        sub = res['df'][res['df']['section'] == sec_name]['dev_ms'].values
        data.append(sub)
        labels.append(name.split()[0])   # first name only
        colors.append(res['color'])

    bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                    medianprops=dict(color='black', linewidth=2))
    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)

    ax.set_title(SEC_EN.get(sec_name, sec_name), fontsize=10, fontweight='bold')
    ax.set_xticklabels(labels, fontsize=9)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.grid(True, alpha=0.2, axis='y')
    if ax == axes[0]:
        ax.set_ylabel('Agogic Deviation (ms)', fontsize=10)

fig.suptitle('Agogic Deviation Distribution per Section / 各段落时值偏差分布\n'
             'Box plots — Lang Lang · Li Yundi · Shen Wenyu  |  《彩云追月》',
             fontsize=12, fontweight='bold')
plt.tight_layout()
p_c = out / '19_agogic_boxplot.png'
fig.savefig(p_c, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {p_c}")

# ── Console key findings ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("KEY FINDINGS / 关键发现")
print("=" * 65)
for sec_name, _, _, _ in SECTIONS:
    print(f"\n【{sec_name}】")
    for name, res in all_results.items():
        sub = res['df'][res['df']['section'] == sec_name]['dev_ms']
        med_abs = np.median(np.abs(sub))
        mean    = np.mean(sub)
        trend   = '→ rushes ahead' if mean < -30 else ('→ holds back' if mean > 30 else '→ on-track')
        print(f"  {name:<12}  mean={mean:+6.0f}ms  |dev|={med_abs:.0f}ms  {trend}")

print("\n" + "=" * 65)
print("Done.")
print("=" * 65)
