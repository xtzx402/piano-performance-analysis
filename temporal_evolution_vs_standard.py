#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temporal Evolution Analysis: Three Pianists vs Standard Sheet Music
时间演化对比分析：三位钢琴家与标准乐谱

Shows how RMS energy, spectral centroid, and other features evolve over time
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("\n" + "="*90)
print("时间演化对比分析 / Temporal Evolution Comparative Analysis")
print("="*90)

# Standard reference parameters
STANDARD_DURATION = 150  # Approximate standard duration based on sheet music (Moderato ~90BPM)

performers_data = {
    "langlang_caiyun.wav": "郎朗 (Lang Lang)",
    "liyundi_caiyun.wav": "李云迪 (Li Yundi)",
    "shenwenyu_caiyun.wav": "沈文裕 (Shen Wenyu)"
}

colors = ['#E74C3C', '#F39C12', '#3498DB']  # Red, Yellow, Blue - high contrast

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('《彩云追月》：三位演奏家与标准乐谱的时间演化对比\nTemporal Evolution: Three Pianists vs Standard Sheet Music',
             fontsize=14, fontweight='bold', y=0.995)

# Subplot 1: RMS Energy Evolution (Dynamics)
ax_rms = axes[0, 0]
ax_rms.set_title('力度演化 / RMS Energy Evolution\n(力度随时间的变化)', fontsize=11, fontweight='bold')

# Subplot 2: Spectral Centroid Evolution (Tone Color)
ax_spec = axes[0, 1]
ax_spec.set_title('音色演化 / Spectral Centroid Evolution\n(音色亮度随时间的变化)', fontsize=11, fontweight='bold')

# Subplot 3: Tempo Consistency (IOI - Inter-Onset Intervals)
ax_tempo = axes[1, 0]
ax_tempo.set_title('节奏一致性 / Tempo Consistency (IOI)\n(音符间隔的稳定性)', fontsize=11, fontweight='bold')

# Subplot 4: Summary Statistics
ax_summary = axes[1, 1]
ax_summary.axis('off')

summary_text = "【三位钢琴家与标准乐谱的对比】\n" + "="*40 + "\n\n"
summary_text += "标准乐谱参数 / Standard Parameters:\n"
summary_text += "• 速度标记：Moderato Chiaramente ≈ 90 BPM\n"
summary_text += "• 调号：B Major (五个升号)\n"
summary_text += "• 拍号：4/4\n"
summary_text += "• 标准时长：~150秒\n"
summary_text += "• 力度：p → mf → ff → pp\n\n"

# Analyze each performer
all_data = []

for idx, (filename, performer_name) in enumerate(performers_data.items()):
    print(f"\n分析 {performer_name}...")

    file_path = Path(filename)
    if not file_path.exists():
        print(f"  ⚠ 文件未找到")
        continue

    # Load audio
    y, sr = librosa.load(filename, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    # Extract features with time windows (2 seconds)
    window_size = 2
    hop_length = int(0.5 * sr)  # 0.5 second hop for smooth curves
    n_frames = int(duration / (hop_length / sr))

    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)

    # RMS Energy (Dynamics)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    # Spectral Centroid (Tone Color)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]

    # Trim times to match rms and spec_cent length
    min_len = min(len(times), len(rms), len(spec_cent))
    times_trimmed = times[:min_len]
    rms_trimmed = rms[:min_len]
    spec_cent_trimmed = spec_cent[:min_len]

    # Onset intervals for tempo consistency
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3,
                                          pre_avg=3, post_avg=3, delta=0.1, wait=10)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    ioi = np.diff(onset_times)  # Inter-onset intervals
    ioi = ioi[ioi > 0.1]  # Filter out very small intervals

    # Estimate tempo from IOI
    if len(ioi) > 0:
        avg_ioi = np.mean(ioi)
        estimated_tempo = 60 / avg_ioi
    else:
        estimated_tempo = 0

    # Plot 1: RMS Energy
    ax_rms.plot(times_trimmed, rms_trimmed, color=colors[idx], label=performer_name, linewidth=2, alpha=0.8)

    # Plot 2: Spectral Centroid
    ax_spec.plot(times_trimmed, spec_cent_trimmed, color=colors[idx], label=performer_name, linewidth=2, alpha=0.8)

    # Plot 3: IOI (Tempo Consistency)
    if len(ioi) > 0:
        # Map IOI back to time
        onset_times_filtered = onset_times[:-1][ioi > 0.1]
        ax_tempo.scatter(onset_times_filtered, 60/ioi, color=colors[idx], label=performer_name,
                        alpha=0.6, s=30)

    # Collect data for summary
    all_data.append({
        'Performer': performer_name,
        'Duration (s)': duration,
        'Avg RMS': np.mean(rms_trimmed),
        'RMS Std': np.std(rms_trimmed),
        'Avg Spectral Centroid (Hz)': np.mean(spec_cent_trimmed),
        'Estimated Tempo (BPM)': estimated_tempo,
        'IOI Consistency (CV)': np.std(ioi) / np.mean(ioi) if len(ioi) > 0 else 0
    })

# Configure Plot 1: RMS Energy
ax_rms.set_xlabel('时间 / Time (seconds)', fontsize=10)
ax_rms.set_ylabel('RMS能量 / RMS Energy', fontsize=10)
ax_rms.legend(loc='upper right', fontsize=9)
ax_rms.grid(True, alpha=0.3)
ax_rms.axhline(y=0.04, color='red', linestyle='--', linewidth=1, alpha=0.5, label='平均参考线')

# Configure Plot 2: Spectral Centroid
ax_spec.set_xlabel('时间 / Time (seconds)', fontsize=10)
ax_spec.set_ylabel('谱心频率 / Spectral Centroid (Hz)', fontsize=10)
ax_spec.legend(loc='upper right', fontsize=9)
ax_spec.grid(True, alpha=0.3)

# Configure Plot 3: IOI
ax_tempo.set_xlabel('时间 / Time (seconds)', fontsize=10)
ax_tempo.set_ylabel('估计速度 / Estimated Tempo (BPM)', fontsize=10)
ax_tempo.legend(loc='upper right', fontsize=9)
ax_tempo.grid(True, alpha=0.3)
ax_tempo.axhline(y=90, color='red', linestyle='--', linewidth=2, alpha=0.7, label='标准速度 (90 BPM)')

# Configure Plot 4: Summary Statistics
summary_text += "演奏家数据 / Pianist Data:\n"
summary_text += "-"*40 + "\n"

for data in all_data:
    summary_text += f"\n{data['Performer']}:\n"
    summary_text += f"  时长: {data['Duration (s)']:.1f}秒\n"
    summary_text += f"  平均RMS: {data['Avg RMS']:.4f}\n"
    summary_text += f"  RMS变化: {data['RMS Std']:.4f}\n"
    summary_text += f"  平均音色亮度: {data['Avg Spectral Centroid (Hz)']:.0f} Hz\n"
    summary_text += f"  估计速度: {data['Estimated Tempo (BPM)']:.1f} BPM\n"
    summary_text += f"  节奏稳定性: CV = {data['IOI Consistency (CV)']:.4f}\n"

ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
               fontsize=9, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('results_ultimate/plots/11_temporal_vs_standard.png', dpi=150, bbox_inches='tight')
print("\n✓ 时间演化对比图已保存")
plt.close()

# Save detailed comparison data
comparison_df = pd.DataFrame(all_data)
comparison_df.to_csv('performance_interpretation_analysis.csv', index=False, encoding='utf-8-sig')

print("\n" + "="*90)
print("【对比分析总结 / Comparative Summary】")
print("="*90)
print(comparison_df.to_string(index=False))

print("\n" + "="*90)
print("【关键解释 / Key Interpretations】")
print("="*90)

# Analyze differences from standard
print("\n1. 与标准乐谱的主要差异 / Main Differences from Standard:")
print("   " + "-"*60)

avg_rms = comparison_df['Avg RMS'].mean()
print(f"   • 力度对比（平均RMS）：")
for idx, row in comparison_df.iterrows():
    diff_pct = ((row['Avg RMS'] - avg_rms) / avg_rms) * 100
    print(f"     {row['Performer']:20s}: {diff_pct:+.1f}% {'(较强)' if diff_pct > 0 else '(较弱)'}")

avg_centroid = comparison_df['Avg Spectral Centroid (Hz)'].mean()
print(f"\n   • 音色亮度对比（谱心频率）：")
for idx, row in comparison_df.iterrows():
    diff_hz = row['Avg Spectral Centroid (Hz)'] - avg_centroid
    print(f"     {row['Performer']:20s}: {diff_hz:+.0f} Hz {'(较亮)' if diff_hz > 0 else '(较暗)'}")

print(f"\n   • 节奏自由度对比（IOI稳定性）：")
for idx, row in comparison_df.iterrows():
    if row['IOI Consistency (CV)'] < 0.15:
        flexibility = "严格、精确"
    elif row['IOI Consistency (CV)'] < 0.25:
        flexibility = "适度自由"
    else:
        flexibility = "高度自由、艺术化"
    print(f"     {row['Performer']:20s}: CV={row['IOI Consistency (CV)']:.4f} → {flexibility}")

print("\n" + "="*90)
print("✓ 分析完成 / Analysis complete")
print("="*90)
