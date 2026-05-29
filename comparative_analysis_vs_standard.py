#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparative Analysis: Three Pianists vs Standard Sheet Music
比较分析：三个钢琴家与标准乐谱的对比

This script compares each pianist's performance against the standard interpretation
based on the sheet music (Moderato Chiaramente, 90 BPM, B Major).
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.font_manager as fm

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("="*90)
print("《彩云追月》三位钢琴家 vs 标准乐谱对比分析")
print("Comparative Analysis: Three Pianists vs Standard Sheet Music Reference")
print("="*90)

# Standard reference parameters from sheet music
STANDARD_TEMPO_BPM = 90  # Moderato Chiaramente
STANDARD_KEY = "B Major"
STANDARD_TIME_SIG = "4/4"
STANDARD_DYNAMICS = "p to mf to ff"

performers_data = {
    "langlang_caiyun.wav": "郎朗 (Lang Lang)",
    "liyundi_caiyun.wav": "李云迪 (Li Yundi)",
    "shenwenyu_caiyun.wav": "沈文裕 (Shen Wenyu)"
}

results = []

print("\n【分析参数 / Analysis Parameters】")
print("-"*90)
print(f"标准速度 / Standard Tempo: {STANDARD_TEMPO_BPM} BPM (Moderato Chiaramente)")
print(f"标准调性 / Standard Key: {STANDARD_KEY}")
print(f"标准拍号 / Standard Time: {STANDARD_TIME_SIG}")
print(f"参考文献 / Reference: 王港中(1975)琴谱版本")

print("\n【逐位钢琴家分析 / Individual Pianist Analysis】")
print("="*90)

for filename, performer_name in performers_data.items():
    print(f"\n{performer_name}")
    print("-"*90)

    file_path = Path(filename)
    if not file_path.exists():
        print(f"⚠ 文件未找到 / File not found: {filename}")
        continue

    # Load audio
    y, sr = librosa.load(filename, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    print(f"  文件 / File: {filename}")
    print(f"  采样率 / Sample Rate: {sr} Hz")
    print(f"  时长 / Duration: {duration:.2f} seconds")

    # Extract tempo using onset detection
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3,
                                          pre_avg=3, post_avg=3, delta=0.1, wait=10)

    if len(onset_frames) > 1:
        # Estimate tempo from inter-onset intervals
        times = librosa.frames_to_time(onset_frames, sr=sr)
        ioi = np.diff(times)  # Inter-onset intervals in seconds

        # Average inter-onset interval
        avg_ioi = np.mean(ioi[ioi > 0.1])  # Filter out very small intervals
        estimated_tempo = 60 / avg_ioi if avg_ioi > 0 else 0

        tempo_deviation = ((estimated_tempo - STANDARD_TEMPO_BPM) / STANDARD_TEMPO_BPM) * 100

        print(f"\n  ⏱ 速度分析 / Tempo Analysis:")
        print(f"    估计速度 / Estimated Tempo: {estimated_tempo:.1f} BPM")
        print(f"    标准速度 / Standard Tempo: {STANDARD_TEMPO_BPM} BPM")
        print(f"    偏差 / Deviation: {tempo_deviation:+.1f}% {'(较快 / Faster)' if tempo_deviation > 0 else '(较慢 / Slower)'}")
    else:
        estimated_tempo = 0
        tempo_deviation = 0
        print(f"  ⚠ 无法估计速度 / Cannot estimate tempo")

    # Extract RMS energy (dynamics)
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    print(f"\n  💪 力度分析 / Dynamics Analysis:")
    print(f"    平均RMS能量 / Average RMS: {rms_mean:.4f}")
    print(f"    RMS标准差 / RMS Std Dev: {rms_std:.4f}")
    print(f"    动态范围 / Dynamic Range: {(np.max(rms) - np.min(rms)):.4f}")
    print(f"    力度变化系数 / Loudness Variation CV: {(rms_std / rms_mean):.4f}")

    # Extract spectral features
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = np.mean(centroid)

    print(f"\n  🎵 音色分析 / Tone Color Analysis:")
    print(f"    谱心频率均值 / Spectral Centroid Mean: {centroid_mean:.1f} Hz")

    # Timing variability (using onset intervals)
    if len(ioi) > 0:
        timing_cv = np.std(ioi[ioi > 0.1]) / np.mean(ioi[ioi > 0.1]) if np.mean(ioi[ioi > 0.1]) > 0 else 0
        print(f"\n  📐 节奏自由度 / Rhythmic Flexibility:")
        print(f"    音符间隔标准差 / IOI Std Dev: {np.std(ioi[ioi > 0.1]):.4f}")
        print(f"    节奏变化系数 / Timing CV: {timing_cv:.4f}")
        print(f"    解释 / Interpretation: {'高自由度 (High Rubato)' if timing_cv > 0.2 else '中等自由度 (Moderate)' if timing_cv > 0.1 else '严格节奏 (Strict Timing)'}")

    # Store results
    results.append({
        'Performer': performer_name,
        'Estimated Tempo (BPM)': estimated_tempo,
        'Tempo Deviation (%)': tempo_deviation,
        'RMS Mean': rms_mean,
        'RMS Std Dev': rms_std,
        'Dynamic Range': np.max(rms) - np.min(rms),
        'Spectral Centroid (Hz)': centroid_mean,
        'Timing CV': timing_cv if len(ioi) > 0 else 0,
        'Duration (s)': duration
    })

# Create summary table
print("\n\n【对比总结表 / Comparative Summary】")
print("="*90)

results_df = pd.DataFrame(results)
results_df.to_csv('performance_vs_standard.csv', index=False, encoding='utf-8-sig')

print("\n演奏速度对比 / Tempo Comparison:")
print("-"*90)
for idx, row in results_df.iterrows():
    deviation = row['Tempo Deviation (%)']
    status = "✓ 接近标准" if abs(deviation) < 5 else f"{'⚡ 较快' if deviation > 0 else '🐢 较慢'}"
    print(f"{row['Performer']:20s}: {row['Estimated Tempo (BPM)']:6.1f} BPM (标准90 BPM, {deviation:+6.1f}%) {status}")

print("\n力度对比 / Dynamics Comparison:")
print("-"*90)
for idx, row in results_df.iterrows():
    print(f"{row['Performer']:20s}: 平均RMS={row['RMS Mean']:.4f}, 范围={row['Dynamic Range']:.4f}, 变化CV={row['RMS Std Dev']/row['RMS Mean']:.4f}")

print("\n节奏自由度对比 / Rhythmic Flexibility Comparison:")
print("-"*90)
for idx, row in results_df.iterrows():
    timing_cv = row['Timing CV']
    if timing_cv > 0.2:
        flexibility = "高自由度 (High Rubato)"
    elif timing_cv > 0.1:
        flexibility = "中等自由度 (Moderate Rubato)"
    else:
        flexibility = "严格节奏 (Strict Timing)"
    print(f"{row['Performer']:20s}: CV={timing_cv:.4f} → {flexibility}")

print("\n\n【关键发现 / Key Findings】")
print("="*90)

# Find extremes
fastest = results_df.loc[results_df['Estimated Tempo (BPM)'].idxmax()]
slowest = results_df.loc[results_df['Estimated Tempo (BPM)'].idxmin()]
most_dynamic = results_df.loc[results_df['Dynamic Range'].idxmax()]
most_flexible = results_df.loc[results_df['Timing CV'].idxmax()]

print(f"\n1. 速度最快 / Fastest Tempo:")
print(f"   {fastest['Performer']}: {fastest['Estimated Tempo (BPM)']:.1f} BPM ({fastest['Tempo Deviation (%)']:+.1f}%)")

print(f"\n2. 速度最慢 / Slowest Tempo:")
print(f"   {slowest['Performer']}: {slowest['Estimated Tempo (BPM)']:.1f} BPM ({slowest['Tempo Deviation (%)']:+.1f}%)")

print(f"\n3. 力度最强 / Largest Dynamic Range:")
print(f"   {most_dynamic['Performer']}: {most_dynamic['Dynamic Range']:.4f}")

print(f"\n4. 节奏最自由 / Most Rhythmic Flexibility:")
print(f"   {most_flexible['Performer']}: CV = {most_flexible['Timing CV']:.4f}")

print("\n" + "="*90)
print(f"✓ 对比分析完成 / Comparative analysis complete")
print(f"✓ 结果已保存到 / Results saved to: performance_vs_standard.csv")
print("="*90)
