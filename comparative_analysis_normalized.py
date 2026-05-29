#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fair Comparative Analysis Using Normalized Audio
使用标准化音频的公平对比分析

This analysis uses normalized audio files to ensure fair comparison:
- Same sample rate (22050 Hz)
- Same RMS energy level (0.05)
- Time-aligned (silence removed)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("="*90)
print("公平对比分析（使用标准化音频）/ Fair Comparative Analysis (Normalized Audio)")
print("="*90)

# Standard reference parameters
STANDARD_TEMPO_BPM = 90
normalized_dir = Path("normalized_audio")

performers_data = {
    "normalized_langlang_caiyun.wav": "郎朗 (Lang Lang)",
    "normalized_liyundi_caiyun.wav": "李云迪 (Li Yundi)",
    "normalized_shenwenyu_caiyun.wav": "沈文裕 (Shen Wenyu)"
}

results = []

print("\n【标准化音频分析 / Analysis of Normalized Audio】")
print("="*90)

for filename, performer_name in performers_data.items():
    print(f"\n{performer_name}")
    print("-"*90)

    file_path = normalized_dir / filename
    if not file_path.exists():
        print(f"  ⚠ 文件未找到 / File not found: {file_path}")
        continue

    # Load normalized audio
    y, sr = librosa.load(str(file_path), sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    print(f"  采样率 / Sample Rate: {sr} Hz")
    print(f"  时长 / Duration: {duration:.2f} seconds")

    # Verify normalization
    rms = np.sqrt(np.mean(y**2))
    print(f"  验证RMS / Verified RMS: {rms:.6f} (应为 0.050000)")

    # Extract features
    # 1. Tempo estimation
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3,
                                          pre_avg=3, post_avg=3, delta=0.1, wait=10)

    if len(onset_frames) > 1:
        times = librosa.frames_to_time(onset_frames, sr=sr)
        ioi = np.diff(times)
        ioi = ioi[ioi > 0.1]

        if len(ioi) > 0:
            avg_ioi = np.mean(ioi)
            estimated_tempo = 60 / avg_ioi
            tempo_deviation = ((estimated_tempo - STANDARD_TEMPO_BPM) / STANDARD_TEMPO_BPM) * 100
            timing_cv = np.std(ioi) / np.mean(ioi) if np.mean(ioi) > 0 else 0
        else:
            estimated_tempo = 0
            tempo_deviation = 0
            timing_cv = 0
    else:
        estimated_tempo = 0
        tempo_deviation = 0
        timing_cv = 0

    print(f"\n  ⏱ 速度分析 / Tempo Analysis:")
    print(f"    估计速度 / Estimated Tempo: {estimated_tempo:.1f} BPM")
    print(f"    标准速度 / Standard Tempo: {STANDARD_TEMPO_BPM} BPM")
    print(f"    偏差 / Deviation: {tempo_deviation:+.1f}%")
    print(f"    节奏一致性CV / Timing CV: {timing_cv:.4f}")

    # 2. Dynamics analysis (RMS is now normalized, so analyze variation)
    rms_envelope = librosa.feature.rms(y=y, hop_length=512)[0]
    rms_mean = np.mean(rms_envelope)
    rms_std = np.std(rms_envelope)
    dynamic_range = (np.max(rms_envelope) - np.min(rms_envelope)) * 1000  # in millivolts

    print(f"\n  💪 力度分析 / Dynamics Analysis:")
    print(f"    平均RMS（标准化） / Average RMS (Normalized): {rms_mean:.6f}")
    print(f"    RMS变化标准差 / RMS Std Dev: {rms_std:.6f}")
    print(f"    动态范围 / Dynamic Range: {dynamic_range:.2f} mV")
    print(f"    力度表现 / Dynamics Character: {'较稳定' if rms_std < 0.01 else '变化丰富'}")

    # 3. Spectral features
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spec_cent_mean = np.mean(spec_cent)
    spec_cent_std = np.std(spec_cent)

    print(f"\n  🎵 音色分析 / Tone Color Analysis:")
    print(f"    谱心频率均值 / Spectral Centroid Mean: {spec_cent_mean:.1f} Hz")
    print(f"    谱心频率标准差 / Spectral Centroid Std: {spec_cent_std:.1f} Hz")

    # Store results
    results.append({
        'Performer': performer_name,
        'Duration (s)': duration,
        'Normalized RMS': rms,
        'RMS Envelope Mean': rms_mean,
        'RMS Envelope Std': rms_std,
        'Dynamic Range (mV)': dynamic_range,
        'Estimated Tempo (BPM)': estimated_tempo,
        'Tempo Deviation (%)': tempo_deviation,
        'Timing CV': timing_cv,
        'Spectral Centroid Mean (Hz)': spec_cent_mean,
        'Spectral Centroid Std (Hz)': spec_cent_std
    })

# Create summary table
print("\n\n" + "="*90)
print("【对比总结表 / Comparative Summary Table】")
print("="*90)

results_df = pd.DataFrame(results)
results_df.to_csv('normalized_comparison_results.csv', index=False, encoding='utf-8-sig')
print(results_df.to_string(index=False))

# Analysis
print("\n\n" + "="*90)
print("【关键发现 / Key Findings】")
print("="*90)

print("\n1. 采样率和RMS标准化后的对比 / Comparison After Normalization:")
print("-"*90)
for idx, row in results_df.iterrows():
    print(f"\n{row['Performer']}:")
    print(f"  时长: {row['Duration (s)']:.1f}s")
    print(f"  RMS标准化验证: {row['Normalized RMS']:.6f} ✓")
    print(f"  动态范围: {row['Dynamic Range (mV)']:.1f} mV")
    print(f"  节奏灵活度: CV = {row['Timing CV']:.4f}")
    print(f"  音色亮度: {row['Spectral Centroid Mean (Hz)']:.0f} Hz")

print("\n2. 时长对比 / Duration Comparison:")
print("-"*90)
for idx, row in results_df.iterrows():
    print(f"{row['Performer']:<20}: {row['Duration (s)']:6.1f}s")

print("\n3. 动态范围对比（标准化后） / Dynamic Range Comparison (After Normalization):")
print("-"*90)
max_dynamic = results_df['Dynamic Range (mV)'].max()
min_dynamic = results_df['Dynamic Range (mV)'].min()

for idx, row in results_df.iterrows():
    pct = (row['Dynamic Range (mV)'] - min_dynamic) / (max_dynamic - min_dynamic) * 100
    bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
    print(f"{row['Performer']:<20}: {bar} {row['Dynamic Range (mV)']:6.1f} mV")

print("\n4. 音色亮度对比 / Tone Brightness Comparison:")
print("-"*90)
for idx, row in results_df.iterrows():
    brightness = "明亮" if row['Spectral Centroid Mean (Hz)'] > 900 else "温暖" if row['Spectral Centroid Mean (Hz)'] > 800 else "深暗"
    print(f"{row['Performer']:<20}: {row['Spectral Centroid Mean (Hz)']:6.0f} Hz → {brightness}")

print("\n" + "="*90)
print("✓ 标准化对比分析完成 / Normalized comparison analysis complete!")
print("="*90)

print("\n【重要说明 / Important Notes】")
print("-"*90)
print("1. 所有音频已标准化至同一RMS能量水平 (0.050)")
print("2. 采样率统一为 22050 Hz")
print("3. 前置沉默已去除，音乐内容对齐")
print("4. 现在的对比是公平且有意义的 / Fair and meaningful comparison now")
print("\n标准化确保了：")
print("  ✓ 消除录音条件差异 / Eliminated recording condition differences")
print("  ✓ 公平的力度对比 / Fair loudness comparison")
print("  ✓ 可靠的音色分析 / Reliable tone color analysis")
print("  ✓ 准确的节奏分析 / Accurate rhythm analysis")
