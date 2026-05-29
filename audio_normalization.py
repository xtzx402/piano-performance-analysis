#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Normalization and Alignment for Fair Comparison
音频标准化和对齐 - 确保公平的对比

Normalize three piano recordings to enable fair comparison:
1. Remove leading silence (time alignment)
2. Normalize RMS energy (loudness standardization)
3. Resample to same sample rate
4. Generate normalized audio files
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import pandas as pd

print("="*90)
print("音频标准化处理 / Audio Normalization and Alignment")
print("="*90)

performers_data = {
    "langlang_caiyun.wav": "郎朗 (Lang Lang)",
    "liyundi_caiyun.wav": "李云迪 (Li Yundi)",
    "shenwenyu_caiyun.wav": "沈文裕 (Shen Wenyu)"
}

# Create output directory
output_dir = Path("normalized_audio")
output_dir.mkdir(exist_ok=True)

# Standard parameters
TARGET_SR = 22050  # Standard sample rate
TARGET_RMS = 0.05  # Target RMS level for normalization

normalization_log = []

print("\n【第一步：检测音频开始点 / Step 1: Detect Audio Start Point】")
print("-"*90)

for filename, performer_name in performers_data.items():
    print(f"\n{performer_name}")
    print("  " + "-"*60)

    file_path = Path(filename)
    if not file_path.exists():
        print(f"  ⚠ 文件未找到 / File not found: {filename}")
        continue

    # Load audio
    y, sr = librosa.load(filename, sr=None)
    original_sr = sr
    original_duration = librosa.get_duration(y=y, sr=sr)
    original_rms = np.sqrt(np.mean(y**2))

    print(f"  原始采样率 / Original SR: {sr} Hz")
    print(f"  原始时长 / Original Duration: {original_duration:.2f}s")
    print(f"  原始RMS能量 / Original RMS: {original_rms:.6f}")

    # Step 1: Detect leading silence and trim
    # Use energy-based detection
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)
    energy = np.mean(S_db, axis=0)

    # Find threshold (silence threshold)
    threshold = np.percentile(energy, 5)

    # Find first frame above threshold
    frames_above = np.where(energy > threshold)[0]

    if len(frames_above) > 0:
        first_frame = frames_above[0]
        # Add some margin (0.5 seconds before)
        margin_frames = int(0.5 * sr / (y.shape[0] / (librosa.get_duration(y=y, sr=sr))))
        start_frame = max(0, first_frame - margin_frames)
        start_sample = librosa.frames_to_samples(start_frame, hop_length=512)

        # Trim
        y_trimmed = y[start_sample:]
        trimmed_duration = librosa.get_duration(y=y_trimmed, sr=sr)
        trimmed_rms = np.sqrt(np.mean(y_trimmed**2))

        print(f"\n  检测结果 / Detection Result:")
        print(f"  • 检测到的开始点 / Detected start: {start_sample/sr:.2f}s")
        print(f"  • 去除沉默后时长 / Trimmed duration: {trimmed_duration:.2f}s")
        print(f"  • 去除沉默后RMS / Trimmed RMS: {trimmed_rms:.6f}")
    else:
        y_trimmed = y
        trimmed_duration = original_duration
        trimmed_rms = original_rms
        print(f"  ⚠ 无法检测到明显开始点 / Could not detect clear start point")

    # Step 2: Normalize RMS energy
    # Scale to target RMS
    if trimmed_rms > 0:
        scale_factor = TARGET_RMS / trimmed_rms
        y_normalized = y_trimmed * scale_factor

        # Soft limiting to prevent clipping (avoid hard clipping artifacts)
        max_val = np.max(np.abs(y_normalized))
        if max_val > 1.0:
            # Use soft clipping with tanh function for smoother limiting
            y_normalized = np.tanh(y_normalized / max_val) * 0.99
            scale_factor = scale_factor / max_val
            print(f"  ⚠ 应用软限制 / Soft limiting applied (tanh-based)")
    else:
        y_normalized = y_trimmed
        scale_factor = 1.0

    normalized_rms = np.sqrt(np.mean(y_normalized**2))
    print(f"\n  标准化结果 / Normalization Result:")
    print(f"  • 缩放因子 / Scale factor: {scale_factor:.4f}")
    print(f"  • 标准化后RMS / Normalized RMS: {normalized_rms:.6f}")

    # Step 3: Resample to standard sample rate
    if original_sr != TARGET_SR:
        y_resampled = librosa.resample(y_normalized, orig_sr=original_sr, target_sr=TARGET_SR)
        print(f"\n  重采样 / Resampling:")
        print(f"  • 从 {original_sr} Hz 到 {TARGET_SR} Hz")
    else:
        y_resampled = y_normalized
        print(f"\n  采样率已匹配 / Sample rate already matched")

    # Step 4: Save normalized audio
    output_filename = output_dir / f"normalized_{filename}"
    sf.write(output_filename, y_resampled, TARGET_SR)

    final_duration = librosa.get_duration(y=y_resampled, sr=TARGET_SR)
    final_rms = np.sqrt(np.mean(y_resampled**2))

    print(f"\n  保存结果 / Saved:")
    print(f"  • 文件路径 / Path: {output_filename}")
    print(f"  • 最终时长 / Final duration: {final_duration:.2f}s")
    print(f"  • 最终RMS / Final RMS: {final_rms:.6f}")

    # Log
    normalization_log.append({
        'Performer': performer_name,
        'Original SR (Hz)': original_sr,
        'Original Duration (s)': original_duration,
        'Original RMS': original_rms,
        'Start Point Detected (s)': start_sample/sr if len(frames_above) > 0 else 0,
        'Trimmed Duration (s)': trimmed_duration,
        'Trimmed RMS': trimmed_rms,
        'Scale Factor': scale_factor,
        'Final SR (Hz)': TARGET_SR,
        'Final Duration (s)': final_duration,
        'Final RMS': final_rms,
        'Output File': str(output_filename)
    })

# Save normalization log
log_df = pd.DataFrame(normalization_log)
log_df.to_csv('normalization_log.csv', index=False, encoding='utf-8-sig')

print("\n\n" + "="*90)
print("【标准化总结 / Normalization Summary】")
print("="*90)

print("\n关键参数对比 / Key Parameters Comparison:")
print("-"*90)

# Before normalization
print("\n【标准化前 / Before Normalization】")
print(f"{'演奏家':<20} {'采样率(Hz)':<15} {'时长(s)':<12} {'RMS能量':<12}")
print("-"*90)
for log in normalization_log:
    print(f"{log['Performer']:<20} {log['Original SR (Hz)']:<15} {log['Original Duration (s)']:<12.2f} {log['Original RMS']:<12.6f}")

# After normalization
print("\n【标准化后 / After Normalization】")
print(f"{'演奏家':<20} {'采样率(Hz)':<15} {'时长(s)':<12} {'RMS能量':<12}")
print("-"*90)
for log in normalization_log:
    print(f"{log['Performer']:<20} {log['Final SR (Hz)']:<15} {log['Final Duration (s)']:<12.2f} {log['Final RMS']:<12.6f}")

# Analysis
print("\n【标准化效果分析 / Normalization Effect Analysis】")
print("-"*90)

for log in normalization_log:
    rms_change = ((log['Final RMS'] - log['Original RMS']) / log['Original RMS']) * 100
    duration_change = ((log['Final Duration (s)'] - log['Original Duration (s)']) / log['Original Duration (s)']) * 100

    print(f"\n{log['Performer']}:")
    print(f"  • RMS变化 / RMS change: {rms_change:+.1f}%")
    print(f"  • 时长变化 / Duration change: {duration_change:+.1f}%")
    print(f"  • 缩放因子 / Scale factor: {log['Scale Factor']:.4f}x")
    if log['Start Point Detected (s)'] > 0.1:
        print(f"  • 去除了 {log['Start Point Detected (s)']:.2f}s 的前置沉默")

print("\n" + "="*90)
print("✓ 标准化完成 / Normalization complete!")
print("="*90)
print("\n【使用标准化音频的优势 / Benefits of Using Normalized Audio】")
print("-"*90)
print("✓ 公平对比 - 消除录音条件差异 / Fair comparison - eliminates recording condition differences")
print("✓ 采样率统一 - 便于信号处理 / Unified sample rate - facilitates signal processing")
print("✓ RMS标准化 - 力度可比 / RMS normalized - loudness is comparable")
print("✓ 时间对齐 - 音乐内容从同一点开始 / Time aligned - musical content starts at same point")

print("\n【后续分析建议 / Recommendations for Analysis】")
print("-"*90)
print("在进行任何对比分析时，请使用 'normalized_audio' 文件夹中的文件：")
print("For any comparative analysis, please use files from 'normalized_audio' directory:")
print(f"  • normalized_langlang_caiyun.wav")
print(f"  • normalized_liyundi_caiyun.wav")
print(f"  • normalized_shenwenyu_caiyun.wav")

print("\n" + "="*90)
