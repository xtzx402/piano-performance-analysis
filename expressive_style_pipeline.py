#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expressive Performance Style Analysis Pipeline
分析演奏家相对于"标准"演奏的表现性风格偏差
维度：Timing, Dynamics, Articulation, Vibrato, Tone Color, Attack, Sustain, Rubato, Agogic Accent
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import signal
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("\n" + "="*90)
print("EXPRESSIVE PERFORMANCE STYLE ANALYSIS".center(90))
print("分析演奏家的表现性风格特征维度".center(90))
print("="*90)

base_path = Path(__file__).parent
results_path = base_path / "results_expressive_style"
results_path.mkdir(exist_ok=True)
plots_path = results_path / "plots"
plots_path.mkdir(exist_ok=True)

performers = {
    "langlang_caiyun.wav": "郎朗 (Lang Lang)",
    "liyundi_caiyun.wav": "李云迪 (Li Yundi)",
    "shenwenyu_caiyun.wav": "沈文裕 (Shen Wenyu)"
}

def analyze_timing_deviations(y, sr):
    """分析Timing（节奏）特征"""
    hop_length = 512

    # Onset detection - 检测每个音的开始
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_frames = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.1, wait=10)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    # 计算相邻音符间隔（Inter-onset interval）
    if len(onset_times) > 1:
        ioi = np.diff(onset_times)  # Inter-Onset Interval
        ioi_mean = np.mean(ioi)
        ioi_std = np.std(ioi)
        ioi_cv = ioi_std / ioi_mean  # Coefficient of Variation
        tempo = 60 / ioi_mean  # 推算的Tempo (beats per minute)
    else:
        ioi = np.array([0])
        ioi_mean = 0
        ioi_std = 0
        ioi_cv = 0
        tempo = 0

    return {
        'onset_times': onset_times,
        'ioi': ioi,
        'ioi_mean': ioi_mean,
        'ioi_std': ioi_std,
        'ioi_cv': ioi_cv,  # Timing variability
        'tempo': tempo,
        'onset_count': len(onset_frames)
    }

def analyze_dynamics(y, sr):
    """分析Dynamics（动态强度）特征"""
    hop_length = 512

    # RMS能量
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)
    rms_max = np.max(rms)
    rms_min = np.min(rms)

    # 动态范围
    dynamic_range = 20 * np.log10(rms_max / (rms_min + 1e-10)) if rms_min > 0 else 0

    # Loudness variation (标准化后的动态变化)
    rms_normalized = (rms - rms_min) / (rms_max - rms_min + 1e-10)
    loudness_variation = np.std(rms_normalized)

    return {
        'rms': rms,
        'rms_mean': rms_mean,
        'rms_std': rms_std,
        'rms_max': rms_max,
        'rms_min': rms_min,
        'dynamic_range': dynamic_range,  # dB
        'loudness_variation': loudness_variation  # 动态变化程度 (0-1)
    }

def analyze_articulation(y, sr):
    """分析Articulation（音符连接方式）特征"""
    hop_length = 512
    n_fft = 2048

    # 计算频谱通量（Spectral Flux）- 衡量音符之间的连续性
    S = librosa.magphase(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))[0]
    spectral_flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))

    spectral_flux_mean = np.mean(spectral_flux)
    spectral_flux_std = np.std(spectral_flux)

    # 高通量 -> 快速音色变化 -> staccato倾向
    # 低通量 -> 平滑音色变化 -> legato倾向
    staccato_tendency = spectral_flux_mean  # 高值=staccato, 低值=legato

    return {
        'spectral_flux': spectral_flux,
        'spectral_flux_mean': spectral_flux_mean,
        'spectral_flux_std': spectral_flux_std,
        'staccato_tendency': staccato_tendency,  # 0-1 scale (estimated)
        'articulation_clarity': spectral_flux_std / (spectral_flux_mean + 1e-10)  # 清晰度
    }

def analyze_vibrato(y, sr):
    """Analyze vibrato as amplitude modulation (correct for piano — piano has no pitch vibrato)"""
    hop_length = 512

    # Piano vibrato manifests as amplitude modulation, not pitch/frequency variation.
    # Use RMS envelope and detect periodic fluctuations via autocorrelation.
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    rms_norm = (rms - np.mean(rms)) / (np.std(rms) + 1e-10)
    autocorr = np.correlate(rms_norm, rms_norm, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]
    autocorr = autocorr / (autocorr[0] + 1e-10)

    # Look for periodicity in 3–8 Hz range (typical vibrato rate)
    frame_rate = sr / hop_length
    min_lag = max(1, int(frame_rate / 8))   # 8 Hz upper bound
    max_lag = int(frame_rate / 3)            # 3 Hz lower bound

    if max_lag > min_lag and max_lag < len(autocorr):
        peak_lag = np.argmax(autocorr[min_lag:max_lag]) + min_lag
        vibrato_rate = frame_rate / peak_lag          # Hz
        vibrato_depth = float(autocorr[peak_lag])     # Correlation strength (0–1)
    else:
        vibrato_rate = 0.0
        vibrato_depth = 0.0

    # Overall dynamic modulation (coefficient of variation of RMS)
    rms_variation = np.std(rms) / (np.mean(rms) + 1e-10)

    return {
        'rms_envelope': rms,
        'vibrato_depth': vibrato_depth,      # Amplitude modulation depth (0–1)
        'vibrato_rate': vibrato_rate,         # Estimated periodicity (Hz)
        'vibrato_prevalence': rms_variation   # Overall dynamic modulation (CV)
    }

def analyze_tone_color(y, sr):
    """分析Tone Color（音色）特征"""
    hop_length = 512
    n_fft = 2048

    # Spectral Centroid - 亮度
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]

    # Spectral Contrast - 音色特性
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)

    # Timbre Balance - MFCC的第2系数（代表亮度）
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)

    return {
        'spectral_centroid_mean': np.mean(spectral_centroid),
        'spectral_centroid_std': np.std(spectral_centroid),
        'tone_brightness': np.mean(spectral_centroid),  # Hz，高=亮
        'spectral_contrast_mean': np.mean(spectral_contrast),
        'tone_darkness': np.mean(mfcc[1]),  # MFCC1 代表音色亮度
        'tone_richness': np.mean(spectral_contrast)  # 音色丰富度
    }

def analyze_attack(y, sr):
    """分析Attack（音符起始速率）特征"""
    hop_length = 512

    # Onset strength envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_frames = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.1, wait=10)

    # 分析每个onset周围的能量上升速率
    attack_rates = []
    for frame in onset_frames:
        start = max(0, frame - 10)
        end = min(len(onset_env), frame + 10)

        if end - start > 1:
            env_slice = onset_env[start:end]
            max_idx = np.argmax(env_slice)

            if max_idx > 0:
                attack_rate = (env_slice[max_idx] - env_slice[0]) / max_idx
                attack_rates.append(attack_rate)

    if attack_rates:
        attack_mean = np.mean(attack_rates)
        attack_std = np.std(attack_rates)
        attack_sharpness = attack_mean  # 高值=快速attack=sharp
    else:
        attack_mean = 0
        attack_std = 0
        attack_sharpness = 0

    return {
        'attack_rates': np.array(attack_rates) if attack_rates else np.array([0]),
        'attack_mean': attack_mean,
        'attack_std': attack_std,
        'attack_sharpness': attack_sharpness,  # 0-1 scale
        'attack_consistency': 1 - (attack_std / (attack_mean + 1e-10))  # 一致性
    }

def analyze_sustain(y, sr):
    """分析Sustain（音符持续时间）特征"""
    hop_length = 512

    # RMS envelope - 衡量音符的衰减
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    # Onset detection
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_frames = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.1, wait=10)

    sustain_times = []
    for i, frame in enumerate(onset_frames[:-1]):
        next_frame = onset_frames[i + 1]

        # 从onset到下一个onset的时间
        sustain_frames = next_frame - frame
        sustain_time = sustain_frames * hop_length / sr

        # 衰减率
        rms_slice = rms[frame:next_frame]
        if len(rms_slice) > 1:
            decay_rate = (rms_slice[0] - rms_slice[-1]) / (rms_slice[0] + 1e-10)
            sustain_times.append({
                'sustain_time': sustain_time,
                'decay_rate': decay_rate
            })

    if sustain_times:
        sustain_mean = np.mean([s['sustain_time'] for s in sustain_times])
        sustain_std = np.std([s['sustain_time'] for s in sustain_times])
        decay_mean = np.mean([s['decay_rate'] for s in sustain_times])
    else:
        sustain_mean = 0
        sustain_std = 0
        decay_mean = 0

    return {
        'sustain_mean': sustain_mean,  # 秒
        'sustain_std': sustain_std,
        'sustain_consistency': 1 - (sustain_std / (sustain_mean + 1e-10)) if sustain_mean > 0 else 0,
        'decay_rate': decay_mean,  # 衰减速率（0-1）
        'sustain_length': sustain_mean  # 音符平均持续时间
    }

def analyze_rubato(y, sr):
    """分析Rubato（节奏自由处理）特征"""
    hop_length = 512

    # Tempo estimation
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_frames = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.1, wait=10)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    if len(onset_times) > 1:
        ioi = np.diff(onset_times)
        ioi_std = np.std(ioi)
        ioi_mean = np.mean(ioi)

        # Rubato = 相对的tempo变化
        rubato_coefficient = ioi_std / ioi_mean  # 高值=high rubato
    else:
        rubato_coefficient = 0

    return {
        'rubato_coefficient': rubato_coefficient,  # Rubato程度 (0=strict, 高=flexible)
        'tempo_flexibility': rubato_coefficient,
        'timing_elasticity': np.std(np.diff(ioi)) / (np.mean(np.diff(ioi)) + 1e-10) if len(onset_times) > 2 else 0
    }

def analyze_agogic_accent(y, sr):
    """分析Agogic Accent（时间重音）特征"""
    hop_length = 512

    # 通过timing的变化来检测agogic accent
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_frames = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.1, wait=10)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    if len(onset_times) > 2:
        ioi = np.diff(onset_times)
        ioi_ratio = ioi / np.mean(ioi)

        # Agogic accents: IOI significantly above local spread (IQR-based, not fixed 110%)
        q75 = np.percentile(ioi_ratio, 75)
        q25 = np.percentile(ioi_ratio, 25)
        accent_threshold = q75 + 0.5 * (q75 - q25)  # Q3 + 0.5*IQR
        accents = np.where(ioi_ratio > accent_threshold)[0]
        accent_frequency = len(accents) / len(ioi) if len(ioi) > 0 else 0
        accent_magnitude = np.mean(ioi_ratio[accents]) if len(accents) > 0 else 1
    else:
        accent_frequency = 0
        accent_magnitude = 0

    return {
        'agogic_accent_frequency': accent_frequency,  # 时间重音的频率
        'agogic_accent_magnitude': accent_magnitude,  # 时间重音的幅度
        'expressive_timing_deviations': accent_frequency * accent_magnitude
    }

# ============================================================================
print("\n[STEP 1] MULTI-DIMENSIONAL EXPRESSIVE FEATURE EXTRACTION".center(90))
print("="*90)

print("\nLoading audio and analyzing 9 expressive dimensions...")

all_expressive_features = {}
audio_info = {}

for filename, performer_name in performers.items():
    filepath = base_path / filename
    print(f"\n  {performer_name}:")
    print(f"    Loading...", end=" ")

    y, sr = librosa.load(filepath, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    audio_info[performer_name] = {'sr': sr, 'duration': duration}
    print("OK")

    # 提取9个维度
    features = {}

    print(f"    Analyzing Timing...", end=" ")
    features['Timing'] = analyze_timing_deviations(y, sr)
    print("OK")

    print(f"    Analyzing Dynamics...", end=" ")
    features['Dynamics'] = analyze_dynamics(y, sr)
    print("OK")

    print(f"    Analyzing Articulation...", end=" ")
    features['Articulation'] = analyze_articulation(y, sr)
    print("OK")

    print(f"    Analyzing Vibrato...", end=" ")
    features['Vibrato'] = analyze_vibrato(y, sr)
    print("OK")

    print(f"    Analyzing Tone Color...", end=" ")
    features['Tone Color'] = analyze_tone_color(y, sr)
    print("OK")

    print(f"    Analyzing Attack...", end=" ")
    features['Attack'] = analyze_attack(y, sr)
    print("OK")

    print(f"    Analyzing Sustain...", end=" ")
    features['Sustain'] = analyze_sustain(y, sr)
    print("OK")

    print(f"    Analyzing Rubato...", end=" ")
    features['Rubato'] = analyze_rubato(y, sr)
    print("OK")

    print(f"    Analyzing Agogic Accent...", end=" ")
    features['Agogic Accent'] = analyze_agogic_accent(y, sr)
    print("OK")

    all_expressive_features[performer_name] = features

print("\n✓ Feature extraction completed!")

# ============================================================================
print("\n[STEP 2] EXPRESSIVE STYLE PROFILE COMPARISON".center(90))
print("="*90)

# 创建综合对比表
style_comparison = []

for performer_name in performers.values():
    features = all_expressive_features[performer_name]

    style_comparison.append({
        'Performer': performer_name,

        # Timing Dimension
        'Tempo (BPM)': features['Timing']['tempo'],
        'Timing Variability (CV)': features['Timing']['ioi_cv'],

        # Dynamics Dimension
        'Dynamic Range (dB)': features['Dynamics']['dynamic_range'],
        'Loudness Variation': features['Dynamics']['loudness_variation'],

        # Articulation Dimension
        'Staccato Tendency': features['Articulation']['staccato_tendency'],
        'Articulation Clarity': features['Articulation']['articulation_clarity'],

        # Vibrato Dimension
        'Vibrato Depth (Hz)': features['Vibrato']['vibrato_depth'],
        'Vibrato Prevalence': features['Vibrato']['vibrato_prevalence'],

        # Tone Color Dimension
        'Tone Brightness (Hz)': features['Tone Color']['tone_brightness'],
        'Tone Richness': features['Tone Color']['tone_richness'],

        # Attack Dimension
        'Attack Sharpness': features['Attack']['attack_sharpness'],
        'Attack Consistency': features['Attack']['attack_consistency'],

        # Sustain Dimension
        'Sustain Length (s)': features['Sustain']['sustain_mean'],
        'Sustain Consistency': features['Sustain']['sustain_consistency'],

        # Rubato Dimension
        'Rubato Coefficient': features['Rubato']['rubato_coefficient'],

        # Agogic Accent
        'Agogic Accent Frequency': features['Agogic Accent']['agogic_accent_frequency']
    })

style_df = pd.DataFrame(style_comparison)

print("\n9维度表现性风格对比：")
print("="*90)
print(style_df.to_string(index=False))

# 保存
style_df.to_csv(results_path / "expressive_style_9dimensions.csv", index=False, encoding='utf-8-sig')

# ============================================================================
print("\n[STEP 3] STYLE INTERPRETATION & NARRATIVE".center(90))
print("="*90)

print("\n【演奏家风格特征总结】\n")

for performer_name in performers.values():
    features = all_expressive_features[performer_name]
    profile_data = style_df[style_df['Performer'] == performer_name].iloc[0]

    print(f"\n{performer_name}：")
    print("-" * 80)

    # Timing
    print(f"\n  [Timing维度 - 节奏感]")
    print(f"    Tempo: {features['Timing']['tempo']:.1f} BPM")
    print(f"    Timing Variability: {features['Timing']['ioi_cv']:.4f}", end="")
    if features['Timing']['ioi_cv'] > 0.15:
        print(" → 高度灵活的节奏处理 (High Rubato)")
    elif features['Timing']['ioi_cv'] > 0.08:
        print(" → 适度的节奏表现性 (Moderate Rubato)")
    else:
        print(" → 严格的节奏控制 (Strict Timing)")

    # Dynamics
    print(f"\n  [Dynamics维度 - 力度感]")
    print(f"    Dynamic Range: {features['Dynamics']['dynamic_range']:.2f} dB", end="")
    if features['Dynamics']['dynamic_range'] > 50:
        print(" → 宽广的力度范围")
    else:
        print(" → 相对均匀的力度")

    print(f"    Loudness Variation: {features['Dynamics']['loudness_variation']:.4f}", end="")
    if features['Dynamics']['loudness_variation'] > 0.2:
        print(" → 大幅度的力度变化 (Expressive)")
    else:
        print(" → 相对稳定的力度")

    # Articulation
    print(f"\n  [Articulation维度 - 音符连接]")
    print(f"    Staccato Tendency: {features['Articulation']['staccato_tendency']:.4f}", end="")
    if features['Articulation']['staccato_tendency'] > np.mean(style_df['Staccato Tendency']):
        print(" → 倾向分离的音符 (Detached)")
    else:
        print(" → 倾向连贯的音符 (Legato)")

    # Vibrato
    print(f"\n  [Vibrato维度 - 颤音]")
    print(f"    Vibrato Depth: {features['Vibrato']['vibrato_depth']:.2f} Hz")
    print(f"    Vibrato Prevalence: {features['Vibrato']['vibrato_prevalence']:.4f}", end="")
    if features['Vibrato']['vibrato_prevalence'] > np.mean(style_df['Vibrato Prevalence']):
        print(" → 明显的颤音使用")
    else:
        print(" → 适度的颤音使用")

    # Tone Color
    print(f"\n  [Tone Color维度 - 音色]")
    print(f"    Tone Brightness: {features['Tone Color']['tone_brightness']:.0f} Hz", end="")
    if features['Tone Color']['tone_brightness'] > np.mean(style_df['Tone Brightness (Hz)']):
        print(" → 明亮的音色 (Bright)")
    else:
        print(" → 深暗的音色 (Dark)")

    print(f"    Tone Richness: {features['Tone Color']['tone_richness']:.4f}", end="")
    if features['Tone Color']['tone_richness'] > np.mean(style_df['Tone Richness']):
        print(" → 丰富的音色层次")
    else:
        print(" → 简洁的音色呈现")

    # Attack
    print(f"\n  [Attack维度 - 音符起始]")
    print(f"    Attack Sharpness: {features['Attack']['attack_sharpness']:.4f}", end="")
    if features['Attack']['attack_sharpness'] > np.mean(style_df['Attack Sharpness']):
        print(" → 清晰有力的起音 (Sharp)")
    else:
        print(" → 柔和的起音 (Soft)")

    print(f"    Attack Consistency: {features['Attack']['attack_consistency']:.4f}")

    # Sustain
    print(f"\n  [Sustain维度 - 音符持续]")
    print(f"    Sustain Length: {features['Sustain']['sustain_mean']:.3f} 秒")
    print(f"    Sustain Consistency: {features['Sustain']['sustain_consistency']:.4f}", end="")
    if features['Sustain']['sustain_consistency'] > 0.7:
        print(" → 一致的音符长度")
    else:
        print(" → 多样化的音符长度")

    # Rubato
    print(f"\n  [Rubato维度 - 节奏自由度]")
    print(f"    Rubato Coefficient: {features['Rubato']['rubato_coefficient']:.4f}", end="")
    if features['Rubato']['rubato_coefficient'] > 0.1:
        print(" → 高度艺术化的节奏处理")
    elif features['Rubato']['rubato_coefficient'] > 0.05:
        print(" → 适度的艺术化处理")
    else:
        print(" → 严谨的节奏遵守")

    # Agogic Accent
    print(f"\n  [Agogic Accent维度 - 时间重音]")
    print(f"    Agogic Accent Frequency: {features['Agogic Accent']['agogic_accent_frequency']:.4f}")

# ============================================================================
print("\n[STEP 4] VISUALIZATION".center(90))
print("="*90)

print("\n  Generating 9-dimension radar chart...", end=" ")

fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(projection='polar'))

dimensions = [
    'Timing Variability (CV)',
    'Loudness Variation',
    'Staccato Tendency',
    'Vibrato Prevalence',
    'Tone Brightness (Hz)',
    'Attack Sharpness',
    'Sustain Consistency',
    'Rubato Coefficient',
    'Agogic Accent Frequency'
]

# 标准化数据（0-1）
for col in dimensions:
    max_val = style_df[col].max()
    min_val = style_df[col].min()
    if max_val > min_val:
        style_df[col + '_norm'] = (style_df[col] - min_val) / (max_val - min_val)
    else:
        style_df[col + '_norm'] = 0.5

for idx, performer in enumerate(performers.values()):
    ax = axes[idx]
    perf_data = style_df[style_df['Performer'] == performer].iloc[0]

    values = [perf_data[col + '_norm'] for col in dimensions]
    values += values[:1]

    angles = np.linspace(0, 2*np.pi, len(dimensions), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    ax.plot(angles, values, 'o-', linewidth=2, markersize=6, label=performer)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        ['节奏\nTiming', '力度\nDynamics', '音符\nArticulation', '颤音\nVibrato', '音色\nTone',
         '起音\nAttack', '持续\nSustain', '速度\nRubato', '重音\nAgogic'],
        size=8
    )
    ax.set_ylim(0, 1)
    ax.set_title(performer, fontsize=12, fontweight='bold', pad=20)
    ax.grid(True)

plt.tight_layout()
plt.savefig(plots_path / "09_expressive_style_radar.png", dpi=150, bbox_inches='tight')
plt.close()
print("OK")

# 并列对比图
print("  Generating dimension comparison chart...", end=" ")

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()

dimension_cn_mapping = {
    'Timing Variability (CV)': '节奏灵活度',
    'Loudness Variation': '力度变化',
    'Staccato Tendency': '断奏倾向',
    'Vibrato Prevalence': '颤音频率',
    'Tone Brightness (Hz)': '音色明亮度',
    'Attack Sharpness': '起音清晰',
    'Sustain Consistency': '音符持续',
    'Rubato Coefficient': '速度自由度',
    'Agogic Accent Frequency': '时间重音'
}

for idx, dimension in enumerate(dimensions):
    ax = axes[idx]

    data = style_df[['Performer', dimension]].copy()
    colors = ['#E74C3C', '#F39C12', '#3498DB']  # Red, Yellow, Blue - high contrast

    bars = ax.bar(data['Performer'], data[dimension], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    cn_title = dimension_cn_mapping.get(dimension, dimension)
    ax.set_title(f'{cn_title}\n{dimension}', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlabel('Performers / 演奏家', fontsize=9)
    ax.set_ylabel('Value / 数值', fontsize=9)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(plots_path / "10_dimensions_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("OK")

# ============================================================================
print("\n[STEP 5] COMPREHENSIVE STYLE REPORT".center(90))
print("="*90)

report = []
report.append("\n" + "="*90)
report.append("EXPRESSIVE PERFORMANCE STYLE ANALYSIS - COMPREHENSIVE REPORT")
report.append("="*90)

report.append("\n【Nine Dimensional Expressive Analysis】")
report.append("-"*90)

report.append("\nFramework: Each dimension represents a distinct aspect of musical expression:")
report.append("""
1. TIMING (节奏感) - How flexible is the performer with tempo?
2. DYNAMICS (力度感) - How varied is the volume/intensity?
3. ARTICULATION (音符连接) - Legato vs Staccato tendencies?
4. VIBRATO (颤音) - How much pitch modulation is used?
5. TONE COLOR (音色) - Brightness and richness of tone?
6. ATTACK (音符起始) - How sharp/clear are note onsets?
7. SUSTAIN (音符持续) - How long are notes held?
8. RUBATO (节奏自由) - How much artistic freedom in timing?
9. AGOGIC ACCENT (时间重音) - Expressive timing deviations?
""")

report.append("\n【Key Findings】")
report.append("-"*90)

# 找出最突出的特征
for performer in performers.values():
    perf_data = style_df[style_df['Performer'] == performer].iloc[0]
    report.append(f"\n{performer}:")

    # 找出最高和最低的维度
    values_dict = {}
    for dim in dimensions:
        norm_col = dim + '_norm'
        if norm_col in perf_data.index:
            values_dict[dim] = perf_data[norm_col]

    if values_dict:
        max_dim = max(values_dict, key=values_dict.get)
        min_dim = min(values_dict, key=values_dict.get)

        report.append(f"  Strongest Characteristic: {max_dim} ({perf_data[max_dim]:.4f})")
        report.append(f"  Weakest Characteristic: {min_dim} ({perf_data[min_dim]:.4f})")

report.append("\n" + "="*90)

report_text = "\n".join(report)

with open(results_path / "EXPRESSIVE_STYLE_REPORT.txt", 'w', encoding='utf-8') as f:
    f.write(report_text)

print(report_text)

# ============================================================================
print("\n" + "="*90)
print("EXPRESSIVE STYLE ANALYSIS COMPLETE".center(90))
print("="*90)
print(f"\nAll results saved to: {results_path}")
print("\nOutput files:")
print("  - expressive_style_9dimensions.csv")
print("  - 09_expressive_style_radar.png")
print("  - 10_dimensions_comparison.png")
print("  - EXPRESSIVE_STYLE_REPORT.txt")
