"""
钢琴家风格对比分析 - 完整pipeline
ICASSP 2026 投稿项目
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 第一部分：特征提取
# ============================================================================

print("\n" + "="*70)
print("第一部分：特征提取".center(70))
print("="*70)

# 定义文件路径
base_path = Path(r"C:\Users\wenli\OneDrive\Desktop\Sound project")
results_path = base_path / "results"
results_path.mkdir(exist_ok=True)

performers = {
    "langlang_caiyun.wav": "郎朗",
    "liyundi_caiyun.wav": "李云迪",
    "shenwenyu_caiyun.wav": "沈文裕"
}

# 字典存储所有特征
audio_data = {}
features_data = {}

print("\n正在加载音频文件...")
for filename, performer_name in performers.items():
    filepath = base_path / filename
    print(f"  加载 {performer_name}...", end=" ")

    # 加载音频
    y, sr = librosa.load(filepath, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    audio_data[performer_name] = {
        'y': y,
        'sr': sr,
        'duration': duration,
        'filename': filename
    }

    print(f"✓ (采样率: {sr}Hz, 时长: {duration:.2f}s)")

print("\n提取音频特征...")

for performer_name, audio_info in audio_data.items():
    y = audio_info['y']
    sr = audio_info['sr']
    print(f"\n  {performer_name}:")

    # 1. MFCC特征 (13维)
    print(f"    - MFCC特征...", end=" ")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    print("✓")

    # 2. Chroma特征
    print(f"    - Chroma特征...", end=" ")
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)
    print("✓")

    # 3. Onset起音检测
    print(f"    - Onset检测...", end=" ")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    # 使用阈值方式检测起音点
    onset_frames = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.1, wait=10)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
    onset_count = len(onset_frames)
    print(f"✓ ({onset_count} 个起音点)")

    # 4. RMS声强
    print(f"    - RMS声强...", end=" ")
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)
    print("✓")

    # 5. 零交叉率 (额外特征，有助于分类)
    print(f"    - 零交叉率...", end=" ")
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=512)[0]
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)
    print("✓")

    # 存储特征
    features_data[performer_name] = {
        'mfcc': mfcc,
        'mfcc_mean': mfcc_mean,
        'mfcc_std': mfcc_std,
        'chroma': chroma,
        'chroma_mean': chroma_mean,
        'chroma_std': chroma_std,
        'onset_times': onset_times,
        'onset_count': onset_count,
        'rms': rms,
        'rms_mean': rms_mean,
        'rms_std': rms_std,
        'zcr': zcr,
        'zcr_mean': zcr_mean,
        'zcr_std': zcr_std
    }

print("\n✓ 特征提取完成！")

# ============================================================================
# 第二部分：DTW时间对齐
# ============================================================================

print("\n" + "="*70)
print("第二部分：DTW时间对齐".center(70))
print("="*70)

print("\n计算MFCC之间的动态时间规划(DTW)...")

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

# 提取MFCC特征（转置为帧数 x 特征数）
mfcc_langlang = features_data["郎朗"]['mfcc'].T
mfcc_liyundi = features_data["李云迪"]['mfcc'].T
mfcc_shenwenyu = features_data["沈文裕"]['mfcc'].T

# 计算DTW距离和路径
print("  计算郎朗 vs 李云迪...", end=" ")
distance_ll_ly, path_ll_ly = fastdtw(mfcc_langlang, mfcc_liyundi, dist=euclidean)
print(f"✓ (距离: {distance_ll_ly:.2f})")

print("  计算郎朗 vs 沈文裕...", end=" ")
distance_ll_sw, path_ll_sw = fastdtw(mfcc_langlang, mfcc_shenwenyu, dist=euclidean)
print(f"✓ (距离: {distance_ll_sw:.2f})")

print("  计算李云迪 vs 沈文裕...", end=" ")
distance_ly_sw, path_ly_sw = fastdtw(mfcc_liyundi, mfcc_shenwenyu, dist=euclidean)
print(f"✓ (距离: {distance_ly_sw:.2f})")

# 存储DTW结果
dtw_results = {
    '郎朗 vs 李云迪': distance_ll_ly,
    '郎朗 vs 沈文裕': distance_ll_sw,
    '李云迪 vs 沈文裕': distance_ly_sw
}

print("\n✓ DTW对齐完成！")

# ============================================================================
# 第三部分：风格差异分析
# ============================================================================

print("\n" + "="*70)
print("第三部分：风格差异分析".center(70))
print("="*70)

print("\n分析三个维度的差异...")

# 计算timing差异（基于onset）
print("  1. Timing差异（起音间隔）...", end=" ")
onset_intervals = {
    '郎朗': np.diff(features_data['郎朗']['onset_times']) if len(features_data['郎朗']['onset_times']) > 1 else np.array([0]),
    '李云迪': np.diff(features_data['李云迪']['onset_times']) if len(features_data['李云迪']['onset_times']) > 1 else np.array([0]),
    '沈文裕': np.diff(features_data['沈文裕']['onset_times']) if len(features_data['沈文裕']['onset_times']) > 1 else np.array([0])
}

timing_variance = {
    '郎朗': np.var(onset_intervals['郎朗']),
    '李云迪': np.var(onset_intervals['李云迪']),
    '沈文裕': np.var(onset_intervals['沈文裕'])
}
print("✓")

# 计算dynamics差异（基于RMS）
print("  2. Dynamics差异（声强变化）...", end=" ")
dynamics_variance = {
    '郎朗': np.var(features_data['郎朗']['rms']),
    '李云迪': np.var(features_data['李云迪']['rms']),
    '沈文裕': np.var(features_data['沈文裕']['rms'])
}
print("✓")

# 计算timbre差异（基于MFCC）
print("  3. Timbre差异（音色变化）...", end=" ")
timbre_variance = {}
for performer in ['郎朗', '李云迪', '沈文裕']:
    mfcc_var = np.mean(np.var(features_data[performer]['mfcc'], axis=1))
    timbre_variance[performer] = mfcc_var
print("✓")

# 创建差异汇总表
print("\n差异分析汇总:")
print("-" * 70)
diff_summary = pd.DataFrame({
    'Timing Variance': timing_variance,
    'Dynamics Variance': dynamics_variance,
    'Timbre Variance': timbre_variance
})
print(diff_summary)

print("\n✓ 风格差异分析完成！")

# 保存到CSV
diff_summary.to_csv(results_path / "style_differences.csv", encoding='utf-8-sig')
print(f"  已保存: results/style_differences.csv")

# ============================================================================
# 第四部分：演奏者分类器（SVM）
# ============================================================================

print("\n" + "="*70)
print("第四部分：演奏者SVM分类器".center(70))
print("="*70)

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

print("\n准备训练数据...")

# 使用帧级MFCC特征，将每个帧作为一个样本
hop_length = 512
X_all = []
y_all = []

for performer_name in ['郎朗', '李云迪', '沈文裕']:
    mfcc = features_data[performer_name]['mfcc'].T  # (n_frames, 13)
    X_all.append(mfcc)
    y_all.extend([performer_name] * len(mfcc))

X = np.vstack(X_all)
y = np.array(y_all)

print(f"  总样本数: {len(X)}")
print(f"  特征维度: {X.shape[1]}")
print(f"  样本分布:")
for performer in ['郎朗', '李云迪', '沈文裕']:
    count = np.sum(y == performer)
    print(f"    - {performer}: {count} 帧")

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print("\n训练SVM分类器...")
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)

# 预测
y_pred_train = svm_model.predict(X_train)
y_pred_test = svm_model.predict(X_test)

# 计算准确率
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"  训练集准确率: {train_accuracy:.4f}")
print(f"  测试集准确率: {test_accuracy:.4f}")

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred_test, labels=['郎朗', '李云迪', '沈文裕'])
print("\n混淆矩阵:")
cm_df = pd.DataFrame(cm,
    index=['郎朗', '李云迪', '沈文裕'],
    columns=['郎朗', '李云迪', '沈文裕']
)
print(cm_df)

# 分类报告
print("\n分类详细报告:")
print(classification_report(y_test, y_pred_test,
    target_names=['郎朗', '李云迪', '沈文裕']))

# 保存分类结果
classification_results = pd.DataFrame({
    'Metric': ['Train Accuracy', 'Test Accuracy'],
    'Score': [train_accuracy, test_accuracy]
})
classification_results.to_csv(results_path / "classification_accuracy.csv", index=False, encoding='utf-8-sig')

cm_df.to_csv(results_path / "confusion_matrix.csv", encoding='utf-8-sig')

print("\n✓ 分类器训练完成！")

# ============================================================================
# 第五部分：可视化
# ============================================================================

print("\n" + "="*70)
print("第五部分：可视化生成".center(70))
print("="*70)

# 创建图表存储目录
plots_path = results_path / "plots"
plots_path.mkdir(exist_ok=True)

print("\n1. 特征对比图表...")

# 1.1 MFCC均值对比
fig, ax = plt.subplots(figsize=(10, 6))
for performer in ['郎朗', '李云迪', '沈文裕']:
    ax.plot(features_data[performer]['mfcc_mean'], marker='o', label=performer)
ax.set_xlabel('MFCC系数索引 / MFCC Coefficient Index')
ax.set_ylabel('均值 / Mean')
ax.set_title('MFCC特征对比 / MFCC Comparison')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plots_path / "01_mfcc_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ MFCC对比图")

# 1.2 RMS包络对比
fig, axes = plt.subplots(3, 1, figsize=(12, 8))
sr = audio_data['郎朗']['sr']
hop_length = 512
for idx, performer in enumerate(['郎朗', '李云迪', '沈文裕']):
    times = librosa.frames_to_time(np.arange(len(features_data[performer]['rms'])),
                                   sr=sr, hop_length=hop_length)
    axes[idx].plot(times, features_data[performer]['rms'], linewidth=0.8)
    axes[idx].set_ylabel(performer)
    axes[idx].grid(True, alpha=0.3)
axes[-1].set_xlabel('时间 (秒) / Time (seconds)')
plt.suptitle('RMS包络对比', fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig(plots_path / "02_rms_envelope.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ RMS包络图")

# 1.3 MFCC时频表示（Spectrogram）
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
for idx, performer in enumerate(['郎朗', '李云迪', '沈文裕']):
    mfcc = features_data[performer]['mfcc']
    im = axes[idx].imshow(mfcc, aspect='auto', origin='lower', cmap='viridis')
    axes[idx].set_ylabel(performer)
    axes[idx].set_ylabel('MFCC系数')
    plt.colorbar(im, ax=axes[idx], label='幅度')
axes[-1].set_xlabel('时间帧 / Time Frame')
plt.suptitle('MFCC时频表示', fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig(plots_path / "03_mfcc_spectrogram.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ MFCC时频谱图")

# 2. 风格差异热力图
print("\n2. 风格差异热力图...")
fig, ax = plt.subplots(figsize=(8, 6))
heatmap_data = pd.DataFrame({
    'Timing': timing_variance,
    'Dynamics': dynamics_variance,
    'Timbre': timbre_variance
})
sns.heatmap(heatmap_data.T, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax, cbar_kws={'label': '方差'})
ax.set_title('风格差异热力图 / Style Difference Heatmap')
ax.set_ylabel('差异维度 / Difference Dimension')
ax.set_xlabel('演奏者 / Performer')
plt.tight_layout()
plt.savefig(plots_path / "04_style_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 风格差异热力图")

# 3. 混淆矩阵热力图
print("\n3. 分类混淆矩阵...")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': '样本数'})
ax.set_title(f'分类混淆矩阵 / Confusion Matrix (测试准确率 / Test Accuracy: {test_accuracy:.2%})')
ax.set_ylabel('真实标签 / True Label')
ax.set_xlabel('预测标签 / Predicted Label')
plt.tight_layout()
plt.savefig(plots_path / "05_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 混淆矩阵图")

# 4. DTW距离对比
print("\n4. DTW距离对比...")
fig, ax = plt.subplots(figsize=(8, 6))
dtw_names = ['郎朗 vs\n李云迪', '郎朗 vs\n沈文裕', '李云迪 vs\n沈文裕']
dtw_distances = [distance_ll_ly, distance_ll_sw, distance_ly_sw]
colors = ['#E74C3C', '#F39C12', '#3498DB']  # Red, Yellow, Blue - high contrast
bars = ax.bar(dtw_names, dtw_distances, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('DTW距离 / DTW Distance')
ax.set_title('演奏风格相似度（DTW距离）/ Performance Style Similarity (DTW Distance)\n距离越小越相似 / Smaller distance = More similar')
ax.grid(True, alpha=0.3, axis='y')
for bar, distance in zip(bars, dtw_distances):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{distance:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(plots_path / "06_dtw_distances.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ DTW距离对比图")

# 5. Chroma特征对比
print("\n5. Chroma特征对比...")
fig, axes = plt.subplots(3, 1, figsize=(12, 9))
for idx, performer in enumerate(['郎朗', '李云迪', '沈文裕']):
    chroma = features_data[performer]['chroma']
    im = axes[idx].imshow(chroma, aspect='auto', origin='lower', cmap='viridis')
    axes[idx].set_ylabel(performer)
    axes[idx].set_yticks(np.arange(12))
    axes[idx].set_yticklabels(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
    plt.colorbar(im, ax=axes[idx], label='能量')
axes[-1].set_xlabel('时间帧 / Time Frame')
plt.suptitle('Chroma特征分析', fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig(plots_path / "07_chroma_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Chroma特征图")

print("\n✓ 所有可视化生成完成！")

# ============================================================================
# 第六部分：结果汇总
# ============================================================================

print("\n" + "="*70)
print("第六部分：结果汇总".center(70))
print("="*70)

# 创建综合报告
report = []
report.append("=" * 70)
report.append("钢琴家演奏风格对比分析 - ICASSP 2026")
report.append("=" * 70)
report.append("")

report.append("1. 音频基本信息")
report.append("-" * 70)
for performer, info in audio_data.items():
    report.append(f"{performer}:")
    report.append(f"  采样率: {info['sr']} Hz")
    report.append(f"  时长: {info['duration']:.2f} 秒")
    report.append(f"  文件: {info['filename']}")
report.append("")

report.append("2. 特征统计")
report.append("-" * 70)
for performer in ['郎朗', '李云迪', '沈文裕']:
    report.append(f"\n{performer}:")
    report.append(f"  MFCC均值: {features_data[performer]['mfcc_mean']}")
    report.append(f"  MFCC标准差: {features_data[performer]['mfcc_std']}")
    report.append(f"  起音点数量: {features_data[performer]['onset_count']}")
    report.append(f"  RMS均值: {features_data[performer]['rms_mean']:.4f}")
    report.append(f"  RMS标准差: {features_data[performer]['rms_std']:.4f}")
report.append("")

report.append("3. DTW相似度分析")
report.append("-" * 70)
for pair, distance in dtw_results.items():
    report.append(f"{pair}: {distance:.2f}")
report.append("")

report.append("4. 风格差异分析")
report.append("-" * 70)
report.append("Timing差异（起音间隔方差）:")
for performer, var in timing_variance.items():
    report.append(f"  {performer}: {var:.6f}")
report.append("\nDynamics差异（声强方差）:")
for performer, var in dynamics_variance.items():
    report.append(f"  {performer}: {var:.6f}")
report.append("\nTimbre差异（音色方差）:")
for performer, var in timbre_variance.items():
    report.append(f"  {performer}: {var:.6f}")
report.append("")

report.append("5. 分类器性能")
report.append("-" * 70)
report.append(f"训练集准确率: {train_accuracy:.4f}")
report.append(f"测试集准确率: {test_accuracy:.4f}")
report.append(f"总样本数: {len(X)}")
report.append("")

report.append("6. 关键发现")
report.append("-" * 70)

# 分析关键发现
print("\n" + "="*70)
print("关键发现分析".center(70))
print("="*70)

findings = []

# 发现1：相似度排序
sorted_dtw = sorted(dtw_results.items(), key=lambda x: x[1])
print("\n【发现1】演奏风格相似度排序（基于DTW）:")
for i, (pair, distance) in enumerate(sorted_dtw, 1):
    print(f"  {i}. {pair}: {distance:.2f}")
    findings.append(f"演奏风格相似度排序#{i}: {pair} (DTW距离={distance:.2f})")

# 发现2：风格特征差异
print("\n【发现2】三个维度的风格差异排序:")
print("  Timing (起音规律性):")
timing_sorted = sorted(timing_variance.items(), key=lambda x: x[1], reverse=True)
for i, (performer, var) in enumerate(timing_sorted, 1):
    print(f"    {i}. {performer}: {var:.6f}")

print("  Dynamics (动态范围):")
dynamics_sorted = sorted(dynamics_variance.items(), key=lambda x: x[1], reverse=True)
for i, (performer, var) in enumerate(dynamics_sorted, 1):
    print(f"    {i}. {performer}: {var:.6f}")

print("  Timbre (音色):")
timbre_sorted = sorted(timbre_variance.items(), key=lambda x: x[1], reverse=True)
for i, (performer, var) in enumerate(timbre_sorted, 1):
    print(f"    {i}. {performer}: {var:.6f}")

# 发现3：分类准确率
print(f"\n【发现3】SVM分类器性能:")
print(f"  测试集准确率: {test_accuracy:.2%}")
print(f"  模型可以有效区分三位演奏家的演奏风格")

# 构建最重要的三个发现
key_findings = []

# 发现1：最相似的两个演奏家
most_similar = sorted_dtw[0]
key_findings.append(f"最相似的演奏风格：{most_similar[0]} (DTW距离={most_similar[1]:.2f})")

# 发现2：最独特的演奏家（以Timing特征为例）
most_unique_timing = timing_sorted[0]
key_findings.append(f"Timing最独特的演奏家：{most_unique_timing[0]} (方差={most_unique_timing[1]:.6f}，起音节奏更不规律)")

# 发现3：分类结果
key_findings.append(f"机器学习分类准确度：{test_accuracy:.2%} (SVM可有效区分三位演奏家)")

report.append("\n")
for i, finding in enumerate(key_findings, 1):
    report.append(f"{i}. {finding}")

# 保存报告
report_text = "\n".join(report)
with open(results_path / "analysis_report.txt", 'w', encoding='utf-8') as f:
    f.write(report_text)

print("\n" + "="*70)
print("📋 最重要的三个发现".center(70))
print("="*70)
for i, finding in enumerate(key_findings, 1):
    print(f"\n【发现{i}】{finding}")

print("\n" + "="*70)
print("所有结果已保存到 results/ 文件夹".center(70))
print("="*70)

print("\n📁 输出文件清单:")
print("  数据文件:")
print("    - style_differences.csv (风格差异数据)")
print("    - classification_accuracy.csv (分类准确率)")
print("    - confusion_matrix.csv (混淆矩阵)")
print("  图表文件:")
print("    - 01_mfcc_comparison.png")
print("    - 02_rms_envelope.png")
print("    - 03_mfcc_spectrogram.png")
print("    - 04_style_heatmap.png")
print("    - 05_confusion_matrix.png")
print("    - 06_dtw_distances.png")
print("    - 07_chroma_analysis.png")
print("  报告文件:")
print("    - analysis_report.txt (完整分析报告)")

print("\n✓ Pipeline执行完成！")
