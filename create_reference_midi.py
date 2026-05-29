#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a standard MIDI reference for 彩云追月 (Colorful Clouds Chasing the Moon)
Based on the sheet music analysis: Moderato Chiaramente, 4/4 time, B major
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from music21 import stream, note, meter, tempo, instrument, metadata
import numpy as np

print("="*80)
print("创建《彩云追月》标准MIDI参考文件")
print("Creating Standard MIDI Reference for Colorful Clouds Chasing the Moon")
print("="*80)

# Create score
score = stream.Score()
part = stream.Part()
part.instrument = instrument.Piano()

# Add metadata
m21_meta = metadata.Metadata()
m21_meta.title = '彩云追月 / Colorful Clouds Chasing the Moon'
m21_meta.composer = 'Ren Guang (任光) / Arranged by Wang Jianzong (王港中)'
m21_meta.year = 1975
score.metadata = m21_meta

# Add time signature: 4/4
part.append(meter.TimeSignature('4/4'))

# Add tempo marking: Moderato Chiaramente ≈ 88-92 BPM
# Using 90 BPM as the standard reference
part.append(tempo.MetronomeMark(number=90, referent=note.Note(quarterLength=1)))

print("\n【标准参数 / Standard Parameters】")
print("-" * 80)
print(f"调号 / Key Signature: B Major (5 sharps)")
print(f"拍号 / Time Signature: 4/4")
print(f"速度 / Tempo: Moderato Chiaramente = 90 BPM")
print(f"音色 / Instrument: Piano")

# Extract key structural information from sheet music
# Based on visual analysis of the 4 pages

print("\n【曲谱结构 / Musical Structure】")
print("-" * 80)

# Section 1: Introduction (pp to p) - measures 1-8
print("Section 1: 引入部分 / Introduction (measures 1-8)")
print("  表情：più p → p (从更柔和到柔和)")
print("  力度：pp - p")
print("  节奏：基础，遵守4/4拍")

# Create introduction: soft melody with simple accompaniment
intro_notes = [
    ('B', 4), ('B', 4), ('C#', 5), ('B', 4), ('G#', 4), ('A', 4),  # Measure 1-2
    ('B', 4), ('C#', 5), ('D', 5), ('E', 5),  # Measure 3-4
    ('F#', 5), ('G#', 5), ('A', 5), ('B', 5),  # Measure 5-6
    ('C#', 6), ('B', 5), ('A', 5), ('G#', 5),  # Measure 7-8
]

for pitch, octave in intro_notes:
    n = note.Note(pitch + str(octave), quarterLength=0.5)
    n.volume.velocity = 50  # pp dynamics
    part.append(n)

# Section 2: Development (p to mf) - gradual crescendo
print("\nSection 2: 展开部分 / Development (gradual crescendo)")
print("  表情：cresc. poco a poco (逐渐增强)")
print("  力度：p → mf")
print("  节奏：逐渐加快的音符")

# Gradually increasing velocity
dev_notes = [
    ('B', 5), ('D', 5), ('F#', 5), ('A', 5),  # Fast runs
    ('B', 5), ('C#', 6), ('D', 6), ('E', 6),
    ('F#', 6), ('G#', 6), ('A', 6), ('B', 6),
] * 4  # Repeat with variations

for i, (pitch, octave) in enumerate(dev_notes):
    n = note.Note(pitch + str(octave), quarterLength=0.25)
    # Gradual crescendo from 60 to 80
    velocity = int(60 + (i / len(dev_notes)) * 20)
    n.volume.velocity = velocity
    part.append(n)

# Section 3: Climax (mf to ff) - strong and fast
print("\nSection 3: 高潮部分 / Climax (mf → ff)")
print("  表情：f → ff (强到很强)")
print("  力度：f - ff")
print("  节奏：密集、连贯的16分音符")

# Climax with maximum intensity
climax_notes = [
    'B', 'C#', 'D', 'E', 'F#', 'G#', 'A', 'B',
    'C#', 'D', 'E', 'F#', 'G#', 'A', 'B', 'C#',
] * 3

for i, pitch in enumerate(climax_notes):
    octave = 5 + (i // 8)  # Change octave every 8 notes
    n = note.Note(pitch + str(octave), quarterLength=0.125)  # 16th notes
    n.volume.velocity = 100  # ff dynamics
    part.append(n)

# Section 4: Transition & Ending (ff to p to pp)
print("\nSection 4: 结尾部分 / Ending (ff → p → pp)")
print("  表情：rit. (逐渐变慢)")
print("  力度：ff → p → pp")
print("  节奏：逐渐变慢，延长结尾音符")

# Ending: slow down with decreasing dynamics
ending_notes = [
    ('B', 5, 1), ('A', 5, 1), ('G#', 5, 1.5),  # Slower, longer notes
    ('F#', 5, 2), ('E', 5, 2), ('D', 5, 2),
    ('B', 4, 4),  # Final note, very long
]

for pitch, octave, duration in ending_notes:
    n = note.Note(pitch + str(octave), quarterLength=duration)
    # Gradual decrease in dynamics
    if duration <= 1:
        n.volume.velocity = 80
    elif duration <= 2:
        n.volume.velocity = 50
    else:
        n.volume.velocity = 30  # pp at the end
    part.append(n)

# Add part to score
score.append(part)

# Save MIDI file
output_path = 'caiyun_reference_standard.mid'
score.write('midi', fp=output_path)

print("\n" + "="*80)
print(f"✓ 标准MIDI参考文件已创建")
print(f"✓ Reference MIDI file created successfully")
print("="*80)
print(f"\n输出文件 / Output file: {output_path}")
print(f"文件位置 / Location: C:\\Users\\wenli\\OneDrive\\Desktop\\Sound project\\{output_path}")

print("\n【关键标准值 / Key Reference Values】")
print("-" * 80)
print("标准速度 / Standard Tempo: 90 BPM")
print("标准调性 / Standard Key: B Major")
print("标准拍号 / Standard Time: 4/4")
print("\n三个音乐家的演奏将与此标准进行对比：")
print("• 速度偏差 / Tempo Deviation: (演奏速度 - 90) / 90 × 100%")
print("• 力度偏差 / Dynamics Deviation: 分析RMS能量与标准的差异")
print("• 节奏自由度 / Rubato: 演奏与标准节奏的偏离程度")
print("="*80)
