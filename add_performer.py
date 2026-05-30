#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add_performer.py  —  Add a new recording to the dataset
---------------------------------------------------------
Usage:
    python add_performer.py <raw_audio_file> <performer_name> [options]

Examples:
    python add_performer.py downloads/yinchengzong.wav "Yin Chengzong"
    python add_performer.py downloads/niemczuk.mp3 "Niemczuk" --color "#9B59B6" --type "Studio"

The script will:
    1. Load and resample to 22050 Hz
    2. Trim leading silence (threshold configurable)
    3. Save to normalized_audio/
    4. Append the entry to performers.json
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import argparse
import json
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path

# ── Default colour palette for new performers ─────────────────────────────────
DEFAULT_COLORS = [
    '#2ECC71',   # green
    '#9B59B6',   # purple
    '#1ABC9C',   # teal
    '#E67E22',   # dark orange
    '#34495E',   # slate
    '#F1C40F',   # yellow
    '#E91E63',   # pink
    '#00BCD4',   # cyan
]

SR = 22050

def trim_leading_silence(y, sr, silence_db=-50, min_music_rms=0.01):
    """
    Remove leading silence/noise.
    Finds first frame where RMS exceeds min_music_rms of the peak RMS.
    """
    hop   = 512
    rms   = librosa.feature.rms(y=y, hop_length=hop)[0]
    peak  = np.max(rms)
    thresh = max(peak * min_music_rms, 1e-5)
    onset_frame = np.argmax(rms > thresh)
    onset_sample = max(0, onset_frame * hop - sr // 10)   # 100ms before onset
    trimmed = y[onset_sample:]
    trimmed_sec = onset_sample / sr
    return trimmed, trimmed_sec

def main():
    parser = argparse.ArgumentParser(description='Add a new recording to the dataset')
    parser.add_argument('audio_file',   help='Path to raw audio file (WAV/MP3/M4A/etc.)')
    parser.add_argument('name',         help='Performer name (used as key in performers.json)')
    parser.add_argument('--color',      default=None, help='Hex colour for plots (e.g. #2ECC71)')
    parser.add_argument('--type',       default='Unknown', help='Recording type (Studio/Live/Home)')
    parser.add_argument('--year',       default=None, help='Year of recording')
    parser.add_argument('--notes',      default='', help='Free-text notes')
    parser.add_argument('--no-trim',    action='store_true', help='Skip silence trimming')
    args = parser.parse_args()

    base     = Path(__file__).parent
    cfg_path = base / 'performers.json'
    out_dir  = base / 'normalized_audio'
    out_dir.mkdir(exist_ok=True)

    # Load existing config
    cfg = json.loads(cfg_path.read_text(encoding='utf-8')) if cfg_path.exists() else {}

    if args.name in cfg:
        print(f"⚠️  '{args.name}' already exists in performers.json. Overwrite? [y/N] ", end='')
        if input().strip().lower() != 'y':
            print("Aborted.")
            return

    # Assign colour
    color = args.color
    if not color:
        used = {v['color'] for v in cfg.values()}
        for c in DEFAULT_COLORS:
            if c not in used:
                color = c
                break
        else:
            color = '#888888'

    # Load audio
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: file not found: {audio_path}")
        sys.exit(1)

    print(f"Loading {audio_path.name} ...")
    y, sr_orig = librosa.load(str(audio_path), sr=SR, mono=True)
    dur_orig = len(y) / SR
    print(f"  Duration: {dur_orig:.1f}s  |  SR: {SR}")

    # Trim leading silence
    trimmed_sec = 0.0
    if not args.no_trim:
        y, trimmed_sec = trim_leading_silence(y, SR)
        if trimmed_sec > 0.5:
            print(f"  Trimmed {trimmed_sec:.1f}s of leading silence")
        else:
            print(f"  No significant leading silence detected ({trimmed_sec:.2f}s)")

    dur_final = len(y) / SR
    print(f"  Final duration: {dur_final:.1f}s")

    # Safe filename
    safe_name = args.name.lower().replace(' ', '_').replace('/', '_')
    out_file  = f'normalized_{safe_name}_caiyun.wav'
    out_path  = out_dir / out_file

    sf.write(str(out_path), y, SR, subtype='PCM_16')
    print(f"  Saved: {out_path}")

    # Update performers.json
    cfg[args.name] = {
        'file':           out_file,
        'color':          color,
        'recording_type': args.type,
        'year':           args.year,
        'notes':          args.notes + (f'; {trimmed_sec:.1f}s leading silence trimmed' if trimmed_sec > 0.5 else ''),
    }
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"  Updated performers.json  ({len(cfg)} performers total)")

    print(f"\n✓  '{args.name}' added. Run the analysis scripts to include this performer.")
    print(f"   Suggested next step:")
    print(f"   python section_analysis.py")
    print(f"   python note_alignment.py")
    print(f"   python attack_analysis.py")

if __name__ == '__main__':
    main()
