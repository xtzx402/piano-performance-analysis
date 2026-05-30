#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py  —  Shared configuration for all analysis scripts
------------------------------------------------------------
All scripts import PERFORMERS and SECTIONS from here.
Adding a new recording only requires updating performers.json.
"""

import json
from pathlib import Path

BASE = Path(__file__).parent

# ── Load performers from JSON ─────────────────────────────────────────────────
def load_performers(base: Path = BASE) -> dict:
    """
    Returns a dict: {name: (wav_path, color)}
    compatible with the existing PERFORMERS format in all scripts.
    """
    cfg_path = base / 'performers.json'
    if not cfg_path.exists():
        raise FileNotFoundError(f"performers.json not found at {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    out = {}
    for name, info in cfg.items():
        wav = base / 'normalized_audio' / info['file']
        out[name] = (wav, info['color'])
    return out

PERFORMERS = load_performers()

# ── Section boundaries (score seconds, Wang Jianzhong 114 BPM MIDI) ──────────
SCORE_DURATION = 151.6

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

SCORE_MIDI = BASE / 'cai-yun-zhui-yue-ren-guang-qu-wang-jian-zhong-gai-bian.mid'
SCORE_WAV  = BASE / 'reference_score.wav'
SCORE_BPM  = 114.0

SR  = 22050
HOP = 512

if __name__ == '__main__':
    print(f"Performers loaded ({len(PERFORMERS)}):")
    for name, (path, color) in PERFORMERS.items():
        status = 'OK' if path.exists() else 'MISSING'
        print(f"  {status}  {name:<20}  {color}  {path.name}")
