#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthesize reference_caiyun.mid → reference_caiyun.wav
Uses additive sine-wave synthesis (no external tools needed).
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import soundfile as sf
import mido
from pathlib import Path

SR = 22050          # sample rate
DECAY = 0.5         # exponential decay time constant (seconds) — piano-like envelope

def midi_note_to_freq(note):
    return 440.0 * 2 ** ((note - 69) / 12)

def synthesize_note(freq, duration_sec, velocity, sr=SR, decay=DECAY):
    """Generate a piano-like tone: sine + harmonics + exponential decay."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # Fundamental + harmonics with decreasing amplitude
    wave = (
        1.00 * np.sin(2 * np.pi * 1 * freq * t) +
        0.50 * np.sin(2 * np.pi * 2 * freq * t) +
        0.25 * np.sin(2 * np.pi * 3 * freq * t) +
        0.12 * np.sin(2 * np.pi * 4 * freq * t) +
        0.06 * np.sin(2 * np.pi * 5 * freq * t)
    )
    # Exponential decay envelope (piano-like)
    envelope = np.exp(-t / decay)
    # Short attack (5ms)
    attack_samples = int(0.005 * sr)
    if attack_samples > 0 and attack_samples < len(t):
        envelope[:attack_samples] *= np.linspace(0, 1, attack_samples)
    amplitude = (velocity / 127.0) * 0.3
    return wave * envelope * amplitude

def midi_to_wav(midi_path, wav_path, sr=SR):
    mid = mido.MidiFile(midi_path)
    ticks_per_beat = mid.ticks_per_beat

    # Find tempo (default 120 BPM)
    tempo = 500000
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break

    def ticks_to_sec(ticks):
        return ticks * tempo / (ticks_per_beat * 1_000_000)

    # Collect all note events across all tracks
    events = []
    for track in mid.tracks:
        abs_ticks = 0
        active = {}
        for msg in track:
            abs_ticks += msg.time
            t_sec = ticks_to_sec(abs_ticks)
            if msg.type == 'note_on' and msg.velocity > 0:
                active[(msg.channel, msg.note)] = (t_sec, msg.velocity)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in active:
                    start, vel = active.pop(key)
                    dur = t_sec - start
                    if dur > 0.01:
                        events.append((start, msg.note, dur, vel))

    if not events:
        print("  Warning: no note events found in MIDI")
        return

    total_duration = max(start + dur for start, _, dur, _ in events) + 1.0
    total_samples = int(total_duration * sr)
    audio = np.zeros(total_samples)

    print(f"  Synthesizing {len(events)} notes, {total_duration:.1f}s total...")
    for i, (start_sec, note, dur_sec, velocity) in enumerate(events):
        freq = midi_note_to_freq(note)
        if freq < 20 or freq > 20000:
            continue
        tone = synthesize_note(freq, dur_sec, velocity, sr=sr)
        start_sample = int(start_sec * sr)
        end_sample = start_sample + len(tone)
        if end_sample > total_samples:
            tone = tone[:total_samples - start_sample]
            end_sample = total_samples
        audio[start_sample:end_sample] += tone

    # Normalize to prevent clipping
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.85

    sf.write(wav_path, audio, sr)
    print(f"  Saved: {wav_path}")
    print(f"  Duration: {total_duration:.1f}s  |  Sample rate: {sr} Hz")

if __name__ == '__main__':
    base = Path(__file__).parent
    midi_path = base / 'reference_caiyun.mid'
    wav_path  = base / 'reference_caiyun.wav'

    print("=" * 60)
    print("Synthesizing MIDI → WAV")
    print("=" * 60)
    midi_to_wav(str(midi_path), str(wav_path))
    print("=" * 60)
    print("Done. Use reference_caiyun.wav for comparison analysis.")
    print("=" * 60)
