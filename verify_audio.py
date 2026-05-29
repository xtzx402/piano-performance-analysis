import librosa
import os
import sys

# Set encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# File mapping
files = {
    "langlang_caiyun.wav": "Lang Lang",
    "liyundi_caiyun.wav": "Li Yundi",
    "shenwenyu_caiyun.wav": "Shen Wenyu"
}

base_path = r"C:\Users\wenli\OneDrive\Desktop\Sound project"

print("=" * 70)
print("Audio File Verification".center(70))
print("=" * 70)
print()

for filename, artist in files.items():
    filepath = os.path.join(base_path, filename)

    if not os.path.exists(filepath):
        print(f"✗ {filename} - File not found")
        continue

    try:
        # Load audio
        y, sr = librosa.load(filepath, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)

        # Get file size
        file_size = os.path.getsize(filepath) / 1024 / 1024

        print(f"✓ {filename}")
        print(f"  Performer: {artist}")
        print(f"  Duration:  {int(duration // 60)}m{int(duration % 60)}s ({duration:.2f}s)")
        print(f"  Sample Rate: {sr} Hz")
        print(f"  File Size: {file_size:.2f} MB")
        print()

    except Exception as e:
        print(f"✗ {filename} - Error: {e}")
        print()

print("=" * 70)
