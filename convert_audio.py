import librosa
import soundfile as sf
import os

webm_file = r"C:\Users\wenli\OneDrive\Desktop\Sound project\shenwenyu_caiyun.webm"
wav_file = r"C:\Users\wenli\OneDrive\Desktop\Sound project\shenwenyu_caiyun.wav"

print(f"Loading: {webm_file}")
try:
    # Load audio with librosa
    y, sr = librosa.load(webm_file, sr=None)
    print(f"✓ Loaded successfully!")
    print(f"  Duration: {librosa.get_duration(y=y, sr=sr):.2f} seconds")
    print(f"  Sample rate: {sr} Hz")

    # Save as WAV
    sf.write(wav_file, y, sr)
    print(f"✓ Saved to: {wav_file}")
    print(f"  File size: {os.path.getsize(wav_file) / 1024 / 1024:.2f} MB")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
