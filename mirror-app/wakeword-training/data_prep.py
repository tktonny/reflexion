#!/usr/bin/env python3
"""Prepare local training data (no `datasets` library). Order: small robust downloads first.
  * mit_rirs/ — MIT RIR survey, resampled to 16 kHz (reverb augmentation)
  * background/ — ESC-50 environmental noise subset, resampled to 16 kHz
  * neg.bin  — ~600 MB HTTP-Range prefix of openWakeWord's 2000-hr negative FEATURES [K,16,96] float16,
               downloaded in resumable 32 MB chunks over HTTP/1.1 (HF resets long HTTP/2 streams).
validation_set_features.npy is already present from earlier.
"""
import os, glob, zipfile, urllib.request, subprocess
import numpy as np, soundfile as sf
from scipy.signal import resample_poly

ITEM = 16 * 96 * 2          # bytes per [16,96] float16 window
K = 200_000                 # ~600 MB of real negative windows (plenty for a small head)
NEED = 128 + K * ITEM       # npy header is 128 bytes
NEG_URL = ("https://huggingface.co/datasets/davidscripka/openwakeword_features/"
           "resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy")


def resample_dir(wavs, out, limit=None):
    os.makedirs(out, exist_ok=True)
    n = 0
    for f in (wavs[:limit] if limit else wavs):
        try:
            x, sr = sf.read(f, dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            if sr != 16000:
                x = resample_poly(x, 16000, sr)
            sf.write(os.path.join(out, f"{n:05d}.wav"),
                     (np.clip(x, -1, 1) * 32767).astype(np.int16), 16000, subtype="PCM_16")
            n += 1
        except Exception:
            pass
    return n


# 1. MIT room impulse responses -> 16 kHz  (small, verified URL)
if len(glob.glob("mit_rirs/*.wav")) < 5:
    urllib.request.urlretrieve("https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip", "mit.zip")
    zipfile.ZipFile("mit.zip").extractall("mit_raw")
    resample_dir(glob.glob("mit_raw/**/*.wav", recursive=True), "mit_rirs")
print("RIRs (16k):", len(glob.glob("mit_rirs/*.wav")), flush=True)

# 2. ESC-50 background noise subset -> 16 kHz  (small, verified URL)
if len(glob.glob("background/*.wav")) < 20:
    urllib.request.urlretrieve("https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip", "esc.zip")
    zipfile.ZipFile("esc.zip").extractall("esc_raw")
    resample_dir(sorted(glob.glob("esc_raw/**/audio/*.wav", recursive=True)), "background", limit=800)
print("background (16k):", len(glob.glob("background/*.wav")), flush=True)

# 3. Negative features: chunked + resumable (HF kills long streams; curl exit 92 = HTTP/2 reset)
CH = 32 * 1024 * 1024
have = os.path.getsize("neg.bin") if os.path.exists("neg.bin") else 0
while have < NEED:
    end = min(have + CH, NEED) - 1
    subprocess.run(["curl", "-sL", "--fail", "--http1.1",
                    "--retry", "8", "--retry-delay", "2", "--retry-all-errors",
                    "-r", f"{have}-{end}", "-o", "chunk.tmp", NEG_URL], check=True)
    with open("neg.bin", "ab") as f:
        f.write(open("chunk.tmp", "rb").read())
    have = os.path.getsize("neg.bin")
    print(f"neg.bin {have/1e6:.0f}/{NEED/1e6:.0f} MB", flush=True)
os.path.exists("chunk.tmp") and os.remove("chunk.tmp")
print("neg.bin bytes:", os.path.getsize("neg.bin"), flush=True)

print("DATA PREP DONE", flush=True)
