#!/usr/bin/env python3
"""Assemble HelloAria_Colab.ipynb (valid nbformat v4). Run: python build_notebook.py

Design notes (learned the hard way against a 2026 Colab):
  * No condacolab: recent condacolab installs Python 3.12 anyway, and all training deps
    (acoustics, speechbrain, etc.) install fine on 3.12 — verified. Avoids a needless restart.
  * All pip installs use {sys.executable} -m pip so packages land in the KERNEL's interpreter
    (a %%bash cell can install into a different Python, which silently breaks imports).
  * No `datasets` library: openWakeWord's train.py never imports it; it was only used to pull
    background/RIR audio from dataset repos that have since bit-rotted. We download the MIT RIR
    survey + ESC-50 as plain zips and resample to 16 kHz instead.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
cells = []

def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.rstrip("\n") + "\n"})

def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.rstrip("\n")})

# ------------------------------------------------------------------ intro
md("""# Train the **"Hello Aria"** wake word (openWakeWord → ONNX)

Produces `hello_aria.onnx`, the swappable detector head for the mirror app
(`mirror-app/assets/wakeword/wakeword.onnx`). The mel + embedding front-end is shared and **not**
retrained, so this head is compatible with the app's bundled models (verified locally against the
stock model's feature space).

**How to run**
1. `Runtime ▸ Change runtime type ▸ GPU` (T4 is fine).
2. `Runtime ▸ Run all`. No restart, no conda — runs on Colab's native Python 3.12.
3. The **preflight cell** renders one test clip; if it says `OK — Piper TTS works.`, the long run will too.
4. Total ≈ **1.5–3 h** (mostly Piper TTS + the 2000-hr negative-feature download). Free Colab is enough.
5. It self-validates, prints a recommended threshold, and downloads `hello_aria.onnx` + `hello_aria_metrics.json`.
6. Send `hello_aria.onnx` back — it's validated on the device-exact pipeline and swapped in.

**Robustness choices** (vs. openWakeWord's stock notebook): installs into the kernel interpreter via
`sys.executable`; uses dscripka's generator fork (espeak-phonemizer, no piper-phonemize wheel needed);
no `datasets` library — RIR/noise pulled as plain zips; TensorFlow/tflite export removed (ONNX only);
openWakeWord pinned to `v0.6.0`; config tuned for elderly / far-field / TV-noise (strict false-accept
target + near-miss negatives). **If a fresh runtime is in a weird state, `Runtime ▸ Disconnect and
delete runtime` and Run all again — every cell is idempotent.**
""")

# ------------------------------------------------------------------ gpu check
md("## 0. Confirm GPU runtime")
code("""# GPU check without importing numpy/torch (setup pins numpy<2; a pre-loaded numpy would shadow it).
import subprocess
r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
print(r.stdout.strip() or r.stderr.strip())
assert "GPU" in r.stdout, "No GPU: Runtime > Change runtime type > GPU, then Run all again."
print("GPU OK.")
""")

# ------------------------------------------------------------------ setup
md("""## 1. Environment setup
Clones openWakeWord (`v0.6.0`) + Piper's sample generator (dscripka fork), installs everything into
**this kernel's** Python, and verifies the two import-sensitive packages actually landed.""")
code(r'''import sys, os, subprocess
def sh(cmd):  subprocess.run(cmd, shell=True, check=True)
def pip(spec): subprocess.run([sys.executable, "-m", "pip", "install", "-q", *spec.split()], check=True)

# Piper neural TTS generator — dscripka's fork keeps generate_samples.py at root and phonemizes via
# espeak-phonemizer (pure Python + espeak-ng), so no piper-phonemize wheel is needed. Self-heal a wrong clone.
if not os.path.exists("piper-sample-generator/generate_samples.py"):
    sh("rm -rf piper-sample-generator && git clone -q https://github.com/dscripka/piper-sample-generator")
if not os.path.exists("piper-sample-generator/models/en-us-libritts-high.pt"):
    sh("wget -q -O piper-sample-generator/models/en-us-libritts-high.pt "
       "'https://github.com/rhasspy/piper-sample-generator/releases/download/v1.0.0/en-us-libritts-high.pt'")

# openWakeWord pinned to v0.6.0 (the whole notebook is built against it).
if not os.path.exists("openwakeword"):
    sh("git clone -q --depth 1 --branch v0.6.0 https://github.com/dscripka/openwakeword")

sh("apt-get -qq update && apt-get -qq install -y espeak-ng")   # system lib for espeak-phonemizer

# ALL pip installs target THIS kernel's interpreter (sys.executable) — a %%bash cell can install into a
# different Python and silently break imports.
# CONSTRAINTS LOCK: pin the ABI-sensitive trio so NO transitive dependency can bump them into a skew.
# A torch/torchaudio version mismatch throws "undefined symbol" at import; numpy<2 (old np APIs in
# speechbrain/openWakeWord). Passing -c on every install makes these versions immovable.
open("constraints.txt", "w").write("torch==2.3.1\ntorchaudio==2.3.1\nnumpy==1.26.4\n")
def pipc(spec): pip("-c constraints.txt " + spec)
pipc("torch==2.3.1 torchaudio==2.3.1 numpy==1.26.4")
pipc("espeak-phonemizer webrtcvad-wheels")   # webrtcvad-wheels = prebuilt (no C compiler needed)
pipc("-e ./openwakeword")
pipc("scipy soundfile onnx onnxruntime pyyaml tqdm "
     "mutagen==1.47.0 torchinfo==1.8.0 torchmetrics==1.2.0 speechbrain==0.5.14 "
     "audiomentations==0.33.0 torch-audiomentations==0.11.0 acoustics==0.2.6 pronouncing==0.2.0")

# Shared front-end models openWakeWord uses to compute features (same family as the app's bundled ones).
os.makedirs("openwakeword/openwakeword/resources/models", exist_ok=True)
for f in ("embedding_model.onnx", "melspectrogram.onnx"):
    dst = f"openwakeword/openwakeword/resources/models/{f}"
    if not os.path.exists(dst):
        sh(f"wget -q 'https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/{f}' -O '{dst}'")

# Import the ABI-sensitive stack HERE so a skew fails loudly in setup, not 3 cells later.
import numpy, torch, torchaudio, webrtcvad, espeak_phonemizer
print(f"setup OK — torch {torch.__version__} | torchaudio {torchaudio.__version__} | "
      f"numpy {numpy.__version__} | CUDA {torch.cuda.is_available()}")
''')

# ------------------------------------------------------------------ patch tflite
md("""## 2. Patch out the TensorFlow/tflite export
`hello_aria.onnx` is written before the tflite step, so we neutralize it and skip TensorFlow entirely.""")
code('''import pathlib
p = pathlib.Path("openwakeword/openwakeword/train.py")
s = p.read_text()
needle = ('def convert_onnx_to_tflite(onnx_model_path, output_path):\\n'
          '    """Converts an ONNX version of an openwakeword model to the Tensorflow tflite format."""')
repl = ('def convert_onnx_to_tflite(onnx_model_path, output_path):\\n'
        '    """PATCHED no-op: the mirror app loads .onnx directly; TensorFlow/tflite skipped."""\\n'
        '    return None')
if "PATCHED no-op" in s:
    print("Already patched — nothing to do (safe to re-run).")
elif needle in s:
    p.write_text(s.replace(needle, repl, 1))
    print("Patched: convert_onnx_to_tflite -> no-op (no TensorFlow needed).")
else:
    print("NOTE: tflite converter not found to patch. Harmless: hello_aria.onnx is written BEFORE the\\n"
          "tflite step, so even if that step errors at the very end, the .onnx you need already exists.")
''')

# ------------------------------------------------------------------ preflight
md("""## 3. Preflight — verify Piper works *before* the long run
Renders 4 "hello aria" clips. If it fails on an import, re-run Step 1 (it prints which package is missing).""")
code('''import sys, os
sys.path.insert(0, os.path.abspath("piper-sample-generator"))
from generate_samples import generate_samples
import inspect
print("generate_samples signature:\\n ", inspect.signature(generate_samples))
os.makedirs("preflight", exist_ok=True)
generate_samples(text=["hello aria"], max_samples=4, batch_size=4,
                 noise_scales=[0.98], noise_scale_ws=[0.98], length_scales=[0.9, 1.0, 1.1],
                 output_dir="preflight", auto_reduce_batch_size=True,
                 file_names=[f"pf_{i}.wav" for i in range(4)])
clips = [f for f in os.listdir("preflight") if f.endswith(".wav")]
print("Preflight clips:", sorted(clips))
assert len(clips) >= 1, "Piper produced no clips"
print("OK — Piper TTS works.")
''')

# ------------------------------------------------------------------ data (datasets-free)
md("""## 4. Download training data (no `datasets` library)
RIRs (reverb) from the MIT survey, background noise from ESC-50 — both plain zips, resampled to 16 kHz —
plus openWakeWord's pre-computed 2000-hr real-audio negatives + an 11-hr false-positive validation set.
The negatives are what make the model resist false accepts in a noisy room.""")
code('''import os, glob, zipfile, urllib.request
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm

def _dl(url, path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)

def _resample_dir(src_wavs, out_dir, limit=None):
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for f in tqdm(src_wavs[:limit] if limit else src_wavs, desc=out_dir):
        try:
            x, sr = sf.read(f, dtype="float32")
            if x.ndim > 1: x = x.mean(axis=1)
            if sr != 16000: x = resample_poly(x, 16000, sr)
            sf.write(os.path.join(out_dir, f"{n:05d}.wav"),
                     (np.clip(x, -1, 1) * 32767).astype(np.int16), 16000, subtype="PCM_16")
            n += 1
        except Exception:
            pass
    return n

# 4a. MIT room impulse responses -> mit_rirs/ (16 kHz)
if len(glob.glob("mit_rirs/*.wav")) < 5:
    _dl("https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip", "mit_rirs.zip")
    zipfile.ZipFile("mit_rirs.zip").extractall("mit_rirs_raw")
    wavs = glob.glob("mit_rirs_raw/**/*.wav", recursive=True) + glob.glob("mit_rirs_raw/**/*.WAV", recursive=True)
    print("RIR wavs in zip:", len(wavs)); _resample_dir(wavs, "mit_rirs")
print("RIRs (16k):", len(glob.glob("mit_rirs/*.wav")))
''')
code('''# 4b. ESC-50 environmental noise -> background/ (16 kHz). Subset for speed; raise limit for more variety.
import os, glob, zipfile, urllib.request
if len(glob.glob("background/*.wav")) < 20:
    _dl("https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip", "esc50.zip")
    zipfile.ZipFile("esc50.zip").extractall("esc50_raw")
    wavs = sorted(glob.glob("esc50_raw/**/audio/*.wav", recursive=True))
    print("ESC-50 wavs:", len(wavs)); _resample_dir(wavs, "background", limit=800)
print("background (16k):", len(glob.glob("background/*.wav")))
''')
code('''# 4c. Pre-computed openWakeWord features: 2000-hr negatives (~16 GB) + 11-hr FP validation set
import os
if not os.path.exists("openwakeword_features_ACAV100M_2000_hrs_16bit.npy"):
    !wget -q --show-progress https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy
if not os.path.exists("validation_set_features.npy"):
    !wget -q --show-progress https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy
import numpy as np
print("negatives:", np.load("openwakeword_features_ACAV100M_2000_hrs_16bit.npy", mmap_mode="r").shape)
print("validation:", np.load("validation_set_features.npy", mmap_mode="r").shape)
''')

# ------------------------------------------------------------------ config
md("""## 5. Training configuration ("Hello Aria")
Tuned for the mirror's deployment. **Quality vs. time knobs** are marked.""")
code('''import yaml
config = yaml.load(open("openwakeword/examples/custom_model.yml").read(), yaml.Loader)

config["target_phrase"] = ["hello aria"]
config["model_name"]    = "hello_aria"

# --- QUALITY vs TIME knobs ---
config["n_samples"]     = 5000     # positives via Piper. Production: 20000-40000 (longer TTS time)
config["n_samples_val"] = 2000
config["steps"]         = 40000    # Production: 50000

# --- deployment tuning: elderly + far-field + TV noise => punish false accepts ---
config["target_false_positives_per_hour"] = 0.2
config["max_negative_weight"]             = 1500
# near-miss phrases we explicitly do NOT want to fire on (extra TTS negatives). NOT the target phrase.
config["custom_negative_phrases"] = ["hello maria", "hello area", "hi aria", "hello ariel",
                                     "hello arya", "hello ara", "yellow aria", "olio aria"]

config["rir_paths"] = ["./mit_rirs"]
config["background_paths"] = ["./background"]
config["background_paths_duplication_rate"] = [1]
config["false_positive_validation_data_path"] = "./validation_set_features.npy"
config["feature_data_files"] = {"ACAV100M_sample": "./openwakeword_features_ACAV100M_2000_hrs_16bit.npy"}
config["piper_sample_generator_path"] = "./piper-sample-generator"
config["output_dir"] = "./my_custom_model"

with open("my_model.yaml", "w") as f:
    yaml.dump(config, f, sort_keys=False)
print(yaml.dump(config, sort_keys=False))
''')

# ------------------------------------------------------------------ train
md("""## 6. Train (3 steps)
Separate script invocations (openWakeWord's design). Steps 1–2 resume where they left off, so a Colab
hiccup during generation is recoverable — just re-run the cell.""")
code('''import sys
# Step 1: synthesize positive "hello aria" clips + adversarial negatives (Piper). Longest step.
!{sys.executable} openwakeword/openwakeword/train.py --training_config my_model.yaml --generate_clips
''')
code('''import sys
# Step 2: augment (reverb + background mix) and compute openWakeWord features.
!{sys.executable} openwakeword/openwakeword/train.py --training_config my_model.yaml --augment_clips
''')
code('''import sys
# Step 3: train the head against the 2000-hr negatives, select best checkpoint by FP/hour, export ONNX.
!{sys.executable} openwakeword/openwakeword/train.py --training_config my_model.yaml --train_model
''')

# ------------------------------------------------------------------ validate
md("""## 7. Validate & choose a detection threshold
openWakeWord emits no metrics file, so we compute one: recall on held-out positive windows, and
false-accepts/hour on the 11-hr validation set, across thresholds. Recommends the operating point that
meets the FP/hour target with the best recall — this is `EXPO_PUBLIC_WAKE_WORD_THRESHOLD`.""")
code('''import numpy as np, onnxruntime as ort, json, os, yaml as _yaml

_cfg = _yaml.safe_load(open("my_model.yaml"))
TARGET_FP = _cfg["target_false_positives_per_hour"]
onnx_path = "my_custom_model/hello_aria.onnx"
assert os.path.exists(onnx_path), "training did not produce hello_aria.onnx — check Step 3 output"
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
print("model:", inp.name, inp.shape, "->", out.name, out.shape, "| size", os.path.getsize(onnx_path), "bytes")

def score(windows):  # windows: [N,16,96] -> [N] scores (model is fixed batch=1)
    s = np.empty(len(windows), np.float32)
    for i in range(len(windows)):
        s[i] = np.asarray(sess.run([out.name], {inp.name: windows[i:i+1].astype(np.float32)})[0]).reshape(-1)[-1]
    return s

pos = np.load("my_custom_model/hello_aria/positive_features_test.npy").astype(np.float32).reshape(-1, 16, 96)
pos_scores = score(pos)

val = np.load("validation_set_features.npy").astype(np.float32)  # [N,96]
N = min(200_000, val.shape[0] - 16)
neg = np.stack([val[i:i+16] for i in range(N)]).astype(np.float32)
neg_scores = score(neg)
EMB_PER_SEC = 12.5
val_hours = len(neg) / EMB_PER_SEC / 3600.0

print(f"\\npositives: n={len(pos_scores)}  neg windows: {len(neg_scores)} (~{val_hours:.2f} h)\\n")
print(f"{'thresh':>7} {'recall':>8} {'FP/hour':>9}")
rows, best = [], None
for t in [0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]:
    recall = float((pos_scores >= t).mean())
    fp_hr = float((neg_scores >= t).sum() / max(val_hours, 1e-9))
    rows.append({"threshold": t, "recall": round(recall,4), "fp_per_hour": round(fp_hr,3)})
    print(f"{t:>7.2f} {recall:>8.3f} {fp_hr:>9.3f}")
    if fp_hr <= TARGET_FP and best is None:
        best = t
if best is None:
    best = 0.99
    print("\\n[warn] no threshold met the FP/hour target; recommending 0.99. Consider more n_samples/steps.")
print(f"\\nRECOMMENDED EXPO_PUBLIC_WAKE_WORD_THRESHOLD = {best}")

metrics = {"model": "hello_aria", "phrase": "Hello Aria", "input_shape": list(inp.shape),
           "recommended_threshold": best, "target_fp_per_hour": TARGET_FP,
           "validation_hours_used": round(val_hours,2), "roc": rows,
           "positive_score_percentiles": {p: float(np.percentile(pos_scores,p)) for p in (5,25,50,75,95)}}
json.dump(metrics, open("my_custom_model/hello_aria_metrics.json","w"), indent=2)
print("wrote hello_aria_metrics.json")
''')

# ------------------------------------------------------------------ download
md("## 8. Download the model + metrics\nSend `hello_aria.onnx` back; I'll validate it on the device-exact pipeline and swap it into the app.")
code('''from google.colab import files
files.download("my_custom_model/hello_aria.onnx")
files.download("my_custom_model/hello_aria_metrics.json")
''')

for i, c in enumerate(cells):
    c["id"] = f"cell{i:02d}"
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"},
                   "colab": {"provenance": []}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}

out = os.path.join(HERE, "HelloAria_Colab.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out, "with", len(cells), "cells")
