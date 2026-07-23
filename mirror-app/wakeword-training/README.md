# "Hello Aria" wake word — train & swap

The mirror ships openWakeWord's stock **"hey jarvis"** model at
`mirror-app/assets/wakeword/wakeword.onnx`. This directory trains a custom **"Hello Aria"** detector
and swaps that one file. The `melspectrogram.onnx` + `embedding_model.onnx` front-end is **shared and
never retrained**; the app loads `wakeword.onnx` unchanged (`src/native/wakeWord.ts`), so no code edit
is needed — the new phrase is live once the APK is rebuilt.

Contract (verified locally against the stock model's feature space): `wakeword.onnx` takes
`[1,16,96]` float32 → scalar score in `[0,1]`. A Colab-trained openWakeWord head produces exactly this.

## Why Colab (not this Mac)
Two hard blockers on Apple-Silicon macOS: **`piper-phonemize` has no arm64 wheel** (openWakeWord's
neural positive generator can't install), and the canonical flow wants ~8 GB of data. Colab is Linux +
free GPU + ample disk, and gives ~900 Piper LibriTTS-R speakers (vs. ~30 macOS `say` voices) — far
better recall on unfamiliar (e.g. elderly) voices. So we **train on Colab, validate + swap locally.**

## 1. Train on Colab → `hello_aria.onnx`
1. Upload **`HelloAria_Colab.ipynb`** to https://colab.research.google.com (File ▸ Upload notebook).
2. `Runtime ▸ Change runtime type ▸ GPU`, then `Runtime ▸ Run all`. No conda, no restart — runs on
   Colab's native Python 3.12.
3. ~1.5–3 h (mostly Piper TTS + the 2000-hr negative-feature download). It self-validates, prints a
   recommended threshold, and downloads `hello_aria.onnx` + `hello_aria_metrics.json`.
4. If the runtime is in a weird state from a prior attempt, `Runtime ▸ Disconnect and delete runtime`
   and Run all again — every cell is idempotent.

The notebook is openWakeWord's official flow, hardened against 2026-Colab drift: installs into the
**kernel** interpreter via `sys.executable` (a `%%bash` cell can install into a different Python);
uses dscripka's generator fork (espeak-phonemizer, so no `piper-phonemize` wheel needed); drops the
`datasets` library — RIRs (MIT survey) + noise (ESC-50) are pulled as plain zips and resampled to
16 kHz, avoiding bit-rotted dataset repos; TensorFlow/tflite export removed (ONNX only); openWakeWord
pinned to `v0.6.0`; a Piper preflight check; config tuned for an elderly / far-field / TV-noise room
(`target_false_positives_per_hour: 0.2`, near-miss negatives). Quality-vs-time knobs (`n_samples`,
`steps`) are marked in the config cell.

## 2. Validate on the device-exact pipeline + swap
`owwfeat.py` mirrors `src/native/wakeWord.ts` byte-for-byte, so scores here are what the device will
produce. The venv is already set up (`.venv`, Python 3.12 + torch/onnxruntime).

```bash
cd mirror-app/wakeword-training
./.venv/bin/python validate_hello_aria.py /path/to/hello_aria.onnx          # validate only
./.venv/bin/python validate_hello_aria.py /path/to/hello_aria.onnx --swap   # + back up stock + install
```
It prints a threshold table (say-recall / clip false-accepts / validation FP-per-hour) and a
recommended operating point. `--swap` backs up the stock model to
`assets/wakeword/wakeword.stock-hey-jarvis.onnx` and installs the new one.

## 3. Set the operating point + rebuild
Phrase **"Hello Aria"** already matches `wakeWord.ts` (`WAKE_WORD_PHRASE` default), so no code change.
Set the threshold from the validator/metrics on the device build (env, no rebuild-of-code needed):
```
EXPO_PUBLIC_WAKE_WORD_THRESHOLD=0.6   # use the recommended value
EXPO_PUBLIC_WAKE_WORD_HITS=3
EXPO_PUBLIC_WAKE_WORD_PHRASE="Hello Aria"
```
Then rebuild the APK (`npx expo run:android` or an EAS dev build). The app falls back to tap-to-start
if the model/runtime can't load, so a bad model can't brick the mirror.

## 4. Verify on the real device
`say`-voice recall in the validator is an out-of-distribution sanity check (the model trains on Piper
voices). The real test: physical mirror at ~1–2 m with a TV playing — several "Hello Aria" utterances
vs. a control period of unrelated speech; confirm detections vs. false accepts against your target.

## Files
| file | role |
|---|---|
| `HelloAria_Colab.ipynb` | the training notebook (run on Colab) |
| `build_notebook.py` | regenerates the notebook (`python build_notebook.py`) |
| `owwfeat.py` | device-exact front-end (mirror of `wakeWord.ts`) |
| `validate_hello_aria.py` | local validation + backup-and-swap |
| `verify_frontend.py` | one-off proof that the bundled front-end = openWakeWord's feature space |
| `train_hey_aria.py` | ⚠️ legacy/local attempt — blocked on arm64 (no `piper-phonemize` wheel). Use Colab. |
