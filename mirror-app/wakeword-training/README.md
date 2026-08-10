# "Hello Aria" wake word — train & swap (fully local on this Mac)

The mirror ships openWakeWord's stock **"hey jarvis"** model at
`mirror-app/assets/wakeword/wakeword.onnx`. This directory trains a custom **"Hello Aria"** detector
and swaps that one file. The `melspectrogram.onnx` + `embedding_model.onnx` front-end is **shared and
never retrained**; the app loads `wakeword.onnx` unchanged (`src/native/wakeWord.ts`), so no code edit
is needed — the new phrase is live once the APK is rebuilt.

Contract (verified against the stock model through a byte-for-byte Python mirror of the app pipeline):
`wakeword.onnx` takes `[1,16,96]` float32 → scalar score in `[0,1]`.

## The pipeline (all local, no Colab)

```
gen_local.py      Piper VITS (~800 LibriTTS speakers) on the Apple GPU (MPS) →
                  local_clips.noindex/{pos_train,pos_test,neg_adv}
data_prep.py      MIT RIRs + ESC-50 noise (16 kHz) + 200k real-negative feature
                  windows (neg.bin, resumable Range-download of openWakeWord's 2000-hr set)
train_local.py    augment (reverb/noise/offset) → features via the BUNDLED front-end →
                  openWakeWord-style DNN head vs real negatives (negative-weight ramp) →
                  threshold table → hello_aria.onnx + hello_aria_metrics.json
validate_hello_aria.py --swap
                  device-exact validation (chunked path, exactly wakeWord.ts) →
                  backs up stock model → installs the new one
run_rest.sh       chains gen → train unattended (nohup-safe, resume-safe)
```

Run order (each step resumes if interrupted):
```bash
cd mirror-app/wakeword-training
./.venv/bin/python data_prep.py                      # one-time data
ESPEAK_DATA_PATH=/opt/homebrew/share/espeak-ng-data PYTORCH_ENABLE_MPS_FALLBACK=1 \
  ./.venv/bin/python gen_local.py                    # clips (GPU, ~15 min)
./.venv/bin/python train_local.py                    # train + export (~15-20 min)
./.venv/bin/python validate_hello_aria.py hello_aria.onnx --swap
```

## Hard-won environment facts (do not rediscover these)
- **Piper runs locally on Apple Silicon.** `piper-phonemize` has no arm64/py3.12 wheel, but the
  **dscripka fork** of piper-sample-generator uses `espeak-phonemizer` (pure Python + brew's
  `espeak-ng`) instead. Two Linux-isms are patched **inside `.venv`'s copy** of `espeak_phonemizer`:
  it loads `libespeak-ng` and libc via `ctypes.util.find_library` (absolute paths) instead of
  hardcoded `.so` names. If the venv is rebuilt, re-apply (see git history) or run
  `validate_hello_aria.py` first — generation will fail loudly if the patch is missing.
- **MPS (Apple GPU) is ~10–30× faster** than the contended CPU for VITS. The generator only
  special-cases CUDA, so `gen_local.py` aliases `torch.cuda.*`/`Tensor.cuda` to MPS. Launch with
  `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- **Spotlight will storm-index thousands of fresh WAVs** and crater throughput — clip output lives in
  `local_clips.noindex/` (the `.noindex` suffix opts the directory out of indexing).
- **HuggingFace kills long unauthenticated streams** (curl exit 92). `data_prep.py` downloads
  `neg.bin` in resumable 32 MB HTTP/1.1 chunks.
- Feature space parity is PROVEN: the stock "hey jarvis" head scores our computed features correctly
  (fires ~0.99 on "hey jarvis" clips; ≤0.003 on `neg.bin` samples) — training features computed by the
  bundled front-end are interchangeable with openWakeWord's published feature sets.

## Set the operating point + rebuild
Phrase **"Hello Aria"** already matches `wakeWord.ts` (`WAKE_WORD_PHRASE` default). Set the threshold
from the validator/metrics on the device build:
```
EXPO_PUBLIC_WAKE_WORD_THRESHOLD=<from hello_aria_metrics.json>
EXPO_PUBLIC_WAKE_WORD_HITS=3
EXPO_PUBLIC_WAKE_WORD_PHRASE="Hello Aria"
```
Then rebuild the APK (`npx expo run:android` or an EAS dev build). The app falls back to tap-to-start
if the model/runtime can't load, so a bad model can't brick the mirror.

## Verify on the real device — Singlish accent is acceptance risk #1
Synthetic-voice recall is a sanity check only. **All training voices are native-English TTS, but the
end users are Singaporean elderly speaking Singlish-accented English** — a distribution with zero
training/validation coverage, and no TTS engine we have can synthesize it. Field acceptance must be
judged with target-accent speakers: physical mirror at ~1–2 m with a TV playing — several
"Hello Aria" utterances vs. a control period of unrelated speech; confirm detections vs. false
accepts against the target (≤0.2 false accepts/hour).

If device recall is poor for target users, fixes ranked by ROI:
1. **Record 10–20 real target users** saying "Hello Aria" → WAVs into
   `local_clips.noindex/pos_real/` → add a `pos_real` block in `train_local.py` mirroring the
   `pos_say` pattern (features + batch slots + composite selection metric) → retrain (~10 min,
   existing features are cached). Beats tens of thousands more synthetic clips.
2. **Lower the threshold** 0.7 → 0.6 in `mirror-app/.env` (recall 74–80%, ~0.9 FP/h per-window
   before the HITS=3 suppression). No retrain — just rebuild the APK.
3. **Scale Piper positives to 5k–20k** (MPS makes it 30–60 min) and add weight decay
   (`Adam(..., weight_decay=1e-4)`) to prevent the late-training saturation that made FP-eligible
   thresholds unreachable after ~step 8000.

## Files
| file | role |
|---|---|
| `gen_local.py` / `data_prep.py` / `train_local.py` | the local pipeline (above) |
| `run_rest.sh` | unattended gen→train chain |
| `owwfeat.py` | device-exact front-end (mirror of `wakeWord.ts`) + whole-clip training path |
| `validate_hello_aria.py` | final validation + backup-and-swap |
| `verify_frontend.py` | one-off feature-space parity proof |
| `HelloAria_Colab.ipynb` / `build_notebook.py` | Colab alternative (works, but local is proven) |
| `train_hey_aria.py` | ⚠️ legacy first attempt — superseded by the local pipeline |
