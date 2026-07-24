#!/usr/bin/env python3
"""Disambiguate: same training wavs, same model — training-path scores vs validator-path scores."""
import numpy as np, glob
from owwfeat import FrontEnd, Detector, load_wav16

front = FrontEnd()
det = Detector("hello_aria.onnx", front)
wavs = sorted(glob.glob("local_clips.noindex/pos_say/*.wav"))[:24]
TOTAL = 32000

a_scores, b_scores = [], []
for w in wavs:
    x = load_wav16(w)
    # (a) training-style: centered in 2 s buffer at 0.6 peak, WHOLE-CLIP features, one window
    xa = x / (np.abs(x).max() + 1e-9) * 0.6
    buf = np.zeros(TOTAL, np.float32); L = min(len(xa), TOTAL); off = (TOTAL - L) // 2
    buf[off:off + L] = xa[:L]
    xa = (buf * 32767).astype(np.int16).astype(np.float32)
    Wa = front.windows(front.embeddings_wholeclip(xa))
    a_scores.append(float(det.score_windows(Wa).max()) if len(Wa) else np.nan)
    # (b) validator-style: 4 s centered pad, STREAMING features, max over sliding windows
    tgt = 64000; pad = tgt - len(x)
    xb = np.concatenate([np.zeros(pad // 2, np.float32), x, np.zeros(pad - pad // 2, np.float32)])
    sb = det.scores(xb)
    b_scores.append(float(sb.max()) if len(sb) else np.nan)

a, b = np.array(a_scores), np.array(b_scores)
print(f"(a) training-path: median={np.nanmedian(a):.3f}  >=0.6: {np.nanmean(a >= 0.6):.2f}")
print(f"(b) validator-path: median={np.nanmedian(b):.3f}  >=0.6: {np.nanmean(b >= 0.6):.2f}")
if np.nanmedian(a) > 0.5 and np.nanmedian(b) < 0.2:
    print("VERDICT: TRANSFORM MISMATCH — model learned the voices; streaming/padding differs")
elif np.nanmedian(a) < 0.2:
    print("VERDICT: MODEL NEVER LEARNED say voices — checkpoint too early / weighting")
else:
    print("VERDICT: mixed — inspect per-clip")
