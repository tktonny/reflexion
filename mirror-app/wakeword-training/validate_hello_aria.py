#!/usr/bin/env python3
"""Validate a Colab-trained hello_aria.onnx on the DEVICE-EXACT pipeline, then optionally swap it in.

    python validate_hello_aria.py path/to/hello_aria.onnx            # validate only
    python validate_hello_aria.py path/to/hello_aria.onnx --swap     # validate + back up stock + install

Validation uses owwfeat.py (a byte-for-byte mirror of mirror-app/src/native/wakeWord.ts), so the
scores here are what the model will actually produce on-device. It checks:
  * I/O contract is [1,16,96] -> scalar  (else the app can't load it)
  * recall on macOS `say` "Hello Aria" clips  (out-of-distribution vs Piper training voices)
  * false accepts on distractor + near-miss clips, and on the 11-hr validation feature set (FP/hour)
"""
import argparse, os, shutil, sys, json
import numpy as np
from owwfeat import FrontEnd, Detector, say_to_wav, load_wav16, ASSETS, WW_WINDOW

HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE = os.path.join(HERE, "output", "val_clips")
STOCK = os.path.join(ASSETS, "wakeword.onnx")
BACKUP = os.path.join(ASSETS, "wakeword.stock-hey-jarvis.onnx")
VAL_FEATURES = os.path.join(HERE, "validation_set_features.npy")
EMB_PER_SEC = 12.5

VOICES = ["Samantha", "Daniel", "Karen", "Moira", "Tessa", "Fred", "Rishi", "Albert",
          "Kathy", "Ralph", "Grandma", "Grandpa", "Reed", "Sandy", "Shelley"]
NEGATIVE_PHRASES = ["what time is it", "turn on the television", "good morning everyone",
                    "hello maria", "hi aria", "hello area", "call my daughter", "i love you",
                    "hello there", "yellow area rug"]


def padded_clip(text, voice, target_s=4.0, rate=None):
    os.makedirs(SMOKE, exist_ok=True)
    wav = os.path.join(SMOKE, f"{voice}_{abs(hash((text, rate))) % 99999}.wav")
    if not say_to_wav(text, voice, wav, rate):
        return None
    x = load_wav16(wav)
    tgt = int(target_s * 16000)
    if len(x) < tgt:
        pad = tgt - len(x); left = pad // 2
        x = np.concatenate([np.zeros(left, np.float32), x, np.zeros(pad - left, np.float32)])
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="path to trained hello_aria.onnx")
    ap.add_argument("--swap", action="store_true", help="back up stock model and install this one")
    ap.add_argument("--target-fp", type=float, default=0.2, help="max false-accepts/hour (default 0.2)")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        sys.exit(f"model not found: {args.model}")

    front = FrontEnd()
    det = Detector(args.model, front)
    ishape, oshape = det.ww.get_inputs()[0].shape, det.ww.get_outputs()[0].shape
    print(f"=== I/O contract ===\n  in {det.ww_in}{ishape} -> out {det.ww_out}{oshape}")
    ok_shape = list(ishape)[-2:] == [16, 96]
    print(f"  [1,16,96]->scalar contract: {'OK' if ok_shape else 'WRONG — app cannot use this model'}")
    if not ok_shape:
        sys.exit(2)

    print("\n=== recall: say 'Hello Aria' (OOD vs Piper voices; higher is better) ===")
    pos = []
    for v in VOICES:
        for r in (None, 170):
            x = padded_clip("Hello Aria", v, rate=r)
            if x is None:
                continue
            s = det.scores(x)
            m = float(s.max()) if len(s) else float("nan")
            pos.append(m)
            print(f"  Hello Aria / {v:8} r={str(r):4} -> {m:.3f}")

    print("\n=== false accepts: distractors + near-misses (lower is better) ===")
    neg = []
    for t in NEGATIVE_PHRASES:
        for v in VOICES[:5]:
            x = padded_clip(t, v)
            if x is None:
                continue
            s = det.scores(x)
            m = float(s.max()) if len(s) else float("nan")
            neg.append(m)
        print(f"  {t:24} max over voices -> {max(neg[-5:]):.3f}")

    pos, neg = np.array(pos), np.array(neg)
    fp_ok = os.path.exists(VAL_FEATURES) and os.path.getsize(VAL_FEATURES) > 1_000_000
    if fp_ok:
        val = np.load(VAL_FEATURES).astype(np.float32)
        n = min(200_000, val.shape[0] - WW_WINDOW)
        wins = np.stack([val[i:i + WW_WINDOW] for i in range(n)]).astype(np.float32)
        vsc = det.score_windows(wins)
        hrs = len(wins) / EMB_PER_SEC / 3600.0

    print(f"\n=== threshold table (target FP/hour <= {args.target_fp}) ===")
    print(f"{'thresh':>7} {'say-recall':>11} {'clip-FA':>8}" + (f" {'val FP/hr':>10}" if fp_ok else ""))
    best = None
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        rec = float((pos >= t).mean()) if len(pos) else float("nan")
        cfa = int((neg >= t).sum())
        line = f"{t:>7.2f} {rec:>11.3f} {cfa:>8d}"
        if fp_ok:
            fphr = float((vsc >= t).sum() / max(hrs, 1e-9))
            line += f" {fphr:>10.3f}"
            if fphr <= args.target_fp and best is None:
                best = t
        print(line)
    if best is None:
        best = 0.7 if not fp_ok else 0.95
    print(f"\nRECOMMENDED: EXPO_PUBLIC_WAKE_WORD_THRESHOLD={best}  EXPO_PUBLIC_WAKE_WORD_HITS=3")
    if len(pos) and (pos >= best).mean() < 0.5:
        print("[note] say-recall at the recommended threshold is low — expected, since the model trained on\n"
              "       Piper voices, not macOS `say`. Real-device recall with human voices is the true test.")

    if args.swap:
        if not os.path.exists(BACKUP):
            shutil.copy2(STOCK, BACKUP)
            print(f"\nbacked up stock model -> {BACKUP}")
        shutil.copy2(args.model, STOCK)
        print(f"installed -> {STOCK}")
        print("Rebuild the APK (npx expo run:android / EAS dev build) for the new phrase to take effect.")
    else:
        print("\n(validation only — re-run with --swap to install once you're happy)")


if __name__ == "__main__":
    main()
