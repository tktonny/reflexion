#!/usr/bin/env python3
"""Trust anchor for the Colab path: prove the BUNDLED mel/emb/wakeword ONNX reproduce
openWakeWord's feature space, using our Python mirror of wakeWord.ts (owwfeat.py).

Pass criteria:
  * stock 'hey jarvis' head fires (high score) on say 'hey jarvis' clips through the bundled front-end
  * 'hello aria' + a distractor sentence score LOW on the stock head
  * downloaded validation frames (real negatives) score LOW on the stock head
If all hold, a Colab-trained 'Hello Aria' head (trained on openWakeWord features) will work
with these exact bundled front-end models on-device.
"""
import os
import numpy as np
from owwfeat import FrontEnd, Detector, say_to_wav, load_wav16, ASSETS, WW_WINDOW

HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE = os.path.join(HERE, "output", "smoke")
os.makedirs(SMOKE, exist_ok=True)

STOCK_WW = os.path.join(ASSETS, "wakeword.onnx")

VOICES = ["Samantha", "Daniel", "Karen", "Moira", "Tessa", "Fred", "Rishi", "Albert"]


def clip_max(det, text, voice, rate=None):
    wav = os.path.join(SMOKE, f"{voice}_{abs(hash((text,rate)))%9999}.wav")
    if not say_to_wav(text, voice, wav, rate):
        return None
    s = det.scores(load_wav16(wav))
    return float(s.max()) if len(s) else float("nan")


def main():
    front = FrontEnd()
    det = Detector(STOCK_WW, front)
    print("=== bundled model I/O ===")
    for name, sess in (("mel", front.mel), ("emb", front.emb), ("ww", det.ww)):
        i = sess.get_inputs()[0]
        o = sess.get_outputs()[0]
        print(f"  {name:3} in {i.name}{i.shape} -> out {o.name}{o.shape}")

    # embeddings/sec (needed later to convert validation windows -> hours for FP/hr)
    probe = os.path.join(SMOKE, "probe.wav")
    say_to_wav("one two three four five six seven eight nine ten", "Samantha", probe)
    x = load_wav16(probe)
    emb = front.embeddings(x)
    eps = len(emb) / (len(x) / 16000.0)
    print(f"\nembeddings/sec ~= {eps:.2f}  ({len(emb)} embeddings over {len(x)/16000:.2f}s)")

    print("\n=== POSITIVE for stock model: 'hey jarvis' (expect HIGH) ===")
    pos = []
    for v in VOICES:
        for r in (None, 160, 200):
            m = clip_max(det, "hey jarvis", v, r)
            if m is not None:
                pos.append(m)
                print(f"  hey jarvis / {v:9} r={str(r):4} -> {m:.3f}")

    print("\n=== NEGATIVE for stock model (expect LOW) ===")
    neg = []
    for text in ["hello aria", "hello maria", "what time is it", "turn on the television", "good morning everyone"]:
        for v in VOICES[:4]:
            m = clip_max(det, text, v)
            if m is not None:
                neg.append(m)
                print(f"  {text:22} / {v:9} -> {m:.3f}")

    print("\n=== SUMMARY (clips) ===")
    if pos:
        print(f"  'hey jarvis' max scores: min={min(pos):.3f} median={np.median(pos):.3f} max={max(pos):.3f}")
    if neg:
        print(f"  negative max scores:     min={min(neg):.3f} median={np.median(neg):.3f} max={max(neg):.3f}")
    ok_pos = pos and np.median(pos) > 0.5
    ok_neg = neg and max(neg) < 0.5
    print(f"  -> stock fires on 'hey jarvis': {'YES' if ok_pos else 'NO'} | stays quiet on negatives: {'YES' if ok_neg else 'NO'}")

    # Validation frames (real negatives) if downloaded
    valp = os.path.join(HERE, "validation_set_features.npy")
    if os.path.exists(valp) and os.path.getsize(valp) > 1_000_000:
        print("\n=== validation frames scored by stock model (expect very few > 0.5) ===")
        val = np.load(valp)                      # [N, 96] float32
        n = min(60000, val.shape[0])
        wins = np.stack([val[i:i + WW_WINDOW] for i in range(0, n - WW_WINDOW)]).astype(np.float32)
        sc = det.score_windows(wins)
        hrs = len(wins) / eps / 3600.0
        fp = int((sc > 0.5).sum())
        print(f"  windows={len(wins)}  (~{hrs:.2f} h)  max={sc.max():.3f}  mean={sc.mean():.4f}")
        print(f"  false accepts >0.5: {fp}  (~{fp/max(hrs,1e-9):.2f}/h)  >0.9: {int((sc>0.9).sum())}")
    else:
        print("\n(validation_set_features.npy not ready yet — re-run for the real-negative check)")


if __name__ == "__main__":
    main()
