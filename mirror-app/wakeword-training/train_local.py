#!/usr/bin/env python3
"""Train the "Hello Aria" head locally (Strategy B, fully controlled):
  Piper clips -> numpy augmentation (reverb/noise/offset) -> features via the BUNDLED front-end
  (whole-clip, matching openWakeWord's space + the downloaded negatives) -> openWakeWord-style DNN head
  trained vs 2000-hr real negatives with a negative-weight schedule -> export [1,16,96]->score ONNX.

Run after gen_local.py + data_prep.py finish:
  ./.venv/bin/python train_local.py
"""
import os, glob, json, time
import numpy as np
import torch, torch.nn as nn
from scipy.signal import fftconvolve
import soundfile as sf
from owwfeat import FrontEnd, Detector, ASSETS, WW_WINDOW

SR = 16000
TOTAL_LEN = 32000              # 2 s -> whole-clip -> 16 embeddings -> one [16,96] window
EMB_PER_SEC = 12.5             # openWakeWord whole-clip embedding rate (for FP/hour)
TARGET_FP = 0.2
RNG = np.random.default_rng(0)
torch.manual_seed(0)

front = FrontEnd()

# ---------------- augmentation resources ----------------
RIRS = [sf.read(f, dtype="float32")[0] for f in sorted(glob.glob("mit_rirs/*.wav"))]
BGS = [sf.read(f, dtype="float32")[0] for f in sorted(glob.glob("background/*.wav"))]
print(f"augmentation: {len(RIRS)} RIRs | {len(BGS)} background clips", flush=True)


def load16(path):
    x, sr = sf.read(path, dtype="float32")
    return x.mean(1) if x.ndim > 1 else x


def augment(x):
    """Utterance -> random reverb -> placed in a 2 s buffer at random offset -> background at random SNR.
    Returns raw-int16-valued float32 (the exact scale wakeWord.ts feeds the mel model)."""
    x = x.astype(np.float32)
    x = x / (np.max(np.abs(x)) + 1e-9) * RNG.uniform(0.3, 0.9)
    if RIRS and RNG.random() < 0.5:
        rir = RIRS[RNG.integers(len(RIRS))]
        x = fftconvolve(x, rir)
        x = x / (np.max(np.abs(x)) + 1e-9) * RNG.uniform(0.3, 0.9)
    buf = np.zeros(TOTAL_LEN, dtype=np.float32)
    L = min(len(x), TOTAL_LEN)
    off = int(RNG.integers(0, max(1, TOTAL_LEN - L)))
    buf[off:off + L] = x[:L]
    if BGS and RNG.random() < 0.7:
        bg = BGS[RNG.integers(len(BGS))]
        if len(bg) < TOTAL_LEN:
            bg = np.tile(bg, int(np.ceil(TOTAL_LEN / len(bg))))
        s = int(RNG.integers(0, max(1, len(bg) - TOTAL_LEN)))
        bg = bg[s:s + TOTAL_LEN]
        snr = RNG.uniform(3, 20)
        g = np.sqrt((np.mean(buf ** 2) + 1e-9) / (np.mean(bg ** 2) + 1e-9) / (10 ** (snr / 10)))
        buf = buf + g * bg
    peak = np.max(np.abs(buf)) + 1e-9
    if peak > 1:
        buf = buf / peak * 0.98
    return (buf * 32767.0).astype(np.int16).astype(np.float32)


def clips_to_features(wavs, rounds, desc):
    cache = f"feats_{desc}_r{rounds}_n{len(wavs)}.npy"
    if os.path.exists(cache):
        arr = np.load(cache)
        print(f"  {desc}: loaded {arr.shape} from {cache}", flush=True)
        return arr
    feats = []
    t = time.time()
    for r in range(rounds):
        for w in wavs:
            W = front.windows(front.embeddings_wholeclip(augment(load16(w))))
            if len(W):
                feats.append(W[0])
        print(f"  {desc} round {r + 1}/{rounds}: {len(feats)} feats ({time.time() - t:.0f}s)", flush=True)
    arr = np.asarray(feats, dtype=np.float32)
    np.save(cache, arr)
    return arr


# ---------------- build feature sets ----------------
# Two TTS engines: Piper (~800 voices, bulk) + macOS say (16 voices, over-rounded to ~10% weight).
# say voices Samantha/Daniel/Karen/Moira/Tessa are HELD OUT for validation — never trained on.
pos_train = clips_to_features(sorted(glob.glob("local_clips.noindex/pos_train/*.wav")), rounds=3, desc="pos_train")
pos_say = clips_to_features(sorted(glob.glob("local_clips.noindex/pos_say/*.wav")), rounds=8, desc="pos_say")
# say_eval: SAME trained voices, FRESH augmentations — measures whether the say engine got learned.
# (The 5 held-out voices stay untouched; they are only ever seen by validate_hello_aria.py.)
say_eval = clips_to_features(sorted(glob.glob("local_clips.noindex/pos_say/*.wav")), rounds=2, desc="say_eval")
pos_test = clips_to_features(sorted(glob.glob("local_clips.noindex/pos_test/*.wav")), rounds=1, desc="pos_test")
adv_neg = clips_to_features(sorted(glob.glob("local_clips.noindex/neg_adv/*.wav")), rounds=1, desc="adv_neg")
neg_say = clips_to_features(sorted(glob.glob("local_clips.noindex/neg_say/*.wav")), rounds=3, desc="neg_say")
if len(neg_say):
    adv_neg = np.concatenate([adv_neg, neg_say])
print(f"features: pos_train {pos_train.shape} | pos_test {pos_test.shape} | adv_neg {adv_neg.shape}", flush=True)
assert len(pos_train) > 100 and len(pos_test) > 20, "too few positives — did gen_local.py finish?"

ITEM = 16 * 96 * 2
Ksz = (os.path.getsize("neg.bin") - 128) // ITEM
neg_real = np.memmap("neg.bin", dtype=np.float16, mode="r", offset=128, shape=(Ksz, 16, 96))
print("neg_real windows:", neg_real.shape, flush=True)

val = np.load("validation_set_features.npy").astype(np.float32)
Nval = min(150_000, val.shape[0] - WW_WINDOW)
val_wins = np.stack([val[i:i + WW_WINDOW] for i in range(Nval)]).astype(np.float32)
val_hours = len(val_wins) / EMB_PER_SEC / 3600.0
print(f"validation windows: {len(val_wins)} (~{val_hours:.2f} h)", flush=True)


# ---------------- openWakeWord-style DNN head ----------------
class Net(nn.Module):
    def __init__(self, layer=96):
        super().__init__()
        self.flatten = nn.Flatten()
        self.l1, self.n1, self.r1 = nn.Linear(16 * 96, layer), nn.LayerNorm(layer), nn.ReLU()
        self.lb, self.nb, self.rb = nn.Linear(layer, layer), nn.LayerNorm(layer), nn.ReLU()
        self.lo, self.sig = nn.Linear(layer, 1), nn.Sigmoid()

    def forward(self, x):
        x = self.r1(self.n1(self.l1(self.flatten(x))))
        x = self.rb(self.nb(self.lb(x)))
        return self.sig(self.lo(x))


net = Net()
opt = torch.optim.Adam(net.parameters(), lr=1e-4)
bce = nn.BCELoss(reduction="none")
pos_t = torch.from_numpy(pos_train)                                   # Piper positives
say_t = torch.from_numpy(pos_say) if len(pos_say) else None           # say positives (2nd engine)
adv_t = torch.from_numpy(adv_neg)

# Explicit engine mix per batch: say gets 25% of positive slots so it actually gets LEARNED
# (at 10% implicit weight the best checkpoint was blind to it). Gentler negative weight (hold at
# 300 after warm-up) — the old 1->1500 ramp destabilized late training so selection always picked
# a step-500 underfit checkpoint.
STEPS, B_ADV, B_NEG, MAXW = 16000, 32, 512, 300
B_PIPER, B_SAY = (48, 16) if say_t is not None else (64, 0)
B_POS = B_PIPER + B_SAY


def sample_neg(n):
    idx = RNG.integers(0, neg_real.shape[0], size=n)
    return torch.from_numpy(np.asarray(neg_real[np.sort(idx)], dtype=np.float32))


pts_t = torch.from_numpy(pos_test)
sev_t = torch.from_numpy(say_eval) if len(say_eval) else None
THRS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

def op_point(model):
    """(combined, threshold, piper_recall, say_recall) — combined = min of the two engine recalls
    at the best FP-eligible threshold, so a checkpoint only wins by learning BOTH engines."""
    model.eval()
    with torch.no_grad():
        p = model(pts_t).squeeze(1).numpy()
        s = model(sev_t).squeeze(1).numpy() if sev_t is not None else None
        v = np.concatenate([model(torch.from_numpy(val_wins[i:i + 8192])).squeeze(1).numpy()
                            for i in range(0, len(val_wins), 8192)])
    model.train()
    best = (-1.0, None, 0.0, 0.0)
    for t in THRS:
        if (v >= t).sum() / max(val_hours, 1e-9) <= TARGET_FP:
            rp = float((p >= t).mean())
            rs = float((s >= t).mean()) if s is not None else rp
            comb = min(rp, rs)
            if comb > best[0]:
                best = (comb, t, rp, rs)
    return best

print("training...", flush=True)
t0 = time.time()
best_recall, best_thr, best_state = -1.0, None, None
net.train()
for step in range(STEPS):
    pi = RNG.integers(0, len(pos_t), B_PIPER)
    parts = [pos_t[pi]]
    if say_t is not None:
        parts.append(say_t[RNG.integers(0, len(say_t), B_SAY)])
    ai = RNG.integers(0, len(adv_t), B_ADV)
    parts += [adv_t[ai], sample_neg(B_NEG)]
    xb = torch.cat(parts, 0)
    yb = torch.cat([torch.ones(B_POS), torch.zeros(B_ADV + B_NEG)])
    negw = 1.0 + (MAXW - 1.0) * min(1.0, step / 4000)   # warm-up ramp, then hold at MAXW
    w = torch.cat([torch.ones(B_POS), torch.full((B_ADV + B_NEG,), negw)])
    opt.zero_grad()
    out = net(xb).squeeze(1)
    (bce(out, yb) * w).mean().backward()
    opt.step()
    if step % 500 == 0 and step > 0:
        r, thr, rp, rs = op_point(net)
        mark = ""
        if r > best_recall:
            best_recall, best_thr = r, thr
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            mark = "  <-- new best"
        print(f"  step {step:5d}  loss {(bce(out, yb) * w).mean().item():.4f}  negw {negw:.0f}"
              f"  min-recall@FP<={TARGET_FP}/h: {r:.3f} (piper {rp:.3f} / say {rs:.3f}, thr {thr}){mark}",
              flush=True)
if best_state is not None:
    net.load_state_dict(best_state)
    print(f"restored BEST checkpoint: recall {best_recall:.3f} @ thr {best_thr}", flush=True)
torch.save(net.state_dict(), "hello_aria_best.pt")   # crash-proof: weights survive any export failure
print(f"trained in {time.time() - t0:.0f}s", flush=True)

# ---------------- evaluate + choose threshold ----------------
net.eval()
with torch.no_grad():
    ps = net(pos_t).squeeze(1).numpy()                       # train recall (sanity)
    pts = net(torch.from_numpy(pos_test)).squeeze(1).numpy()  # held-out recall
    vs = np.concatenate([net(torch.from_numpy(val_wins[i:i + 4096])).squeeze(1).numpy()
                         for i in range(0, len(val_wins), 4096)])
print(f"\n{'thresh':>7} {'recall(test)':>12} {'FP/hour':>9}", flush=True)
rows, best = [], None
for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
    recall = float((pts >= t).mean())
    fp_hr = float((vs >= t).sum() / max(val_hours, 1e-9))
    rows.append({"threshold": t, "recall": round(recall, 4), "fp_per_hour": round(fp_hr, 3)})
    print(f"{t:>7.2f} {recall:>12.3f} {fp_hr:>9.3f}", flush=True)
    if fp_hr <= TARGET_FP and best is None:
        best = t
if best is None:
    best = 0.95
    print("[warn] no threshold met FP target; recommending 0.95", flush=True)
print(f"\nRECOMMENDED EXPO_PUBLIC_WAKE_WORD_THRESHOLD = {best}", flush=True)

# ---------------- export ONNX [1,16,96] -> score ----------------
torch.onnx.export(net, torch.rand(1, WW_WINDOW, 96), "hello_aria.onnx",
                  input_names=["x"], output_names=["score"], opset_version=13)
metrics = {"model": "hello_aria", "phrase": "Hello Aria", "recommended_threshold": best,
           "target_fp_per_hour": TARGET_FP, "validation_hours_used": round(val_hours, 2),
           "n_pos_train": len(pos_train), "n_pos_test": len(pos_test), "n_adv_neg": len(adv_neg),
           "n_real_neg": int(neg_real.shape[0]), "roc": rows,
           "pos_test_score_pct": {p: float(np.percentile(pts, p)) for p in (5, 25, 50, 75, 95)}}
json.dump(metrics, open("hello_aria_metrics.json", "w"), indent=2)
print("wrote hello_aria.onnx + hello_aria_metrics.json", flush=True)
print("DONE", flush=True)
