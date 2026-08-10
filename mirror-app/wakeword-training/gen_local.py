#!/usr/bin/env python3
"""Generate 'Hello Aria' positives + adversarial/near-miss negatives locally via the Piper LibriTTS
(~900-speaker) generator (dscripka fork). Run with the espeak env set:

  DYLD_LIBRARY_PATH=/opt/homebrew/lib ESPEAK_DATA_PATH=/opt/homebrew/share/espeak-ng-data \
    ./.venv/bin/python gen_local.py
"""
import sys, os, time
sys.path.insert(0, os.path.abspath("piper-sample-generator"))
import torch
# Run VITS on Apple's GPU: the generator only special-cases CUDA, so alias those paths to MPS.
# Measured ~10x over the contended CPU (0.94 s/clip). Launch with PYTORCH_ENABLE_MPS_FALLBACK=1.
if torch.backends.mps.is_available():
    torch.cuda.is_available = lambda: True
    torch.Tensor.cuda = lambda self, *a, **k: self.to("mps")
    torch.nn.Module.cuda = lambda self, *a, **k: self.to("mps")
    torch.cuda.empty_cache = lambda: None
    print("[gen] using Apple GPU (MPS)", flush=True)
from generate_samples import generate_samples

OUT = "local_clips.noindex"   # .noindex = Spotlight skips it (indexing storms killed gen throughput)

def gen(text, n, subdir, batch=16):
    d = os.path.join(OUT, subdir)
    os.makedirs(d, exist_ok=True)
    have = len([f for f in os.listdir(d) if f.endswith(".wav")])
    if have >= 0.95 * n:
        print(f"[{subdir}] already {have}, skip", flush=True)
        return
    t = time.time()
    generate_samples(
        text=text if isinstance(text, list) else [text],
        max_samples=n - have, batch_size=batch,
        # broad speaker + prosody variety; cap speakers <904 (later ones have artifacts)
        max_speakers=800,
        noise_scales=[0.667, 0.85, 1.0], noise_scale_ws=[0.8], length_scales=[0.8, 1.0, 1.2],
        output_dir=d, auto_reduce_batch_size=True,
        # resume-safe: name new clips AFTER the ones already on disk (never overwrite progress)
        file_names=[f"{subdir}_{i:06d}.wav" for i in range(have, n)],
    )
    got = len([f for f in os.listdir(d) if f.endswith(".wav")])
    print(f"[{subdir}] -> {got} clips in {time.time()-t:.0f}s", flush=True)

# Positives (1500 is enough: augmentation rounds in train_local.py multiply acoustic variety;
# max_speakers=800 still gives wide voice coverage at this count)
gen(["hello aria"], 1500, "pos_train", batch=32)
gen(["hello aria"], 300, "pos_test", batch=32)

# Adversarial / near-miss negatives: speech that must NOT trigger (near-misses + everyday elderly-home
# phrases). The bulk of negative coverage comes from the 2000-hr real-audio features; these harden
# precision on confusables.
NEG = ["hello maria", "hi aria", "hello area", "hello arya", "hello ariel", "hello ara", "yellow aria",
       "hello sara", "hello clara", "hello darya", "hello mira", "hola aria", "hello aria's",
       "how are you", "what time is it", "turn on the light", "good morning", "good night", "i love you",
       "call my daughter", "where are my glasses", "what's the weather", "play some music", "thank you",
       "hello there", "hey google", "okay", "yes please", "no thanks", "see you later", "let me think",
       "the television is too loud", "dinner is ready", "take your medicine", "it is cold today",
       "open the door", "close the window", "read the news", "set an alarm", "how is everyone"]
gen(NEG, 800, "neg_adv", batch=32)
print("GEN DONE", flush=True)
