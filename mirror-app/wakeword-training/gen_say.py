#!/usr/bin/env python3
"""Add macOS `say` voice diversity to training (second TTS engine; openWakeWord's own guidance is
that more engines beats more clips from one engine).

VOICE-DISJOINT EVAL: Samantha/Daniel/Karen/Moira/Tessa are NEVER used for training — they stay a
pure held-out generalization probe in validate_hello_aria.py.
"""
import os, subprocess, sys
from owwfeat import say_to_wav

TRAIN_VOICES = ["Albert", "Aman", "Eddy", "Flo", "Fred", "Grandma", "Grandpa", "Junior",
                "Kathy", "Ralph", "Reed", "Rishi", "Rocko", "Sandy", "Shelley", "Tara"]
RATES = [None, 150, 180, 210]

NEG_PHRASES = ["hello maria", "hello area", "hi aria", "hello ariel", "hello arya",
               "yellow aria", "hello sara", "yellow area rug", "hello there", "hola aria"]

def gen(texts, voices, rates, outdir):
    os.makedirs(outdir, exist_ok=True)
    made = 0
    for t in texts:
        for v in voices:
            for r in rates:
                name = f"{outdir}/{v}_{r or 'def'}_{abs(hash(t)) % 9999}.wav"
                if os.path.exists(name):
                    made += 1
                    continue
                if say_to_wav(t, v, name, r):
                    made += 1
    return made

n_pos = gen(["Hello Aria"], TRAIN_VOICES, RATES, "local_clips.noindex/pos_say")
n_neg = gen(NEG_PHRASES, TRAIN_VOICES[:8], [None, 180], "local_clips.noindex/neg_say")
print(f"say clips: {n_pos} positives | {n_neg} negatives")
