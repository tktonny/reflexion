"""Reproduce the EXACT on-device wake-word front-end (mirror-app/src/native/wakeWord.ts) in Python.

Pipeline (openWakeWord specs, mirrored byte-for-byte from wakeWord.ts):
  16 kHz 16-bit PCM (raw int16 values fed as float32, NOT normalized)
    -> melspectrogram.onnx  ([1,1280] chunks -> (frames,32), scaled x/10+2)
    -> embedding_model.onnx ([1,76,32,1] windows, stride 8 -> 96-dim)
    -> wakeword.onnx        ([1,16,96] -> score)

Computing training features with this SAME code guarantees train/infer parity with the device.
"""
from __future__ import annotations
import os
import subprocess
import tempfile
import numpy as np
import onnxruntime as ort

MEL_BINS = 32
EMB_WINDOW = 76      # mel frames per embedding window
EMB_STRIDE = 8       # mel-frame step between successive embeddings
WW_WINDOW = 16       # embeddings per wakeword prediction
CHUNK = 1280         # samples per melspec call (80 ms @ 16 kHz)

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.abspath(os.path.join(HERE, "..", "assets", "wakeword"))


def _sess(path: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, (os.cpu_count() or 2))
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])


class FrontEnd:
    """mel + embedding front-end (shared, never retrained)."""

    def __init__(self, assets_dir: str = ASSETS):
        self.mel = _sess(os.path.join(assets_dir, "melspectrogram.onnx"))
        self.emb = _sess(os.path.join(assets_dir, "embedding_model.onnx"))
        self.mel_in, self.mel_out = self.mel.get_inputs()[0].name, self.mel.get_outputs()[0].name
        self.emb_in, self.emb_out = self.emb.get_inputs()[0].name, self.emb.get_outputs()[0].name

    def embeddings(self, samples: np.ndarray) -> np.ndarray:
        """samples: 1D float32 of RAW int16 values. Returns [T, 96] embeddings (stride-8)."""
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        mel_frames = []
        i = 0
        n = len(samples)
        while i + CHUNK <= n:                     # drop trailing <1280 remainder, exactly like the app
            chunk = samples[i:i + CHUNK]
            i += CHUNK
            out = self.mel.run([self.mel_out], {self.mel_in: chunk.reshape(1, CHUNK)})[0]
            data = np.asarray(out, dtype=np.float32).reshape(-1)
            frames = len(data) // MEL_BINS
            for f in range(frames):
                mel_frames.append(data[f * MEL_BINS:(f + 1) * MEL_BINS] / 10.0 + 2.0)
        if len(mel_frames) < EMB_WINDOW:
            return np.zeros((0, 96), dtype=np.float32)
        mel = np.asarray(mel_frames, dtype=np.float32)          # [F, 32]
        embs = []
        start = 0
        while len(mel) >= start + EMB_WINDOW:
            win = mel[start:start + EMB_WINDOW].reshape(1, EMB_WINDOW, MEL_BINS, 1)
            e = self.emb.run([self.emb_out], {self.emb_in: win})[0]
            embs.append(np.asarray(e, dtype=np.float32).reshape(-1))   # 96
            start += EMB_STRIDE
        return np.asarray(embs, dtype=np.float32)               # [T, 96]

    @staticmethod
    def windows(embs: np.ndarray) -> np.ndarray:
        """[T,96] -> [W,16,96] sliding windows (stride 1). Empty if T<16."""
        if len(embs) < WW_WINDOW:
            return np.zeros((0, WW_WINDOW, 96), dtype=np.float32)
        return np.stack([embs[i:i + WW_WINDOW] for i in range(0, len(embs) - WW_WINDOW + 1)]).astype(np.float32)


class Detector:
    """Full pipeline with a given wakeword head; returns per-window scores."""

    def __init__(self, ww_path: str, front: FrontEnd | None = None):
        self.front = front or FrontEnd()
        self.ww = _sess(ww_path)
        self.ww_in, self.ww_out = self.ww.get_inputs()[0].name, self.ww.get_outputs()[0].name
        self.ww_in_shape = self.ww.get_inputs()[0].shape

    def score_windows(self, wins: np.ndarray) -> np.ndarray:
        if len(wins) == 0:
            return np.zeros((0,), dtype=np.float32)
        # Handle fixed batch=1 exports by looping; dynamic batch runs in one shot.
        batchable = not isinstance(self.ww_in_shape[0], int) or self.ww_in_shape[0] in (None, -1) or self.ww_in_shape[0] == 0
        if batchable:
            try:
                out = self.ww.run([self.ww_out], {self.ww_in: wins.astype(np.float32)})[0]
                return np.asarray(out, dtype=np.float32).reshape(len(wins), -1)[:, -1]
            except Exception:
                pass
        scores = np.empty(len(wins), dtype=np.float32)
        for k in range(len(wins)):
            out = self.ww.run([self.ww_out], {self.ww_in: wins[k:k + 1].astype(np.float32)})[0]
            scores[k] = np.asarray(out, dtype=np.float32).reshape(-1)[-1]
        return scores

    def scores(self, samples: np.ndarray) -> np.ndarray:
        return self.score_windows(self.front.windows(self.front.embeddings(samples)))


def load_wav16(path: str) -> np.ndarray:
    """Read a 16 kHz mono WAV as float32 of RAW int16 values (matches wakeWord.ts feed())."""
    import soundfile as sf
    data, sr = sf.read(path, dtype="int16")
    if data.ndim > 1:
        data = data[:, 0]
    if sr != 16000:
        raise ValueError(f"{path}: expected 16 kHz, got {sr}")
    return data.astype(np.float32)


def say_to_wav(text: str, voice: str, wav_path: str, rate: int | None = None) -> bool:
    """Render `text` with macOS `say` (voice, optional words/min) -> 16 kHz mono s16 WAV. False on failure."""
    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
        aiff = tmp.name
    try:
        cmd = ["say", "-v", voice, "-o", aiff]
        if rate:
            cmd += ["-r", str(rate)]
        cmd.append(text)
        if subprocess.run(cmd, capture_output=True).returncode != 0:
            return False
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", aiff,
             "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", wav_path],
            capture_output=True,
        )
        return r.returncode == 0 and os.path.exists(wav_path)
    finally:
        try:
            os.remove(aiff)
        except OSError:
            pass
