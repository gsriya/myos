#!/usr/bin/env python3
"""
MyOS voice service: speaks any text in YOUR voice, cloned from a short recording.

Runs on your Mac only (Apple Silicon, F5-TTS via MLX). Nothing is uploaded anywhere.
It is started automatically by server.py, using the separate .voice-venv Python.

    GET  /health  -> {"ready": bool, "has_voice": bool}
    POST /tts     {"text": "..."} -> audio/wav in your voice
"""
from __future__ import annotations

import io
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mlx.core as mx
import numpy as np
import soundfile as sf
from f5_tts_mlx.cfm import F5TTS
from f5_tts_mlx.utils import convert_char_to_pinyin

HERE = os.path.dirname(os.path.abspath(__file__))
VOICE_DIR = os.path.join(HERE, "voice")
REF_WAV = os.path.join(VOICE_DIR, "ref.wav")
REF_TXT = os.path.join(VOICE_DIR, "ref.txt")
PORT = int(os.getenv("MYOS_VOICE_PORT", "8766"))
SR = 24_000
TARGET_RMS = 0.1

MODELS = os.path.join(VOICE_DIR, "models")
KOKORO_FILES = (os.path.join(MODELS, "kokoro-v1.0.onnx"), os.path.join(MODELS, "voices-v1.0.bin"))
kokoro = {"k": None}
klock = threading.Lock()


def natural_available() -> bool:
    return all(os.path.exists(f) for f in KOKORO_FILES)


def synth_natural(text: str, voice: str) -> bytes:
    """Fast, natural-sounding built-in voices (Kokoro)."""
    with klock:
        if kokoro["k"] is None:
            from kokoro_onnx import Kokoro
            kokoro["k"] = Kokoro(*KOKORO_FILES)
        samples, sr = kokoro["k"].create(text, voice=voice, speed=1.0,
                                         lang="en-gb" if voice.startswith("b") else "en-us")
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


state = {"model": None, "ref": None, "ref_text": "", "ref_mtime": 0.0, "ref_gain": 1.0}
lock = threading.Lock()


def load_model() -> None:
    t = time.time()
    state["model"] = F5TTS.from_pretrained("lucasnewman/f5-tts-mlx")
    print(f"voice model loaded in {time.time() - t:.1f}s", flush=True)


def load_ref() -> bool:
    """(Re)load your voice sample whenever the file changes."""
    if not (os.path.exists(REF_WAV) and os.path.exists(REF_TXT)):
        return False
    mtime = os.path.getmtime(REF_WAV)
    if mtime != state["ref_mtime"]:
        audio, sr = sf.read(REF_WAV, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            raise ValueError("voice sample must be 24 kHz")
        a = mx.array(audio)
        rms = float(mx.sqrt(mx.mean(mx.square(a))))
        gain = 1.0
        if 0 < rms < TARGET_RMS:
            gain = TARGET_RMS / rms
            a = a * gain
        state.update(ref=a, ref_text=open(REF_TXT).read().strip(), ref_mtime=mtime, ref_gain=gain)
        print(f"loaded your voice sample ({audio.shape[0] / SR:.1f}s)", flush=True)
    return True


def synth(text: str) -> bytes:
    with lock:
        if state["model"] is None:      # the cloning model is only loaded if you use "My voice"
            load_model()
        if not load_ref():
            raise RuntimeError("no voice sample recorded yet")
        ref = state["ref"]
        full = convert_char_to_pinyin([state["ref_text"] + " " + text])
        wave, _ = state["model"].sample(
            mx.expand_dims(ref, axis=0), text=full, duration=None,
            steps=8, method="euler", speed=1.0, cfg_strength=2.0, sway_sampling_coef=-1.0, seed=None,
        )
        wave = wave[ref.shape[0]:]           # drop the copy of your sample
        mx.eval(wave)
        out = np.array(wave) / state["ref_gain"]   # back to your natural loudness
        peak = float(np.max(np.abs(out))) if out.size else 0
        if peak > 0.99:
            out = out * (0.99 / peak)
        buf = io.BytesIO()
        sf.write(buf, out.astype(np.float32), SR, format="WAV", subtype="PCM_16")
        return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"ready": True, "natural": natural_available(),
                             "has_voice": os.path.exists(REF_WAV) and os.path.exists(REF_TXT)})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/tts":
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n) or b"{}")
            text = (req.get("text") or "").strip()
            voice = req.get("voice") or "mine"
            if not text:
                return self._json(400, {"error": "no text"})
            t = time.time()
            wav = synth_natural(text[:600], voice[4:]) if voice.startswith("nat:") else synth(text[:400])
            print(f"spoke {len(text)} chars in {time.time() - t:.1f}s", flush=True)
        except Exception as e:
            print("tts error:", e, flush=True)
            return self._json(503, {"error": str(e)})
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav)))
        self.end_headers()
        self.wfile.write(wav)

    def log_message(self, *args):  # keep the log quiet
        pass


if __name__ == "__main__":
    os.makedirs(VOICE_DIR, exist_ok=True)
    if natural_available():  # warm up so the first sentence is instant
        threading.Thread(target=lambda: synth_natural("Hello.", "af_heart"), daemon=True).start()
    print(f"MyOS voice service on http://127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
