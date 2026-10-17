"""
voice_scoring.py
================
SIH26104 VoiceGuard -- file-level voice authenticity scoring for the
transaction-fraud integration.

    from voice_scoring import analyze_voice
    result = analyze_voice("call.wav", model_path="track_a.int8.onnx")

returns exactly this dict (JSON-serializable: plain bool / float):

    {
      "is_genuine":            False,  # bool
      "genuine_probability":   0.12,   # float in [0, 1]
      "synthetic_probability": 0.88,   # float in [0, 1]
      "audio_confidence":      0.92    # float in [0, 1]
    }

FIELD DEFINITIONS
-----------------
synthetic_probability
    Mean, over the speech windows of the file, of the Track A model's
    P(synthetic) -- the softmax of the two logits [bonafide, spoof] (class
    index 1 = spoof), as trained by train_track_a.py on ASVspoof 2019 LA.
    "Synthetic" means ANY machine-generated speech: text-to-speech (TTS) and
    voice conversion (VC) are NOT told apart (see CLASSES below).
    It is a raw model score, not a calibrated frequency: the network was
    trained with class weights on a 90%-spoof dataset, so 0.88 does not mean
    "88% of such calls are fake". Calibrate on your own labelled calls
    (e.g. temperature/Platt scaling) before treating it as a literal
    probability or setting a fraud-model weight on it.

genuine_probability
    1 - synthetic_probability. The model is a two-class softmax, so the two
    are NOT independent: they always sum to 1.0 (both rounded to 4 places).

is_genuine
    genuine_probability >= threshold (default 0.5), compared on the rounded
    values returned. Raise `threshold` to be stricter about calling a voice
    genuine. It is a decision on the model score only; it does NOT look at
    audio_confidence. Gate on audio_confidence yourself (see below).

audio_confidence
    0-1 score of how reliable the INPUT AUDIO is for this model. It is computed
    from the signal alone and never uses the model's output, so it is
    independent of genuine/synthetic. It is a heuristic quality score, NOT a
    probability that the verdict is correct. It is the weighted geometric mean
    of four components (weights in brackets):
      speech_duration [0.25]  seconds of active speech / 4 s, capped at 1
      snr             [0.25]  rough SNR (90th-percentile active-frame level minus
                              10th-percentile frame level): 5 dB -> 0, 30 dB -> 1.
                              Underestimates SNR for speech with no pauses.
      clipping        [0.15]  share of samples at full scale: 0% -> 1, >=1% -> 0
      bandwidth       [0.35]  is the audio really wideband (the model trained on 16 kHz
                              speech)? Looks for a sharp spectral cliff at 4 kHz
                              (power 3-4 kHz vs 5-8 kHz: <=20 dB drop -> 1, >=40 dB
                              -> 0.1) and caps the score when the file's own sample
                              rate is below 16 kHz (8 kHz telephone audio -> 0.1).
                              Telephone-band audio therefore lands near 0.45 overall.
    The mappings and weights are hand-chosen and have NOT been tuned against
    actual accuracy. Pass return_details=True to see every component, and
    validate by measuring error rate per audio_confidence bin on your data.
    Suggested use: treat verdicts with audio_confidence < 0.5 as "uncertain".

CLASSES
-------
The model has a BINARY head: genuine (bonafide) vs synthetic (spoof). It does
NOT output a separate voice-converted class. TTS and VC speech are both
"synthetic". The ASVspoof 2019 LA protocol files carry a per-utterance attack
id that would allow 3-class training (genuine / TTS / VC), but that has not
been done.

INPUT FORMAT
------------
audio_path : str | os.PathLike
    16-bit PCM WAV, any sample rate, mono or multi-channel (always readable).
    FLAC / OGG / float or 24-bit WAV are readable only if the optional
    `soundfile` package is installed. Anything else raises AudioFormatError.

PREPROCESSING (all done inside analyze_voice; nothing for the caller to do)
-------------
 1. Downmix to mono (mean of channels); scale to float32 in [-1, 1].
 2. Only the first MAX_SECONDS (30 s) are analysed.
 3. Resample to 16 kHz (scipy resample_poly). Training audio was 16 kHz.
 4. Split into 20 ms frames; mark a frame "active" if its level is above
    max(-50 dBFS, noise_floor + 4 dB), noise_floor = 10th-percentile level.
 5. Slide a 1.0 s window with a 0.5 s hop (training used 1.0 s windows); score
    only windows where >= 50% of frames are active.
 6. Per window: 20 LFCCs (dsp_pipeline.compute_lfcc, the same front-end used in
    training) -> ONNX model -> P(synthetic). Average over windows.

RAISES (so the fraud pipeline can route to manual review instead of getting a
meaningless score)
------
 AudioFormatError    missing / unreadable / unsupported file
 AudioTooShortError  shorter than 1.0 s
 NoSpeechError       no window with enough active speech
 ModelNotLoadedError no trained model loaded (see below)

MODEL
-----
model_path, else the TRACK_A_MODEL environment variable (the same variable the
server uses), pointing at the INT8/FP32 ONNX file from train_track_a.py. With
no loadable model the function RAISES, because the built-in DSP fallback is a
placeholder, not a detector. Pass allow_heuristic_fallback=True to use it
anyway (a RuntimeWarning is emitted; the numbers are NOT meaningful).

KNOWN LIMITS (be explicit about these in the fraud-model docs)
------------
 * Trained only on ASVspoof 2019 LA (English, clean 16 kHz, a fixed set of
   attack systems). Behaviour on telephony codecs, Indian languages, noise and
   newer voice-cloning systems is unmeasured.
 * That dataset has a known silence shortcut (silence length correlates with
   the label); a model can partly learn it. Scoring only active-speech windows
   reduces, but does not remove, the effect.
 * Scores from a model trained on real data must be calibrated; scores from
   the toy/synthetic-data model used in tests are meaningless.

Byte order: WAV samples come from Python's `wave` module, which returns native
byte order on both little- and big-endian hosts, so this module is safe on
IBM Z (s390x).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
import wave
from math import gcd
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.signal import resample_poly

from acoustic_model import AcousticModel
from dsp_pipeline import SAMPLE_RATE, compute_lfcc

__all__ = [
    "analyze_voice",
    "VoiceScoringError",
    "AudioFormatError",
    "AudioTooShortError",
    "NoSpeechError",
    "ModelNotLoadedError",
]

# ---- tunables ---------------------------------------------------------------
WINDOW_S = 1.0
HOP_S = 0.5
MIN_SECONDS = 1.0
MAX_SECONDS = 30.0
FRAME = 320                     # 20 ms @ 16 kHz
MIN_ACTIVE_FRACTION = 0.5       # share of active frames a window needs to be scored
ACTIVE_ABOVE_NOISE_DB = 4.0
ACTIVE_FLOOR_DBFS = -50.0

SPEECH_S_FULL = 4.0
SNR_DB_ZERO, SNR_DB_FULL = 5.0, 30.0
CLIP_FRACTION_ZERO = 0.01
BW_DROP_DB_FULL, BW_DROP_DB_ZERO = 20.0, 40.0   # 3-4 kHz vs 5-8 kHz power drop
BANDWIDTH_FLOOR = 0.1
CONFIDENCE_WEIGHTS = {"speech_duration": 0.25, "snr": 0.25, "clipping": 0.15, "bandwidth": 0.35}


# ---- errors -----------------------------------------------------------------
class VoiceScoringError(ValueError):
    """Base class for input problems that make a score meaningless."""


class AudioFormatError(VoiceScoringError):
    pass


class AudioTooShortError(VoiceScoringError):
    pass


class NoSpeechError(VoiceScoringError):
    pass


class ModelNotLoadedError(RuntimeError):
    pass


# ---- audio loading ----------------------------------------------------------
class _NotPcm16(Exception):
    pass


def _read_wav_pcm16(path: Path) -> tuple[np.ndarray, int, float]:
    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise _NotPcm16()
        sr, ch = wf.getframerate(), wf.getnchannels()
        if sr <= 0 or ch < 1:
            raise AudioFormatError("WAV header has an invalid sample rate or channel count")
        raw = wf.readframes(min(wf.getnframes(), int(MAX_SECONDS * sr)))
    pcm = np.frombuffer(raw, dtype=np.int16)  # wave returns NATIVE byte order, also on big-endian
    pcm = pcm[: len(pcm) // ch * ch].reshape(-1, ch)
    clip_fraction = float(np.mean(np.abs(pcm.astype(np.int32)) >= 32700)) if pcm.size else 0.0
    mono = pcm.astype(np.float32).mean(axis=1) / 32768.0
    return mono.astype(np.float32), sr, clip_fraction


def _read_with_soundfile(path: Path) -> tuple[np.ndarray, int, float]:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise AudioFormatError(
            "Unsupported audio format. 16-bit PCM WAV is always supported; for FLAC/OGG/float "
            "or 24-bit WAV install the optional 'soundfile' package."
        ) from exc
    try:
        with sf.SoundFile(str(path)) as f:
            sr = f.samplerate
            data = f.read(frames=int(MAX_SECONDS * sr), dtype="float32", always_2d=True)
    except Exception as exc:  # libsndfile raises RuntimeError / LibsndfileError
        raise AudioFormatError(f"Could not decode audio file: {exc}") from exc
    clip_fraction = float(np.mean(np.abs(data) >= 0.999)) if data.size else 0.0
    return data.mean(axis=1).astype(np.float32), int(sr), clip_fraction


def _load_audio(path: Path) -> tuple[np.ndarray, int, float]:
    if not path.is_file():
        raise AudioFormatError(f"audio file not found: {path}")
    if path.suffix.lower() in (".wav", ".wave"):
        try:
            return _read_wav_pcm16(path)
        except (_NotPcm16, wave.Error, EOFError):
            pass  # float / 24-bit / odd WAV: try soundfile
    return _read_with_soundfile(path)


def _to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == SAMPLE_RATE:
        return x
    g = gcd(SAMPLE_RATE, sr)
    return resample_poly(x, SAMPLE_RATE // g, sr // g).astype(np.float32)


# ---- audio quality (independent of the model) --------------------------------
def _clip01(v: float) -> float:
    return float(min(1.0, max(0.0, v)))


def _quality(frame_db: np.ndarray, frames: np.ndarray, active: np.ndarray, noise_db: float,
             clip_fraction: float, sr_in: int) -> tuple[float, dict[str, float], dict[str, float]]:
    speech_s = float(active.sum() * FRAME / SAMPLE_RATE)
    snr_db = float(np.clip(np.percentile(frame_db[active], 90) - noise_db, -20.0, 60.0))

    act = frames[active]
    power = (np.abs(np.fft.rfft(act * np.hanning(FRAME), n=512, axis=1)) ** 2).mean(axis=0)
    freqs = np.fft.rfftfreq(512, 1.0 / SAMPLE_RATE)
    p_mid = power[(freqs >= 3000) & (freqs < 4000)].sum()
    p_high = power[(freqs >= 5000) & (freqs <= 8000)].sum()
    drop_db = float(10.0 * math.log10((p_mid + 1e-20) / (p_high + 1e-20)))
    bw = BANDWIDTH_FLOOR + (1.0 - BANDWIDTH_FLOOR) * _clip01(
        (BW_DROP_DB_ZERO - drop_db) / (BW_DROP_DB_ZERO - BW_DROP_DB_FULL))
    if sr_in < SAMPLE_RATE:  # the file itself cannot hold content above its Nyquist
        cap = BANDWIDTH_FLOOR + (1.0 - BANDWIDTH_FLOOR) * _clip01((sr_in / 2.0 - 4000.0) / 4000.0)
        bw = min(bw, cap)

    comps = {
        "speech_duration": _clip01(speech_s / SPEECH_S_FULL),
        "snr": _clip01((snr_db - SNR_DB_ZERO) / (SNR_DB_FULL - SNR_DB_ZERO)),
        "clipping": _clip01(1.0 - clip_fraction / CLIP_FRACTION_ZERO),
        "bandwidth": _clip01(bw),
    }
    log_sum = sum(CONFIDENCE_WEIGHTS[k] * math.log(max(v, 1e-6)) for k, v in comps.items())
    measures = {"speech_seconds": speech_s, "snr_db": snr_db, "clipped_fraction": clip_fraction,
                "spectral_drop_3to4k_vs_5to8k_db": drop_db}
    return _clip01(math.exp(log_sum)), comps, measures


# ---- model resolution ---------------------------------------------------------
_MODEL_CACHE: dict[str, AcousticModel] = {}


def _resolve_model(model: Any, model_path: Optional[str], allow_fallback: bool) -> tuple[Any, str]:
    if model is not None:
        return model, getattr(model, "backend_name", type(model).__name__)

    path = model_path or os.environ.get("TRACK_A_MODEL")
    problem = "no model path given and TRACK_A_MODEL is not set"
    if path:
        key = os.path.abspath(os.fspath(path))
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached, f"onnx:{os.path.basename(key)}"
        loaded = AcousticModel(model_path=key)
        if not loaded.using_fallback:
            _MODEL_CACHE[key] = loaded
            return loaded, f"onnx:{os.path.basename(key)}"
        problem = (f"could not load '{key}' (file missing, or onnxruntime is not installed)")

    if allow_fallback:
        warnings.warn(
            f"VoiceGuard: {problem}. Using the built-in DSP heuristic, which is a placeholder and "
            "NOT a trained detector; the returned probabilities are not meaningful.",
            RuntimeWarning, stacklevel=3)
        return AcousticModel(model_path=None), "dsp-heuristic-fallback"
    raise ModelNotLoadedError(
        f"No trained Track A model loaded: {problem}. Pass model_path=... or set TRACK_A_MODEL to the "
        "ONNX file from train_track_a.py (or allow_heuristic_fallback=True for a non-meaningful placeholder).")


# ---- public API ----------------------------------------------------------------
def analyze_voice(
    audio_path: "str | os.PathLike[str]",
    *,
    model_path: Optional[str] = None,
    threshold: float = 0.5,
    return_details: bool = False,
    allow_heuristic_fallback: bool = False,
    model: Any = None,
) -> dict[str, Any]:
    """
    Score one audio file. Returns the 4-field dict described in the module
    docstring; with return_details=True a fifth key "details" is added.

    threshold                  is_genuine = genuine_probability >= threshold
    model_path                 ONNX file; defaults to $TRACK_A_MODEL
    allow_heuristic_fallback   use the placeholder DSP heuristic if no model loads
    model                      advanced/testing: any object with
                               predict_vocoder_probability(lfcc) -> P(synthetic)
    """
    if not (0.0 < float(threshold) < 1.0):
        raise ValueError("threshold must be strictly between 0 and 1")
    scorer, backend = _resolve_model(model, model_path, allow_heuristic_fallback)

    raw, sr_in, clip_fraction = _load_audio(Path(os.fspath(audio_path)))
    x = _to_16k(raw, sr_in)
    duration_s = len(x) / SAMPLE_RATE
    if duration_s < MIN_SECONDS:
        raise AudioTooShortError(f"audio is {duration_s:.2f} s; at least {MIN_SECONDS:.1f} s is required")
    x = np.clip(x[: int(MAX_SECONDS * SAMPLE_RATE)], -1.0, 1.0).astype(np.float32)

    n_frames = len(x) // FRAME
    frames = x[: n_frames * FRAME].reshape(n_frames, FRAME)
    frame_db = 20.0 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-10)
    noise_db = float(np.percentile(frame_db, 10))
    active = frame_db > max(noise_db + ACTIVE_ABOVE_NOISE_DB, ACTIVE_FLOOR_DBFS)

    win, hop = int(WINDOW_S * SAMPLE_RATE), int(HOP_S * SAMPLE_RATE)
    frames_per_win = win // FRAME
    probs: list[float] = []
    n_windows = 0
    for start in range(0, len(x) - win + 1, hop):
        n_windows += 1
        f0 = start // FRAME
        if active[f0:f0 + frames_per_win].mean() < MIN_ACTIVE_FRACTION:
            continue
        p = float(scorer.predict_vocoder_probability(compute_lfcc(x[start:start + win])))
        if not math.isfinite(p):
            raise VoiceScoringError("model returned a non-finite score")
        probs.append(_clip01(p))
    if not probs:
        raise NoSpeechError("no window contains enough active speech above the noise floor (silence, or SNR too low to find the speech)")

    synthetic = round(float(np.mean(probs)), 4)
    genuine = round(1.0 - synthetic, 4)
    confidence, comps, measures = _quality(frame_db, frames, active, noise_db, clip_fraction, sr_in)

    result: dict[str, Any] = {
        "is_genuine": bool(genuine >= threshold),
        "genuine_probability": float(genuine),
        "synthetic_probability": float(synthetic),
        "audio_confidence": round(float(confidence), 4),
    }
    if return_details:
        result["details"] = {
            "backend": backend,
            "threshold": float(threshold),
            "duration_seconds": round(duration_s, 3),
            "input_sample_rate": int(sr_in),
            "windows_total": n_windows,
            "windows_scored": len(probs),
            "per_window_synthetic": [round(p, 4) for p in probs],
            "window_std": round(float(np.std(probs)), 4),
            "confidence_components": {k: round(v, 4) for k, v in comps.items()},
            "measurements": {k: round(float(v), 4) for k, v in measures.items()},
        }
    return result


def _main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Score one audio file with VoiceGuard Track A.")
    ap.add_argument("audio")
    ap.add_argument("--model", default=None, help="ONNX model (default: $TRACK_A_MODEL)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--details", action="store_true")
    ap.add_argument("--allow-heuristic-fallback", action="store_true")
    args = ap.parse_args(argv)
    try:
        out = analyze_voice(args.audio, model_path=args.model, threshold=args.threshold,
                            return_details=args.details,
                            allow_heuristic_fallback=args.allow_heuristic_fallback)
    except (VoiceScoringError, ModelNotLoadedError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
