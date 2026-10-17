"""
train_track_a.py
=================
Trains a lightweight LFCC-based spoof/vocoder classifier for SIH26104
Track A — a real replacement for the DSP-heuristic placeholder in
`app/acoustic_model.py`.

FEATURE / MODEL CONTRACT (locked to what acoustic_model.py already expects
— do not change these without also changing dsp_pipeline.py / acoustic_model.py):
    - 16 kHz mono float32 audio, 1.0s windows (16000 samples)
    - LFCC via app/dsp_pipeline.py::compute_lfcc -> shape (98, 20) per window
      (N_FFT=512, HOP=160, WIN=400, 40 linear filters, 20 LFCC via DCT-II)
    - ONNX input: (batch, 98, 20) float32
    - ONNX output: 2 logits [bonafide, spoof] (raw logits, NOT softmaxed —
      acoustic_model.py applies softmax itself and reads probs[-1])

DATASET: ASVspoof2019 LA (registration required, not fetchable from a
sandbox — download on your own machine):
    https://datashare.ed.ac.uk/handle/10283/3336
Expected layout after extraction (adjust --data-root if yours differs):
    ASVspoof2019_LA_train/flac/*.flac
    ASVspoof2019_LA_dev/flac/*.flac
    ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt
    ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt
Protocol line format (whitespace-separated):
    SPEAKER_ID  AUDIO_FILE_NAME  -  SYSTEM_ID  KEY
where KEY is "bonafide" or "spoof" (last column).

INSTALL (training only — does not touch the server's requirements.txt):
    # RTX 50-series (Blackwell, e.g. 5070 Ti) needs a CUDA 12.8+ build of PyTorch;
    # older cu121/cu118 wheels fail with "no kernel image is available".
    pip install torch --index-url https://download.pytorch.org/whl/cu128   # or the newest cuXXX shown on pytorch.org
    pip install soundfile scikit-learn onnx onnxruntime
    # Verify the GPU is used:  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

USAGE:
    # Train
    python train_track_a.py --data-root /path/to/ASVspoof2019_root --epochs 20 \
        --out-dir runs/track_a

    # Export best checkpoint to ONNX (+ optional INT8 quantization)
    python train_track_a.py --export-only --checkpoint runs/track_a/best.pt \
        --onnx-out track_a.onnx --quantize

Then run the server with the model switched on (needs the env-var patch in
s390x-endian-and-model-env.patch, or edit server.py's AcousticModel(model_path=...)):
    TRACK_A_MODEL=track_a.int8.onnx uvicorn server:app --app-dir app --port 8000
Check /healthz: "using_model_fallback" must read false.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# --- Reuse the server's own LFCC front-end so train/serve features never drift ---
def _find_app_dir() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [os.environ.get("VOICEGUARD_APP_DIR"), here / "app", here.parent / "app", Path.cwd() / "app"]
    for c in candidates:
        if c and (Path(c) / "dsp_pipeline.py").exists():
            return Path(c)
    sys.exit("Cannot find app/dsp_pipeline.py. Put this script next to the repo's app/ folder "
             "or set VOICEGUARD_APP_DIR=/path/to/repo/app")


sys.path.insert(0, str(_find_app_dir()))
from dsp_pipeline import compute_lfcc, SAMPLE_RATE  # noqa: E402

WINDOW_SAMPLES = SAMPLE_RATE  # 1.0s, matches the server's ring-buffer stride
N_FRAMES, N_LFCC = 98, 20  # locked shape — see module docstring


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def _parse_protocol(protocol_path: Path) -> dict[str, int]:
    """Returns {utt_id: label} with label 1=spoof, 0=bonafide."""
    labels = {}
    with open(protocol_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            utt_id, key = parts[1], parts[-1]
            labels[utt_id] = 0 if key == "bonafide" else 1
    return labels


class ASVspoofWindows:
    """
    Lazily slices each utterance into non-overlapping 1.0s windows and
    computes LFCC per window. Utterances shorter than 1.0s are zero-padded
    once (matches dsp_pipeline's own padding behavior for short buffers).
    """

    def __init__(self, flac_dir: Path, protocol_path: Path):
        import soundfile as sf

        self.labels = _parse_protocol(protocol_path)
        self.items: list[tuple[Path, int, int]] = []  # (path, window_idx, label)

        for utt_id, label in self.labels.items():
            path = flac_dir / f"{utt_id}.flac"
            if not path.exists():
                continue
            info = sf.info(str(path))
            n_windows = max(1, info.frames // WINDOW_SAMPLES)
            for w in range(n_windows):
                self.items.append((path, w, label))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        import soundfile as sf  # imported here so the Dataset stays picklable for spawn workers

        path, w, label = self.items[idx]
        audio, sr = sf.read(str(path), dtype="float32")
        if sr != SAMPLE_RATE:
            raise ValueError(f"{path} is {sr} Hz, expected {SAMPLE_RATE} Hz — resample your data first.")
        start = w * WINDOW_SAMPLES
        chunk = audio[start:start + WINDOW_SAMPLES]
        if len(chunk) < WINDOW_SAMPLES:
            chunk = np.pad(chunk, (0, WINDOW_SAMPLES - len(chunk)))
        lfcc = compute_lfcc(chunk)  # (98, 20)
        return lfcc, label


def _collate(batch):
    import torch

    x = torch.from_numpy(np.stack([b[0] for b in batch]))
    y = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return x, y


def make_loader(flac_dir, protocol_path, batch_size, shuffle, workers):
    from torch.utils.data import DataLoader

    ds = ASVspoofWindows(Path(flac_dir), Path(protocol_path))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      persistent_workers=workers > 0, collate_fn=_collate, drop_last=shuffle)


# ---------------------------------------------------------------------------
# Model — small LCNN-style CNN over the (98, 20) LFCC map.
# Compact and fast on a single mobile GPU; a well-precedented architecture
# family for LFCC-based spoof detection (lighter than full AASIST/RawNet3,
# and doesn't require raw-waveform SincNet front-ends).
# ---------------------------------------------------------------------------
def build_model():
    import torch
    import torch.nn as nn

    class MFM(nn.Module):
        """Max-Feature-Map: split channels in half, take element-wise max."""
        def forward(self, x):
            a, b = x.chunk(2, dim=1)
            return torch.maximum(a, b)

    class LCNN(nn.Module):
        def __init__(self, n_lfcc=N_LFCC, n_classes=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 32, 5, padding=2), MFM(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 64, 3, padding=1), MFM(),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 96, 3, padding=1), MFM(),
                nn.BatchNorm2d(48),
                nn.MaxPool2d(2),
                nn.Conv2d(48, 64, 3, padding=1), MFM(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Sequential(
                nn.Flatten(), nn.Dropout(0.3),
                nn.Linear(32, 64), nn.ReLU(),
                nn.Linear(64, n_classes),
            )

        def forward(self, x):  # x: (B, 98, 20)
            x = x.unsqueeze(1)  # (B, 1, 98, 20)
            return self.head(self.net(x))

    return LCNN()


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2)


def train(args):
    import torch
    import torch.nn as nn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    root = Path(args.data_root)
    train_loader = make_loader(
        root / "ASVspoof2019_LA_train/flac",
        root / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
        args.batch_size, shuffle=True, workers=args.workers,
    )
    dev_loader = make_loader(
        root / "ASVspoof2019_LA_dev/flac",
        root / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
        args.batch_size, shuffle=False, workers=args.workers,
    )

    model = build_model().to(device)
    # ASVspoof2019 LA train is heavily spoof-skewed (~9:1) — weight the loss.
    class_weights = torch.tensor([9.0, 1.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_eer = 1.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for x, y in dev_loader:
                x = x.to(device)
                probs = torch.softmax(model(x), dim=1)[:, 1].cpu().numpy()
                all_scores.append(probs)
                all_labels.append(y.numpy())
        eer = compute_eer(np.concatenate(all_scores), np.concatenate(all_labels))
        print(f"epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f}  dev_eer={eer:.4f}")

        if eer < best_eer:
            best_eer = eer
            torch.save(model.state_dict(), out_dir / "best.pt")
            print(f"  -> saved new best (EER={eer:.4f})")

    print(f"done. best dev EER: {best_eer:.4f}. checkpoint: {out_dir / 'best.pt'}")


def export_onnx(args):
    import torch

    model = build_model()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    dummy = torch.zeros(1, N_FRAMES, N_LFCC, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, args.onnx_out,
        input_names=["lfcc"], output_names=["logits"],
        dynamic_axes={"lfcc": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,  # new dynamo exporter splits weights into a .onnx.data sidecar and breaks INT8 quantization
    )
    print(f"exported {args.onnx_out}")

    if args.quantize:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        int8_path = str(Path(args.onnx_out).with_suffix(".int8.onnx"))
        quantize_dynamic(args.onnx_out, int8_path, weight_type=QuantType.QInt8)
        print(f"quantized -> {int8_path}  (this is the file acoustic_model.py should load)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", help="Path to extracted ASVspoof2019 LA root")
    p.add_argument("--out-dir", default="runs/track_a")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1), help="DataLoader workers (0 = in-process)")
    p.add_argument("--export-only", action="store_true")
    p.add_argument("--checkpoint", help="Path to a saved .pt checkpoint (for --export-only)")
    p.add_argument("--onnx-out", default="track_a.onnx")
    p.add_argument("--quantize", action="store_true", help="Also emit an INT8 quantized copy")
    args = p.parse_args()

    if args.export_only:
        export_onnx(args)
    else:
        train(args)
