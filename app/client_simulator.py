"""Authenticated real-time WAV streaming simulator for the demo gateway."""
from __future__ import annotations
import argparse, asyncio, base64, json, random, sys, urllib.parse, urllib.request, wave
from contextlib import suppress
from pathlib import Path
import numpy as np
import websockets

def _http_json(url: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST" if body else "GET", headers={"Content-Type": "application/json"} if body else {})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))

def _load_wav(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getsampwidth() != 2:
            raise ValueError("Simulator expects a 16-bit PCM WAV input.")
        sample_rate, channels = wav_file.getframerate(), wav_file.getnchannels()
        raw = wav_file.readframes(wav_file.getnframes())
    if channels < 1:
        raise ValueError("WAV has no audio channels.")
    if channels > 1:
        raw = np.frombuffer(raw, dtype=np.int16).reshape(-1, channels)[:, 0].astype(np.int16).tobytes()
    return raw, sample_rate

def _print_event(event: dict) -> None:
    kind = event.get("event")
    if kind == "risk_update":
        print("risk_update " + " ".join(f"{key}={event.get(key)}" for key in (
            "risk_score", "speaker_similarity", "packet_loss", "mitigation_action", "processing_latency_ms", "risk_level")))
    elif kind in {"mitigation_verdict", "liveness_challenge", "enforcement_action"}:
        print(f"{kind} {json.dumps(event, sort_keys=True)}")
    else:
        print(f"{kind or 'event'} {json.dumps(event, sort_keys=True)}")

async def stream_file(audio_path: Path, host: str, port: int, token: str, user_id: str, chunk_ms: int, loss: float) -> None:
    audio, sample_rate = _load_wav(audio_path)
    chunk_bytes = max(2, int(sample_rate * chunk_ms / 1000) * 2)
    ws_url = f"ws://{host}:{port}/ws/stream?" + urllib.parse.urlencode({"token": token})
    async with websockets.connect(ws_url, additional_headers={"Authorization": f"Bearer {token}"}, max_size=None) as ws:
        _print_event(json.loads(await ws.recv()))
        await ws.send(json.dumps({"type": "handshake", "sample_rate": sample_rate, "user_id": user_id}))
        async def receiver() -> None:
            async for message in ws:
                _print_event(json.loads(message))
        receiver_task = asyncio.create_task(receiver())
        try:
            for offset in range(0, len(audio), chunk_bytes):
                packet_loss = random.random() < loss
                await ws.send(json.dumps({"type": "rtp_status", "packet_loss": packet_loss}))
                await ws.send(audio[offset:offset + chunk_bytes])
                await asyncio.sleep(chunk_ms / 1000.0)
            await asyncio.sleep(1.0)
        finally:
            receiver_task.cancel()
            with suppress(asyncio.CancelledError):
                await receiver_task

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_pos", nargs="?", default=None, help="Path to 16-bit PCM WAV audio file")
    parser.add_argument("--audio", default=None)
    parser.add_argument("--token"); parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000); parser.add_argument("--chunk-ms", type=int, default=200)
    parser.add_argument("--loss", type=float, default=0.0); parser.add_argument("--user-id", default="dev-user")
    parser.add_argument("--enroll", action="store_true")
    args = parser.parse_args()
    if args.chunk_ms <= 0: parser.error("--chunk-ms must be positive")
    if not 0.0 <= args.loss <= 1.0: parser.error("--loss must be between 0.0 and 1.0")
    raw_audio = args.audio_pos or args.audio or "app/sample_call.wav"
    audio_path = Path(raw_audio)
    if not audio_path.is_file():
        for candidate in [Path("app") / raw_audio, Path("app/sample_call.wav"), Path("sample_call.wav")]:
            if candidate.is_file():
                audio_path = candidate
                break
    if not audio_path.is_file(): parser.error(f"audio file not found: {raw_audio}")
    base_url = f"http://{args.host}:{args.port}"
    try:
        token = args.token or _http_json(f"{base_url}/dev/token?" + urllib.parse.urlencode({"sub": args.user_id}))["token"]
        if args.enroll:
            enrollment = _http_json(f"{base_url}/enroll?" + urllib.parse.urlencode({"token": token}), {"user_id": args.user_id, "audio_b64": base64.b64encode(audio_path.read_bytes()).decode("ascii")})
            print(f"enrolled {json.dumps(enrollment, sort_keys=True)}")
        asyncio.run(stream_file(audio_path, args.host, args.port, token, args.user_id, args.chunk_ms, args.loss))
    except KeyboardInterrupt:
        print("\nStreaming interrupted.")
    except (OSError, urllib.error.URLError, websockets.exceptions.WebSocketException, KeyError, ValueError) as exc:
        print(f"Unable to reach or use the server: {exc}", file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
