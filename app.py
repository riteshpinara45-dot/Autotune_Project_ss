"""
Autotune Web Server
===================
Flask backend that wraps the autotune DSP engine.

Endpoints:
    GET  /                  → serves the frontend
    POST /api/upload        → upload a .wav file, returns file_id
    POST /api/record        → upload recorded audio blob, returns file_id
    POST /api/autotune      → process a file with given settings
    GET  /api/audio/<id>    → stream an audio file
    GET  /api/download/<id> → download an audio file
"""

import os
import sys
import uuid
import tempfile
import traceback
from pathlib import Path

import numpy as np
from flask import (
    Flask, request, jsonify, send_file, send_from_directory, abort
)

# Add project dir so we can import the DSP engine
sys.path.insert(0, str(Path(__file__).parent))
from autotune_system_main_phase import (
    AudioProcessingWorker, LiveAudioSystem, SCALE_PRESETS, FS, BLOCK_SIZE,
    CHANNELS, SCALE_SEMITONES, PITCH_CORRECTION_STRENGTH
)

import soundfile as sf
import sounddevice as sd

# ──────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=None)

UPLOAD_DIR = Path(tempfile.gettempdir()) / "autotune_web"
UPLOAD_DIR.mkdir(exist_ok=True)

# In-memory store:  file_id → {path, filename, type}
audio_store: dict[str, dict] = {}


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


# ──────────────────────────────────────────────────────────
# Serve frontend
# ──────────────────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "web_static"


@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(str(STATIC_DIR), filename)


# ──────────────────────────────────────────────────────────
# API: Upload a file
# ──────────────────────────────────────────────────────────
@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify(error="No file provided"), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify(error="Empty filename"), 400

    file_id = _new_id()
    ext = Path(f.filename).suffix or ".wav"
    save_path = UPLOAD_DIR / f"{file_id}{ext}"
    f.save(str(save_path))

    # Read to get info
    try:
        audio, sr = sf.read(str(save_path), dtype="float32", always_2d=True)
        duration = len(audio) / sr
        channels = audio.shape[1]
    except Exception as e:
        save_path.unlink(missing_ok=True)
        return jsonify(error=f"Could not read audio: {e}"), 400

    audio_store[file_id] = {
        "path": str(save_path),
        "filename": f.filename,
        "type": "original",
        "sr": sr,
        "duration": round(duration, 2),
        "channels": channels,
    }

    return jsonify(
        file_id=file_id,
        filename=f.filename,
        duration=round(duration, 2),
        sample_rate=sr,
        channels=channels,
    )


# ──────────────────────────────────────────────────────────
# API: Upload recorded audio (from browser MediaRecorder)
# ──────────────────────────────────────────────────────────
@app.route("/api/record", methods=["POST"])
def api_record():
    if "audio" not in request.files:
        return jsonify(error="No audio blob provided"), 400

    blob = request.files["audio"]
    file_id = _new_id()

    wav_path = UPLOAD_DIR / f"{file_id}.wav"
    blob.save(str(wav_path))

    # Read with soundfile directly (native WAV, FLAC, OGG)
    try:
        audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    except Exception as e:
        # Fallback if an external codec was uploaded (e.g. webm)
        try:
            import subprocess
            temp_raw = UPLOAD_DIR / f"{file_id}_raw.tmp"
            if wav_path.exists():
                wav_path.rename(temp_raw)
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(temp_raw), "-ar", "48000", "-ac", "1", str(wav_path)],
                capture_output=True, timeout=30, check=True
            )
            temp_raw.unlink(missing_ok=True)
            audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        except Exception as e2:
            if wav_path.exists():
                wav_path.unlink()
            return jsonify(error=f"Could not decode audio: {e}"), 400

    if audio.shape[1] > 1:
        audio = audio.mean(axis=1, keepdims=True)
        sf.write(str(wav_path), audio, sr)

    duration = len(audio) / sr

    audio_store[file_id] = {
        "path": str(wav_path),
        "filename": "recording.wav",
        "type": "original",
        "sr": sr,
        "duration": round(duration, 2),
        "channels": 1,
    }

    return jsonify(
        file_id=file_id,
        filename="recording.wav",
        duration=round(duration, 2),
        sample_rate=sr,
        channels=1,
    )

    return jsonify(
        file_id=file_id,
        filename="recording.wav",
        duration=round(duration, 2),
        sample_rate=sr,
        channels=1,
    )


# Root note lookup
ROOT_MAP = {
    "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3,
    "E": 4, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8,
    "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11
}

# ──────────────────────────────────────────────────────────
# API: Autotune processing
# ──────────────────────────────────────────────────────────
@app.route("/api/autotune", methods=["POST"])
def api_autotune():
    data = request.get_json(silent=True) or {}
    file_id = data.get("file_id")
    scale_name = data.get("scale", "major")
    root_name = data.get("root", "C")
    strength = float(data.get("strength", 1.0))
    transpose = int(data.get("transpose", 0))

    if not file_id or file_id not in audio_store:
        return jsonify(error="Invalid file_id"), 400

    info = audio_store[file_id]

    root = ROOT_MAP.get(root_name.strip().upper(), 0)
    scale = SCALE_PRESETS.get(scale_name, SCALE_PRESETS["major"])
    strength = max(0.0, min(1.0, strength))

    try:
        audio, sr = sf.read(info["path"], dtype="float32", always_2d=True)
        if audio.shape[1] > 1:
            audio = audio.mean(axis=1, keepdims=True)

        original_frames = len(audio)
        processor = AudioProcessingWorker(
            fs=sr,
            scale=scale,
            strength=strength,
            root=root,
            transpose=transpose,
            yin_threshold=0.20
        )
        processor.start()

        latency = processor.latency_samples
        if latency:
            audio = np.concatenate(
                [audio, np.zeros((latency, audio.shape[1]), dtype=audio.dtype)], axis=0
            )

        total = len(audio)
        processed_blocks = []
        block_id = 0

        for start in range(0, total, BLOCK_SIZE):
            end = min(start + BLOCK_SIZE, total)
            block = audio[start:end]
            processor.submit(block, block_id=block_id, block_when_full=True)
            result = processor.get_processed(timeout=10.0)
            if result:
                _, pb = result
                processed_blocks.append(pb)
            block_id += 1

        processor.stop()

        if not processed_blocks:
            return jsonify(error="Processing produced no output"), 500

        result_audio = np.concatenate(processed_blocks, axis=0)
        result_audio = result_audio[latency:latency + original_frames]

        # Save result
        out_id = _new_id()
        out_path = UPLOAD_DIR / f"{out_id}_autotuned.wav"
        sf.write(str(out_path), result_audio, sr)

        peak_original = float(np.max(np.abs(audio[:original_frames])))
        peak_output = float(np.max(np.abs(result_audio)))
        duration = original_frames / sr

        audio_store[out_id] = {
            "path": str(out_path),
            "filename": f"autotuned_{info['filename']}",
            "type": "processed",
            "sr": sr,
            "duration": round(duration, 2),
            "channels": 1,
        }

        # Generate waveform data for visualization (downsampled)
        def waveform_points(samples, num_points=300):
            mono = samples.ravel()
            step = max(1, len(mono) // num_points)
            ds = mono[::step][:num_points]
            return [round(float(v), 4) for v in ds]

        return jsonify(
            output_id=out_id,
            duration=round(duration, 2),
            peak_original=round(peak_original, 4),
            peak_output=round(peak_output, 4),
            scale=scale_name,
            strength=strength,
            waveform_original=waveform_points(audio[:original_frames]),
            waveform_processed=waveform_points(result_audio),
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify(error=str(e)), 500


# ──────────────────────────────────────────────────────────
# API: Serve / download audio
# ──────────────────────────────────────────────────────────
@app.route("/api/audio/<file_id>")
def api_audio(file_id):
    if file_id not in audio_store:
        abort(404)
    info = audio_store[file_id]
    return send_file(info["path"], mimetype="audio/wav", conditional=True)


@app.route("/api/download/<file_id>")
def api_download(file_id):
    if file_id not in audio_store:
        abort(404)
    info = audio_store[file_id]
    return send_file(info["path"], as_attachment=True,
                     download_name=info["filename"])


# ──────────────────────────────────────────────────────────
# API: Waveform data for an existing file
# ──────────────────────────────────────────────────────────
@app.route("/api/waveform/<file_id>")
def api_waveform(file_id):
    if file_id not in audio_store:
        abort(404)
    info = audio_store[file_id]
    try:
        audio, sr = sf.read(info["path"], dtype="float32")
        mono = audio.ravel() if audio.ndim > 1 else audio
        num_points = 300
        step = max(1, len(mono) // num_points)
        ds = mono[::step][:num_points]
        points = [round(float(v), 4) for v in ds]
        return jsonify(waveform=points, duration=info["duration"], sr=sr)
    except Exception as e:
        return jsonify(error=str(e)), 500


# ──────────────────────────────────────────────────────────
# API: Hardware Live Microphone Autotune (Python sounddevice)
# ──────────────────────────────────────────────────────────
active_live_stream = None
active_live_system = None

@app.route("/api/live/start", methods=["POST"])
def api_live_start():
    global active_live_stream, active_live_system
    if active_live_stream is not None and getattr(active_live_stream, 'active', False):
        return jsonify(status="running", message="Hardware stream already active")

    data = request.get_json(silent=True) or {}
    scale_name = data.get("scale", "major")
    root_name = data.get("root", "C")
    strength = float(data.get("strength", 1.0))
    transpose = int(data.get("transpose", 0))

    root = ROOT_MAP.get(root_name.strip().upper(), 0)
    scale = SCALE_PRESETS.get(scale_name, SCALE_PRESETS["major"])
    strength = max(0.0, min(1.0, strength))

    try:
        if active_live_stream:
            try:
                active_live_stream.stop()
                active_live_stream.close()
            except Exception:
                pass
            active_live_stream = None

        if active_live_system:
            try:
                active_live_system.processor.stop()
            except Exception:
                pass
            active_live_system = None

        active_live_system = LiveAudioSystem(
            scale=scale, strength=strength, root=root, transpose=transpose
        )
        active_live_system.start()

        active_live_stream = sd.Stream(
            samplerate=FS,
            blocksize=BLOCK_SIZE,
            channels=CHANNELS,
            dtype="float32",
            callback=active_live_system.audio_callback,
            latency="low"
        )
        active_live_stream.start()
        return jsonify(status="started", root=root_name, scale=scale_name)
    except Exception as e:
        traceback.print_exc()
        if active_live_system:
            active_live_system.processor.stop()
            active_live_system = None
        active_live_stream = None
        return jsonify(error=str(e)), 500


@app.route("/api/live/stop", methods=["POST"])
def api_live_stop():
    global active_live_stream, active_live_system
    if active_live_stream is not None:
        try:
            active_live_stream.stop()
            active_live_stream.close()
        except Exception:
            pass
        active_live_stream = None

    if active_live_system is not None:
        try:
            active_live_system.processor.stop()
        except Exception:
            pass
        active_live_system = None

    return jsonify(status="stopped")


@app.route("/api/live/status", methods=["GET"])
def api_live_status():
    global active_live_stream, active_live_system
    is_running = active_live_stream is not None and getattr(active_live_stream, 'active', False)
    stats = {}
    if active_live_system and active_live_system.processor:
        p = active_live_system.processor
        stats = {
            "detected_hz": round(p.last_detected_hz, 1),
            "target_hz": round(p.last_target_hz, 1),
            "ratio": round(p.last_shift_ratio, 3),
            "processed_blocks": p.processed_blocks
        }
    return jsonify(running=is_running, stats=stats)


# ──────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Force UTF-8 output on Windows
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    print("\n" + "=" * 55)
    print("  AUTOTUNE WEB SERVER")
    print("=" * 55)
    print(f"  Open in browser: http://localhost:5000")
    print(f"  Temp files:      {UPLOAD_DIR}")
    print("=" * 55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
