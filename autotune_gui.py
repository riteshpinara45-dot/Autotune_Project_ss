"""
Autotune Voice Processor — Simple Bright‑Theme GUI
====================================================
A tkinter-based graphical front-end for the autotune DSP engine.

Features:
  • Record voice from microphone (with countdown)
  • Load a .wav file from disk
  • Choose scale (Chromatic / Major / Minor / Pentatonic / Blues)
  • Adjust correction strength with a slider
  • One-click "Autotune" processing
  • Play original / autotuned audio side-by-side
  • Save the autotuned result
  • Live waveform display (original vs autotuned)

Dependencies: numpy, scipy, sounddevice, soundfile, matplotlib, tkinter (stdlib)
"""

import os
import sys
import threading
import time
from pathlib import Path
from tkinter import (
    Tk, Frame, Label, Button, Scale, StringVar, OptionMenu,
    filedialog, messagebox, HORIZONTAL, LEFT, RIGHT, BOTH, X, Y,
    TOP, BOTTOM, DISABLED, NORMAL, END, W, E, N, S, CENTER,
    Canvas, PhotoImage
)
from tkinter import ttk
import numpy as np

# --------------- optional heavy imports with graceful error ---------------
_MISSING = []

try:
    import sounddevice as sd
except ImportError:
    _MISSING.append("sounddevice")

try:
    import soundfile as sf
except ImportError:
    _MISSING.append("soundfile")

try:
    from scipy.signal import butter, sosfilt, sosfilt_zi, lfilter, lfilter_zi
except ImportError:
    _MISSING.append("scipy")

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Re-use the DSP engine from the main file — it sits in the same folder.
sys.path.insert(0, str(Path(__file__).parent))
from autotune_system_main_phase import (
    AudioProcessingWorker, FilterBank,
    SCALE_PRESETS, FS, BLOCK_SIZE, CHANNELS,
    PITCH_CORRECTION_STRENGTH, SCALE_SEMITONES,
    _hz_to_midi
)

# ═══════════════════════════════════════════════════════════
# COLOUR PALETTE  — clean, bright theme
# ═══════════════════════════════════════════════════════════
BG           = "#F7F8FA"   # page background
CARD_BG      = "#FFFFFF"   # card surfaces
PRIMARY      = "#4F6DF5"   # main accent (indigo-blue)
PRIMARY_DARK = "#3B54C4"
ACCENT       = "#00C896"   # success / play green
DANGER       = "#EF4444"   # record / stop red
TEXT         = "#1E293B"   # main text
TEXT_LIGHT   = "#64748B"   # secondary text
BORDER       = "#E2E8F0"   # card border
SLIDER_BG    = "#CBD5E1"

FONT_TITLE = ("Segoe UI", 18, "bold")
FONT_HEAD  = ("Segoe UI", 12, "bold")
FONT_BODY  = ("Segoe UI", 10)
FONT_SMALL = ("Segoe UI", 9)
FONT_BTN   = ("Segoe UI", 10, "bold")


# ═══════════════════════════════════════════════════════════
# HELPER — simple waveform canvas (no matplotlib needed)
# ═══════════════════════════════════════════════════════════
class WaveformView(Canvas):
    """Lightweight waveform drawing on a tkinter Canvas."""

    def __init__(self, master, label="Waveform", **kw):
        kw.setdefault("bg", CARD_BG)
        kw.setdefault("height", 100)
        kw.setdefault("highlightthickness", 0)
        super().__init__(master, **kw)
        self._label = label
        self._draw_empty()

    # ----- public -----
    def set_data(self, samples: np.ndarray, sr: int = FS):
        self.delete("all")
        w = self.winfo_width() or 400
        h = self.winfo_height() or 100
        if len(samples) == 0:
            self._draw_empty()
            return

        mono = samples.ravel()
        # Downsample to canvas width
        step = max(1, len(mono) // w)
        ds = mono[::step]
        peak = max(np.max(np.abs(ds)), 1e-6)
        ds = ds / peak

        mid = h // 2
        points = []
        for i, v in enumerate(ds[:w]):
            x = i
            y = mid - int(v * (mid - 4))
            points.append((x, y))

        if len(points) < 2:
            self._draw_empty()
            return

        # Draw filled area
        fill_pts = [(0, mid)]
        fill_pts.extend(points)
        fill_pts.append((len(points) - 1, mid))
        flat = [c for p in fill_pts for c in p]
        self.create_polygon(flat, fill="#DDE4FF", outline="")

        # Draw line
        flat_line = [c for p in points for c in p]
        self.create_line(flat_line, fill=PRIMARY, width=1.5, smooth=True)

        # Label
        self.create_text(6, 6, text=self._label, anchor="nw",
                         fill=TEXT_LIGHT, font=FONT_SMALL)
        dur_s = len(mono) / sr
        self.create_text(w - 6, 6, text=f"{dur_s:.1f}s", anchor="ne",
                         fill=TEXT_LIGHT, font=FONT_SMALL)

    def _draw_empty(self):
        self.delete("all")
        w = self.winfo_width() or 400
        h = self.winfo_height() or 100
        self.create_text(w // 2, h // 2, text=f"No {self._label.lower()} loaded",
                         fill=BORDER, font=FONT_BODY)


# ═══════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════
class AutotuneApp:

    def __init__(self, root: Tk):
        self.root = root
        root.title("🎤 Signal & System — Voice Autotune")
        root.configure(bg=BG)
        root.minsize(720, 620)
        root.resizable(True, True)

        # state
        self.original_audio: np.ndarray | None = None
        self.processed_audio: np.ndarray | None = None
        self.sample_rate = FS
        self._recording = False
        self._playing = False
        self._rec_thread: threading.Thread | None = None

        self._build_ui()

    # ──────────────────────────────────────────────────────
    # UI CONSTRUCTION
    # ──────────────────────────────────────────────────────
    def _build_ui(self):
        # Title bar
        title_frame = Frame(self.root, bg=PRIMARY, pady=12)
        title_frame.pack(fill=X)
        Label(title_frame, text="🎵  Voice Autotune System",
              font=FONT_TITLE, fg="white", bg=PRIMARY).pack()
        Label(title_frame, text="Signal & System Project  —  Pitch Detection · Correction · Filtering",
              font=FONT_SMALL, fg="#C7D2FE", bg=PRIMARY).pack()

        # Main content area
        content = Frame(self.root, bg=BG, padx=20, pady=14)
        content.pack(fill=BOTH, expand=True)

        # ---- LEFT: Controls ----
        ctrl = Frame(content, bg=BG)
        ctrl.pack(side=LEFT, fill=Y, padx=(0, 14))

        self._card(ctrl, "Input Source", self._build_input_card)
        self._card(ctrl, "Autotune Settings", self._build_settings_card)
        self._card(ctrl, "Process & Output", self._build_process_card)

        # ---- RIGHT: Waveforms ----
        wave_frame = Frame(content, bg=BG)
        wave_frame.pack(side=RIGHT, fill=BOTH, expand=True)

        self.wave_orig = WaveformView(wave_frame, label="Original")
        self.wave_orig.pack(fill=BOTH, expand=True, pady=(0, 8))

        self.wave_proc = WaveformView(wave_frame, label="Autotuned")
        self.wave_proc.pack(fill=BOTH, expand=True)

        # Status bar
        self.status_var = StringVar(value="Ready — record or load a file to begin.")
        status_bar = Label(self.root, textvariable=self.status_var,
                           font=FONT_SMALL, fg=TEXT_LIGHT, bg="#EEF0F4",
                           anchor=W, padx=10, pady=4)
        status_bar.pack(fill=X, side=BOTTOM)

        # Redraw waveforms on resize
        self.root.bind("<Configure>", self._on_resize)

    # ---- card helper ----
    def _card(self, parent, title, builder_fn):
        outer = Frame(parent, bg=BORDER, bd=0, padx=1, pady=1)
        outer.pack(fill=X, pady=(0, 10))
        inner = Frame(outer, bg=CARD_BG, padx=14, pady=10)
        inner.pack(fill=X)
        Label(inner, text=title, font=FONT_HEAD, fg=TEXT, bg=CARD_BG).pack(anchor=W)
        sep = Frame(inner, bg=BORDER, height=1)
        sep.pack(fill=X, pady=(4, 8))
        builder_fn(inner)

    # ---- Input card ----
    def _build_input_card(self, parent):
        row = Frame(parent, bg=CARD_BG)
        row.pack(fill=X)

        self.btn_record = Button(
            row, text="⏺  Record", font=FONT_BTN,
            fg="white", bg=DANGER, activebackground="#DC2626",
            relief="flat", padx=12, pady=5, cursor="hand2",
            command=self._toggle_record)
        self.btn_record.pack(side=LEFT, padx=(0, 6))

        self.btn_load = Button(
            row, text="📂  Load File", font=FONT_BTN,
            fg=PRIMARY, bg="#EEF2FF", activebackground="#DDE4FF",
            relief="flat", padx=12, pady=5, cursor="hand2",
            command=self._load_file)
        self.btn_load.pack(side=LEFT)

        self.lbl_input_info = Label(parent, text="No audio loaded",
                                     font=FONT_SMALL, fg=TEXT_LIGHT, bg=CARD_BG)
        self.lbl_input_info.pack(anchor=W, pady=(6, 0))

    # ---- Settings card ----
    def _build_settings_card(self, parent):
        # Scale selector
        row1 = Frame(parent, bg=CARD_BG)
        row1.pack(fill=X, pady=(0, 6))
        Label(row1, text="Scale:", font=FONT_BODY, fg=TEXT, bg=CARD_BG).pack(side=LEFT)
        self.scale_var = StringVar(value="chromatic")
        scales = list(SCALE_PRESETS.keys())
        menu = OptionMenu(row1, self.scale_var, *scales)
        menu.config(font=FONT_BODY, bg=CARD_BG, fg=TEXT, relief="flat",
                    highlightthickness=0, activebackground="#EEF2FF")
        menu.pack(side=LEFT, padx=(8, 0))

        # Strength slider
        row2 = Frame(parent, bg=CARD_BG)
        row2.pack(fill=X)
        Label(row2, text="Strength:", font=FONT_BODY, fg=TEXT, bg=CARD_BG).pack(side=LEFT)
        self.strength_var = StringVar(value="85%")
        self.strength_slider = Scale(
            row2, from_=0, to=100, orient=HORIZONTAL,
            bg=CARD_BG, fg=TEXT, troughcolor=SLIDER_BG,
            highlightthickness=0, sliderrelief="flat",
            length=160, showvalue=False,
            command=lambda v: self.strength_var.set(f"{int(float(v))}%"))
        self.strength_slider.set(85)
        self.strength_slider.pack(side=LEFT, padx=(8, 4))
        Label(row2, textvariable=self.strength_var, font=FONT_BODY,
              fg=PRIMARY, bg=CARD_BG, width=4).pack(side=LEFT)

    # ---- Process & playback card ----
    def _build_process_card(self, parent):
        row = Frame(parent, bg=CARD_BG)
        row.pack(fill=X, pady=(0, 6))

        self.btn_process = Button(
            row, text="✨  Autotune!", font=FONT_BTN,
            fg="white", bg=PRIMARY, activebackground=PRIMARY_DARK,
            relief="flat", padx=16, pady=6, cursor="hand2",
            command=self._run_autotune)
        self.btn_process.pack(side=LEFT, padx=(0, 6))

        self.btn_save = Button(
            row, text="💾 Save", font=FONT_BTN,
            fg=TEXT, bg="#F1F5F9", activebackground=BORDER,
            relief="flat", padx=12, pady=6, cursor="hand2",
            command=self._save_output, state=DISABLED)
        self.btn_save.pack(side=LEFT)

        row2 = Frame(parent, bg=CARD_BG)
        row2.pack(fill=X)

        self.btn_play_orig = Button(
            row2, text="▶ Original", font=FONT_BODY,
            fg=ACCENT, bg="#ECFDF5", activebackground="#D1FAE5",
            relief="flat", padx=10, pady=4, cursor="hand2",
            command=lambda: self._play(self.original_audio), state=DISABLED)
        self.btn_play_orig.pack(side=LEFT, padx=(0, 6))

        self.btn_play_proc = Button(
            row2, text="▶ Autotuned", font=FONT_BODY,
            fg=PRIMARY, bg="#EEF2FF", activebackground="#DDE4FF",
            relief="flat", padx=10, pady=4, cursor="hand2",
            command=lambda: self._play(self.processed_audio), state=DISABLED)
        self.btn_play_proc.pack(side=LEFT, padx=(0, 6))

        self.btn_stop = Button(
            row2, text="⏹ Stop", font=FONT_BODY,
            fg=DANGER, bg="#FEF2F2", activebackground="#FEE2E2",
            relief="flat", padx=10, pady=4, cursor="hand2",
            command=self._stop_play, state=DISABLED)
        self.btn_stop.pack(side=LEFT)

        self.lbl_proc_info = Label(parent, text="",
                                    font=FONT_SMALL, fg=TEXT_LIGHT, bg=CARD_BG)
        self.lbl_proc_info.pack(anchor=W, pady=(6, 0))

    # ──────────────────────────────────────────────────────
    # RECORDING
    # ──────────────────────────────────────────────────────
    def _toggle_record(self):
        if self._recording:
            self._recording = False
            self.btn_record.config(text="⏺  Record", bg=DANGER)
            self.status_var.set("Processing recording…")
        else:
            self._recording = True
            self.btn_record.config(text="⏹  Stop", bg="#7C3AED")
            self.status_var.set("🔴  Recording… click Stop when done.")
            self._rec_thread = threading.Thread(target=self._record_worker, daemon=True)
            self._rec_thread.start()

    def _record_worker(self):
        chunks = []
        try:
            with sd.InputStream(samplerate=FS, channels=CHANNELS,
                                blocksize=BLOCK_SIZE, dtype="float32") as stream:
                while self._recording:
                    data, _ = stream.read(BLOCK_SIZE)
                    chunks.append(data.copy())
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Recording Error", str(e)))
            self._recording = False
            self.root.after(0, lambda: self.btn_record.config(text="⏺  Record", bg=DANGER))
            return

        if chunks:
            self.original_audio = np.concatenate(chunks, axis=0)
            self.sample_rate = FS
            dur = len(self.original_audio) / FS
            self.root.after(0, lambda: self._on_audio_loaded(
                f"Recorded {dur:.1f}s @ {FS} Hz"))
        else:
            self.root.after(0, lambda: self.status_var.set("Recording was empty."))

    # ──────────────────────────────────────────────────────
    # FILE LOADING
    # ──────────────────────────────────────────────────────
    def _load_file(self):
        path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("WAV files", "*.wav"), ("All audio", "*.wav *.flac *.ogg *.mp3"),
                       ("All files", "*.*")])
        if not path:
            return
        try:
            audio, sr = sf.read(path, dtype="float32", always_2d=True)
            if audio.shape[1] > 1:
                audio = audio.mean(axis=1, keepdims=True)
            self.original_audio = audio
            self.sample_rate = sr
            dur = len(audio) / sr
            self._on_audio_loaded(f"{Path(path).name}  —  {dur:.1f}s @ {sr} Hz")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def _on_audio_loaded(self, info_text: str):
        self.processed_audio = None
        self.lbl_input_info.config(text=info_text)
        self.btn_play_orig.config(state=NORMAL)
        self.btn_play_proc.config(state=DISABLED)
        self.btn_save.config(state=DISABLED)
        self.status_var.set(f"Audio loaded. Adjust settings and hit Autotune!")
        self._redraw_waveforms()

    # ──────────────────────────────────────────────────────
    # AUTOTUNE PROCESSING
    # ──────────────────────────────────────────────────────
    def _run_autotune(self):
        if self.original_audio is None:
            messagebox.showinfo("No Audio", "Record or load an audio file first.")
            return

        self.btn_process.config(state=DISABLED, text="⏳ Processing…")
        self.status_var.set("Autotuning in progress…")
        self.root.update_idletasks()

        threading.Thread(target=self._autotune_worker, daemon=True).start()

    def _autotune_worker(self):
        try:
            scale_name = self.scale_var.get()
            scale = SCALE_PRESETS.get(scale_name, SCALE_SEMITONES)
            strength = self.strength_slider.get() / 100.0

            audio = self.original_audio.copy()
            sr = self.sample_rate

            processor = AudioProcessingWorker(scale=scale, strength=strength)
            processor.start()

            latency = processor.latency_samples
            original_frames = len(audio)
            if latency:
                audio = np.concatenate(
                    [audio, np.zeros((latency, audio.shape[1]), dtype=audio.dtype)], axis=0)

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

            if processed_blocks:
                result = np.concatenate(processed_blocks, axis=0)
                result = result[latency:latency + original_frames]
                self.processed_audio = result
                self.root.after(0, self._on_autotune_done)
            else:
                self.root.after(0, lambda: self.status_var.set("Processing produced no output."))
                self.root.after(0, lambda: self.btn_process.config(
                    state=NORMAL, text="✨  Autotune!"))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Processing Error", str(e)))
            self.root.after(0, lambda: self.btn_process.config(
                state=NORMAL, text="✨  Autotune!"))

    def _on_autotune_done(self):
        self.btn_process.config(state=NORMAL, text="✨  Autotune!")
        self.btn_play_proc.config(state=NORMAL)
        self.btn_save.config(state=NORMAL)
        dur = len(self.processed_audio) / self.sample_rate
        peak = float(np.max(np.abs(self.processed_audio)))
        self.lbl_proc_info.config(text=f"Output: {dur:.1f}s · Peak: {peak:.3f}")
        self.status_var.set("✅  Autotune complete! Play or save the result.")
        self._redraw_waveforms()

    # ──────────────────────────────────────────────────────
    # PLAYBACK
    # ──────────────────────────────────────────────────────
    def _play(self, audio):
        if audio is None:
            return
        self._stop_play()
        try:
            sd.play(audio, samplerate=self.sample_rate)
            self._playing = True
            self.btn_stop.config(state=NORMAL)
            self.status_var.set("▶  Playing…")
            # Poll until done
            self._poll_playback()
        except Exception as e:
            messagebox.showerror("Playback Error", str(e))

    def _poll_playback(self):
        if sd.get_stream() and sd.get_stream().active:
            self.root.after(200, self._poll_playback)
        else:
            self._playing = False
            self.btn_stop.config(state=DISABLED)
            self.status_var.set("Ready.")

    def _stop_play(self):
        try:
            sd.stop()
        except Exception:
            pass
        self._playing = False
        self.btn_stop.config(state=DISABLED)

    # ──────────────────────────────────────────────────────
    # SAVE
    # ──────────────────────────────────────────────────────
    def _save_output(self):
        if self.processed_audio is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save Autotuned Audio",
            defaultextension=".wav",
            filetypes=[("WAV file", "*.wav")])
        if not path:
            return
        try:
            sf.write(path, self.processed_audio, self.sample_rate)
            self.status_var.set(f"Saved to {Path(path).name}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    # ──────────────────────────────────────────────────────
    # WAVEFORM REDRAW
    # ──────────────────────────────────────────────────────
    def _redraw_waveforms(self):
        if self.original_audio is not None:
            self.wave_orig.set_data(self.original_audio, self.sample_rate)
        else:
            self.wave_orig._draw_empty()

        if self.processed_audio is not None:
            self.wave_proc.set_data(self.processed_audio, self.sample_rate)
        else:
            self.wave_proc._draw_empty()

    def _on_resize(self, event):
        # Debounced redraw — only on root resize
        if event.widget == self.root:
            self.root.after(150, self._redraw_waveforms)


# ═══════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════
def main():
    if _MISSING:
        print(f"Missing packages: {', '.join(_MISSING)}")
        print("Install with:  pip install " + " ".join(_MISSING))
        sys.exit(1)

    root = Tk()
    # Windows DPI awareness
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    app = AutotuneApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
