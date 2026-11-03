"""
Autotune + Filtering Audio System
==================================
Architecture: queue-based worker threads (original design preserved).

Signal chain per block:
    mic/file → [Pre-filter] → [Pitch Detection] → [Pitch Shift] → [Post-filter] → output

Filters implemented (all via scipy IIR/FIR, applied in worker thread):
    - High-pass  : removes DC offset and low rumble below 80 Hz
    - Low-pass   : removes HF noise above 8 kHz (voice intelligibility band)
    - Pre-emphasis: gentle shelving boost above 3 kHz to help pitch detector

Autotune:
    - Pitch detection : YIN autocorrelation algorithm (no external deps)
    - Pitch correction: pitch-shift to nearest semitone in a user-selected scale
    - Pitch shifting  : OLA (Overlap-Add) — no phase-vocoder dependency needed
    - Phase matching  : consecutive frames are aligned (xcorr + sub-sample
                        refinement) and cross-faded so there is no phase jump
                        at block boundaries (see PhaseMatchedPitchShifter)
"""

import queue
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import butter, sosfilt, sosfilt_zi, firwin, lfilter, lfilter_zi


# ============================================================
# 1. AUDIO CONFIGURATION
# ============================================================

FS          = 48000
CHANNELS    = 1
BLOCK_SIZE  = 2048      # larger block = better pitch detection (≈42 ms at 48 kHz)

CLIP_LEVEL  = 0.99
QUEUE_BLOCKS = 24

ORIGINAL_LIVE_FILENAME  = "original_live.wav"
PROCESSED_LIVE_FILENAME = "processed_live.wav"

# Autotune target scale — chromatic by default (snap to every semitone).
# Replace with e.g. [0,2,4,5,7,9,11] for major scale.
SCALE_SEMITONES = list(range(12))   # 0=C, 1=C#, 2=D, ... 11=B

# Pitch correction strength: 1.0 = hard snap, 0.0 = no correction.
PITCH_CORRECTION_STRENGTH = 0.85

# Voice frequency range for the pitch detector (Hz).
PITCH_MIN_HZ = 80.0
PITCH_MAX_HZ = 1000.0

# Phase matching between consecutive frames.
PHASE_MATCH_ENABLED = True                    # False = original OLA shifter (for A/B tests)
PHASE_OVERLAP       = 256                     # cross-fade length in samples (≈5.3 ms)
PHASE_SEARCH        = int(FS / PITCH_MIN_HZ / 2)   # ±search range: covers one period of the lowest pitch
PHASE_MIN_CORR      = 0.20                    # below this the seam is noise/silence → no alignment


# ============================================================
# 2. FILTER BANK
# ============================================================

def _make_highpass(cutoff=80.0, fs=FS, order=4):
    """4th-order Butterworth high-pass; removes DC and sub-bass rumble."""
    sos = butter(order, cutoff / (fs / 2), btype='high', output='sos')
    return sos


def _make_lowpass(cutoff=8000.0, fs=FS, order=4):
    """4th-order Butterworth low-pass; removes HF noise above 8 kHz."""
    sos = butter(order, cutoff / (fs / 2), btype='low', output='sos')
    return sos


def _make_preemphasis(cutoff=3000.0, fs=FS):
    """
    First-order high-shelf FIR pre-emphasis.
    Boosts frequencies above cutoff by ~6 dB to help YIN detect pitch.
    Implemented as a simple first-difference filter scaled by coefficient.
    H(z) = 1 - alpha*z^-1  where alpha controls shelf frequency.
    """
    alpha = 1.0 - (2.0 * np.pi * cutoff / fs)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    # b = [1, -alpha], a = [1]  → direct IIR form
    b = np.array([1.0, -alpha], dtype=np.float64)
    a = np.array([1.0],        dtype=np.float64)
    return b, a


class FilterBank:
    """
    Stateful filter bank — preserves IIR state across blocks so there
    are no clicks or DC jumps at block boundaries.

    Filters applied in order:
        1. Pre-emphasis  (before pitch detection — only used internally)
        2. High-pass     (always on output)
        3. Low-pass      (always on output)
    """

    def __init__(self, fs=FS):
        self.fs = fs

        self._hp_sos = _make_highpass(80.0,  fs)
        self._lp_sos = _make_lowpass(8000.0, fs)
        self._pe_b, self._pe_a = _make_preemphasis(3000.0, fs)

        # Per-section state vectors — one row per biquad section.
        self._hp_zi = sosfilt_zi(self._hp_sos)          # shape (sections, 2)
        self._lp_zi = sosfilt_zi(self._lp_sos)
        n_pe = max(len(self._pe_b), len(self._pe_a)) - 1
        self._pe_zi = np.zeros(n_pe, dtype=np.float64)

    def apply_preemphasis(self, x: np.ndarray) -> np.ndarray:
        """Pre-emphasis only — used before pitch detection."""
        x64 = x.astype(np.float64)
        y, self._pe_zi = lfilter(self._pe_b, self._pe_a, x64, zi=self._pe_zi)
        return y.astype(np.float32)

    def apply_output_filters(self, x: np.ndarray) -> np.ndarray:
        """High-pass then low-pass — applied to the final output."""
        x64 = x.astype(np.float64).ravel()

        y, self._hp_zi = sosfilt(self._hp_sos, x64, zi=self._hp_zi)
        y, self._lp_zi = sosfilt(self._lp_sos, y,   zi=self._lp_zi)

        return np.clip(y, -CLIP_LEVEL, CLIP_LEVEL).astype(np.float32).reshape(x.shape)


# ============================================================
# 3. PITCH DETECTION  —  YIN ALGORITHM
# ============================================================

def _yin_pitch(frame: np.ndarray, fs: float,
               f_min: float = PITCH_MIN_HZ,
               f_max: float = PITCH_MAX_HZ,
               threshold: float = 0.20) -> float:
    """
    YIN autocorrelation pitch estimator.
    Returns detected frequency in Hz, or 0.0 if no pitch found.
    Threshold default 0.20 for robust vocal pitch detection across
    different microphones and acoustic environments.
    """
    N    = len(frame)
    tau_min = max(1, int(fs / f_max))
    tau_max = min(N // 2, int(fs / f_min))

    if tau_max <= tau_min:
        return 0.0

    # Step 1 & 2: cumulative mean normalized difference
    d  = np.zeros(tau_max + 1, dtype=np.float64)
    cm = np.zeros(tau_max + 1, dtype=np.float64)
    cm[0] = 1.0

    running_sum = 0.0
    x64 = frame.astype(np.float64)

    for tau in range(1, tau_max + 1):
        diff = x64[:N - tau] - x64[tau:N]
        d[tau] = float(np.dot(diff, diff))
        running_sum += d[tau]
        cm[tau] = d[tau] * tau / running_sum if running_sum > 0 else 1.0

    # Step 3: find first tau below threshold
    best_tau = -1
    for tau in range(tau_min, tau_max):
        if cm[tau] < threshold:
            # take local minimum
            while tau + 1 < tau_max and cm[tau + 1] < cm[tau]:
                tau += 1
            best_tau = tau
            break

    if best_tau < 1:
        return 0.0

    # Step 4: parabolic interpolation
    if 1 <= best_tau < tau_max:
        s0, s1, s2 = cm[best_tau - 1], cm[best_tau], cm[best_tau + 1]
        denom = 2.0 * (2.0 * s1 - s0 - s2)
        offset = (s2 - s0) / denom if abs(denom) > 1e-10 else 0.0
        refined_tau = best_tau + offset
    else:
        refined_tau = float(best_tau)

    if refined_tau < 1.0:
        return 0.0

    return float(fs / refined_tau)


# ============================================================
# 4. PITCH CORRECTION
# ============================================================

def _hz_to_midi(freq: float) -> float:
    """Convert frequency in Hz to fractional MIDI note number."""
    if freq <= 0.0:
        return 0.0
    return 12.0 * np.log2(freq / 440.0) + 69.0


def _midi_to_hz(midi: float) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def _nearest_scale_note(midi: float, scale: list, root: int = 0) -> float:
    """
    Snap a MIDI pitch to the nearest note in the given scale and root.
    scale is a list of semitone offsets (0–11); octave is preserved.
    root is the tonic note (0=C, 1=C#, 2=D, ... 11=B).
    """
    actual_scale = [(s + root) % 12 for s in scale]
    octave   = int(midi) // 12
    semitone = midi % 12.0
    distances = [(abs(semitone - s), s) for s in actual_scale]
    # also check wrapping to next octave
    distances += [(abs(semitone - (s + 12)), s + 12) for s in actual_scale]
    best = min(distances, key=lambda x: x[0])[1]
    return float(octave * 12 + best)


def _correction_ratio(detected_hz: float,
                      scale: list,
                      strength: float,
                      root: int = 0,
                      transpose: int = 0,
                      robot_note: float | None = None) -> float:
    """
    Return the frequency ratio by which to pitch-shift the block.
    ratio = 1.0 → no shift.
    """
    if detected_hz <= 0.0:
        if transpose != 0:
            return float(2.0 ** (transpose / 12.0))
        return 1.0

    detected_midi = _hz_to_midi(detected_hz)

    if robot_note is not None and robot_note > 0:
        target_midi = float(robot_note)
    else:
        target_midi = _nearest_scale_note(detected_midi, scale, root)
        target_midi += transpose

    # Blend: strength=1.0 → full hard snap, strength=0.0 → no change.
    corrected_midi = detected_midi + strength * (target_midi - detected_midi)
    target_hz      = _midi_to_hz(corrected_midi)

    return float(target_hz / detected_hz)


# ============================================================
# 5. PITCH SHIFTING  —  OVERLAP-ADD (OLA)
# ============================================================

class OLAPitchShifter:
    """
    Time-domain Overlap-Add pitch shifter.

    Algorithm:
        1. Compute desired output length for ratio r: L_out = L_in / r
        2. Resample the block to L_out using linear interpolation
           (time-stretching in frequency domain is not needed for the
           small corrections autotune makes, ±1–2 semitones typically)
        3. The block is then output at the original block rate, which
           effectively changes pitch without changing tempo for small ratios.

    Limitation: for large ratios (>±2 semitones) use a phase-vocoder.
    For autotune on voice this OLA approach is sufficient and has
    no external dependencies.
    """

    def __init__(self, hop: int = BLOCK_SIZE // 4):
        self.hop       = hop
        self._overlap  = np.zeros(0, dtype=np.float32)

    def shift(self, block: np.ndarray, ratio: float) -> np.ndarray:
        """
        Pitch-shift one block by the given ratio.
        ratio > 1.0 → higher pitch, ratio < 1.0 → lower pitch.
        Returns an array of the same length as the input.
        """
        if abs(ratio - 1.0) < 1e-4:
            return block  # nothing to do

        x   = block.ravel().astype(np.float32)
        N   = len(x)

        # Time-stretch: resample x to N/ratio samples, then trim/pad to N.
        # This gives pitch * ratio when played back at the original rate.
        stretched_len = max(1, int(round(N / ratio)))
        old_indices   = np.linspace(0, N - 1, stretched_len)
        x_stretched   = np.interp(old_indices,
                                   np.arange(N),
                                   x).astype(np.float32)

        # Trim or zero-pad to original block length.
        if len(x_stretched) >= N:
            out = x_stretched[:N]
        else:
            out = np.zeros(N, dtype=np.float32)
            out[:len(x_stretched)] = x_stretched

        # Apply a Hann window for smooth block joins.
        window = np.hanning(N).astype(np.float32)
        return out * window


# ============================================================
# 5b. PHASE-MATCHED PITCH SHIFTER  (frame-to-frame phase alignment)
# ============================================================

class PhaseMatchedPitchShifter:
    """
    Pitch shifter that matches and equalises the phase between consecutive
    frames before joining them.

    Why it is needed
    ----------------
    Every block is resampled with its own ratio, so the waveform at the end of
    block k and the start of block k+1 are generally at different phases.
    Joining them directly gives a click / buzz at the block rate (~23 Hz).

    How it works (per block, in order)
    ----------------------------------
    1. Resample the newest samples of the input history by `ratio`, anchored
       on the newest sample.  The window is wider than the block by the
       overlap and by +/- search range, so the join point can slide.
    2. Phase search: normalised cross-correlation between the held tail of the
       previous output frame and every candidate start position of the new
       frame (integer lag, +/-PHASE_SEARCH samples).
    3. Sub-sample refinement: parabolic interpolation of the correlation peak
       gives a fractional lag, so residual phase error is < 1/2 sample.
       The final frame is re-read from the input at that fractional offset.
    4. Equal-gain raised-cosine cross-fade over PHASE_OVERLAP samples.  Since
       the two signals are now in phase the gains sum to 1 with no
       comb-filtering / amplitude dip.
    5. Emit exactly len(block) samples and hold the last PHASE_OVERLAP samples
       as the tail to match against in the next call.

    Constant latency = overlap + search samples (see `latency`).

    State lives inside this object, so it MUST be called from one thread, in
    block order.  The worker thread already guarantees that.
    """

    def __init__(self, overlap=PHASE_OVERLAP, search=PHASE_SEARCH,
                 max_block=BLOCK_SIZE, min_corr=PHASE_MIN_CORR):
        self.O        = int(overlap)
        self.L        = int(search)
        self.min_corr = float(min_corr)

        # Fade-in curve; fade-out = 1 - fade-in  (equal-gain, sums to 1).
        i = np.arange(self.O, dtype=np.float64)
        self._fade_in = 0.5 - 0.5 * np.cos(np.pi * (i + 0.5) / self.O)

        # History must cover the widest read: window * max ratio (2.0).
        window         = max_block + self.O + 2 * self.L
        self._hist_len = 2 * window + 8
        self._hist     = np.zeros(self._hist_len, dtype=np.float64)
        self._tail     = np.zeros(self.O, dtype=np.float64)

        # Diagnostics for the status line.
        self.last_lag  = 0.0
        self.last_corr = 0.0
        self._prev_k   = float(self.L)        # last chosen start (L = no shift)

    @property
    def latency(self) -> int:
        """Fixed delay (samples) between input and output."""
        return self.O + self.L

    # ----------------------------------------------------------
    def _find_alignment(self, R: np.ndarray):
        """
        Find where in R the new frame should start so that its head is in
        phase with the held tail.  Returns (start_index, correlation) where
        start_index is fractional and in [0, 2L]; L means "no shift".
        """
        O, L  = self.O, self.L
        tail  = self._tail
        e_t   = float(np.dot(tail, tail))
        if e_t < 1e-10:                       # previous tail is silent
            return float(L), 0.0

        # Cross-correlation for starts k = 0..2L.
        c   = np.correlate(R, tail, mode="valid")
        r2  = np.concatenate(([0.0], np.cumsum(R * R)))
        e_r = r2[O:O + 2 * L + 1] - r2[:2 * L + 1]
        norm = c / np.sqrt(e_r * e_t + 1e-12)

        # A periodic signal correlates equally well one period away, so bias
        # slightly towards the previous lag: this keeps the alignment stable
        # (constant ratio → constant lag, no needless jumps between periods).
        ks    = np.arange(2 * L + 1)
        score = norm - 0.05 * np.abs(ks - self._prev_k) / L
        k     = int(np.argmax(score))
        corr  = float(norm[k])
        if corr < self.min_corr:              # unvoiced / noisy: don't chase noise
            return float(L), corr

        frac = 0.0
        if 0 < k < 2 * L:                     # parabolic peak interpolation
            y0, y1, y2 = norm[k - 1], norm[k], norm[k + 1]
            d = y0 - 2.0 * y1 + y2
            if d < -1e-12:
                frac = float(np.clip(0.5 * (y0 - y2) / d, -0.5, 0.5))
        self._prev_k = k + frac
        return k + frac, corr

    # ----------------------------------------------------------
    def shift(self, block: np.ndarray, ratio: float) -> np.ndarray:
        """
        Pitch-shift `block` by `ratio` (>1 up, <1 down) with phase matching.
        Returns an array with the same number of samples as `block`.
        """
        x  = np.asarray(block, dtype=np.float64).ravel()
        n  = len(x)
        O, L = self.O, self.L
        ratio = float(np.clip(ratio, 0.5, 2.0))

        self._hist = np.concatenate((self._hist, x))[-self._hist_len:]
        hist   = self._hist
        Hn     = len(hist)
        grid   = np.arange(Hn, dtype=np.float64)
        W      = n + O + 2 * L                # resampled window length

        # Input position for resampled index j, anchored on the newest sample.
        def read(j):
            return np.interp((Hn - 1) - (W - 1 - j) * ratio, grid, hist)

        # Step 1-2: coarse resample of the region used for the search.
        R = read(np.arange(2 * L + O, dtype=np.float64))

        # Step 2-3: integer + fractional phase alignment.
        start, corr = self._find_alignment(R)
        self.last_lag  = start - L
        self.last_corr = corr

        # Re-read the frame at the exact fractional start (single interpolation).
        seg = read(start + np.arange(n + O, dtype=np.float64))

        # Step 4: equal-gain cross-fade with the previous frame's tail.
        seg[:O] = self._tail * (1.0 - self._fade_in) + seg[:O] * self._fade_in

        # Step 5: emit n samples, hold the last O for the next call.
        self._tail = seg[n:n + O].copy()
        return seg[:n].astype(np.float32)


# ============================================================
# 6. AUDIO PROCESSING WORKER
# ============================================================

class AudioProcessingWorker:
    """
    Background thread that applies the full signal chain to each block:
        pre-emphasis → pitch detect → pitch shift → output filters

    Input queue item  : (block_id, np.ndarray)
    Output queue item : (block_id, np.ndarray)
    """

    def __init__(self,
                 fs: int = FS,
                 scale=SCALE_SEMITONES,
                 strength=PITCH_CORRECTION_STRENGTH,
                 root: int = 0,
                 transpose: int = 0,
                 robot_note: float | None = None,
                 yin_threshold: float = 0.20,
                 queue_blocks=QUEUE_BLOCKS):

        self.fs            = int(fs)
        self.scale         = scale
        self.strength      = strength
        self.root          = root
        self.transpose     = transpose
        self.robot_note    = robot_note
        self.yin_threshold = yin_threshold

        self.input_queue  = queue.Queue(maxsize=queue_blocks)
        self.output_queue = queue.Queue(maxsize=queue_blocks)
        self.stop_event   = threading.Event()
        self.worker_thread = None

        self.filter_bank = FilterBank(self.fs)
        self.shifter     = (PhaseMatchedPitchShifter() if PHASE_MATCH_ENABLED
                            else OLAPitchShifter())

        self.processed_blocks    = 0
        self.dropped_input_blocks  = 0
        self.dropped_output_blocks = 0

        # Stats for live display.
        self.last_detected_hz  = 0.0
        self.last_target_hz    = 0.0
        self.last_shift_ratio  = 1.0

        # Pitch continuity tracker (prevents micro-gaps from fluttering)
        self._prev_detected_hz = 0.0
        self._hold_frames      = 0

    @property
    def latency_samples(self) -> int:
        """Extra delay introduced by phase matching (0 when disabled)."""
        return getattr(self.shifter, "latency", 0)

    # ----------------------------------------------------------
    def start(self):
        if self.worker_thread is not None and self.worker_thread.is_alive():
            return
        self.stop_event.clear()
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            name="AudioProcessingWorker",
            daemon=True
        )
        self.worker_thread.start()

    def stop(self):
        self.stop_event.set()
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=5.0)

    # ----------------------------------------------------------
    def submit(self, audio_block, block_id=None, block_when_full=False):
        item = (block_id, np.asarray(audio_block, dtype=np.float32).copy())
        try:
            if block_when_full:
                self.input_queue.put(item, timeout=5.0)
            else:
                self.input_queue.put_nowait(item)
            return True
        except queue.Full:
            self.dropped_input_blocks += 1
            return False

    def get_processed(self, timeout=0.0):
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ----------------------------------------------------------
    def _put_output(self, item):
        try:
            self.output_queue.put_nowait(item)
        except queue.Full:
            try:
                self.output_queue.get_nowait()
                self.dropped_output_blocks += 1
            except queue.Empty:
                pass
            try:
                self.output_queue.put_nowait(item)
            except queue.Full:
                self.dropped_output_blocks += 1

    # ----------------------------------------------------------
    def _process_block(self, audio_block: np.ndarray) -> np.ndarray:
        """
        Full signal chain for one block.

            1. Pre-emphasis  → exposes pitch harmonics for YIN
            2. YIN pitch detection on pre-emphasized copy
            3. Compute correction ratio (scale snap, hard autotune, robot or transpose)
            4. Pitch shift on original block with phase alignment
            5. Output high-pass + low-pass filtering
        """
        mono = audio_block.ravel()
        rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))

        # 1. Pre-emphasis
        pe = self.filter_bank.apply_preemphasis(mono)

        # 2. Pitch detection (skip silent/sub-noise regions to prevent noise chasing)
        if rms > 0.003:
            detected_hz = _yin_pitch(pe, float(self.fs), threshold=self.yin_threshold)
        else:
            detected_hz = 0.0

        # Pitch continuity: bridge momentary 1-2 block vocal drops during speech/singing
        if detected_hz > 0.0:
            self._prev_detected_hz = detected_hz
            self._hold_frames = 0
            effective_pitch = detected_hz
        elif self._hold_frames < 2 and self._prev_detected_hz > 0.0 and rms > 0.005:
            effective_pitch = self._prev_detected_hz
            self._hold_frames += 1
        else:
            effective_pitch = 0.0
            self._hold_frames = 0

        # 3. Correction ratio
        ratio = _correction_ratio(
            effective_pitch,
            self.scale,
            self.strength,
            root=self.root,
            transpose=self.transpose,
            robot_note=self.robot_note
        )

        # Store stats for reporting.
        self.last_detected_hz = effective_pitch
        if effective_pitch > 0.0 or self.transpose != 0:
            self.last_target_hz   = effective_pitch * ratio if effective_pitch > 0 else 0.0
            self.last_shift_ratio = ratio

        # 4. Pitch shift the original block.
        shifted = self.shifter.shift(mono, ratio)

        # 5. Output filters — reshape to match original channel layout.
        filtered = self.filter_bank.apply_output_filters(
            shifted.reshape(audio_block.shape)
        )

        return filtered

    # ----------------------------------------------------------
    def _worker_loop(self):
        while not self.stop_event.is_set() or not self.input_queue.empty():
            try:
                item = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                self.input_queue.task_done()
                break

            block_id, audio_block = item
            processed = self._process_block(audio_block)
            self._put_output((block_id, processed))
            self.processed_blocks += 1
            self.input_queue.task_done()


# ============================================================
# 7. LIVE MICROPHONE INPUT AND PLAYBACK
# ============================================================

class LiveAudioSystem:
    """
    Full-duplex microphone-to-speaker pipeline.
    The PortAudio callback only moves data; all DSP runs in the worker.
    """

    def __init__(self, scale=SCALE_SEMITONES, strength=PITCH_CORRECTION_STRENGTH, root: int = 0, transpose: int = 0):
        self.processor = AudioProcessingWorker(scale=scale, strength=strength, root=root, transpose=transpose, yin_threshold=0.20)
        self.original_blocks  = []
        self.processed_blocks = []

        self.next_input_id  = 0
        self.pending_output = np.zeros((0, CHANNELS), dtype=np.float32)

        self.last_report_time      = time.monotonic()
        self.callback_status_count = 0

    def start(self):
        self.processor.start()

    def audio_callback(self, indata, outdata, frames, time_info, status):
        if status:
            self.callback_status_count += 1

        input_block = indata.copy()
        self.original_blocks.append(input_block.copy())

        self.processor.submit(
            input_block,
            block_id=self.next_input_id,
            block_when_full=False
        )
        self.next_input_id += 1

        # Assemble exactly 'frames' output samples from completed blocks.
        output_parts   = []
        samples_needed = frames

        while samples_needed > 0:
            if len(self.pending_output) == 0:
                item = self.processor.get_processed(timeout=0.0)
                if item is None:
                    break
                _, processed_block = item
                self.processed_blocks.append(processed_block.copy())
                self.pending_output = processed_block

            take = min(samples_needed, len(self.pending_output))
            output_parts.append(self.pending_output[:take])
            self.pending_output = self.pending_output[take:]
            samples_needed -= take

        if output_parts:
            available = np.concatenate(output_parts, axis=0)
            outdata[:len(available)] = available
            if len(available) < frames:
                outdata[len(available):] = 0.0
        else:
            outdata.fill(0.0)

        # Periodic status report every 2 seconds.
        now = time.monotonic()
        if now - self.last_report_time >= 2.0:
            p = self.processor
            detected = p.last_detected_hz
            target   = p.last_target_hz
            ratio    = p.last_shift_ratio
            phase_txt = ""
            if isinstance(p.shifter, PhaseMatchedPitchShifter):
                phase_txt = (f" | Phase lag: {p.shifter.last_lag:+.1f} smp "
                             f"(corr {p.shifter.last_corr:.2f})")
            if detected > 0.0:
                detected_midi = _hz_to_midi(detected)
                target_midi   = _hz_to_midi(target)
                note_names    = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
                det_note   = note_names[int(round(detected_midi)) % 12]
                tgt_note   = note_names[int(round(target_midi))  % 12]
                print(
                    f"Detected: {detected:.1f} Hz ({det_note}) → "
                    f"Target: {target:.1f} Hz ({tgt_note}) | "
                    f"Ratio: {ratio:.4f} | "
                    f"Processed: {p.processed_blocks} | "
                    f"Dropped in: {p.dropped_input_blocks} out: {p.dropped_output_blocks}"
                    f"{phase_txt}"
                )
            else:
                print(
                    f"No pitch detected | "
                    f"Processed: {p.processed_blocks} | "
                    f"Dropped in: {p.dropped_input_blocks} out: {p.dropped_output_blocks}"
                )
            self.last_report_time = now

    def save_recordings(self):
        if self.original_blocks:
            original = np.concatenate(self.original_blocks, axis=0)
            sf.write(ORIGINAL_LIVE_FILENAME, original, FS)
            print("Saved original microphone audio:", ORIGINAL_LIVE_FILENAME)

        if self.processed_blocks:
            processed = np.concatenate(self.processed_blocks, axis=0)
            sf.write(PROCESSED_LIVE_FILENAME, processed, FS)
            print("Saved autotuned audio:", PROCESSED_LIVE_FILENAME)


def run_live_microphone(scale=SCALE_SEMITONES, strength=PITCH_CORRECTION_STRENGTH):
    system = LiveAudioSystem(scale=scale, strength=strength)
    system.start()

    note_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    scale_notes = [note_names[s % 12] for s in scale]

    print("\nStarting live autotune.")
    print("Target scale :", scale_notes)
    print("Correction   :", f"{strength * 100:.0f}%")
    print("Sample rate  :", FS, "Hz")
    print("Block size   :", BLOCK_SIZE, "samples")
    print("Filters      : 80 Hz HP + 8 kHz LP + 3 kHz pre-emphasis")
    if PHASE_MATCH_ENABLED:
        lat = PHASE_OVERLAP + PHASE_SEARCH
        print(f"Phase match  : on (overlap {PHASE_OVERLAP}, search ±{PHASE_SEARCH}, "
              f"+{1000.0 * lat / FS:.1f} ms latency)")
    print("Press Ctrl+C to stop.\n")

    try:
        with sd.Stream(
            samplerate=FS,
            blocksize=BLOCK_SIZE,
            channels=CHANNELS,
            dtype="float32",
            callback=system.audio_callback,
            latency="low"
        ):
            while True:
                sd.sleep(200)
    except KeyboardInterrupt:
        print("\nStopping live audio...")
    finally:
        system.processor.stop()
        system.save_recordings()
        print("Live audio system stopped.")


# ============================================================
# 8. RECORDED FILE PROCESSING
# ============================================================

def process_recorded_file(input_filename,
                           scale=SCALE_SEMITONES,
                           strength=PITCH_CORRECTION_STRENGTH):
    input_path = Path(input_filename).expanduser()

    if not input_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    audio, samplerate = sf.read(str(input_path), dtype="float32", always_2d=True)

    if audio.size == 0:
        raise ValueError("The selected audio file contains no samples.")

    original_frames = len(audio)
    processor       = AudioProcessingWorker(scale=scale, strength=strength)
    processor.start()

    # Phase matching delays the output by a fixed number of samples.  Pad the
    # input with silence so the delayed tail is flushed, then trim the same
    # amount from the front so the result lines up with the input.
    latency = processor.latency_samples
    if latency:
        audio = np.concatenate(
            [audio, np.zeros((latency, audio.shape[1]), dtype=audio.dtype)], axis=0)
    total_frames = len(audio)

    processed_blocks  = []
    submitted_blocks  = 0
    received_blocks   = 0

    print(f"\nReading: {input_path}")
    print(f"Sample rate : {samplerate} Hz")
    print(f"Channels    : {audio.shape[1]}")
    print(f"Duration    : {original_frames / samplerate:.2f} s")
    print(f"Correction  : {strength * 100:.0f}%")

    try:
        for start in range(0, total_frames, BLOCK_SIZE):
            end   = min(start + BLOCK_SIZE, total_frames)
            block = audio[start:end]

            if not processor.submit(block, block_id=submitted_blocks, block_when_full=True):
                raise RuntimeError("Could not submit audio block.")
            submitted_blocks += 1

            result = processor.get_processed(timeout=5.0)
            if result is None:
                raise TimeoutError("Timed out waiting for processed block.")

            block_id, processed_block = result
            if block_id != received_blocks:
                raise RuntimeError(
                    f"Block order error: expected {received_blocks}, got {block_id}."
                )

            processed_blocks.append(processed_block)
            received_blocks += 1

            if received_blocks % 100 == 0:
                pct = 100.0 * received_blocks / max(1, submitted_blocks)
                print(f"  {pct:.0f}% — {received_blocks} blocks done")

    finally:
        processor.stop()

    processed_audio = np.concatenate(processed_blocks, axis=0)
    processed_audio = processed_audio[latency:latency + original_frames]
    output_path     = input_path.with_name(input_path.stem + "_autotuned.wav")
    sf.write(str(output_path), processed_audio, samplerate)

    print(f"\nDone.")
    print(f"Input  peak : {float(np.max(np.abs(audio[:original_frames]))):.4f}")
    print(f"Output peak : {float(np.max(np.abs(processed_audio))):.4f}")
    print(f"Saved to    : {output_path}")

    play_choice = input("Play autotuned output now? (y/n): ").strip().lower()
    if play_choice == "y":
        print("Playing. Ctrl+C to stop.")
        try:
            sd.play(processed_audio, samplerate=samplerate)
            sd.wait()
        except KeyboardInterrupt:
            sd.stop()
            print("\nPlayback stopped.")

    return output_path


# ============================================================
# 9. SCALE PRESETS
# ============================================================

SCALE_PRESETS = {
    "chromatic" : list(range(12)),
    "major"     : [0, 2, 4, 5, 7, 9, 11],
    "minor"     : [0, 2, 3, 5, 7, 8, 10],
    "pentatonic": [0, 2, 4, 7, 9],
    "blues"     : [0, 3, 5, 6, 7, 10],
}


# ============================================================
# 10. MAIN MENU
# ============================================================

def _choose_scale() -> list:
    print("\nAvailable scales:")
    for i, (name, notes) in enumerate(SCALE_PRESETS.items(), 1):
        note_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        print(f"  {i}. {name:12s} {[note_names[n] for n in notes]}")

    raw = input("Select scale [1–5, default=1 chromatic]: ").strip()
    try:
        idx  = int(raw) - 1
        name = list(SCALE_PRESETS.keys())[idx]
        print(f"Scale selected: {name}")
        return SCALE_PRESETS[name]
    except (ValueError, IndexError):
        print("Defaulting to chromatic.")
        return SCALE_PRESETS["chromatic"]


def _choose_strength() -> float:
    raw = input(f"Correction strength 0.0–1.0 [default {PITCH_CORRECTION_STRENGTH}]: ").strip()
    try:
        v = float(raw)
        return float(np.clip(v, 0.0, 1.0))
    except ValueError:
        return PITCH_CORRECTION_STRENGTH


def main():
    print("=" * 55)
    print("AUTOTUNE + FILTER AUDIO SYSTEM")
    print("=" * 55)
    print("Signal chain: HP filter → pitch detect (YIN) →")
    print("              phase-matched pitch shift → LP filter")
    print("=" * 55)
    print("1. Live microphone autotune")
    print("2. Process a recorded audio file")
    print("3. Exit")

    choice = input("\nSelect (1/2/3): ").strip()

    if choice == "1":
        scale    = _choose_scale()
        strength = _choose_strength()
        run_live_microphone(scale=scale, strength=strength)

    elif choice == "2":
        filename = input("Audio file path (e.g. vocals.wav): ").strip()
        scale    = _choose_scale()
        strength = _choose_strength()
        process_recorded_file(filename, scale=scale, strength=strength)

    elif choice == "3":
        print("Exiting.")

    else:
        print("Invalid option.")


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print("\nError:", err)
        raise