"""Violin DSP Workstation - upgraded offline Tkinter audio processor.

Signal chain:
    noise reduction/gate -> HP/LP filters -> 5-band parametric EQ
    -> compressor -> algorithmic or convolution reverb -> peak safety

Dependencies: numpy, scipy, soundfile, sounddevice, matplotlib (plus tkinter).
Run this file directly, choose a WAV file, adjust controls, then Process audio.
"""

from __future__ import annotations

import os
import json
import threading
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy import signal

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


DOWNLOADS_DIR = os.path.join(os.path.expanduser("~"), "Downloads")
EPS = 1e-12


def _as_2d(x):
    """Return samples x channels and remember whether input was mono."""
    x = np.asarray(x, dtype=np.float64)
    mono = x.ndim == 1
    return (x[:, None] if mono else x), mono


def _restore_shape(x, mono):
    return x[:, 0] if mono else x


def _sos_filter(x, sos):
    return signal.sosfilt(sos, x, axis=0)


def peaking_coefficients(fs, freq, gain_db, q):
    """RBJ audio-EQ-cookbook peaking biquad coefficients."""
    freq = float(np.clip(freq, 10.0, fs * 0.49))
    q = max(float(q), 0.05)
    a_gain = 10.0 ** (float(gain_db) / 40.0)
    w0 = 2.0 * np.pi * freq / fs
    alpha = np.sin(w0) / (2.0 * q)
    b = np.array([1 + alpha * a_gain, -2 * np.cos(w0), 1 - alpha * a_gain])
    a = np.array([1 + alpha / a_gain, -2 * np.cos(w0), 1 - alpha / a_gain])
    return b / a[0], a / a[0]


def multiband_parametric_eq(x, fs, bands):
    """Cascade enabled (frequency, gain dB, Q) parametric EQ bands."""
    y, mono = _as_2d(x)
    for enabled, freq, gain_db, q in bands:
        if enabled and abs(gain_db) > 1e-6:
            b, a = peaking_coefficients(fs, freq, gain_db, q)
            y = signal.lfilter(b, a, y, axis=0)
    return _restore_shape(y, mono)


def high_low_pass(x, fs, hp_enabled=False, hp_hz=70.0,
                  lp_enabled=False, lp_hz=16000.0, order=4):
    y, mono = _as_2d(x)
    nyquist = fs / 2.0
    if hp_enabled:
        cutoff = np.clip(hp_hz, 10.0, nyquist * 0.95)
        y = _sos_filter(y, signal.butter(order, cutoff, btype="highpass", fs=fs, output="sos"))
    if lp_enabled:
        cutoff = np.clip(lp_hz, 20.0, nyquist * 0.98)
        y = _sos_filter(y, signal.butter(order, cutoff, btype="lowpass", fs=fs, output="sos"))
    return _restore_shape(y, mono)


def spectral_noise_reduction(x, fs, strength=0.65, noise_seconds=0.25):
    """Light spectral subtraction using the opening audio as a noise profile."""
    data, mono = _as_2d(x)
    if len(data) < 64 or strength <= 0:
        return x.copy()
    nperseg = min(2048, len(data))
    noverlap = nperseg * 3 // 4
    profile_samples = max(nperseg, min(len(data), int(noise_seconds * fs)))
    out = np.zeros_like(data)
    for ch in range(data.shape[1]):
        _, _, z = signal.stft(data[:, ch], fs=fs, nperseg=nperseg,
                              noverlap=noverlap, boundary="zeros")
        _, _, nz = signal.stft(data[:profile_samples, ch], fs=fs, nperseg=nperseg,
                               noverlap=noverlap, boundary="zeros")
        noise_mag = np.median(np.abs(nz), axis=1, keepdims=True)
        mag = np.abs(z)
        floor = 0.08 * mag
        cleaned = np.maximum(mag - strength * noise_mag, floor)
        z_clean = cleaned * np.exp(1j * np.angle(z))
        _, rebuilt = signal.istft(z_clean, fs=fs, nperseg=nperseg,
                                  noverlap=noverlap, input_onesided=True)
        out[:, ch] = rebuilt[:len(data)]
    return _restore_shape(out, mono)


def noise_gate(x, fs, threshold_db=-52.0, attack_ms=8.0,
               release_ms=120.0, floor_db=-70.0):
    """Linked-channel gate with smoothed attack/release gain."""
    data, mono = _as_2d(x)
    detector = np.max(np.abs(data), axis=1)
    threshold = 10.0 ** (threshold_db / 20.0)
    floor = 10.0 ** (floor_db / 20.0)
    target = np.where(detector >= threshold, 1.0, floor)
    attack = np.exp(-1.0 / max(1.0, fs * attack_ms / 1000.0))
    release = np.exp(-1.0 / max(1.0, fs * release_ms / 1000.0))
    gain = np.empty(len(target))
    state = target[0] if len(target) else 1.0
    for n, wanted in enumerate(target):
        coeff = attack if wanted > state else release
        state = coeff * state + (1.0 - coeff) * wanted
        gain[n] = state
    return _restore_shape(data * gain[:, None], mono)


def compressor(x, fs, threshold_db=-18.0, ratio=3.0, attack_ms=15.0,
               release_ms=120.0, knee_db=6.0, makeup_db=2.0):
    """Feed-forward linked-channel RMS-like compressor with soft knee."""
    data, mono = _as_2d(x)
    level = np.max(np.abs(data), axis=1)
    level_db = 20.0 * np.log10(level + EPS)
    over = level_db - threshold_db
    slope = 1.0 / max(ratio, 1.0) - 1.0
    reduction_db = np.zeros_like(level_db)
    if knee_db > 0:
        lower, upper = -knee_db / 2.0, knee_db / 2.0
        mid = (over > lower) & (over < upper)
        high = over >= upper
        reduction_db[mid] = slope * (over[mid] - lower) ** 2 / (2.0 * knee_db)
        reduction_db[high] = slope * over[high]
    else:
        high = over > 0
        reduction_db[high] = slope * over[high]
    wanted = 10.0 ** (reduction_db / 20.0)
    a = np.exp(-1.0 / max(1.0, fs * attack_ms / 1000.0))
    r = np.exp(-1.0 / max(1.0, fs * release_ms / 1000.0))
    smooth = np.empty_like(wanted)
    state = 1.0
    for n, target in enumerate(wanted):
        coeff = a if target < state else r
        state = coeff * state + (1.0 - coeff) * target
        smooth[n] = state
    y = data * smooth[:, None] * 10.0 ** (makeup_db / 20.0)
    return _restore_shape(y, mono)


def _feedback_comb(x, delay, gain, damping):
    """Feedback comb. A low-pass input makes the tail progressively darker."""
    cutoff = 0.48 - 0.38 * np.clip(damping, 0.0, 1.0)
    b_lp, a_lp = signal.butter(1, cutoff)
    dark = signal.lfilter(b_lp, a_lp, x, axis=0)
    b = np.zeros(delay + 1); b[0] = 1.0
    a = np.zeros(delay + 1); a[0] = 1.0; a[delay] = -gain
    return signal.lfilter(b, a, dark, axis=0)


def _allpass(x, delay, gain=0.7):
    b = np.zeros(delay + 1); b[0] = -gain; b[delay] = 1.0
    a = np.zeros(delay + 1); a[0] = 1.0; a[delay] = -gain
    return signal.lfilter(b, a, x, axis=0)


def schroeder_reverb_rt60(x, fs, rt60=1.8, damping=0.45,
                          predelay_ms=20.0, wet_level=0.25):
    """Schroeder reverb whose comb feedback reaches -60 dB at RT60."""
    data, mono = _as_2d(x)
    predelay = max(0, int(fs * predelay_ms / 1000.0))
    driven = np.pad(data, ((predelay, 0), (0, 0)))[:len(data)]
    delays_ms = [29.7, 37.1, 41.1, 43.7]
    wet = np.zeros_like(data)
    for delay_ms in delays_ms:
        delay = max(1, int(fs * delay_ms / 1000.0))
        delay_seconds = delay / fs
        gain = 10.0 ** (-3.0 * delay_seconds / max(rt60, 0.1))
        wet += _feedback_comb(driven, delay, gain, damping)
    wet /= len(delays_ms)
    for delay_ms in (5.0, 1.7):
        wet = _allpass(wet, max(1, int(fs * delay_ms / 1000.0)))
    peak = np.max(np.abs(wet))
    if peak > 4.0:
        wet *= 4.0 / peak
    y = (1.0 - wet_level) * data + wet_level * wet
    return _restore_shape(y, mono)


def convolution_reverb(x, fs, ir, ir_fs, wet_level=0.25):
    """FFT convolution with mono/stereo IR, resampling and peak-safe IR normalization."""
    data, mono = _as_2d(x)
    response, _ = _as_2d(ir)
    if ir_fs != fs:
        divisor = np.gcd(int(ir_fs), int(fs))
        response = signal.resample_poly(response, fs // divisor, ir_fs // divisor, axis=0)
    response = response[:max(1, int(12 * fs))]
    response -= np.mean(response, axis=0, keepdims=True)
    norm = np.sqrt(np.sum(response * response, axis=0, keepdims=True))
    response /= np.maximum(norm, EPS)
    channels = max(data.shape[1], response.shape[1])
    if data.shape[1] == 1 and channels == 2:
        data = np.repeat(data, 2, axis=1)
    if response.shape[1] == 1 and channels > 1:
        response = np.repeat(response, channels, axis=1)
    wet = np.column_stack([
        signal.fftconvolve(data[:, c], response[:, c], mode="full")[:len(data)]
        for c in range(channels)
    ])
    wet_peak = np.max(np.abs(wet))
    dry_peak = np.max(np.abs(data))
    if wet_peak > EPS:
        wet *= max(dry_peak, 0.1) / wet_peak
    y = (1.0 - wet_level) * data + wet_level * wet
    return _restore_shape(y, mono and channels == 1)


@dataclass
class EQControls:
    enabled: tk.BooleanVar
    freq: tk.DoubleVar
    gain: tk.DoubleVar
    q: tk.DoubleVar


class DSPApp:
    EQ_DEFAULTS = [
        ("Low", 150, -2.0, 0.8), ("Body", 400, 3.0, 1.2),
        ("Nasal", 1000, -2.0, 2.0), ("Presence", 3000, 2.0, 1.5),
        ("Brilliance", 7000, 2.0, 1.0),
    ]

    def __init__(self, root):
        self.root = root
        root.title("Violin DSP Workstation")
        root.geometry("900x920")
        root.minsize(760, 700)
        self.input_path = tk.StringVar(value="No file selected")
        self.ir_path = tk.StringVar(value="No impulse response selected")
        self.status = tk.StringVar(value="Ready.")
        self.audio = self.processed = None
        self.fs = None
        self.output_path = None
        self._build_gui()

    def _build_gui(self):
        file_frame = ttk.LabelFrame(self.root, text="1. Choose audio file")
        file_frame.pack(fill="x", padx=10, pady=(10, 5))
        ttk.Button(file_frame, text="Browse...", command=self.browse_file).pack(side="left", padx=6, pady=6)
        ttk.Label(file_frame, textvariable=self.input_path, wraplength=700).pack(side="left", padx=5)

        controls = ttk.LabelFrame(self.root, text="2. Adjust parameters")
        controls.pack(fill="x", padx=10, pady=5)
        notebook = ttk.Notebook(controls)
        notebook.pack(fill="x", padx=5, pady=5)
        cleanup = ttk.Frame(notebook); eq = ttk.Frame(notebook)
        dynamics = ttk.Frame(notebook); reverb = ttk.Frame(notebook)
        experiment = ttk.Frame(notebook)
        notebook.add(experiment, text="Experiment / Isolation")
        notebook.add(cleanup, text="Cleanup / Filters")
        notebook.add(eq, text="5-band EQ")
        notebook.add(dynamics, text="Compressor")
        notebook.add(reverb, text="Reverb")
        self._build_cleanup(cleanup)
        self._build_eq(eq)
        self._build_compressor(dynamics)
        self._build_reverb(reverb)
        self._build_experiment(experiment)

        settings_row = ttk.Frame(controls)
        settings_row.pack(fill="x", padx=5, pady=(0, 6))
        ttk.Label(settings_row, text="Settings preset:").pack(side="left", padx=(3, 8))
        ttk.Button(settings_row, text="Save settings...", command=self.save_settings).pack(side="left", padx=3)
        ttk.Button(settings_row, text="Load settings...", command=self.load_settings).pack(side="left", padx=3)

        actions = ttk.LabelFrame(self.root, text="3. Process & compare")
        actions.pack(fill="x", padx=10, pady=5)
        ttk.Button(actions, text="Process audio", command=self.process_audio).pack(fill="x", padx=5, pady=4)
        row = ttk.Frame(actions); row.pack(fill="x", padx=4, pady=2)
        for text, command in (("Play original", self.play_original),
                              ("Play enhanced", self.play_enhanced),
                              ("Stop playback", sd.stop)):
            ttk.Button(row, text=text, command=command).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Label(self.root, textvariable=self.status, foreground="gray").pack(fill="x", padx=12, pady=3)

        analysis = ttk.LabelFrame(self.root, text="4. Spectrum & spectrogram analyzer")
        analysis.pack(fill="both", expand=True, padx=10, pady=(5, 10))
        self.plot_tabs = ttk.Notebook(analysis); self.plot_tabs.pack(fill="both", expand=True)
        spectrum_tab = ttk.Frame(self.plot_tabs); spectrogram_tab = ttk.Frame(self.plot_tabs)
        self.plot_tabs.add(spectrum_tab, text="Spectrum")
        self.plot_tabs.add(spectrogram_tab, text="Spectrogram")
        self.fig_spectrum, self.ax_spectrum = plt.subplots(figsize=(7.5, 3.1))
        self.fig_specgram, (self.ax_spec_original, self.ax_spec_processed) = plt.subplots(2, 1, figsize=(7.5, 4.6), sharex=True)
        self.canvas_spectrum = FigureCanvasTkAgg(self.fig_spectrum, master=spectrum_tab)
        self.canvas_spectrum.get_tk_widget().pack(fill="both", expand=True)
        self.canvas_specgram = FigureCanvasTkAgg(self.fig_specgram, master=spectrogram_tab)
        self.canvas_specgram.get_tk_widget().pack(fill="both", expand=True)
        self.update_plots()

    def _check(self, parent, text, default=False):
        var = tk.BooleanVar(value=default)
        ttk.Checkbutton(parent, text=text, variable=var).pack(anchor="w", padx=6, pady=2)
        return var

    def _slider(self, parent, label, lo, hi, default, digits=2):
        row = ttk.Frame(parent); row.pack(fill="x", padx=5, pady=2)
        ttk.Label(row, text=label, width=24).pack(side="left")
        var = tk.DoubleVar(value=default)
        ttk.Scale(row, from_=lo, to=hi, variable=var).pack(side="left", fill="x", expand=True, padx=5)
        value = ttk.Label(row, width=8); value.pack(side="left")
        var.trace_add("write", lambda *_: value.config(text=f"{var.get():.{digits}f}"))
        value.config(text=f"{default:.{digits}f}")
        return var

    def _build_cleanup(self, parent):
        cols = ttk.Frame(parent); cols.pack(fill="x", pady=4)
        left = ttk.LabelFrame(cols, text="Noise"); left.pack(side="left", fill="both", expand=True, padx=3)
        right = ttk.LabelFrame(cols, text="Filters"); right.pack(side="left", fill="both", expand=True, padx=3)
        self.denoise_on = self._check(left, "Spectral noise reduction", False)
        self.denoise_strength = self._slider(left, "Reduction strength", 0, 1.5, 0.65)
        self.gate_on = self._check(left, "Noise gate", True)
        self.gate_threshold = self._slider(left, "Gate threshold (dB)", -80, -20, -52, 1)
        self.hp_on = self._check(right, "High-pass filter", True)
        self.hp_hz = self._slider(right, "HP cutoff (Hz)", 20, 400, 70, 0)
        self.lp_on = self._check(right, "Low-pass filter", False)
        self.lp_hz = self._slider(right, "LP cutoff (Hz)", 4000, 22000, 16000, 0)

    def _build_eq(self, parent):
        header = ttk.Frame(parent); header.pack(fill="x", padx=5, pady=(5, 1))
        for text, width in (("On / Band", 15), ("Frequency (Hz)", 20), ("Gain (dB)", 18), ("Q", 16)):
            ttk.Label(header, text=text, width=width).pack(side="left")
        self.eq_controls = []
        for name, freq, gain, q in self.EQ_DEFAULTS:
            row = ttk.Frame(parent); row.pack(fill="x", padx=5, pady=1)
            enabled = tk.BooleanVar(value=True)
            ttk.Checkbutton(row, text=name, variable=enabled, width=13).pack(side="left")
            vars_ = []
            for lo, hi, default, width in ((40, 18000, freq, 9), (-12, 12, gain, 7), (0.3, 8, q, 6)):
                var = tk.DoubleVar(value=default); vars_.append(var)
                ttk.Entry(row, textvariable=var, width=width).pack(side="left", padx=(6, 75))
            self.eq_controls.append(EQControls(enabled, *vars_))

    def _build_compressor(self, parent):
        self.comp_on = self._check(parent, "Enable compressor", True)
        self.comp_threshold = self._slider(parent, "Threshold (dB)", -50, 0, -18, 1)
        self.comp_ratio = self._slider(parent, "Ratio", 1, 12, 3, 1)
        self.comp_attack = self._slider(parent, "Attack (ms)", 1, 100, 15, 1)
        self.comp_release = self._slider(parent, "Release (ms)", 20, 800, 120, 0)
        self.comp_knee = self._slider(parent, "Soft knee (dB)", 0, 18, 6, 1)
        self.comp_makeup = self._slider(parent, "Makeup gain (dB)", 0, 12, 2, 1)

    def _build_reverb(self, parent):
        self.reverb_on = self._check(parent, "Enable reverb", True)
        mode_row = ttk.Frame(parent); mode_row.pack(fill="x", padx=6, pady=2)
        ttk.Label(mode_row, text="Type", width=24).pack(side="left")
        self.reverb_mode = tk.StringVar(value="Schroeder (RT60)")
        ttk.Combobox(mode_row, textvariable=self.reverb_mode, state="readonly",
                     values=("Schroeder (RT60)", "Convolution IR"), width=22).pack(side="left")
        self.rt60 = self._slider(parent, "RT60 (seconds)", 0.25, 6.0, 1.8)
        self.damping = self._slider(parent, "HF damping", 0, 1, 0.45)
        self.predelay = self._slider(parent, "Pre-delay (ms)", 0, 100, 20, 0)
        self.wet = self._slider(parent, "Wet level", 0, 1, 0.25)
        row = ttk.Frame(parent); row.pack(fill="x", padx=6, pady=4)
        ttk.Button(row, text="Choose impulse response...", command=self.browse_ir).pack(side="left")
        ttk.Label(row, textvariable=self.ir_path, wraplength=500).pack(side="left", padx=6)

    # Reference values used when one parameter is isolated.  They make a
    # parameter audible while keeping every other parameter fixed.
    EXPERIMENT_COMP_REFERENCE = {
        "threshold": -18.0,
        "ratio": 3.0,
        "attack": 15.0,
        "release": 120.0,
        "knee": 6.0,
        "makeup": 0.0,
    }
    EXPERIMENT_REVERB_REFERENCE = {
        "rt60": 1.5,
        "damping": 0.50,
        "predelay": 15.0,
        "wet": 0.12,
    }

    def _build_experiment(self, parent):
        """Controls for one-variable-at-a-time qualitative testing."""
        self.experiment_mode = tk.BooleanVar(value=False)
        self.experiment_vars = {}

        top = ttk.Frame(parent)
        top.pack(fill="x", padx=8, pady=(7, 4))
        ttk.Checkbutton(
            top,
            text="Enable isolation mode",
            variable=self.experiment_mode,
            command=self._experiment_status,
        ).pack(side="left")
        ttk.Button(top, text="Clear selections", command=self._clear_experiment).pack(side="left", padx=8)

        ttk.Label(
            parent,
            text=(
                "Isolation mode is intended for controlled qualitative tests. Check one variable "
                "to hear that variable while all unrelated DSP stages are bypassed. Parameters that "
                "cannot operate alone (for example compressor attack or RT60) use fixed reference "
                "values for the other required parameters. Multiple boxes may be checked if you "
                "deliberately want a combined-condition test. Spectral denoising is file-mode only."
            ),
            wraplength=820,
            justify="left",
        ).pack(fill="x", padx=8, pady=(0, 7))

        body = ttk.Frame(parent)
        body.pack(fill="x", padx=5, pady=2)
        left = ttk.Frame(body); left.pack(side="left", fill="both", expand=True, padx=3)
        right = ttk.Frame(body); right.pack(side="left", fill="both", expand=True, padx=3)

        cleanup = ttk.LabelFrame(left, text="Cleanup / filters")
        cleanup.pack(fill="x", pady=3)
        for key, label in (
            ("denoise_strength", "Spectral denoise strength (offline)"),
            ("gate_threshold", "Noise-gate threshold"),
            ("hp_hz", "High-pass cutoff"),
            ("lp_hz", "Low-pass cutoff"),
        ):
            self._experiment_check(cleanup, key, label)

        eq_box = ttk.LabelFrame(left, text="EQ parameters")
        eq_box.pack(fill="x", pady=3)
        ttk.Label(eq_box, text="Band", width=14).grid(row=0, column=0, sticky="w", padx=4)
        for col, name in enumerate(("Frequency", "Gain", "Q"), start=1):
            ttk.Label(eq_box, text=name, width=11).grid(row=0, column=col, padx=2)
        for i, (name, *_rest) in enumerate(self.EQ_DEFAULTS, start=1):
            ttk.Label(eq_box, text=name, width=14).grid(row=i, column=0, sticky="w", padx=4)
            for col, suffix in enumerate(("freq", "gain", "q"), start=1):
                key = f"eq{i-1}_{suffix}"
                var = tk.BooleanVar(value=False)
                self.experiment_vars[key] = var
                ttk.Checkbutton(eq_box, variable=var, command=self._experiment_status).grid(
                    row=i, column=col, padx=12
                )

        comp = ttk.LabelFrame(right, text="Compressor parameters")
        comp.pack(fill="x", pady=3)
        for key, label in (
            ("threshold", "Threshold"), ("ratio", "Ratio"),
            ("attack", "Attack"), ("release", "Release"),
            ("knee", "Soft knee"), ("makeup", "Makeup gain"),
        ):
            self._experiment_check(comp, f"comp_{key}", label)

        reverb = ttk.LabelFrame(right, text="Schroeder reverb parameters")
        reverb.pack(fill="x", pady=3)
        for key, label in (
            ("rt60", "RT60"), ("damping", "HF damping"),
            ("predelay", "Pre-delay"), ("wet", "Wet level"),
        ):
            self._experiment_check(reverb, f"reverb_{key}", label)

        self.experiment_status = tk.StringVar(value="Isolation mode off — normal DSP controls are active.")
        ttk.Label(
            parent, textvariable=self.experiment_status, foreground="gray",
            wraplength=820, justify="left"
        ).pack(fill="x", padx=8, pady=(6, 8))

    def _experiment_check(self, parent, key, label):
        var = tk.BooleanVar(value=False)
        self.experiment_vars[key] = var
        ttk.Checkbutton(parent, text=label, variable=var, command=self._experiment_status).pack(
            anchor="w", padx=6, pady=1
        )

    def _clear_experiment(self):
        for var in self.experiment_vars.values():
            var.set(False)
        self._experiment_status()

    def _experiment_status(self):
        if not hasattr(self, "experiment_status"):
            return
        selected = [name for name, var in self.experiment_vars.items() if var.get()]
        if not self.experiment_mode.get():
            self.experiment_status.set("Isolation mode off — normal DSP controls are active.")
        elif not selected:
            self.experiment_status.set("Isolation mode on — no variables selected, so the DSP path is neutral/raw.")
        else:
            pretty = ", ".join(selected)
            self.experiment_status.set(f"Isolation mode on — active test variable(s): {pretty}")

    def _experiment_metadata(self):
        return {
            "enabled": bool(self.experiment_mode.get()),
            "selected": {name: bool(var.get()) for name, var in self.experiment_vars.items()},
        }

    def _apply_experiment_isolation(self, settings):
        """Return effective settings for controlled one-variable-at-a-time tests."""
        exp = settings.get("experiment", {})
        if not exp.get("enabled"):
            return settings
        selected = exp.get("selected", {})
        s = dict(settings)

        # Unrelated stages are off unless their test variable is selected.
        s["denoise"] = bool(selected.get("denoise_strength"))
        s["gate"] = bool(selected.get("gate_threshold"))
        s["hp"] = bool(selected.get("hp_hz"))
        s["lp"] = bool(selected.get("lp_hz"))

        # Each EQ parameter may be isolated.  Disabled parameters for an active
        # band are held at stable reference values so frequency/Q remain audible.
        isolated_bands = []
        for i, (enabled, freq, gain, q) in enumerate(settings["bands"]):
            use_freq = bool(selected.get(f"eq{i}_freq"))
            use_gain = bool(selected.get(f"eq{i}_gain"))
            use_q = bool(selected.get(f"eq{i}_q"))
            active = use_freq or use_gain or use_q
            ref_name, ref_freq, ref_gain, ref_q = self.EQ_DEFAULTS[i]
            # If the reference gain is exactly zero in a future preset, use a
            # modest +2 dB so frequency and Q tests remain perceptible.
            audible_ref_gain = ref_gain if abs(ref_gain) > 1e-6 else 2.0
            isolated_bands.append((
                active,
                freq if use_freq else ref_freq,
                gain if use_gain else audible_ref_gain,
                q if use_q else ref_q,
            ))
        s["bands"] = isolated_bands

        # Compressor: selecting any one parameter enables the compressor. Other
        # parameters are held at the fixed reference values above.
        comp_keys = ("threshold", "ratio", "attack", "release", "knee", "makeup")
        comp_active = any(selected.get(f"comp_{key}") for key in comp_keys)
        s["compress"] = bool(comp_active)
        for key in comp_keys:
            if not selected.get(f"comp_{key}"):
                s[key] = self.EXPERIMENT_COMP_REFERENCE[key]

        # Reverb: same principle.  A 12% reference wet mix makes RT60, damping,
        # and pre-delay audible without requiring the Wet variable to be selected.
        reverb_keys = ("rt60", "damping", "predelay", "wet")
        reverb_active = any(selected.get(f"reverb_{key}") for key in reverb_keys)
        s["reverb"] = bool(reverb_active)
        s["mode"] = "Schroeder (RT60)"
        for key in reverb_keys:
            if not selected.get(f"reverb_{key}"):
                s[key] = self.EXPERIMENT_REVERB_REFERENCE[key]

        return s

    def browse_file(self):
        path = filedialog.askopenfilename(title="Select a WAV file", initialdir=DOWNLOADS_DIR,
                                          filetypes=[("WAV files", "*.wav"), ("Audio files", "*.wav *.flac *.aiff")])
        if not path: return
        try:
            audio, fs = sf.read(path, always_2d=False)
            self.audio = np.asarray(audio, dtype=np.float64)
            self.fs = fs; self.processed = None
            self.input_path.set(path)
            channels = 1 if self.audio.ndim == 1 else self.audio.shape[1]
            self.status.set(f"Loaded {os.path.basename(path)}: {len(self.audio)/fs:.1f}s, {fs} Hz, {channels} channel(s)")
            self.update_plots()
        except Exception as exc:
            messagebox.showerror("Error loading file", str(exc))

    def browse_ir(self):
        path = filedialog.askopenfilename(title="Select impulse-response WAV",
                                          filetypes=[("WAV files", "*.wav"), ("Audio files", "*.wav *.flac *.aiff")])
        if path: self.ir_path.set(path)

    def _settings(self, effective=True):
        settings = {
            "bands": [(b.enabled.get(), b.freq.get(), b.gain.get(), b.q.get()) for b in self.eq_controls],
            "denoise": self.denoise_on.get(), "denoise_strength": self.denoise_strength.get(),
            "gate": self.gate_on.get(), "gate_threshold": self.gate_threshold.get(),
            "hp": self.hp_on.get(), "hp_hz": self.hp_hz.get(), "lp": self.lp_on.get(), "lp_hz": self.lp_hz.get(),
            "compress": self.comp_on.get(), "threshold": self.comp_threshold.get(), "ratio": self.comp_ratio.get(),
            "attack": self.comp_attack.get(), "release": self.comp_release.get(), "knee": self.comp_knee.get(),
            "makeup": self.comp_makeup.get(), "reverb": self.reverb_on.get(), "mode": self.reverb_mode.get(),
            "rt60": self.rt60.get(), "damping": self.damping.get(), "predelay": self.predelay.get(), "wet": self.wet.get(),
            "ir_path": self.ir_path.get(),
            "experiment": self._experiment_metadata(),
        }
        return self._apply_experiment_isolation(settings) if effective else settings

    def save_settings(self):
        """Save every Section 2 control to a portable JSON preset."""
        path = filedialog.asksaveasfilename(
            title="Save DSP settings",
            initialdir=DOWNLOADS_DIR,
            initialfile="violin_dsp_settings.json",
            defaultextension=".json",
            filetypes=[("DSP settings", "*.json"), ("JSON files", "*.json")],
        )
        if not path:
            return
        document = {
            "format": "violin-dsp-settings",
            "version": 1,
            "settings": self._settings(effective=False),
        }
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(document, handle, indent=2)
                handle.write("\n")
            self.status.set(f"Settings saved: {path}")
        except Exception as exc:
            messagebox.showerror("Could not save settings", str(exc))

    def load_settings(self):
        """Load a JSON preset and update every Section 2 control."""
        path = filedialog.askopenfilename(
            title="Load DSP settings",
            initialdir=DOWNLOADS_DIR,
            filetypes=[("DSP settings", "*.json"), ("JSON files", "*.json")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                document = json.load(handle)
            if not isinstance(document, dict) or document.get("format") != "violin-dsp-settings":
                raise ValueError("This is not a Violin DSP settings file.")
            if document.get("version") != 1 or not isinstance(document.get("settings"), dict):
                raise ValueError("This settings-file version is not supported.")
            self._apply_settings(document["settings"])
            self.status.set(f"Settings loaded: {path}")
        except Exception as exc:
            messagebox.showerror("Could not load settings", str(exc))

    def _apply_settings(self, settings):
        required = {
            "bands", "denoise", "denoise_strength", "gate", "gate_threshold",
            "hp", "hp_hz", "lp", "lp_hz", "compress", "threshold", "ratio",
            "attack", "release", "knee", "makeup", "reverb", "mode", "rt60",
            "damping", "predelay", "wet", "ir_path",
        }
        missing = sorted(required.difference(settings))
        if missing:
            raise ValueError("Settings file is missing: " + ", ".join(missing))
        bands = settings["bands"]
        if not isinstance(bands, list) or len(bands) != len(self.eq_controls):
            raise ValueError(f"Settings must contain exactly {len(self.eq_controls)} EQ bands.")

        scalar_controls = {
            "denoise": self.denoise_on, "denoise_strength": self.denoise_strength,
            "gate": self.gate_on, "gate_threshold": self.gate_threshold,
            "hp": self.hp_on, "hp_hz": self.hp_hz,
            "lp": self.lp_on, "lp_hz": self.lp_hz,
            "compress": self.comp_on, "threshold": self.comp_threshold,
            "ratio": self.comp_ratio, "attack": self.comp_attack,
            "release": self.comp_release, "knee": self.comp_knee,
            "makeup": self.comp_makeup, "reverb": self.reverb_on,
            "mode": self.reverb_mode, "rt60": self.rt60,
            "damping": self.damping, "predelay": self.predelay,
            "wet": self.wet, "ir_path": self.ir_path,
        }
        for name, variable in scalar_controls.items():
            variable.set(settings[name])
        for control, values in zip(self.eq_controls, bands):
            if not isinstance(values, (list, tuple)) or len(values) != 4:
                raise ValueError("Each EQ band must contain enabled, frequency, gain, and Q values.")
            enabled, frequency, gain, q = values
            control.enabled.set(enabled)
            control.freq.set(frequency)
            control.gain.set(gain)
            control.q.set(q)

        experiment = settings.get("experiment")
        if isinstance(experiment, dict):
            self.experiment_mode.set(bool(experiment.get("enabled", False)))
            selected = experiment.get("selected", {})
            if isinstance(selected, dict):
                for name, var in self.experiment_vars.items():
                    var.set(bool(selected.get(name, False)))
            self._experiment_status()

    def process_audio(self):
        if self.audio is None:
            messagebox.showwarning("No file", "Choose an audio file first."); return
        settings = self._settings()
        if settings["reverb"] and settings["mode"] == "Convolution IR" and not os.path.isfile(settings["ir_path"]):
            messagebox.showwarning("No impulse response", "Choose an impulse-response audio file first."); return
        self.status.set("Processing...")
        threading.Thread(target=self._process_worker, args=(settings,), daemon=True).start()

    def _process_worker(self, s):
        try:
            y = self.audio.copy()
            if s["denoise"]: y = spectral_noise_reduction(y, self.fs, s["denoise_strength"])
            if s["gate"]: y = noise_gate(y, self.fs, s["gate_threshold"])
            y = high_low_pass(y, self.fs, s["hp"], s["hp_hz"], s["lp"], s["lp_hz"])
            y = multiband_parametric_eq(y, self.fs, s["bands"])
            if s["compress"]:
                y = compressor(y, self.fs, s["threshold"], s["ratio"], s["attack"], s["release"], s["knee"], s["makeup"])
            if s["reverb"]:
                if s["mode"] == "Convolution IR":
                    ir, ir_fs = sf.read(s["ir_path"], always_2d=False)
                    y = convolution_reverb(y, self.fs, ir, ir_fs, s["wet"])
                else:
                    y = schroeder_reverb_rt60(y, self.fs, s["rt60"], s["damping"], s["predelay"], s["wet"])
            peak = np.max(np.abs(y)) if len(y) else 0.0
            if peak > 0.98: y = y * (0.98 / peak)
            src = self.input_path.get(); base = os.path.splitext(os.path.basename(src))[0]
            out = os.path.join(DOWNLOADS_DIR, f"{base}_enhanced.wav")
            sf.write(out, y, self.fs, subtype="PCM_24")
            self.processed = y; self.output_path = out
            self.root.after(0, self._processing_done)
        except Exception as exc:
            error = str(exc)
            self.root.after(0, lambda error=error: self._processing_failed(error))

    def _processing_done(self):
        self.status.set(f"Saved: {self.output_path}")
        self.update_plots()

    def _processing_failed(self, error):
        messagebox.showerror("Processing error", error); self.status.set("Processing failed.")

    @staticmethod
    def _mono_for_plot(x):
        return np.mean(x, axis=1) if x is not None and x.ndim > 1 else x

    def update_plots(self):
        ax = self.ax_spectrum; ax.clear()
        ax.set(title="Frequency spectrum", xlabel="Frequency (Hz)", ylabel="Power (dB)", xscale="log", xlim=(30, 20000))
        for audio, color, label in ((self.audio, "steelblue", "Original"), (self.processed, "darkorange", "Enhanced")):
            if audio is not None and len(audio) > 8:
                mono = self._mono_for_plot(audio); nseg = min(4096, len(mono))
                f, psd = signal.welch(mono, self.fs, nperseg=nseg)
                ax.semilogx(f, 10 * np.log10(psd + EPS), color=color, label=label)
        if self.audio is not None: ax.legend()
        ax.grid(True, which="both", alpha=.25); self.fig_spectrum.tight_layout(); self.canvas_spectrum.draw()
        for spec_ax, audio, title in ((self.ax_spec_original, self.audio, "Original"),
                                      (self.ax_spec_processed, self.processed, "Enhanced")):
            spec_ax.clear(); spec_ax.set_ylabel("Hz"); spec_ax.set_title(title, loc="left", fontsize=9)
            if audio is not None and len(audio) > 64:
                mono = self._mono_for_plot(audio); nfft = min(2048, len(mono))
                spec_ax.specgram(mono, NFFT=nfft, Fs=self.fs, noverlap=nfft * 3 // 4, cmap="magma", scale="dB")
                spec_ax.set_ylim(0, min(16000, self.fs / 2))
            else:
                spec_ax.text(.5, .5, "Process audio to populate", ha="center", va="center", transform=spec_ax.transAxes)
        self.ax_spec_processed.set_xlabel("Time (s)")
        self.fig_specgram.tight_layout(); self.canvas_specgram.draw()

    def play_original(self):
        if self.audio is None: messagebox.showwarning("No file", "Choose an audio file first."); return
        sd.stop(); sd.play(self.audio, self.fs)

    def play_enhanced(self):
        if self.processed is None: messagebox.showwarning("Not processed", "Click Process audio first."); return
        sd.stop(); sd.play(self.processed, self.fs)


if __name__ == "__main__":
    root = tk.Tk()
    DSPApp(root)
    root.mainloop()
