"""
GUI for the offline violin DSP script
--------------------------------------
Same processing as violin_dsp_starter.py (peaking EQ + Schroeder reverb),
wrapped in a simple Tkinter window so you can:
  - pick a .wav file (defaults to your Downloads folder)
  - adjust room_size / damping / wet_level / EQ with sliders
  - process it and save the result
  - play the original and enhanced versions back to compare

Uses only tkinter (built into Python) + your existing packages
(numpy, scipy, soundfile, sounddevice) -- nothing new to install.

Run as a full script (F5 in Spyder).
"""

import os
import numpy as np
import soundfile as sf
import sounddevice as sd
from scipy import signal
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

DOWNLOADS_DIR = os.path.join(os.path.expanduser("~"), "Downloads")


# ----------------------------------------------------------------------
# DSP functions (same as violin_dsp_starter.py)
# ----------------------------------------------------------------------
def schroeder_reverb(x, fs, room_size=0.6, damping=0.5, wet_level=0.3):
    comb_delays_ms = [29.7, 37.1, 41.1, 43.7]
    comb_gains = [room_size * 0.84 for _ in comb_delays_ms]

    wet = np.zeros(len(x))
    for delay_ms, g in zip(comb_delays_ms, comb_gains):
        D = int(fs * delay_ms / 1000)
        b = np.zeros(D + 1)
        b[0] = 1.0
        a = np.zeros(D + 1)
        a[0] = 1.0
        a[D] = -g * (1 - damping * 0.3)
        wet += signal.lfilter(b, a, x)
    wet /= len(comb_delays_ms)

    for delay_ms in [5.0, 1.7]:
        D = int(fs * delay_ms / 1000)
        g = 0.7
        b = np.zeros(D + 1)
        b[0] = -g
        b[D] = 1.0
        a = np.zeros(D + 1)
        a[0] = 1.0
        a[D] = -g
        wet = signal.lfilter(b, a, wet)

    return (1 - wet_level) * x + wet_level * wet


def peaking_eq(x, fs, freq, gain_db, Q=1.0):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq / fs
    alpha = np.sin(w0) / (2 * Q)
    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A
    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0
    return signal.lfilter(b, a, x)


# ----------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------
class DSPApp:
    def __init__(self, root):
        self.root = root
        root.title("Violin Room Enhancer")
        root.geometry("560x860")

        self.input_path = tk.StringVar(value="No file selected")
        self.audio = None
        self.fs = None
        self.processed = None
        self.output_path = None

        # --- File selection ---
        file_frame = ttk.LabelFrame(root, text="1. Choose audio file")
        file_frame.pack(fill="x", padx=10, pady=10)

        ttk.Button(file_frame, text="Browse...", command=self.browse_file).pack(
            side="left", padx=5, pady=5
        )
        ttk.Label(file_frame, textvariable=self.input_path, wraplength=350).pack(
            side="left", padx=5
        )

        # --- Parameters ---
        param_frame = ttk.LabelFrame(root, text="2. Adjust parameters")
        param_frame.pack(fill="x", padx=10, pady=10)

        self.room_size = self._add_slider(param_frame, "Room size", 0.0, 1.0, 0.6)
        self.damping = self._add_slider(param_frame, "Damping", 0.0, 1.0, 0.4)
        self.wet_level = self._add_slider(param_frame, "Wet level (dry <-> wet)", 0.0, 1.0, 0.28)
        self.eq_freq = self._add_slider(param_frame, "EQ frequency (Hz)", 100, 2000, 500)
        self.eq_gain = self._add_slider(param_frame, "EQ gain (dB)", -12, 12, 3)
        self.eq_q = self._add_slider(param_frame, "EQ Q (bandwidth)", 0.3, 5.0, 1.2)

        # --- Actions ---
        action_frame = ttk.LabelFrame(root, text="3. Process & compare")
        action_frame.pack(fill="x", padx=10, pady=10)

        ttk.Button(action_frame, text="Process audio", command=self.process_audio).pack(
            fill="x", padx=5, pady=3
        )

        btn_row = ttk.Frame(action_frame)
        btn_row.pack(fill="x", padx=5, pady=3)
        ttk.Button(btn_row, text="Play original", command=self.play_original).pack(
            side="left", expand=True, fill="x", padx=2
        )
        ttk.Button(btn_row, text="Play enhanced", command=self.play_enhanced).pack(
            side="left", expand=True, fill="x", padx=2
        )
        ttk.Button(action_frame, text="Stop playback", command=sd.stop).pack(
            fill="x", padx=5, pady=3
        )

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(root, textvariable=self.status, foreground="gray").pack(
            fill="x", padx=10, pady=5
        )

        # --- Spectrum analyzer ---
        spectrum_frame = ttk.LabelFrame(root, text="4. Spectrum analyzer")
        spectrum_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.fig, self.ax = plt.subplots(figsize=(5.2, 3.2))
        self.ax.set_title("Frequency spectrum (process audio to populate)")
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Power (dB)")
        self.ax.set_xscale("log")
        self.ax.set_xlim(50, 20000)
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=spectrum_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

    def update_spectrum_plot(self):
        """Overlay original vs. enhanced spectra using Welch's method."""
        self.ax.clear()
        self.ax.set_title("Frequency spectrum: original vs. enhanced")
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Power (dB)")
        self.ax.set_xscale("log")
        self.ax.set_xlim(50, 20000)

        if self.audio is not None:
            freqs, psd = signal.welch(self.audio, self.fs, nperseg=4096)
            self.ax.semilogx(
                freqs, 10 * np.log10(psd + 1e-12), color="steelblue", label="Original"
            )
        if self.processed is not None:
            freqs, psd = signal.welch(self.processed, self.fs, nperseg=4096)
            self.ax.semilogx(
                freqs, 10 * np.log10(psd + 1e-12), color="darkorange", label="Enhanced"
            )

        self.ax.legend()
        self.ax.grid(True, which="both", alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw()

    def _add_slider(self, parent, label, lo, hi, default):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=5, pady=3)
        ttk.Label(row, text=label, width=22).pack(side="left")
        var = tk.DoubleVar(value=default)
        scale = ttk.Scale(row, from_=lo, to=hi, variable=var, orient="horizontal")
        scale.pack(side="left", fill="x", expand=True, padx=5)
        value_label = ttk.Label(row, width=6)
        value_label.pack(side="left")

        def update_label(*_):
            value_label.config(text=f"{var.get():.2f}")

        var.trace_add("write", update_label)
        update_label()
        return var

    def browse_file(self):
        path = filedialog.askopenfilename(
            title="Select a .wav file",
            initialdir=DOWNLOADS_DIR,
            filetypes=[("WAV files", "*.wav")],
        )
        if path:
            self.input_path.set(path)
            self.status.set(f"Loading {os.path.basename(path)}...")
            try:
                self.audio, self.fs = sf.read(path)
                if self.audio.ndim > 1:
                    self.audio = self.audio.mean(axis=1)
                self.status.set(
                    f"Loaded: {os.path.basename(path)} "
                    f"({len(self.audio)/self.fs:.1f}s, {self.fs} Hz)"
                )
                self.processed = None
                self.update_spectrum_plot()
            except Exception as e:
                messagebox.showerror("Error loading file", str(e))
                self.status.set("Failed to load file.")

    def process_audio(self):
        if self.audio is None:
            messagebox.showwarning("No file", "Choose a .wav file first.")
            return

        self.status.set("Processing...")
        self.root.update_idletasks()

        try:
            eq_out = peaking_eq(
                self.audio, self.fs, freq=self.eq_freq.get(), gain_db=self.eq_gain.get(),
                Q=self.eq_q.get()
            )
            wet = schroeder_reverb(
                eq_out,
                self.fs,
                room_size=self.room_size.get(),
                damping=self.damping.get(),
                wet_level=self.wet_level.get(),
            )
            peak = np.max(np.abs(wet))
            self.processed = wet / peak * 0.95 if peak > 0 else wet

            src = self.input_path.get()
            base, _ = os.path.splitext(os.path.basename(src))
            self.output_path = os.path.join(DOWNLOADS_DIR, f"{base}_enhanced.wav")
            sf.write(self.output_path, self.processed, self.fs)

            self.status.set(f"Saved: {self.output_path}")
            self.update_spectrum_plot()
        except Exception as e:
            messagebox.showerror("Processing error", str(e))
            self.status.set("Processing failed.")

    def play_original(self):
        if self.audio is None:
            messagebox.showwarning("No file", "Choose a .wav file first.")
            return
        sd.stop()
        sd.play(self.audio, self.fs)

    def play_enhanced(self):
        if self.processed is None:
            messagebox.showwarning("Not processed", "Click 'Process audio' first.")
            return
        sd.stop()
        sd.play(self.processed, self.fs)


if __name__ == "__main__":
    root = tk.Tk()
    app = DSPApp(root)
    root.mainloop()
