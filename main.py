

from __future__ import annotations

import os
import queue
import threading
from collections import deque

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy import signal
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from violin_dsp_gui_upgraded import DSPApp, peaking_coefficients


EPS = 1e-12


class StatefulSOS:
    def __init__(self, sos, channels):
        self.sos = np.asarray(sos)
        self.zi = np.zeros((len(self.sos), 2, channels))

    def process(self, block):
        y, self.zi = signal.sosfilt(self.sos, block, axis=0, zi=self.zi)
        return y


class StatefulBiquad:
    def __init__(self, fs, freq, gain, q, channels):
        self.b, self.a = peaking_coefficients(fs, freq, gain, q)
        self.zi = np.zeros((2, channels))

    def process(self, block):
        y, self.zi = signal.lfilter(self.b, self.a, block, axis=0, zi=self.zi)
        return y


class LiveGate:
    def __init__(self, fs, threshold_db, channels, attack_ms=8, release_ms=120):
        self.threshold = 10 ** (threshold_db / 20)
        self.floor = 10 ** (-70 / 20)
        self.attack = np.exp(-1 / max(1, fs * attack_ms / 1000))
        self.release = np.exp(-1 / max(1, fs * release_ms / 1000))
        self.gain = 1.0
        self.channels = channels

    def process(self, block):
        out = np.empty_like(block)
        for n in range(len(block)):
            target = 1.0 if np.max(np.abs(block[n])) >= self.threshold else self.floor
            coefficient = self.attack if target > self.gain else self.release
            self.gain = coefficient * self.gain + (1 - coefficient) * target
            out[n] = block[n] * self.gain
        return out


class LiveCompressor:
    def __init__(self, fs, threshold_db, ratio, attack_ms, release_ms,
                 knee_db, makeup_db):
        self.threshold_db = threshold_db
        self.ratio = max(1.0, ratio)
        self.knee_db = max(0.0, knee_db)
        self.makeup = 10 ** (makeup_db / 20)
        self.attack = np.exp(-1 / max(1, fs * attack_ms / 1000))
        self.release = np.exp(-1 / max(1, fs * release_ms / 1000))
        self.gain = 1.0
        self.reduction_db = 0.0

    def _target(self, level_db):
        over = level_db - self.threshold_db
        slope = 1 / self.ratio - 1
        if self.knee_db <= 0:
            reduction = slope * max(over, 0)
        elif over <= -self.knee_db / 2:
            reduction = 0.0
        elif over >= self.knee_db / 2:
            reduction = slope * over
        else:
            reduction = slope * (over + self.knee_db / 2) ** 2 / (2 * self.knee_db)
        self.reduction_db = reduction
        return 10 ** (reduction / 20)

    def process(self, block):
        out = np.empty_like(block)
        for n in range(len(block)):
            level_db = 20 * np.log10(np.max(np.abs(block[n])) + EPS)
            target = self._target(level_db)
            coefficient = self.attack if target < self.gain else self.release
            self.gain = coefficient * self.gain + (1 - coefficient) * target
            out[n] = block[n] * self.gain * self.makeup
        return out


class DelayComb:
    def __init__(self, delay, gain, channels, damping):
        self.buffer = np.zeros((max(1, delay), channels))
        self.index = 0
        self.gain = gain
        self.damping = np.clip(damping, 0, 1)
        self.lowpass = np.zeros(channels)

    def process_sample(self, sample):
        delayed = self.buffer[self.index].copy()
        # One-pole damping in the feedback path: higher damping darkens the tail.
        coefficient = 0.15 + 0.8 * self.damping
        self.lowpass = coefficient * self.lowpass + (1 - coefficient) * delayed
        self.buffer[self.index] = sample + self.gain * self.lowpass
        self.index = (self.index + 1) % len(self.buffer)
        return delayed


class DelayAllpass:
    def __init__(self, delay, gain, channels):
        self.buffer = np.zeros((max(1, delay), channels))
        self.index = 0
        self.gain = gain

    def process_sample(self, sample):
        delayed = self.buffer[self.index].copy()
        output = -self.gain * sample + delayed
        self.buffer[self.index] = sample + self.gain * output
        self.index = (self.index + 1) % len(self.buffer)
        return output


class LiveSchroeder:
    def __init__(self, fs, channels, rt60, damping, predelay_ms, wet):
        self.wet = np.clip(wet, 0, 1)
        self.predelay = np.zeros((max(1, int(fs * predelay_ms / 1000)), channels))
        self.predelay_index = 0
        self.combs = []
        for delay_ms in (29.7, 37.1, 41.1, 43.7):
            delay = max(1, int(fs * delay_ms / 1000))
            gain = 10 ** (-3 * (delay / fs) / max(rt60, 0.1))
            self.combs.append(DelayComb(delay, gain, channels, damping))
        self.allpasses = [DelayAllpass(max(1, int(fs * ms / 1000)), .7, channels)
                          for ms in (5.0, 1.7)]

    def process(self, block):
        out = np.empty_like(block)
        for n, dry in enumerate(block):
            driven = self.predelay[self.predelay_index].copy()
            self.predelay[self.predelay_index] = dry
            self.predelay_index = (self.predelay_index + 1) % len(self.predelay)
            reverberant = sum(comb.process_sample(driven) for comb in self.combs) / len(self.combs)
            for allpass in self.allpasses:
                reverberant = allpass.process_sample(reverberant)
            out[n] = (1 - self.wet) * dry + self.wet * reverberant
        return out


class LiveDSPChain:
    """Block processor whose filter, envelope, and delay state survives callbacks."""

    def __init__(self, fs, channels, settings):
        self.fs = fs
        self.channels = channels
        self.gate = (LiveGate(fs, settings["gate_threshold"], channels)
                     if settings["gate"] else None)
        self.filters = []
        nyquist = fs / 2
        if settings["hp"]:
            cutoff = np.clip(settings["hp_hz"], 10, nyquist * .95)
            self.filters.append(StatefulSOS(signal.butter(4, cutoff, "highpass", fs=fs, output="sos"), channels))
        if settings["lp"]:
            cutoff = np.clip(settings["lp_hz"], 20, nyquist * .98)
            self.filters.append(StatefulSOS(signal.butter(4, cutoff, "lowpass", fs=fs, output="sos"), channels))
        self.eq = [StatefulBiquad(fs, freq, gain, q, channels)
                   for enabled, freq, gain, q in settings["bands"] if enabled and abs(gain) > 1e-6]
        self.compressor = (LiveCompressor(fs, settings["threshold"], settings["ratio"],
                                          settings["attack"], settings["release"],
                                          settings["knee"], settings["makeup"])
                           if settings["compress"] else None)
        self.reverb = None
        if settings["reverb"]:
            # Low-latency live mode deliberately uses Schroeder reverb. The offline
            # mode continues to support long-IR FFT convolution.
            self.reverb = LiveSchroeder(fs, channels, settings["rt60"], settings["damping"],
                                        settings["predelay"], settings["wet"])

    def process(self, block):
        y = np.asarray(block, dtype=np.float64)
        if self.gate: y = self.gate.process(y)
        for audio_filter in self.filters: y = audio_filter.process(y)
        for eq_filter in self.eq: y = eq_filter.process(y)
        if self.compressor: y = self.compressor.process(y)
        if self.reverb: y = self.reverb.process(y)
        # Keep the normal live path linear for controlled listening tests.
        # Only samples outside the valid digital range are clipped.
        return np.clip(y, -0.999, 0.999)


class LiveDSPApp(DSPApp):
    def __init__(self, root):
        self.mode = tk.StringVar(value="File processing")
        mode_frame = ttk.LabelFrame(root, text="Operating mode")
        mode_frame.pack(fill="x", padx=10, pady=(10, 0))
        ttk.Radiobutton(mode_frame, text="Process an existing file", variable=self.mode,
                        value="File processing", command=self._update_mode).pack(side="left", padx=8, pady=5)
        ttk.Radiobutton(mode_frame, text="Use live input", variable=self.mode,
                        value="Live input", command=self._update_mode).pack(side="left", padx=8, pady=5)
        super().__init__(root)
        self.root.title("Violin DSP Workstation - File + Live")
        self.stream = None
        self.live_chain = None
        self.pending_chain = None
        self.live_settings_signature = None
        self.recording = False
        self.recorded_blocks = []
        self.meter_queue = queue.Queue(maxsize=4)
        self._build_live_panel()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(60, self._poll_meter)
        self.root.after(150, self._poll_live_settings)
        self._update_mode()

    def _build_live_panel(self):
        self.live_frame = ttk.LabelFrame(self.root, text="Live input and processed playback")
        # Put live controls before the analysis area.
        analysis_frame = self.plot_tabs.master
        self.live_frame.pack(fill="x", padx=10, pady=5, before=analysis_frame)
        device_row = ttk.Frame(self.live_frame); device_row.pack(fill="x", padx=5, pady=3)
        ttk.Label(device_row, text="Input device", width=14).pack(side="left")
        self.input_device = tk.StringVar()
        self.input_combo = ttk.Combobox(device_row, textvariable=self.input_device, state="readonly", width=32)
        self.input_combo.pack(side="left", padx=3)
        ttk.Label(device_row, text="Output device", width=14).pack(side="left", padx=(12, 0))
        self.output_device = tk.StringVar()
        self.output_combo = ttk.Combobox(device_row, textvariable=self.output_device, state="readonly", width=32)
        self.output_combo.pack(side="left", padx=3)

        options = ttk.Frame(self.live_frame); options.pack(fill="x", padx=5, pady=3)
        self.live_samplerate = tk.IntVar(value=48000)
        self.live_blocksize = tk.IntVar(value=512)
        self.live_channels = tk.IntVar(value=1)
        for label, variable, values in (
            ("Sample rate", self.live_samplerate, (44100, 48000, 96000)),
            ("Buffer", self.live_blocksize, (64, 128, 256, 512, 1024)),
            ("Channels", self.live_channels, (1, 2)),
        ):
            ttk.Label(options, text=label).pack(side="left", padx=(5, 2))
            ttk.Combobox(options, textvariable=variable, values=values, state="readonly", width=8).pack(side="left")
        self.record_live = tk.BooleanVar(value=False)
        ttk.Checkbutton(options, text="Record processed audio", variable=self.record_live).pack(side="left", padx=15)

        buttons = ttk.Frame(self.live_frame); buttons.pack(fill="x", padx=5, pady=4)
        self.start_live_button = ttk.Button(buttons, text="Start live playback", command=self.start_live)
        self.start_live_button.pack(side="left", expand=True, fill="x", padx=2)
        self.stop_live_button = ttk.Button(buttons, text="Stop live playback", command=self.stop_live)
        self.stop_live_button.pack(side="left", expand=True, fill="x", padx=2)
        self.live_level = tk.DoubleVar(value=0)
        ttk.Label(buttons, text="Output").pack(side="left", padx=(12, 3))
        self.meter = ttk.Progressbar(buttons, variable=self.live_level, maximum=1, length=150)
        self.meter.pack(side="left", padx=3)
        self.refresh_devices()

    def refresh_devices(self):
        try:
            devices = sd.query_devices()
            inputs = [f"{i}: {d['name']}" for i, d in enumerate(devices) if d["max_input_channels"] > 0]
            outputs = [f"{i}: {d['name']}" for i, d in enumerate(devices) if d["max_output_channels"] > 0]
            self.input_combo["values"] = inputs; self.output_combo["values"] = outputs
            default_in, default_out = sd.default.device
            self._choose_default(self.input_combo, self.input_device, inputs, default_in)
            self._choose_default(self.output_combo, self.output_device, outputs, default_out)
        except Exception as exc:
            self.status.set(f"Could not enumerate audio devices: {exc}")

    @staticmethod
    def _choose_default(combo, variable, choices, index):
        match = next((item for item in choices if item.startswith(f"{index}:")), None)
        if match: variable.set(match)
        elif choices: combo.current(0)

    @staticmethod
    def _device_index(text):
        return int(text.split(":", 1)[0])

    def _update_mode(self):
        if not hasattr(self, "live_frame"): return
        live = self.mode.get() == "Live input"
        state = "normal" if live else "disabled"
        self.start_live_button.configure(state=state)
        self.stop_live_button.configure(state=state)
        if not live and self.stream is not None: self.stop_live()

    def start_live(self):
        if self.stream is not None: return
        if not self.input_device.get() or not self.output_device.get():
            messagebox.showwarning("Audio device", "Select both an input and output device."); return
        try:
            fs = self.live_samplerate.get(); channels = self.live_channels.get()
            settings = self._settings()
            if settings["denoise"]:
                self.status.set("Live spectral denoising is bypassed to preserve low latency; the gate remains available.")
            if settings["mode"] == "Convolution IR":
                self.status.set("Live monitoring uses RT60 Schroeder reverb; convolution remains available in file mode.")
            self.live_chain = LiveDSPChain(fs, channels, settings)
            self.pending_chain = None
            self.live_settings_signature = repr(settings)
            self.recorded_blocks = []; self.recording = self.record_live.get()
            self.stream = sd.Stream(
                samplerate=fs, blocksize=self.live_blocksize.get(), channels=channels,
                dtype="float32", latency="low",
                device=(self._device_index(self.input_device.get()), self._device_index(self.output_device.get())),
                callback=self._audio_callback, finished_callback=self._stream_finished,
            )
            self.stream.start()
            latency_ms = 1000 * self.live_blocksize.get() / fs
            self.status.set(f"Live processed playback running; one buffer = {latency_ms:.1f} ms. Use headphones.")
        except Exception as exc:
            self.stream = None; self.live_chain = None
            messagebox.showerror("Live audio error", str(exc)); self.status.set("Could not start live playback.")

    def _audio_callback(self, indata, outdata, frames, time_info, status):
        try:
            processed = self.live_chain.process(indata)
            # A replacement chain is prepared by the GUI thread whenever a
            # control changes. Crossfade for one block to avoid an abrupt edge.
            replacement = self.pending_chain
            if replacement is not None:
                changed = replacement.process(indata)
                fade = np.linspace(0, 1, len(indata), endpoint=True)[:, None]
                processed = processed * (1 - fade) + changed * fade
                self.live_chain = replacement
                self.pending_chain = None
            if not np.isfinite(processed).all(): processed = np.zeros_like(indata)
            outdata[:] = processed.astype(np.float32, copy=False)
            if self.recording: self.recorded_blocks.append(outdata.copy())
            level = float(np.max(np.abs(outdata)))
            try: self.meter_queue.put_nowait(level)
            except queue.Full: pass
        except Exception:
            outdata.fill(0)
            raise sd.CallbackAbort

    def _poll_meter(self):
        level = None
        try:
            while True: level = self.meter_queue.get_nowait()
        except queue.Empty:
            pass
        if level is not None: self.live_level.set(min(level, 1.0))
        self.root.after(60, self._poll_meter)

    def _poll_live_settings(self):
        """Apply GUI edits to a running stream without stopping audio."""
        if self.stream is not None:
            try:
                settings = self._settings()
                signature = repr(settings)
                if signature != self.live_settings_signature:
                    self.pending_chain = LiveDSPChain(
                        self.live_samplerate.get(), self.live_channels.get(), settings)
                    self.live_settings_signature = signature
            except Exception as exc:
                self.status.set(f"Live setting could not be applied: {exc}")
        self.root.after(150, self._poll_live_settings)

    def _stream_finished(self):
        pass

    def stop_live(self):
        stream, self.stream = self.stream, None
        if stream is not None:
            try: stream.stop(); stream.close()
            except Exception: pass
        self.live_chain = None; self.live_level.set(0)
        self.pending_chain = None
        if self.recorded_blocks:
            self._save_live_recording()
        else:
            self.status.set("Live playback stopped.")

    def _save_live_recording(self):
        path = filedialog.asksaveasfilename(
            title="Save processed live recording", initialdir=os.path.join(os.path.expanduser("~"), "Downloads"),
            defaultextension=".wav", filetypes=[("WAV files", "*.wav")],
            initialfile="live_processed.wav")
        if path:
            try:
                sf.write(path, np.concatenate(self.recorded_blocks, axis=0), self.live_samplerate.get(), subtype="PCM_24")
                self.status.set(f"Live playback stopped. Recording saved: {path}")
            except Exception as exc:
                messagebox.showerror("Recording error", str(exc))
        else:
            self.status.set("Live playback stopped; recording was not saved.")
        self.recorded_blocks = []

    def _close(self):
        self.recorded_blocks = []
        self.stop_live()
        sd.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    LiveDSPApp(root)
    root.mainloop()
