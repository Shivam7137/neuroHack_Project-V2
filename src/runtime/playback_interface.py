"""Preview-first desktop interface for synthetic OpenBCI text export."""

from __future__ import annotations

import os
import subprocess
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np

from src.generator.inference.sampler import GeneratorCondition, SyntheticSampler
from src.runtime.playback_tools import PlaybackRecording, build_synthetic_recording, write_openbci_txt


class PlaybackInterface(tk.Tk):
    """Simple EEG generator UI with live preview and OpenBCI export."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Synthetic EEG Generator")
        self.geometry("1280x820")
        self.minsize(1080, 720)

        self.export_dir = Path("artifacts/playback/exports")
        self.file_stem_var = tk.StringVar(value="synthetic_openbci")
        self.duration_var = tk.DoubleVar(value=12.0)
        self.concentration_var = tk.DoubleVar(value=0.7)
        self.stress_var = tk.DoubleVar(value=0.25)
        self.seed_var = tk.IntVar(value=42)
        self.status_var = tk.StringVar(value="Ready.")

        self.preview_recording: PlaybackRecording | None = None
        self._preview_job: str | None = None
        self._stream_job: str | None = None
        self._is_streaming = False
        self._stream_window_seconds = 3.0
        self._stream_chunk_seconds = 0.18
        self._streaming_sampler: SyntheticSampler | None = None
        self._stream_carry_state: dict[str, np.ndarray] | None = None
        self._live_channel_names: list[str] = []
        self._live_sample_rate = 250.0
        self._live_elapsed_seconds = 0.0
        self._live_signal_buffer = np.empty((0, 0), dtype=float)
        self._live_time_buffer = np.empty(0, dtype=float)
        self._live_write_index = 0
        self._live_size = 0
        self._max_live_buffer_samples = 0
        self._stream_tick_count = 0
        self._slow_panel_interval = 3
        self._trace_signature: tuple[int, int, tuple[str, ...], int] | None = None
        self._trace_title_item: int | None = None
        self._trace_label_items: list[int] = []
        self._trace_baseline_items: list[int] = []
        self._trace_wave_items: list[int] = []

        self.trace_canvas: tk.Canvas
        self.fft_canvas: tk.Canvas
        self.band_canvas: tk.Canvas
        self.headband_canvas: tk.Canvas

        self._build()
        self.after(50, self._refresh_preview)

    def _build(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        controls = ttk.Frame(self, padding=18)
        controls.grid(row=0, column=0, sticky="nsw")
        preview = ttk.Frame(self, padding=(0, 18, 18, 18))
        preview.grid(row=0, column=1, sticky="nsew")
        preview.columnconfigure(0, weight=1)
        preview.columnconfigure(1, weight=1)
        preview.rowconfigure(0, weight=3)
        preview.rowconfigure(1, weight=2)

        ttk.Label(controls, text="Synthetic EEG Generator", font=("Segoe UI", 18, "bold")).pack(anchor="w")
        ttk.Label(
            controls,
            text="Adjust concentration and stress, then start generation for a live 3-second rolling view. Export duration only affects the OpenBCI .txt file.",
            wraplength=280,
            justify="left",
        ).pack(anchor="w", pady=(8, 20))

        self._build_slider(
            controls,
            "Concentration",
            self.concentration_var,
            value_format="{:.2f}",
            from_=0.0,
            to=1.0,
            resolution=0.01,
        )
        self._build_slider(
            controls,
            "Stress",
            self.stress_var,
            value_format="{:.2f}",
            from_=0.0,
            to=1.0,
            resolution=0.01,
        )
        self._build_slider(
            controls,
            "Export Duration (sec)",
            self.duration_var,
            value_format="{:.1f}",
            from_=2.0,
            to=60.0,
            resolution=0.5,
        )

        seed_row = ttk.Frame(controls)
        seed_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(seed_row, text="Seed").pack(side=tk.LEFT)
        ttk.Spinbox(seed_row, from_=0, to=99999, textvariable=self.seed_var, width=10, command=self._schedule_preview).pack(
            side=tk.RIGHT
        )

        filename = ttk.LabelFrame(controls, text="Export", padding=12)
        filename.pack(fill=tk.X, pady=(20, 0))
        ttk.Label(filename, text="File name").pack(anchor="w")
        stem_entry = ttk.Entry(filename, textvariable=self.file_stem_var)
        stem_entry.pack(fill=tk.X, pady=(4, 8))
        stem_entry.bind("<KeyRelease>", lambda _event: self._set_status("File name updated."))
        ttk.Button(filename, text="Start Generation", command=self._start_stream).pack(fill=tk.X)
        ttk.Button(filename, text="Stop", command=self._stop_stream).pack(fill=tk.X, pady=(8, 0))
        ttk.Button(filename, text="Choose Folder", command=self._choose_export_dir).pack(fill=tk.X)
        ttk.Button(filename, text="Export OpenBCI TXT", command=self._export_openbci).pack(fill=tk.X, pady=(8, 0))
        ttk.Button(filename, text="Open Export Folder", command=self._open_export_folder).pack(fill=tk.X, pady=(8, 0))

        info = ttk.LabelFrame(controls, text="Status", padding=12)
        info.pack(fill=tk.X, pady=(20, 0))
        ttk.Label(info, textvariable=self.status_var, wraplength=280, justify="left").pack(anchor="w")

        trace_frame = ttk.LabelFrame(preview, text="Signals", padding=8)
        trace_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))
        trace_frame.rowconfigure(0, weight=1)
        trace_frame.columnconfigure(0, weight=1)
        self.trace_canvas = tk.Canvas(trace_frame, background="#0f172a", highlightthickness=0)
        self.trace_canvas.grid(row=0, column=0, sticky="nsew")

        fft_frame = ttk.LabelFrame(preview, text="FFT", padding=8)
        fft_frame.grid(row=0, column=1, sticky="nsew", pady=(0, 8))
        fft_frame.rowconfigure(0, weight=1)
        fft_frame.columnconfigure(0, weight=1)
        self.fft_canvas = tk.Canvas(fft_frame, background="#111827", highlightthickness=0)
        self.fft_canvas.grid(row=0, column=0, sticky="nsew")

        band_frame = ttk.LabelFrame(preview, text="Band Power", padding=8)
        band_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        band_frame.rowconfigure(0, weight=1)
        band_frame.columnconfigure(0, weight=1)
        self.band_canvas = tk.Canvas(band_frame, background="#111827", highlightthickness=0)
        self.band_canvas.grid(row=0, column=0, sticky="nsew")

        headband_frame = ttk.LabelFrame(preview, text="Headband Map", padding=8)
        headband_frame.grid(row=1, column=1, sticky="nsew")
        headband_frame.rowconfigure(0, weight=1)
        headband_frame.columnconfigure(0, weight=1)
        self.headband_canvas = tk.Canvas(headband_frame, background="#f8fafc", highlightthickness=0)
        self.headband_canvas.grid(row=0, column=0, sticky="nsew")

        for canvas in (self.trace_canvas, self.fft_canvas, self.band_canvas, self.headband_canvas):
            canvas.bind("<Configure>", lambda _event: self._redraw_preview())

    def _build_slider(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.DoubleVar,
        value_format: str,
        from_: float,
        to: float,
        resolution: float,
    ) -> None:
        wrapper = ttk.Frame(parent)
        wrapper.pack(fill=tk.X, pady=(0, 16))

        value_var = tk.StringVar(value=value_format.format(variable.get()))
        variable.trace_add("write", lambda *_args: value_var.set(value_format.format(variable.get())))

        row = ttk.Frame(wrapper)
        row.pack(fill=tk.X)
        ttk.Label(row, text=label).pack(side=tk.LEFT)
        ttk.Label(row, textvariable=value_var).pack(side=tk.RIGHT)

        scale = tk.Scale(
            wrapper,
            from_=from_,
            to=to,
            orient=tk.HORIZONTAL,
            variable=variable,
            resolution=resolution,
            showvalue=False,
            command=lambda _value: self._schedule_preview(),
            highlightthickness=0,
        )
        scale.pack(fill=tk.X, pady=(6, 0))

    def _schedule_preview(self) -> None:
        if self._preview_job is not None:
            self.after_cancel(self._preview_job)
        self._preview_job = self.after(120, self._refresh_preview)

    def _refresh_preview(self) -> None:
        self._preview_job = None
        if self._is_streaming:
            return
        try:
            self.preview_recording = build_synthetic_recording(
                duration_sec=self._stream_window_seconds,
                concentration=float(self.concentration_var.get()),
                stress=float(self.stress_var.get()),
                seed=int(self.seed_var.get()),
            )
        except Exception as exc:
            self._set_status(f"Preview failed: {exc}")
            return
        self._set_status(
            f"Previewing {self.preview_recording.signal.shape[0]} channels at "
            f"{self.preview_recording.sample_rate:.0f} Hz into {self.export_dir.resolve()}."
        )
        self._redraw_preview()

    def _redraw_preview(self) -> None:
        if self.preview_recording is None:
            return
        visible_recording = self._visible_recording()
        self._redraw_fast_panels(visible_recording)
        self._redraw_slow_panels(visible_recording)

    def _redraw_fast_panels(self, recording: PlaybackRecording) -> None:
        self._draw_signals(recording)

    def _redraw_slow_panels(self, recording: PlaybackRecording) -> None:
        self._draw_fft(recording)
        self._draw_bandpower(recording)
        self._draw_headband(recording)

    def _visible_recording(self) -> PlaybackRecording:
        if self._is_streaming and self._live_size:
            return self._snapshot_live_recording(source="live_preview")
        if self.preview_recording is None:
            raise RuntimeError("No recording available for preview.")
        return self.preview_recording

    def _start_stream(self) -> None:
        self._stop_stream(silent=True)
        self._streaming_sampler = SyntheticSampler(sample_rate=250.0, random_seed=int(self.seed_var.get()))
        self._stream_carry_state = None
        self._live_channel_names = list(self._streaming_sampler.channel_names)
        self._live_sample_rate = float(self._streaming_sampler.sample_rate)
        self._max_live_buffer_samples = max(1, int(round(self._stream_window_seconds * self._live_sample_rate)))
        self._live_signal_buffer = np.zeros((len(self._live_channel_names), self._max_live_buffer_samples), dtype=float)
        self._live_time_buffer = np.zeros(self._max_live_buffer_samples, dtype=float)
        self._live_write_index = 0
        self._live_size = 0
        self._live_elapsed_seconds = 0.0
        self._is_streaming = True
        self._stream_tick_count = 0
        self._set_status(
            f"Streaming live with a {self._stream_window_seconds:.1f}s rolling view. "
            f"Export duration is {float(self.duration_var.get()):.1f}s."
        )
        self._tick_stream()

    def _tick_stream(self) -> None:
        if not self._is_streaming or self._streaming_sampler is None:
            return
        condition = GeneratorCondition(
            concentration_level=float(self.concentration_var.get()),
            stress_level=float(self.stress_var.get()),
        )
        sample = self._streaming_sampler.sample(
            condition,
            duration_sec=self._stream_chunk_seconds,
            carry_state=self._stream_carry_state,
        )
        self._stream_carry_state = dict(sample.carry_state)
        self._append_live_chunk(sample.data, sample.sample_rate)
        visible_recording = self._visible_recording()
        self._redraw_fast_panels(visible_recording)
        self._stream_tick_count += 1
        if self._stream_tick_count == 1 or self._stream_tick_count % self._slow_panel_interval == 0:
            self._redraw_slow_panels(visible_recording)
        self._set_status(
            f"Streaming live {self._stream_window_seconds:.1f}s view at {sample.sample_rate:.0f} Hz. "
            f"Elapsed {self._live_elapsed_seconds:.1f}s. Export duration {float(self.duration_var.get()):.1f}s."
        )
        self._stream_job = self.after(int(round(self._stream_chunk_seconds * 1000.0)), self._tick_stream)

    def _stop_stream(self, silent: bool = False) -> None:
        if self._stream_job is not None:
            self.after_cancel(self._stream_job)
            self._stream_job = None
        was_streaming = self._is_streaming
        self._is_streaming = False
        if self._live_size:
            self.preview_recording = self._snapshot_live_recording(source="stopped_live_preview")
        self._streaming_sampler = None
        self._stream_carry_state = None
        self._redraw_preview()
        if was_streaming and not silent:
            self._set_status("Streaming stopped.")

    def _append_live_chunk(self, chunk: np.ndarray, sample_rate: float) -> None:
        if self._max_live_buffer_samples <= 0:
            return
        chunk_samples = chunk.shape[1]
        if chunk_samples >= self._max_live_buffer_samples:
            chunk = chunk[:, -self._max_live_buffer_samples :]
            chunk_samples = chunk.shape[1]
        sample_times = self._live_elapsed_seconds + (np.arange(chunk_samples, dtype=float) / sample_rate)
        indices = (self._live_write_index + np.arange(chunk_samples, dtype=int)) % self._max_live_buffer_samples
        self._live_signal_buffer[:, indices] = chunk
        self._live_time_buffer[indices] = sample_times
        self._live_write_index = int((self._live_write_index + chunk_samples) % self._max_live_buffer_samples)
        self._live_size = min(self._live_size + chunk_samples, self._max_live_buffer_samples)
        self._live_elapsed_seconds = float(sample_times[-1] + (1.0 / sample_rate))

    def _snapshot_live_recording(self, source: str) -> PlaybackRecording:
        if self._live_size <= 0 or self._max_live_buffer_samples <= 0:
            return PlaybackRecording(
                signal=np.empty((len(self._live_channel_names), 0), dtype=float),
                sample_rate=self._live_sample_rate,
                channel_names=list(self._live_channel_names),
                timestamps=np.empty(0, dtype=float),
                metadata={
                    "source": source,
                    "concentration_level": float(self.concentration_var.get()),
                    "stress_level": float(self.stress_var.get()),
                },
            )
        start = (self._live_write_index - self._live_size) % self._max_live_buffer_samples
        ordered_indices = (start + np.arange(self._live_size, dtype=int)) % self._max_live_buffer_samples
        return PlaybackRecording(
            signal=self._live_signal_buffer[:, ordered_indices].copy(),
            sample_rate=self._live_sample_rate,
            channel_names=list(self._live_channel_names),
            timestamps=self._live_time_buffer[ordered_indices].copy(),
            metadata={
                "source": source,
                "concentration_level": float(self.concentration_var.get()),
                "stress_level": float(self.stress_var.get()),
            },
        )

    def _draw_signals(self, recording: PlaybackRecording) -> None:
        canvas = self.trace_canvas
        width = max(canvas.winfo_width(), 1)
        height = max(canvas.winfo_height(), 1)
        signal = recording.signal
        sample_rate = recording.sample_rate
        window_samples = max(1, min(signal.shape[1], int(round(sample_rate * self._stream_window_seconds))))
        segment = signal[:, -window_samples:]
        colors = ["#60a5fa", "#f472b6", "#34d399", "#fbbf24", "#f97316", "#ef4444", "#a78bfa", "#22d3ee"]
        point_count = min(segment.shape[1], 240)
        signature = (width, height, tuple(recording.channel_names), point_count)
        if signature != self._trace_signature:
            self._rebuild_signal_canvas(signature, recording.channel_names, colors)

        if self._trace_title_item is not None:
            canvas.itemconfigure(self._trace_title_item, text=f"Last {self._stream_window_seconds:.1f} s")
        row_height = (height - 28) / max(segment.shape[0], 1)
        point_indices = np.linspace(0, segment.shape[1] - 1, point_count, dtype=int)
        x = np.linspace(52.0, width - 8.0, point_count, dtype=float)
        for idx, channel_name in enumerate(recording.channel_names):
            top = 28 + idx * row_height
            mid = top + row_height / 2.0
            reduced = segment[idx, point_indices]
            band = float(np.max(np.abs(reduced))) or 1.0
            y = mid - (reduced / band) * (row_height * 0.35)
            points = np.empty(point_count * 2, dtype=float)
            points[0::2] = x
            points[1::2] = y
            if idx < len(self._trace_baseline_items):
                canvas.coords(self._trace_baseline_items[idx], 48, mid, width - 8, mid)
            if idx < len(self._trace_wave_items):
                canvas.coords(self._trace_wave_items[idx], *points.tolist())
            if idx < len(self._trace_label_items):
                canvas.coords(self._trace_label_items[idx], 10, mid)
                canvas.itemconfigure(self._trace_label_items[idx], text=channel_name)

    def _rebuild_signal_canvas(
        self,
        signature: tuple[int, int, tuple[str, ...], int],
        channel_names: list[str],
        colors: list[str],
    ) -> None:
        canvas = self.trace_canvas
        canvas.delete("all")
        self._trace_signature = signature
        self._trace_label_items = []
        self._trace_baseline_items = []
        self._trace_wave_items = []
        width, height, _, _point_count = signature
        self._trace_title_item = canvas.create_text(
            12,
            14,
            text=f"Last {self._stream_window_seconds:.1f} s",
            fill="#cbd5e1",
            anchor="w",
            font=("Segoe UI", 10, "bold"),
        )
        row_height = (height - 28) / max(len(channel_names), 1)
        for idx, channel_name in enumerate(channel_names):
            mid = 28 + idx * row_height + row_height / 2.0
            self._trace_baseline_items.append(
                canvas.create_line(48, mid, width - 8, mid, fill="#233046", width=1)
            )
            self._trace_wave_items.append(
                canvas.create_line(52, mid, width - 8, mid, fill=colors[idx % len(colors)], width=2)
            )
            self._trace_label_items.append(
                canvas.create_text(10, mid, text=channel_name, fill="#e5e7eb", anchor="w", font=("Segoe UI", 9))
            )

    def _draw_fft(self, recording: PlaybackRecording) -> None:
        canvas = self.fft_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 1)
        height = max(canvas.winfo_height(), 1)
        if recording.signal.shape[1] < 16:
            return
        signal = recording.signal - recording.signal.mean(axis=1, keepdims=True)
        freqs = np.fft.rfftfreq(signal.shape[1], d=1.0 / recording.sample_rate)
        spectrum = np.abs(np.fft.rfft(signal, axis=1)).mean(axis=0)
        mask = freqs <= 45.0
        freqs = freqs[mask]
        spectrum = spectrum[mask]

        canvas.create_text(12, 14, text="0-45 Hz average spectrum", fill="#d1d5db", anchor="w", font=("Segoe UI", 10, "bold"))
        plot_left, plot_top, plot_right, plot_bottom = 42, 26, width - 12, height - 24
        canvas.create_rectangle(plot_left, plot_top, plot_right, plot_bottom, outline="#243244")
        if spectrum.size == 0:
            return
        scale = float(np.max(spectrum)) or 1.0
        xs = plot_left + (freqs / max(float(freqs[-1]), 1.0)) * (plot_right - plot_left)
        ys = plot_bottom - (spectrum / scale) * (plot_bottom - plot_top - 12)
        points: list[float] = []
        for px, py in zip(xs, ys, strict=True):
            points.extend((float(px), float(py)))
        canvas.create_line(points, fill="#f59e0b", width=2)
        for marker in (6, 10, 20, 30, 40):
            x = plot_left + (marker / 45.0) * (plot_right - plot_left)
            canvas.create_line(x, plot_top, x, plot_bottom, fill="#1f2937", dash=(3, 4))
            canvas.create_text(x, plot_bottom + 12, text=str(marker), fill="#9ca3af", font=("Segoe UI", 8))

    def _draw_bandpower(self, recording: PlaybackRecording) -> None:
        canvas = self.band_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 1)
        height = max(canvas.winfo_height(), 1)
        if recording.signal.shape[1] < 16:
            return

        signal = recording.signal - recording.signal.mean(axis=1, keepdims=True)
        freqs = np.fft.rfftfreq(signal.shape[1], d=1.0 / recording.sample_rate)
        power = (np.abs(np.fft.rfft(signal, axis=1)) ** 2).mean(axis=0)
        bands = [
            ("Delta", 0.5, 4.0, "#60a5fa"),
            ("Theta", 4.0, 8.0, "#a78bfa"),
            ("Alpha", 8.0, 13.0, "#34d399"),
            ("Beta", 13.0, 32.0, "#f59e0b"),
            ("Gamma", 32.0, 45.0, "#f87171"),
        ]
        values: list[float] = []
        for _label, low, high, _color in bands:
            mask = (freqs >= low) & (freqs < high)
            values.append(float(power[mask].mean()) if np.any(mask) else 0.0)

        canvas.create_text(12, 14, text="Average spectral energy by band", fill="#d1d5db", anchor="w", font=("Segoe UI", 10, "bold"))
        max_value = max(values) if values else 1.0
        bar_width = (width - 40) / max(len(bands), 1)
        for idx, (label, _low, _high, color) in enumerate(bands):
            left = 20 + idx * bar_width
            right = left + bar_width - 12
            bottom = height - 34
            top = bottom - ((values[idx] / max_value) * (height - 70) if max_value > 0 else 0.0)
            canvas.create_rectangle(left, top, right, bottom, fill=color, outline="")
            canvas.create_text((left + right) / 2.0, bottom + 14, text=label, fill="#d1d5db", font=("Segoe UI", 9))

    def _draw_headband(self, recording: PlaybackRecording) -> None:
        canvas = self.headband_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 1)
        height = max(canvas.winfo_height(), 1)
        center_x = width / 2.0
        center_y = height / 2.0 + 6.0
        radius = min(width, height) * 0.34

        canvas.create_oval(
            center_x - radius,
            center_y - radius * 1.08,
            center_x + radius,
            center_y + radius * 0.92,
            outline="#cbd5e1",
            width=2,
        )
        canvas.create_line(center_x, center_y - radius * 1.12, center_x, center_y + radius * 0.86, fill="#e2e8f0")
        canvas.create_line(center_x - radius * 0.92, center_y, center_x + radius * 0.92, center_y, fill="#e2e8f0")
        canvas.create_polygon(
            center_x - 12,
            center_y - radius * 1.08,
            center_x + 12,
            center_y - radius * 1.08,
            center_x,
            center_y - radius * 1.28,
            fill="#cbd5e1",
            outline="",
        )

        positions = {
            "Fp1": (-0.55, -0.82),
            "Fp2": (0.55, -0.82),
            "C3": (-0.72, -0.02),
            "C4": (0.72, -0.02),
            "P3": (-0.55, 0.5),
            "P4": (0.55, 0.5),
            "O1": (-0.38, 0.9),
            "O2": (0.38, 0.9),
        }
        channel_power = np.sqrt(np.mean(np.square(recording.signal), axis=1))
        max_power = float(np.max(channel_power)) or 1.0
        for idx, channel_name in enumerate(recording.channel_names):
            x_factor, y_factor = positions.get(channel_name, (0.0, 0.0))
            x = center_x + radius * x_factor
            y = center_y + radius * y_factor
            value = float(channel_power[idx] / max_power)
            color = self._activity_color(value)
            canvas.create_oval(x - 18, y - 18, x + 18, y + 18, fill=color, outline="#0f172a", width=1)
            canvas.create_text(x, y, text=channel_name, fill="#0f172a", font=("Segoe UI", 9, "bold"))

        canvas.create_text(
            12,
            16,
            text="Relative channel activity",
            fill="#0f172a",
            anchor="w",
            font=("Segoe UI", 10, "bold"),
        )

    def _activity_color(self, value: float) -> str:
        value = float(np.clip(value, 0.0, 1.0))
        low = np.array([191, 219, 254], dtype=float)
        high = np.array([37, 99, 235], dtype=float)
        rgb = np.rint(low + (high - low) * value).astype(int)
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    def _choose_export_dir(self) -> None:
        selected = filedialog.askdirectory(title="Choose export directory", initialdir=str(self.export_dir))
        if selected:
            self.export_dir = Path(selected)
            self._set_status(f"Export folder set to {self.export_dir.resolve()}.")

    def _export_openbci(self) -> None:
        default_path = self.export_dir / f"{self.file_stem_var.get().strip() or 'synthetic_openbci'}.txt"
        selected = filedialog.asksaveasfilename(
            title="Export OpenBCI text file",
            initialdir=str(default_path.parent),
            initialfile=default_path.name,
            defaultextension=".txt",
            filetypes=[("OpenBCI text", "*.txt")],
        )
        if not selected:
            return
        try:
            export_recording = build_synthetic_recording(
                duration_sec=float(self.duration_var.get()),
                concentration=float(self.concentration_var.get()),
                stress=float(self.stress_var.get()),
                seed=int(self.seed_var.get()),
            )
            output = write_openbci_txt(
                export_recording.signal,
                export_recording.sample_rate,
                Path(selected),
            )
        except Exception as exc:
            messagebox.showerror("Export OpenBCI TXT", str(exc))
            return
        self._set_status(f"Exported OpenBCI file: {output}")
        messagebox.showinfo("Export OpenBCI TXT", f"Saved:\n{output}")

    def _open_export_folder(self) -> None:
        export_dir = self.export_dir.resolve()
        export_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(export_dir)  # type: ignore[attr-defined]
        except AttributeError:
            subprocess.run(["explorer", str(export_dir)], check=False)
        self._set_status(f"Opened export folder: {export_dir}")

    def _set_status(self, message: str) -> None:
        self.status_var.set(message)


def main() -> None:
    app = PlaybackInterface()
    app.mainloop()


if __name__ == "__main__":
    main()
