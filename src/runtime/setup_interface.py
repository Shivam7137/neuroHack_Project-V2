"""Dedicated desktop setup UI for adaptive user calibration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk

from src.runtime.calibration_controller import CalibrationController, CalibrationEvent, build_calibration_source
from src.runtime.constants import RUNTIME_SETUP_CHANNELS


@dataclass(slots=True)
class StimulusFrame:
    """Pure animation state for one render frame."""

    scale: float
    jitter_x: float
    jitter_y: float
    fill: str
    outline: str
    glow: str
    label: str = ""


def compute_stimulus_frame(animation_mode: str, elapsed_seconds: float, modifier_level: int) -> StimulusFrame:
    """Return the current animated circle state for the setup UI."""
    t = float(max(elapsed_seconds, 0.0))
    modifier = max(int(modifier_level), 0)
    if animation_mode == "focused":
        period = max(4.0, 5.0 - 0.2 * modifier)
        phase = (t % period) / period
        wave = 0.5 - 0.5 * __import__("math").cos(2.0 * __import__("math").pi * phase)
        label = "inhale" if wave >= 0.5 else "exhale"
        return StimulusFrame(
            scale=0.88 + 0.22 * wave,
            jitter_x=0.0,
            jitter_y=0.0,
            fill="#3b82f6",
            outline="#93c5fd",
            glow="#1d4ed8",
            label=label,
        )
    if animation_mode == "stressed":
        import math

        wave = (
            math.sin((1.6 + modifier * 0.15) * t * 2.0 * math.pi)
            + 0.45 * math.sin((2.7 + modifier * 0.20) * t * 2.0 * math.pi + 0.8)
            + 0.20 * math.sin(4.4 * t * 2.0 * math.pi + 1.6)
        )
        scale = 0.96 + 0.14 * max(-0.8, min(wave, 1.0))
        jitter_x = 8.0 * math.sin(6.8 * t + modifier)
        jitter_y = 5.0 * math.cos(8.1 * t + 0.5 * modifier)
        countdown = max(1, 4 - int((t * (1.15 + modifier * 0.08)) % 4))
        return StimulusFrame(
            scale=scale,
            jitter_x=jitter_x,
            jitter_y=jitter_y,
            fill="#ef4444",
            outline="#fca5a5",
            glow="#991b1b",
            label=str(countdown),
        )
    if animation_mode == "validation":
        import math

        wave = 0.5 - 0.5 * math.cos(2.0 * math.pi * ((t % 3.5) / 3.5))
        return StimulusFrame(
            scale=0.94 + 0.08 * wave,
            jitter_x=0.0,
            jitter_y=0.0,
            fill="#14b8a6",
            outline="#99f6e4",
            glow="#115e59",
            label="live",
        )
    if animation_mode in {"rest", "recovery"}:
        import math

        wave = 0.5 - 0.5 * math.cos(2.0 * math.pi * ((t % 6.0) / 6.0))
        return StimulusFrame(
            scale=0.92 + 0.06 * wave,
            jitter_x=0.0,
            jitter_y=0.0,
            fill="#6b7a90",
            outline="#cbd5e1",
            glow="#334155",
            label="steady",
        )
    if animation_mode == "idle":
        import math

        return StimulusFrame(
            scale=0.98 + 0.01 * math.sin(t * 0.8),
            jitter_x=0.0,
            jitter_y=0.0,
            fill="#94a3b8",
            outline="#cbd5e1",
            glow="#475569",
            label="awake",
        )
    return StimulusFrame(
        scale=1.0,
        jitter_x=0.0,
        jitter_y=0.0,
        fill="#94a3b8",
        outline="#e2e8f0",
        glow="#334155",
        label="check",
    )


class SetupInterface(tk.Tk):
    """Minimal desktop interface for adaptive setup and user profile capture."""

    def __init__(
        self,
        *,
        default_source: str = "generator",
        default_serial_port: str = "COM5",
        default_user_id: str = "demo_user",
        default_artifacts_root: str = "artifacts/runtime_v1",
    ) -> None:
        super().__init__()
        self.title("NeuroHack Setup")
        self.geometry("1120x780")
        self.minsize(960, 700)
        self.configure(background="#0f172a")

        self.source_var = tk.StringVar(value=default_source)
        self.serial_port_var = tk.StringVar(value=default_serial_port)
        self.user_id_var = tk.StringVar(value=default_user_id)
        self.artifacts_root_var = tk.StringVar(value=str(Path(default_artifacts_root).expanduser().resolve()))
        self.status_var = tk.StringVar(value="Ready to start setup.")
        self.phase_title_var = tk.StringVar(value="Setup")
        self.instruction_var = tk.StringVar(value="Choose a source, then start the guided calibration.")
        self.concentration_var = tk.StringVar(value="Concentration --")
        self.stress_var = tk.StringVar(value="Stress --")
        self.quality_var = tk.StringVar(value="Quality --")
        self.stability_var = tk.StringVar(value="Stability --")
        self.footer_var = tk.StringVar(value="The circle will guide each phase.")

        self._event_queue: queue.Queue[CalibrationEvent | tuple[str, str]] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._controller: CalibrationController | None = None
        self._running = False
        self._animation_mode = "signal_check"
        self._modifier_level = 0
        self._animation_started = time.monotonic()
        self._phase_progress = 0.0
        self._completed_profile_path = ""

        self._build()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._poll_events)
        self.after(33, self._animate)

    def _build(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Panel.TFrame", background="#111827")
        style.configure("Panel.TLabel", background="#111827", foreground="#e5e7eb")
        style.configure("Muted.TLabel", background="#111827", foreground="#94a3b8")
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Setup.Horizontal.TProgressbar", troughcolor="#1f2937", background="#38bdf8", bordercolor="#1f2937")

        root = ttk.Frame(self, style="Panel.TFrame", padding=18)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        header = ttk.Frame(root, style="Panel.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(4, weight=1)
        ttk.Label(header, text="Adaptive Setup", style="Panel.TLabel", font=("Segoe UI", 24, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Profile maker for concentration high/low, stress high/low, then a live detection check.",
            style="Muted.TLabel",
        ).grid(row=1, column=0, columnspan=5, sticky="w", pady=(4, 18))

        ttk.Label(header, text="Source", style="Muted.TLabel").grid(row=2, column=0, sticky="w")
        ttk.Combobox(header, textvariable=self.source_var, values=("generator", "cyton"), state="readonly", width=12).grid(
            row=3, column=0, sticky="ew", padx=(0, 12)
        )
        ttk.Label(header, text="Serial Port", style="Muted.TLabel").grid(row=2, column=1, sticky="w")
        ttk.Entry(header, textvariable=self.serial_port_var, width=12).grid(row=3, column=1, sticky="ew", padx=(0, 12))
        ttk.Label(header, text="User ID", style="Muted.TLabel").grid(row=2, column=2, sticky="w")
        ttk.Entry(header, textvariable=self.user_id_var, width=18).grid(row=3, column=2, sticky="ew", padx=(0, 12))
        ttk.Label(header, text="Artifacts Root", style="Muted.TLabel").grid(row=2, column=3, sticky="w")
        ttk.Entry(header, textvariable=self.artifacts_root_var).grid(row=3, column=3, sticky="ew", padx=(0, 12))
        actions = ttk.Frame(header, style="Panel.TFrame")
        actions.grid(row=3, column=4, sticky="e")
        self.start_button = ttk.Button(actions, text="Start Setup", style="Accent.TButton", command=self._start_setup)
        self.start_button.pack(side=tk.LEFT)
        self.cancel_button = ttk.Button(actions, text="Cancel", command=self._cancel_setup)
        self.cancel_button.pack(side=tk.LEFT, padx=(8, 0))

        body = ttk.Frame(root, style="Panel.TFrame")
        body.grid(row=1, column=0, sticky="nsew", pady=(18, 0))
        body.columnconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)

        stage = ttk.Frame(body, style="Panel.TFrame")
        stage.grid(row=0, column=0, sticky="ew")
        ttk.Label(stage, textvariable=self.phase_title_var, style="Panel.TLabel", font=("Segoe UI", 20, "bold")).pack(anchor="center")
        ttk.Label(
            stage,
            textvariable=self.instruction_var,
            style="Muted.TLabel",
            font=("Segoe UI", 11),
            wraplength=760,
            justify="center",
        ).pack(anchor="center", pady=(6, 18))

        canvas_card = tk.Frame(body, background="#111827", highlightthickness=1, highlightbackground="#1f2937")
        canvas_card.grid(row=1, column=0, sticky="nsew")
        canvas_card.columnconfigure(0, weight=1)
        canvas_card.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(canvas_card, background="#111827", highlightthickness=0, width=760, height=420)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        footer = ttk.Frame(body, style="Panel.TFrame")
        footer.grid(row=2, column=0, sticky="ew", pady=(18, 0))
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.status_var, style="Panel.TLabel").grid(row=0, column=0, sticky="w")
        self.progress = ttk.Progressbar(
            footer,
            style="Setup.Horizontal.TProgressbar",
            mode="determinate",
            maximum=100.0,
        )
        self.progress.grid(row=1, column=0, sticky="ew", pady=(12, 12))

        chips = tk.Frame(footer, background="#111827")
        chips.grid(row=2, column=0, sticky="ew")
        self._chip(chips, self.concentration_var, "#0f172a", "#93c5fd").pack(side=tk.LEFT)
        self._chip(chips, self.stress_var, "#0f172a", "#fca5a5").pack(side=tk.LEFT, padx=(10, 0))
        self._chip(chips, self.quality_var, "#0f172a", "#cbd5e1").pack(side=tk.LEFT, padx=(10, 0))
        self._chip(chips, self.stability_var, "#0f172a", "#67e8f9").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Label(footer, textvariable=self.footer_var, style="Muted.TLabel", wraplength=780).grid(row=3, column=0, sticky="w", pady=(12, 0))

    def _chip(self, parent: tk.Misc, variable: tk.StringVar, background: str, foreground: str) -> tk.Label:
        return tk.Label(
            parent,
            textvariable=variable,
            bg=background,
            fg=foreground,
            padx=12,
            pady=6,
            font=("Segoe UI", 10, "bold"),
            highlightthickness=1,
            highlightbackground="#334155",
        )

    def _start_setup(self) -> None:
        if self._running:
            return
        user_id = self.user_id_var.get().strip()
        if not user_id:
            self.status_var.set("User ID is required.")
            return
        self._running = True
        self._completed_profile_path = ""
        self.status_var.set("Starting setup ...")
        self.phase_title_var.set("Preparing")
        self.instruction_var.set("Warming up the source and validating the runtime bundle.")
        self.footer_var.set("The setup will step through concentration high, concentration low, stress high, stress low, then live detection.")
        self.progress.configure(value=0.0)
        self._worker = threading.Thread(target=self._run_setup_worker, daemon=True)
        self._worker.start()

    def _run_setup_worker(self) -> None:
        try:
            source_type = self.source_var.get().strip().lower()
            source = build_calibration_source(
                source_type=source_type,
                serial_port=self.serial_port_var.get().strip(),
                channel_names=list(RUNTIME_SETUP_CHANNELS),
            )
            controller = CalibrationController(
                source=source,
                source_type=source_type,
                user_id=self.user_id_var.get().strip(),
                artifacts_root=Path(self.artifacts_root_var.get().strip() or "artifacts/runtime_v1"),
                expected_channels=list(RUNTIME_SETUP_CHANNELS),
                observer=self._event_queue.put,
            )
            self._controller = controller
            result = controller.run()
            self._event_queue.put(("completed", str(result.profile_files["profile_json"])))
        except Exception as exc:
            self._event_queue.put(("error", str(exc)))

    def _cancel_setup(self) -> None:
        if self._controller is not None:
            self._controller.cancel()
        self.status_var.set("Cancelling setup ...")

    def _poll_events(self) -> None:
        while True:
            try:
                event = self._event_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_event(event)
        self.after(100, self._poll_events)

    def _handle_event(self, event: CalibrationEvent | tuple[str, str]) -> None:
        if isinstance(event, tuple):
            kind, message = event
            self._running = False
            if kind == "completed":
                self._completed_profile_path = message
                self.status_var.set("Calibrated.")
                self.phase_title_var.set("Calibrated")
                self.instruction_var.set("The user profile was saved and is ready for live personalization.")
                self.footer_var.set(f"Saved profile: {message}")
                self.progress.configure(value=100.0)
                self._animation_mode = "recovery"
            else:
                self.status_var.set(f"Setup failed: {message}")
                self.footer_var.set(message)
            return

        self._animation_mode = event.animation_mode or self._animation_mode
        self._modifier_level = event.modifier_level
        self._animation_started = time.monotonic() - float(event.elapsed_seconds or 0.0)
        if event.title:
            self.phase_title_var.set(event.title)
        if event.instruction:
            self.instruction_var.set(event.instruction)
        if event.concentration is not None:
            self.concentration_var.set(f"Concentration {event.concentration:.2f}")
        if event.stress is not None:
            self.stress_var.set(f"Stress {event.stress:.2f}")
        if event.quality is not None:
            self.quality_var.set(f"Quality {event.quality:.2f}")
        if event.stability is not None:
            self.stability_var.set(f"Stability {event.stability:.2f}")
        if event.message:
            self.status_var.set(event.message)

        phase_total = max(event.phase_count, 1)
        phase_progress = event.stability or 0.0
        overall = ((max(event.phase_index, 1) - 1) + min(max(phase_progress, 0.0), 1.0)) / phase_total
        self.progress.configure(value=max(0.0, min(overall * 100.0, 100.0)))

        if event.kind == "phase_started":
            self.footer_var.set("Hold the requested state until the stability meter fills.")
        elif event.kind == "profile_created":
            self.footer_var.set("Profile created. Now checking live detection.")
        elif event.kind == "phase_modifier":
            self.footer_var.set(f"Intensity nudged to level {event.modifier_level}.")
        elif event.kind == "phase_completed":
            if event.payload.get("accepted", False):
                self.footer_var.set(f"{event.title} locked in. Moving to the next state.")
            else:
                self.footer_var.set(f"{event.title} did not stabilize.")
        elif event.kind == "phase_progress" and event.phase_name == "detection_check":
            quadrant_state = str(event.payload.get("quadrant_state", "unknown"))
            target_state = str(event.payload.get("target_state", "unknown")).replace("_", " ")
            self.footer_var.set(f"Live detection: {quadrant_state} | target: {target_state}")
        elif event.kind == "session_completed":
            self.footer_var.set(f"Saved profile: {event.payload.get('profile_path', '')}")

    def _animate(self) -> None:
        canvas = self.canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 1)
        height = max(canvas.winfo_height(), 1)
        elapsed = time.monotonic() - self._animation_started
        frame = compute_stimulus_frame(self._animation_mode, elapsed, self._modifier_level)
        cx = width / 2.0 + frame.jitter_x
        cy = height / 2.0 + frame.jitter_y
        radius = 92.0 * frame.scale
        glow_radius = radius + 26.0
        canvas.create_oval(cx - glow_radius, cy - glow_radius, cx + glow_radius, cy + glow_radius, fill=frame.glow, outline="")
        canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, fill=frame.fill, outline=frame.outline, width=3)
        canvas.create_text(cx, cy, text=frame.label.upper(), fill="#f8fafc", font=("Segoe UI", 16, "bold"))
        canvas.create_text(
            width / 2.0,
            height - 38.0,
            text="Blue for concentration. Red for stress. Teal confirms live detection.",
            fill="#94a3b8",
            font=("Segoe UI", 11),
        )
        self.after(33, self._animate)

    def _on_close(self) -> None:
        if self._controller is not None:
            self._controller.cancel()
        self.destroy()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Launch the adaptive EEG setup screen.")
    parser.add_argument("--source", choices=("generator", "cyton"), default="generator", help="Default setup source.")
    parser.add_argument("--serial-port", default="COM5", help="Default Cyton serial port.")
    parser.add_argument("--user-id", default="demo_user", help="Default user identifier.")
    parser.add_argument("--artifacts-root", default="artifacts/runtime_v1", help="Runtime artifacts bundle root.")
    args = parser.parse_args()

    app = SetupInterface(
        default_source=args.source,
        default_serial_port=args.serial_port,
        default_user_id=args.user_id,
        default_artifacts_root=args.artifacts_root,
    )
    app.mainloop()


if __name__ == "__main__":
    main()
