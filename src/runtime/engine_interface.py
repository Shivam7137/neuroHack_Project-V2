"""Desktop interface for the baseline runtime engine over UDP."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import queue
import socket
import threading
import tkinter as tk
from tkinter import filedialog, ttk
import json

from src.config import get_settings
from src.inference.scorer import RuntimeScorer
from src.runtime.engine import StreamingEngine
from src.runtime.eeg_frame import frame_from_chunk
from src.runtime.stream_transport import DEFAULT_STREAM_HOST, DEFAULT_STREAM_PORT, unpack_chunk_datagram


class EngineInterface(tk.Tk):
    """Simple live monitor for the baseline concentration and stress engine."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Baseline EEG Engine")
        self.geometry("1180x760")
        self.minsize(980, 640)
        settings = get_settings()

        self.host_var = tk.StringVar(value=DEFAULT_STREAM_HOST)
        self.port_var = tk.StringVar(value=str(DEFAULT_STREAM_PORT))
        self.udp_out_host_var = tk.StringVar(value="127.0.0.1")
        self.udp_out_port_var = tk.StringVar(value="5005")
        self.artifacts_root_var = tk.StringVar(value=str(Path("artifacts").resolve()))
        self.calibration_path_var = tk.StringVar(value="")
        self.concentration_cleanup_var = tk.StringVar(value=settings.concentration_cleanup_level)
        self.stress_cleanup_var = tk.StringVar(value=settings.stress_cleanup_level)
        self.status_var = tk.StringVar(value="Ready.")
        self.concentration_var = tk.StringVar(value="--")
        self.stress_var = tk.StringVar(value="--")
        self.probability_var = tk.StringVar(value="--")
        self.quality_var = tk.StringVar(value="--")
        self.state_var = tk.StringVar(value="--")
        self.stress_class_var = tk.StringVar(value="--")
        self.source_var = tk.StringVar(value="--")
        self.sender_var = tk.StringVar(value="--")

        self._receiver_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._event_queue: queue.Queue[dict[str, object]] = queue.Queue()
        self._is_running = False
        self._concentration_history: deque[float] = deque(maxlen=80)
        self._stress_history: deque[float] = deque(maxlen=80)

        self.history_canvas: tk.Canvas
        self.log_text: tk.Text

        self._build()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._poll_events)

    def _build(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        controls = ttk.Frame(self, padding=18)
        controls.grid(row=0, column=0, sticky="nsw")
        main = ttk.Frame(self, padding=(0, 18, 18, 18))
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)
        main.rowconfigure(2, weight=1)

        ttk.Label(controls, text="Baseline EEG Engine", font=("Segoe UI", 18, "bold")).pack(anchor="w")
        ttk.Label(
            controls,
            text="Bind the UDP port, receive synthetic EEG, and run the baseline concentration and stress scorer.",
            wraplength=300,
            justify="left",
        ).pack(anchor="w", pady=(8, 18))

        network = ttk.LabelFrame(controls, text="UDP Listener", padding=12)
        network.pack(fill=tk.X)
        ttk.Label(network, text="Host").pack(anchor="w")
        ttk.Entry(network, textvariable=self.host_var).pack(fill=tk.X, pady=(4, 8))
        ttk.Label(network, text="Port").pack(anchor="w")
        ttk.Entry(network, textvariable=self.port_var).pack(fill=tk.X, pady=(4, 8))
        
        ttk.Label(network, text="Game UDP Out Host").pack(anchor="w", pady=(4, 0))
        ttk.Entry(network, textvariable=self.udp_out_host_var).pack(fill=tk.X, pady=(4, 8))
        ttk.Label(network, text="Game UDP Out Port (0 to disable)").pack(anchor="w")
        ttk.Entry(network, textvariable=self.udp_out_port_var).pack(fill=tk.X, pady=(4, 8))

        ttk.Button(network, text="Start Engine", command=self._start_listener).pack(fill=tk.X)
        ttk.Button(network, text="Stop Engine", command=self._stop_listener).pack(fill=tk.X, pady=(8, 0))

        paths = ttk.LabelFrame(controls, text="Artifacts", padding=12)
        paths.pack(fill=tk.X, pady=(18, 0))
        ttk.Label(paths, text="Artifacts Root").pack(anchor="w")
        ttk.Entry(paths, textvariable=self.artifacts_root_var).pack(fill=tk.X, pady=(4, 8))
        ttk.Button(paths, text="Choose Artifacts Folder", command=self._choose_artifacts_root).pack(fill=tk.X)
        ttk.Button(paths, text="Use Repo Artifacts", command=self._use_repo_artifacts).pack(fill=tk.X, pady=(8, 0))
        ttk.Label(paths, text="Concentration Cleanup").pack(anchor="w", pady=(10, 0))
        ttk.Combobox(
            paths,
            textvariable=self.concentration_cleanup_var,
            state="readonly",
            values=("none", "light", "medium", "heavy"),
        ).pack(fill=tk.X, pady=(4, 8))
        ttk.Label(paths, text="Stress Cleanup").pack(anchor="w")
        ttk.Combobox(
            paths,
            textvariable=self.stress_cleanup_var,
            state="readonly",
            values=("none", "light", "medium", "heavy"),
        ).pack(fill=tk.X, pady=(4, 8))
        ttk.Label(paths, text="Calibration Path (optional)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(paths, textvariable=self.calibration_path_var).pack(fill=tk.X, pady=(4, 8))
        ttk.Button(paths, text="Choose Calibration File", command=self._choose_calibration_path).pack(fill=tk.X)

        status = ttk.LabelFrame(controls, text="Status", padding=12)
        status.pack(fill=tk.X, pady=(18, 0))
        ttk.Label(status, textvariable=self.status_var, wraplength=300, justify="left").pack(anchor="w")

        metrics = ttk.Frame(main)
        metrics.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        for column in range(4):
            metrics.columnconfigure(column, weight=1)

        self._metric_card(metrics, "Concentration", self.concentration_var, 0)
        self._metric_card(metrics, "Stress", self.stress_var, 1)
        self._metric_card(metrics, "Probability", self.probability_var, 2)
        self._metric_card(metrics, "Quality", self.quality_var, 3)

        details = ttk.Frame(main)
        details.grid(row=1, column=0, sticky="nsew", pady=(0, 12))
        details.columnconfigure(0, weight=1)
        details.columnconfigure(1, weight=1)
        details.rowconfigure(0, weight=1)

        info = ttk.LabelFrame(details, text="Current Output", padding=12)
        info.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        ttk.Label(info, text="Decision State", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(info, textvariable=self.state_var, font=("Segoe UI", 15, "bold")).grid(row=1, column=0, sticky="w", pady=(0, 10))
        ttk.Label(info, text="Stress Class", font=("Segoe UI", 10, "bold")).grid(row=2, column=0, sticky="w")
        ttk.Label(info, textvariable=self.stress_class_var).grid(row=3, column=0, sticky="w", pady=(0, 10))
        ttk.Label(info, text="Source", font=("Segoe UI", 10, "bold")).grid(row=4, column=0, sticky="w")
        ttk.Label(info, textvariable=self.source_var).grid(row=5, column=0, sticky="w", pady=(0, 10))
        ttk.Label(info, text="Sender", font=("Segoe UI", 10, "bold")).grid(row=6, column=0, sticky="w")
        ttk.Label(info, textvariable=self.sender_var).grid(row=7, column=0, sticky="w")

        history = ttk.LabelFrame(details, text="Score History", padding=8)
        history.grid(row=0, column=1, sticky="nsew")
        history.rowconfigure(0, weight=1)
        history.columnconfigure(0, weight=1)
        self.history_canvas = tk.Canvas(history, background="#111827", highlightthickness=0)
        self.history_canvas.grid(row=0, column=0, sticky="nsew")
        self.history_canvas.bind("<Configure>", lambda _event: self._redraw_history())

        logs = ttk.LabelFrame(main, text="Live Output Log", padding=8)
        logs.grid(row=2, column=0, sticky="nsew")
        logs.rowconfigure(0, weight=1)
        logs.columnconfigure(0, weight=1)
        self.log_text = tk.Text(logs, height=10, background="#0f172a", foreground="#e5e7eb", wrap="word")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_text.configure(state="disabled")

    def _metric_card(self, parent: ttk.Frame, title: str, variable: tk.StringVar, column: int) -> None:
        card = ttk.LabelFrame(parent, text=title, padding=12)
        card.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 8, 0))
        ttk.Label(card, textvariable=variable, font=("Segoe UI", 20, "bold")).pack(anchor="w")

    def _start_listener(self) -> None:
        if self._is_running:
            return
        host = self.host_var.get().strip() or DEFAULT_STREAM_HOST
        try:
            port = int(self.port_var.get().strip())
        except ValueError:
            self.status_var.set("Port must be an integer.")
            return

        udp_out_host = self.udp_out_host_var.get().strip() or "127.0.0.1"
        try:
            udp_out_port = int(self.udp_out_port_var.get().strip())
        except ValueError:
            udp_out_port = 0

        artifacts_root = Path(self.artifacts_root_var.get().strip() or "artifacts").expanduser().resolve()
        calibration_text = self.calibration_path_var.get().strip()
        calibration_path = Path(calibration_text).expanduser().resolve() if calibration_text else None
        concentration_cleanup_level = self.concentration_cleanup_var.get().strip() or "light"
        stress_cleanup_level = self.stress_cleanup_var.get().strip() or "none"

        self._stop_event.clear()
        self._is_running = True
        self.status_var.set(
            f"Binding UDP listener on {host}:{port} with concentration={concentration_cleanup_level} "
            f"and stress={stress_cleanup_level} cleanup."
        )
        self._receiver_thread = threading.Thread(
            target=self._receiver_loop,
            kwargs={
                "host": host,
                "port": port,
                "artifacts_root": artifacts_root,
                "calibration_path": calibration_path,
                "concentration_cleanup_level": concentration_cleanup_level,
                "stress_cleanup_level": stress_cleanup_level,
                "udp_out_host": udp_out_host,
                "udp_out_port": udp_out_port,
            },
            daemon=True,
        )
        self._receiver_thread.start()

    def _stop_listener(self) -> None:
        if not self._is_running:
            return
        self._stop_event.set()
        self._is_running = False
        self.status_var.set("Stopping engine listener ...")

    def _receiver_loop(
        self,
        *,
        host: str,
        port: int,
        artifacts_root: Path,
        calibration_path: Path | None,
        concentration_cleanup_level: str,
        stress_cleanup_level: str,
        udp_out_host: str,
        udp_out_port: int,
    ) -> None:
        try:
            scorer = RuntimeScorer(
                artifacts_root=artifacts_root,
                calibration_path=calibration_path,
                concentration_cleanup_level=concentration_cleanup_level,
                stress_cleanup_level=stress_cleanup_level,
            )
            engine = StreamingEngine(scorer=scorer, calibration_path=calibration_path)
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_socket, \
                 socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_out_socket:
                udp_socket.bind((host, port))
                udp_socket.settimeout(0.25)
                self._event_queue.put(
                    {
                        "type": "status",
                        "message": (
                            f"Listening for UDP on {host}:{port}. "
                            f"Concentration cleanup={concentration_cleanup_level}, stress cleanup={stress_cleanup_level}."
                        ),
                    }
                )
                while not self._stop_event.is_set():
                    try:
                        packet, address = udp_socket.recvfrom(65535)
                    except socket.timeout:
                        continue
                    chunk, sequence = unpack_chunk_datagram(packet)
                    frame = frame_from_chunk(chunk, source=str(chunk.metadata.get("source_name", "udp")))
                    for output in engine.process_frame(frame):
                        if udp_out_port > 0:
                            out_json = json.dumps({
                                "timestamp": output.timestamp,
                                "source": output.source,
                                "concentration_score": output.concentration_score,
                                "stress_score": output.stress_score,
                                "concentration_probability": output.concentration_probability,
                                "stress_predicted_class": output.stress_predicted_class,
                            })
                            try:
                                udp_out_socket.sendto(out_json.encode("utf-8"), (udp_out_host, udp_out_port))
                            except Exception as e:
                                print(f"Failed to send UDP output: {e}")

                        print(
                            f"Time: {output.timestamp:<10.3f} | Source: {output.source:<12} | "
                            f"Focus: {output.concentration_score:>5.1f} ({output.concentration_probability:.2f}) | "
                            f"Stress: {output.stress_score:>5.1f} ({output.stress_predicted_class})"
                        )

                        self._event_queue.put(
                            {
                                "type": "output",
                                "timestamp": output.timestamp,
                                "source": output.source,
                                "sender": f"{address[0]}:{address[1]}",
                                "chunk_sequence": sequence,
                                "concentration_score": output.concentration_score,
                                "stress_score": output.stress_score,
                                "concentration_probability": output.concentration_probability,
                                "stress_predicted_class": output.stress_predicted_class,
                                "quality_score": output.quality_score,
                                "state": output.metadata.get("decision_state", "unknown"),
                            }
                        )
        except Exception as exc:
            self._event_queue.put({"type": "error", "message": str(exc)})
        finally:
            self._event_queue.put({"type": "stopped"})

    def _poll_events(self) -> None:
        while True:
            try:
                event = self._event_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_event(event)
        self.after(100, self._poll_events)

    def _handle_event(self, event: dict[str, object]) -> None:
        kind = str(event.get("type", ""))
        if kind == "status":
            self.status_var.set(str(event["message"]))
            return
        if kind == "error":
            self.status_var.set(f"Engine error: {event['message']}")
            self._is_running = False
            return
        if kind == "stopped":
            if self._is_running:
                self.status_var.set("Engine stopped.")
            self._is_running = False
            return
        if kind != "output":
            return
        concentration = float(event["concentration_score"])
        stress = float(event["stress_score"])
        probability = float(event["concentration_probability"]) * 100.0
        quality = float(event["quality_score"])
        self.concentration_var.set(f"{concentration:.2f}")
        self.stress_var.set(f"{stress:.2f}")
        self.probability_var.set(f"{probability:.2f}%")
        self.quality_var.set(f"{quality:.2f}")
        self.state_var.set(str(event["state"]))
        self.stress_class_var.set(str(event["stress_predicted_class"]))
        self.source_var.set(str(event["source"]))
        self.sender_var.set(str(event["sender"]))
        self._concentration_history.append(concentration)
        self._stress_history.append(stress)
        self._redraw_history()
        self._append_log(
            f"seq={event['chunk_sequence']}  conc={concentration:.2f}  stress={stress:.2f}  "
            f"state={event['state']}  class={event['stress_predicted_class']}"
        )

    def _redraw_history(self) -> None:
        canvas = self.history_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 1)
        height = max(canvas.winfo_height(), 1)
        if not self._concentration_history:
            canvas.create_text(12, 14, text="Waiting for scored windows ...", fill="#cbd5e1", anchor="w")
            return
        canvas.create_text(12, 14, text="Recent concentration and stress scores", fill="#cbd5e1", anchor="w")
        left, top, right, bottom = 44, 30, width - 12, height - 28
        canvas.create_rectangle(left, top, right, bottom, outline="#243244")
        for marker in range(0, 101, 20):
            y = bottom - ((marker / 100.0) * (bottom - top))
            canvas.create_line(left, y, right, y, fill="#1f2937", dash=(3, 4))
            canvas.create_text(left - 8, y, text=str(marker), fill="#9ca3af", anchor="e", font=("Segoe UI", 8))
        self._draw_series(canvas, list(self._concentration_history), left, top, right, bottom, "#34d399")
        self._draw_series(canvas, list(self._stress_history), left, top, right, bottom, "#f87171")
        canvas.create_text(right - 180, top + 12, text="Concentration", fill="#34d399", anchor="w", font=("Segoe UI", 9, "bold"))
        canvas.create_text(right - 80, top + 12, text="Stress", fill="#f87171", anchor="w", font=("Segoe UI", 9, "bold"))

    def _draw_series(
        self,
        canvas: tk.Canvas,
        values: list[float],
        left: int,
        top: int,
        right: int,
        bottom: int,
        color: str,
    ) -> None:
        if len(values) < 2:
            return
        xs = [left + (idx / (len(values) - 1)) * (right - left) for idx in range(len(values))]
        ys = [bottom - (max(0.0, min(value, 100.0)) / 100.0) * (bottom - top) for value in values]
        points: list[float] = []
        for x_value, y_value in zip(xs, ys, strict=True):
            points.extend((float(x_value), float(y_value)))
        canvas.create_line(points, fill=color, width=2, smooth=True)

    def _append_log(self, line: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line + "\n")
        line_count = int(self.log_text.index("end-1c").split(".")[0])
        while line_count > 120:
            self.log_text.delete("1.0", "2.0")
            line_count -= 1
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _choose_artifacts_root(self) -> None:
        selected = filedialog.askdirectory(
            title="Choose artifacts root",
            initialdir=self.artifacts_root_var.get() or str(Path("artifacts").resolve()),
        )
        if selected:
            self.artifacts_root_var.set(str(Path(selected).resolve()))
            self.status_var.set(f"Artifacts root set to {Path(selected).resolve()}.")

    def _choose_calibration_path(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose calibration file",
            initialdir=str(Path(".").resolve()),
        )
        if selected:
            self.calibration_path_var.set(str(Path(selected).resolve()))
            self.status_var.set(f"Calibration path set to {Path(selected).resolve()}.")

    def _use_repo_artifacts(self) -> None:
        root = Path("artifacts").resolve()
        self.artifacts_root_var.set(str(root))
        self.status_var.set(f"Using artifacts root {root}.")

    def _on_close(self) -> None:
        self._stop_listener()
        self.destroy()


def main() -> None:
    app = EngineInterface()
    app.mainloop()


if __name__ == "__main__":
    main()
