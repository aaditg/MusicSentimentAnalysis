"""Desktop app for scoring songs and browsing results.

Tkinter so it runs as a native window with no server and no extra dependency.

    python -m music_emotion.app
"""

from __future__ import annotations

import csv
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .predict import MODEL_PATH
from .timeline import (
    QUADRANT_COLORS,
    QUADRANT_NAMES,
    Window,
    _axis_limit,
    _timestamp,
    write_csv,
)

BG = "#f7f7f5"
PANEL = "#ffffff"
INK = "#1c1c1c"
MUTED = "#6b6b6b"
LINE = "#dcdcd8"
VALENCE = "#c2643a"
AROUSAL = "#4a6fa5"

RESULTS_SUMMARY = Path("results/results_summary.csv")
TIMELINE_DIR = Path("results/timelines")


class Analyzer(threading.Thread):
    """Runs the heavy librosa/sklearn work off the UI thread."""

    def __init__(self, job: dict, outbox: queue.Queue):
        super().__init__(daemon=True)
        self.job = job
        self.outbox = outbox

    def run(self) -> None:
        try:
            self.outbox.put(("status", "Loading model..."))
            from .predict import train
            from .timeline import analyse

            model_path = self.job["model"]
            if not model_path.exists():
                self.outbox.put(("status", "No model yet - training audio-only model..."))
                train(model_path=model_path, audio_only=True)

            self.outbox.put(("status", "Extracting features (this takes a while)..."))
            windows = analyse(
                self.job["audio"],
                self.job["lyrics"],
                model_path,
                self.job["window"],
                self.job["hop"],
            )
            self.outbox.put(("done", windows))
        except Exception as error:  # surfaced in the UI rather than a traceback
            self.outbox.put(("error", f"{type(error).__name__}: {error}"))


class TimelineChart(ttk.Frame):
    """Valence/arousal curves plus a quadrant strip, drawn on a canvas."""

    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, bg=PANEL, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.windows: list[Window] = []
        self.canvas.bind("<Configure>", lambda _event: self.redraw())
        self.canvas.bind("<Motion>", self._hover)
        self._hover_text = None

    def show(self, windows: list[Window]) -> None:
        self.windows = windows
        self.redraw()

    def _geometry(self):
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        left, right, top, bottom = 56, 90, 44, 76
        return width, height, left, right, top, bottom

    def redraw(self) -> None:
        self.canvas.delete("all")
        if not self.windows:
            self.canvas.create_text(
                self.canvas.winfo_width() / 2,
                self.canvas.winfo_height() / 2,
                text="Choose an audio file and press Analyze",
                fill=MUTED,
                font=("Helvetica", 13),
            )
            return

        width, height, left, right, top, bottom = self._geometry()
        if width < 120 or height < 120:
            return
        plot_w = width - left - right
        strip_h = 22
        plot_h = height - top - bottom - strip_h

        windows = self.windows
        span = max(windows[-1].end - windows[0].start, 1e-6)
        limit = _axis_limit(windows)

        def x_of(t: float) -> float:
            return left + plot_w * ((t - windows[0].start) / span)

        def y_of(v: float) -> float:
            return top + plot_h / 2 - (v / limit) * (plot_h / 2)

        self._x_of = x_of

        for frac in (1.0, 0.5, 0.0, -0.5, -1.0):
            y = y_of(limit * frac)
            self.canvas.create_line(left, y, left + plot_w, y, fill=LINE)
            self.canvas.create_text(
                left - 10,
                y,
                text=f"{limit * frac:+.1f}",
                fill=MUTED,
                anchor="e",
                font=("Helvetica", 9),
            )
        zero = y_of(0.0)
        self.canvas.create_line(left, zero, left + plot_w, zero, fill=MUTED)

        for attr, color, label in (
            ("valence", VALENCE, "valence"),
            ("arousal", AROUSAL, "arousal"),
        ):
            points = []
            for w in windows:
                points.extend([x_of((w.start + w.end) / 2), y_of(getattr(w, attr))])
            if len(points) >= 4:
                self.canvas.create_line(*points, fill=color, width=2, smooth=True)
            for w in windows:
                x, y = x_of((w.start + w.end) / 2), y_of(getattr(w, attr))
                self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=color, outline="")
            last = windows[-1]
            self.canvas.create_text(
                x_of((last.start + last.end) / 2) + 10,
                y_of(getattr(last, attr)),
                text=label,
                fill=color,
                anchor="w",
                font=("Helvetica", 10, "bold"),
            )

        strip_y = top + plot_h + 14
        for w in windows:
            x0, x1 = x_of(w.start), x_of(w.end)
            self.canvas.create_rectangle(
                x0,
                strip_y,
                max(x1 - 1, x0 + 1),
                strip_y + strip_h,
                fill=QUADRANT_COLORS[w.quadrant],
                outline="",
            )

        step = max(1, len(windows) // 10)
        for w in windows[::step]:
            self.canvas.create_text(
                x_of(w.start),
                strip_y + strip_h + 14,
                text=_timestamp(w.start),
                fill=MUTED,
                font=("Helvetica", 9),
            )

        x = left
        for quadrant, color in QUADRANT_COLORS.items():
            y = strip_y + strip_h + 36
            self.canvas.create_rectangle(x, y - 5, x + 10, y + 5, fill=color, outline="")
            self.canvas.create_text(
                x + 16,
                y,
                text=f"{quadrant} {QUADRANT_NAMES[quadrant]}",
                fill=INK,
                anchor="w",
                font=("Helvetica", 9),
            )
            x += 108

        self._marker = None

    def _hover(self, event) -> None:
        if not self.windows or not hasattr(self, "_x_of"):
            return
        width, height, left, right, top, bottom = self._geometry()
        if not (left <= event.x <= width - right):
            return
        nearest = min(
            self.windows,
            key=lambda w: abs(self._x_of((w.start + w.end) / 2) - event.x),
        )
        if self._hover_text is not None:
            self.canvas.delete(self._hover_text)
            self.canvas.delete(self._hover_box)
        text = (
            f"{_timestamp(nearest.start)}-{_timestamp(nearest.end)}   "
            f"{nearest.quadrant} {QUADRANT_NAMES[nearest.quadrant]}   "
            f"v {nearest.valence:+.2f}   a {nearest.arousal:+.2f}"
        )
        self._hover_box = self.canvas.create_rectangle(
            left, 12, left + 420, 32, fill="#f0f0ec", outline=LINE
        )
        self._hover_text = self.canvas.create_text(
            left + 10, 22, text=text, fill=INK, anchor="w", font=("Helvetica", 10)
        )


class Circumplex(ttk.Frame):
    """The same windows plotted as a path through valence/arousal space."""

    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, bg=PANEL, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.windows: list[Window] = []
        self.canvas.bind("<Configure>", lambda _event: self.redraw())

    def show(self, windows: list[Window]) -> None:
        self.windows = windows
        self.redraw()

    def redraw(self) -> None:
        self.canvas.delete("all")
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width < 120 or height < 120:
            return
        if not self.windows:
            self.canvas.create_text(
                width / 2,
                height / 2,
                text="No analysis yet",
                fill=MUTED,
                font=("Helvetica", 13),
            )
            return

        size = min(width, height) - 80
        cx, cy = width / 2, height / 2
        half = size / 2
        limit = _axis_limit(self.windows)

        quadrant_at = {
            "Q1": (cx + half / 2, cy - half / 2),
            "Q2": (cx - half / 2, cy - half / 2),
            "Q3": (cx - half / 2, cy + half / 2),
            "Q4": (cx + half / 2, cy + half / 2),
        }
        for quadrant, (qx, qy) in quadrant_at.items():
            self.canvas.create_text(
                qx,
                qy,
                text=f"{quadrant}\n{QUADRANT_NAMES[quadrant]}",
                fill=QUADRANT_COLORS[quadrant],
                font=("Helvetica", 11, "bold"),
                justify="center",
            )

        self.canvas.create_rectangle(cx - half, cy - half, cx + half, cy + half, outline=LINE)
        self.canvas.create_line(cx - half, cy, cx + half, cy, fill=MUTED)
        self.canvas.create_line(cx, cy - half, cx, cy + half, fill=MUTED)
        self.canvas.create_text(
            cx + half,
            cy + 14,
            text="valence +",
            fill=MUTED,
            anchor="e",
            font=("Helvetica", 9),
        )
        self.canvas.create_text(
            cx + 6,
            cy - half,
            text="arousal +",
            fill=MUTED,
            anchor="w",
            font=("Helvetica", 9),
        )

        points = []
        for w in self.windows:
            px = cx + (w.valence / limit) * half
            py = cy - (w.arousal / limit) * half
            points.extend([px, py])
        if len(points) >= 4:
            self.canvas.create_line(*points, fill=MUTED, width=1, smooth=True)

        for i, w in enumerate(self.windows):
            px = cx + (w.valence / limit) * half
            py = cy - (w.arousal / limit) * half
            radius = 5
            self.canvas.create_oval(
                px - radius,
                py - radius,
                px + radius,
                py + radius,
                fill=QUADRANT_COLORS[w.quadrant],
                outline=PANEL,
            )
            if i == 0 or i == len(self.windows) - 1:
                self.canvas.create_text(
                    px,
                    py - 14,
                    text="start" if i == 0 else "end",
                    fill=INK,
                    font=("Helvetica", 9, "bold"),
                )


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Music Emotion")
        self.geometry("1120x720")
        self.minsize(900, 600)
        self.configure(bg=BG)

        self.audio_path: Path | None = None
        self.lyrics_path: Path | None = None
        self.windows: list[Window] = []
        self.outbox: queue.Queue = queue.Queue()

        self._style()
        self._build()
        self.after(100, self._drain)

    def _style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("aqua")
        except tk.TclError:
            style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=INK)
        style.configure("Muted.TLabel", background=BG, foreground=MUTED)
        style.configure("Title.TLabel", background=BG, font=("Helvetica", 17, "bold"))
        style.configure("Head.TLabel", background=BG, font=("Helvetica", 11, "bold"))
        style.configure("Verdict.TLabel", background=BG, font=("Helvetica", 15, "bold"))

    def _build(self) -> None:
        header = ttk.Frame(self, padding=(18, 14, 18, 8))
        header.pack(fill="x")
        ttk.Label(header, text="Music Emotion", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Quadrant prediction and emotional trajectory for any song",
            style="Muted.TLabel",
        ).pack(anchor="w")

        body = ttk.Frame(self, padding=(18, 4, 18, 18))
        body.pack(fill="both", expand=True)

        sidebar = ttk.Frame(body, width=300)
        sidebar.pack(side="left", fill="y", padx=(0, 16))
        sidebar.pack_propagate(False)
        self._build_sidebar(sidebar)

        notebook = ttk.Notebook(body)
        notebook.pack(side="right", fill="both", expand=True)

        self.chart = TimelineChart(notebook)
        notebook.add(self.chart, text="Timeline")

        self.circumplex = Circumplex(notebook)
        notebook.add(self.circumplex, text="Circumplex")

        self.table = self._build_table(notebook)
        notebook.add(self.table, text="Windows")

        benchmarks = self._build_benchmarks(notebook)
        notebook.add(benchmarks, text="Benchmarks")

    def _build_sidebar(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Song", style="Head.TLabel").pack(anchor="w")
        self.audio_label = ttk.Label(
            parent, text="No audio selected", style="Muted.TLabel", wraplength=280
        )
        self.audio_label.pack(anchor="w", pady=(2, 6))
        ttk.Button(parent, text="Choose audio...", command=self._pick_audio).pack(fill="x")

        self.lyrics_label = ttk.Label(
            parent, text="No lyrics (audio only)", style="Muted.TLabel", wraplength=280
        )
        self.lyrics_label.pack(anchor="w", pady=(12, 6))
        row = ttk.Frame(parent)
        row.pack(fill="x")
        ttk.Button(row, text="Choose lyrics...", command=self._pick_lyrics).pack(
            side="left", expand=True, fill="x"
        )
        ttk.Button(row, text="Clear", width=6, command=self._clear_lyrics).pack(
            side="left", padx=(6, 0)
        )

        ttk.Separator(parent).pack(fill="x", pady=16)

        ttk.Label(parent, text="Window", style="Head.TLabel").pack(anchor="w")
        self.window_var = tk.StringVar(value="30")
        self.hop_var = tk.StringVar(value="30")
        grid = ttk.Frame(parent)
        grid.pack(fill="x", pady=(4, 0))
        ttk.Label(grid, text="length (s)").grid(row=0, column=0, sticky="w")
        ttk.Entry(grid, textvariable=self.window_var, width=7).grid(row=0, column=1)
        ttk.Label(grid, text="hop (s)").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(grid, textvariable=self.hop_var, width=7).grid(row=1, column=1, pady=(6, 0))
        ttk.Label(
            parent,
            text="A smaller hop overlaps windows and smooths the curves.",
            style="Muted.TLabel",
            wraplength=280,
        ).pack(anchor="w", pady=(8, 0))

        ttk.Separator(parent).pack(fill="x", pady=16)

        self.analyze_button = ttk.Button(parent, text="Analyze", command=self._analyze)
        self.analyze_button.pack(fill="x")
        self.progress = ttk.Progressbar(parent, mode="indeterminate")
        self.status = ttk.Label(parent, text="Ready", style="Muted.TLabel", wraplength=280)
        self.status.pack(anchor="w", pady=(8, 0))

        ttk.Separator(parent).pack(fill="x", pady=16)
        self.verdict = ttk.Label(parent, text="", style="Verdict.TLabel", wraplength=280)
        self.verdict.pack(anchor="w")
        self.verdict_detail = ttk.Label(parent, text="", style="Muted.TLabel", wraplength=280)
        self.verdict_detail.pack(anchor="w", pady=(2, 0))

        self.export_button = ttk.Button(
            parent, text="Export CSV", command=self._export, state="disabled"
        )
        self.export_button.pack(fill="x", side="bottom")

    def _build_table(self, parent) -> ttk.Frame:
        frame = ttk.Frame(parent)
        columns = ("time", "quadrant", "valence", "arousal", "confidence")
        tree = ttk.Treeview(frame, columns=columns, show="headings")
        for name, width in zip(columns, (140, 160, 100, 100, 100), strict=True):
            tree.heading(name, text=name.capitalize())
            tree.column(name, width=width, anchor="w")
        scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.window_tree = tree
        return frame

    def _build_benchmarks(self, parent) -> ttk.Frame:
        frame = ttk.Frame(parent)
        columns = ("modality", "model", "accuracy", "macro_f1", "balanced_accuracy")
        tree = ttk.Treeview(frame, columns=columns, show="headings")
        for name, width in zip(columns, (110, 300, 90, 90, 140), strict=True):
            tree.heading(name, text=name.replace("_", " ").title())
            tree.column(name, width=width, anchor="w")
        scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        if RESULTS_SUMMARY.exists():
            with RESULTS_SUMMARY.open(newline="", encoding="utf-8") as handle:
                best = 0.0
                rows = list(csv.DictReader(handle))
                best = max(float(r["macro_f1"]) for r in rows) if rows else 0.0
                for row in rows:
                    values = [row[c] for c in columns]
                    tags = ("best",) if float(row["macro_f1"]) == best else ()
                    tree.insert("", "end", values=values, tags=tags)
                tree.tag_configure("best", background="#fdf1e7")
        else:
            tree.insert("", "end", values=("no results_summary.csv found", "", "", "", ""))
        return frame

    def _pick_audio(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose an audio file",
            filetypes=[("Audio", "*.mp3 *.wav *.flac *.m4a *.aiff *.ogg"), ("All", "*.*")],
        )
        if path:
            self.audio_path = Path(path)
            self.audio_label.configure(text=self.audio_path.name)

    def _pick_lyrics(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose a lyrics file", filetypes=[("Text", "*.txt"), ("All", "*.*")]
        )
        if path:
            self.lyrics_path = Path(path)
            self.lyrics_label.configure(text=f"Lyrics: {self.lyrics_path.name}")

    def _clear_lyrics(self) -> None:
        self.lyrics_path = None
        self.lyrics_label.configure(text="No lyrics (audio only)")

    def _analyze(self) -> None:
        if self.audio_path is None:
            messagebox.showinfo("Pick a song", "Choose an audio file first.")
            return
        try:
            window = float(self.window_var.get())
            hop = float(self.hop_var.get())
            if window <= 0 or hop <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Bad window", "Window and hop must be positive numbers.")
            return

        model = MODEL_PATH if self.lyrics_path else Path("models/audio.joblib")
        self.analyze_button.configure(state="disabled")
        self.progress.pack(fill="x", pady=(8, 0))
        self.progress.start(12)
        self.verdict.configure(text="")
        self.verdict_detail.configure(text="")

        Analyzer(
            {
                "audio": self.audio_path,
                "lyrics": self.lyrics_path,
                "model": model,
                "window": window,
                "hop": hop,
            },
            self.outbox,
        ).start()

    def _drain(self) -> None:
        try:
            while True:
                kind, payload = self.outbox.get_nowait()
                if kind == "status":
                    self.status.configure(text=payload)
                elif kind == "done":
                    self._finish(payload)
                elif kind == "error":
                    self._fail(payload)
        except queue.Empty:
            pass
        self.after(120, self._drain)

    def _finish(self, windows: list[Window]) -> None:
        self.windows = windows
        self.progress.stop()
        self.progress.pack_forget()
        self.analyze_button.configure(state="normal")
        self.export_button.configure(state="normal")
        self.status.configure(text=f"Done - {len(windows)} windows")

        counts: dict[str, int] = {}
        for w in windows:
            counts[w.quadrant] = counts.get(w.quadrant, 0) + 1
        dominant = max(counts, key=lambda q: counts[q])
        self.verdict.configure(
            text=f"{dominant} - {QUADRANT_NAMES[dominant]}",
            foreground=QUADRANT_COLORS[dominant],
        )
        mode = "audio + lyrics" if self.lyrics_path else "audio only"
        self.verdict_detail.configure(text=f"{counts[dominant]} of {len(windows)} windows, {mode}")

        self.chart.show(windows)
        self.circumplex.show(windows)
        self.window_tree.delete(*self.window_tree.get_children())
        for w in windows:
            self.window_tree.insert(
                "",
                "end",
                values=(
                    f"{_timestamp(w.start)}-{_timestamp(w.end)}",
                    f"{w.quadrant} {QUADRANT_NAMES[w.quadrant]}",
                    f"{w.valence:+.2f}",
                    f"{w.arousal:+.2f}",
                    f"{w.confidence:.0%}" if w.calibrated else "-",
                ),
            )

    def _fail(self, message: str) -> None:
        self.progress.stop()
        self.progress.pack_forget()
        self.analyze_button.configure(state="normal")
        self.status.configure(text="Failed")
        messagebox.showerror("Analysis failed", message)

    def _export(self) -> None:
        if not self.windows or self.audio_path is None:
            return
        path = write_csv(self.windows, TIMELINE_DIR / f"{self.audio_path.stem}.csv")
        self.status.configure(text=f"Wrote {path}")


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
