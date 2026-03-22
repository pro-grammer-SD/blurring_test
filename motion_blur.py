"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    motion_blur.py — High-Performance Video Motion Blur       ║
╚══════════════════════════════════════════════════════════════════════════════╝

DESCRIPTION
-----------
Applies cinematic-quality motion blur to MP4 / MKV video files using:
  • Temporal frame averaging  (simulates real camera shutter motion blur)
  • Optional directional blur  (horizontal, vertical, or both axes)
  • Batch-based frame pipeline with worker threads for maximum throughput
  • Vectorised NumPy accumulation — no Python-level per-pixel loops
  • Windows-safe codecs (mp4v / XVID) — no external DLL required
  • Memory-bounded processing — safe for 1080p/4K at any duration

REQUIREMENTS
------------
    pip install opencv-python-headless numpy tqdm

    Python  ≥ 3.9
    OpenCV  ≥ 4.5
    NumPy   ≥ 1.21
    tqdm    ≥ 4.60

USAGE
-----
    # Minimal – defaults to 7-frame temporal blur:
    python motion_blur.py --input video.mp4

    # Full control:
    python motion_blur.py \
        --input  video.mkv           \
        --output blurred_output.mp4  \
        --frames 11                  \
        --dir    horizontal          \
        --ksize  15                  \
        --batch  64                  \
        --workers 4

CLI ARGUMENTS
-------------
  --input    PATH   Source video file (MP4 or MKV). [required]
  --output   PATH   Destination MP4. Default: <input>_blurred.mp4
  --frames   INT    Number of frames to average (odd, ≥ 3). Default: 7
                    More frames → stronger / smoother blur.
  --dir      STR    Directional blur axis: 'none'|'horizontal'|'vertical'|'both'
                    Applied on top of temporal blur. Default: 'none'
  --ksize    INT    Directional blur kernel size (odd, ≥ 3). Default: 15
  --batch    INT    Frames per write-batch. Default: 32
                    Lower → less RAM; Higher → better throughput.
  --workers  INT    I/O reader threads. Default: CPU count (max 8)
  --codec    STR    FourCC codec string: 'mp4v'|'XVID'. Default: 'mp4v'

EXAMPLES
--------
    # Subtle horizontal motion blur (sports footage):
    python motion_blur.py --input race.mp4 --frames 5 --dir horizontal --ksize 21

    # Heavy cinematic blur (music video):
    python motion_blur.py --input clip.mp4 --frames 15 --batch 16

    # 4K memory-conscious run:
    python motion_blur.py --input 4k_video.mp4 --batch 8 --workers 2
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Constants / Tunables
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_TEMPORAL_FRAMES = 7    # odd number; more = stronger blur
DEFAULT_KSIZE           = 15   # directional blur kernel size (px)
DEFAULT_BATCH_SIZE      = 32   # frames buffered before a single writer flush
DEFAULT_WORKERS         = min(os.cpu_count() or 4, 8)
QUEUE_MAX               = 4    # max pending batches in the write queue


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_odd(n: int, name: str) -> int:
    """Guarantee a value is a positive odd integer."""
    n = max(3, int(n))
    if n % 2 == 0:
        n += 1
        print(f"[warn] {name} must be odd — bumped to {n}")
    return n


def _build_directional_kernel(ksize: int, direction: str) -> Optional[np.ndarray]:
    """
    Return a separable 1-D motion-blur kernel, or None if direction=='none'.

    'horizontal' → kernel along columns  (left-right smear)
    'vertical'   → kernel along rows     (up-down  smear)
    'both'       → applied twice in sequence
    """
    if direction == "none":
        return None
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    mid = ksize // 2
    if direction == "horizontal":
        kernel[mid, :] = 1.0 / ksize
    elif direction == "vertical":
        kernel[:, mid] = 1.0 / ksize
    # 'both' is handled in apply_blur by calling twice
    return kernel


def _apply_directional_blur(
    frame: np.ndarray,
    ksize: int,
    direction: str,
) -> np.ndarray:
    """Apply optional directional motion blur with cv2.filter2D (vectorised C++)."""
    if direction == "none":
        return frame

    mid = ksize // 2

    def _make(d: str) -> np.ndarray:
        k = np.zeros((ksize, ksize), dtype=np.float32)
        if d == "horizontal":
            k[mid, :] = 1.0 / ksize
        else:
            k[:, mid] = 1.0 / ksize
        return k

    if direction == "both":
        frame = cv2.filter2D(frame, -1, _make("horizontal"))
        frame = cv2.filter2D(frame, -1, _make("vertical"))
    else:
        frame = cv2.filter2D(frame, -1, _make(direction))

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Frame reader (threaded producer)
# ─────────────────────────────────────────────────────────────────────────────

class FrameProducer:
    """
    Reads frames from a VideoCapture in a background thread and fills a
    thread-safe deque.  Decouples I/O from compute so cores stay busy.
    """

    _SENTINEL = object()  # signals EOF to consumers

    def __init__(self, path: str, prefetch: int = DEFAULT_WORKERS * 2):
        self._cap     = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {path!r}")
        self._queue: Deque = deque()
        self._lock    = threading.Lock()
        self._cond    = threading.Condition(self._lock)
        self._maxlen  = max(prefetch, 8)
        self._done    = False
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # -- public properties ----------------------------------------------------

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) or 25.0

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def frame_count(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # -- iteration ------------------------------------------------------------

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        while True:
            with self._cond:
                while not self._queue and not self._done:
                    self._cond.wait(timeout=0.05)
                if self._queue:
                    item = self._queue.popleft()
                    self._cond.notify_all()
                    if item is self._SENTINEL:
                        raise StopIteration
                    return item
                if self._done and not self._queue:
                    raise StopIteration

    # -- background reader ----------------------------------------------------

    def _run(self):
        try:
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    break
                with self._cond:
                    while len(self._queue) >= self._maxlen:
                        self._cond.wait(timeout=0.05)
                    self._queue.append(frame)
                    self._cond.notify_all()
        finally:
            with self._cond:
                self._queue.append(self._SENTINEL)
                self._done = True
                self._cond.notify_all()
            self._cap.release()


# ─────────────────────────────────────────────────────────────────────────────
# Motion-blur engine
# ─────────────────────────────────────────────────────────────────────────────

class TemporalBlurEngine:
    """
    Applies temporal motion blur by accumulating a sliding window of frames
    and computing their weighted average.

    All arithmetic is performed on float32 arrays; the window is managed as
    a fixed-size ring buffer to avoid repeated memory allocation.

    Parameters
    ----------
    n_frames  : Number of frames to blend (must be odd).
    direction : Directional kernel axis ('none'|'horizontal'|'vertical'|'both').
    ksize     : Directional kernel half-size (pixels).
    workers   : Thread-pool size for batch processing.
    """

    def __init__(
        self,
        n_frames:  int,
        direction: str = "none",
        ksize:     int = DEFAULT_KSIZE,
        workers:   int = DEFAULT_WORKERS,
    ) -> None:
        self.n_frames  = n_frames
        self.direction = direction
        self.ksize     = ksize
        self._executor = ThreadPoolExecutor(max_workers=workers)
        self._weights  = self._compute_weights(n_frames)

    # -- weight profile -------------------------------------------------------

    @staticmethod
    def _compute_weights(n: int) -> np.ndarray:
        """
        Triangle-weighted average: centre frame has the most influence.
        This avoids the 'ghosting' artefact of a flat-average window.
        """
        half   = n // 2
        ramp   = np.arange(1, half + 2, dtype=np.float32)
        w      = np.concatenate([ramp, ramp[-2::-1]])  # symmetric triangle
        return w / w.sum()

    # -- public API -----------------------------------------------------------

    def process_batch(
        self,
        raw_frames: List[np.ndarray],
        ring:       Deque[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Given a list of new raw frames and the current ring buffer of previous
        frames, return blurred versions of each raw frame.

        The ring buffer is updated in-place so state carries across batches.
        """
        results: List[Optional[np.ndarray]] = [None] * len(raw_frames)
        futures = []

        for i, frame in enumerate(raw_frames):
            ring.append(frame.astype(np.float32))
            snapshot = list(ring)          # O(n_frames) copy; n is tiny
            w        = self._weights
            idx      = i
            futures.append(
                self._executor.submit(self._blend_and_blur, snapshot, w, idx)
            )

        for fut in futures:
            blurred, idx = fut.result()
            results[idx] = blurred

        return results  # type: ignore[return-value]

    # -- internal -------------------------------------------------------------

    def _blend_and_blur(
        self,
        window: List[np.ndarray],
        weights: np.ndarray,
        idx: int,
    ) -> Tuple[np.ndarray, int]:
        """
        Vectorised weighted sum of the frame window, then optional directional
        blur.  Runs inside a ThreadPoolExecutor worker.
        """
        n = len(window)
        w = weights

        # Align weights to actual window length (first frames have fewer neighbours)
        if n < len(w):
            w = w[-n:]
            w = w / w.sum()

        # Stack → shape (n, H, W, C); weighted sum along axis-0
        stacked = np.stack(window, axis=0)           # (n, H, W, 3) float32
        blended = np.tensordot(w, stacked, axes=([0], [0]))  # (H, W, 3) float32

        frame_u8 = np.clip(blended, 0, 255).astype(np.uint8)

        if self.direction != "none":
            frame_u8 = _apply_directional_blur(frame_u8, self.ksize, self.direction)

        return frame_u8, idx

    def shutdown(self):
        self._executor.shutdown(wait=True)


# ─────────────────────────────────────────────────────────────────────────────
# Writer (threaded consumer)
# ─────────────────────────────────────────────────────────────────────────────

class BatchWriter:
    """
    Collects processed frames and flushes them to a VideoWriter in batches.
    A background thread handles the write so the main pipeline never stalls
    on disk I/O.
    """

    def __init__(
        self,
        path:      str,
        fps:       float,
        width:     int,
        height:    int,
        codec:     str = "mp4v",
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        fourcc        = cv2.VideoWriter_fourcc(*codec)
        self._writer  = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise IOError(
                f"VideoWriter could not open {path!r} with codec {codec!r}. "
                "Try --codec XVID."
            )
        self._batch_size = batch_size
        self._buffer: List[np.ndarray] = []
        self._queue: Deque[List[np.ndarray]] = deque()
        self._lock   = threading.Lock()
        self._cond   = threading.Condition(self._lock)
        self._done   = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # -- public API -----------------------------------------------------------

    def write(self, frame: np.ndarray) -> None:
        """Buffer one frame; flush automatically at batch_size boundary."""
        self._buffer.append(frame)
        if len(self._buffer) >= self._batch_size:
            self._flush_buffer()

    def close(self) -> None:
        """Flush remaining frames and wait for the writer thread to finish."""
        if self._buffer:
            self._flush_buffer()
        with self._cond:
            self._done = True
            self._cond.notify_all()
        self._thread.join()
        self._writer.release()

    # -- internal -------------------------------------------------------------

    def _flush_buffer(self) -> None:
        batch = self._buffer[:]
        self._buffer.clear()
        with self._cond:
            # Back-pressure: don't let queue grow beyond QUEUE_MAX
            while len(self._queue) >= QUEUE_MAX:
                self._cond.wait(timeout=0.05)
            self._queue.append(batch)
            self._cond.notify_all()

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._done:
                    self._cond.wait(timeout=0.05)
                if self._queue:
                    batch = self._queue.popleft()
                    self._cond.notify_all()
                elif self._done:
                    break
                else:
                    continue
            for frame in batch:
                self._writer.write(frame)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(args: argparse.Namespace) -> None:
    # ── validate input ───────────────────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"[error] Input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix not in {".mp4", ".mkv", ".avi", ".mov"}:
        print(f"[warn] Unrecognised extension {suffix!r}; attempting anyway.")

    # ── derive output path ───────────────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(f"{input_path.stem}_blurred.mp4")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── normalise params ─────────────────────────────────────────────────────
    n_frames  = _ensure_odd(args.frames, "--frames")
    ksize     = _ensure_odd(args.ksize,  "--ksize")
    direction = args.dir.lower()
    if direction not in {"none", "horizontal", "vertical", "both"}:
        sys.exit(f"[error] --dir must be none|horizontal|vertical|both, got {direction!r}")

    batch_size = max(1, args.batch)
    workers    = max(1, args.workers)

    # ── open source ──────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Source  : {input_path}")
    print(f"  Output  : {output_path}")
    print(f"  Temporal: {n_frames} frames  (triangle-weighted)")
    print(f"  Dir blur: {direction}" + (f"  ksize={ksize}" if direction != "none" else ""))
    print(f"  Batch   : {batch_size} frames/flush   Workers: {workers}")
    print(f"{'─'*60}\n")

    producer   = FrameProducer(str(input_path), prefetch=workers * 4)
    total_frames = producer.frame_count
    fps        = producer.fps
    width      = producer.width
    height     = producer.height

    print(f"  Resolution : {width}×{height}  |  FPS: {fps:.3f}")
    print(f"  Est. frames: {total_frames if total_frames > 0 else 'unknown'}")

    # Memory estimate (rough)
    frame_mb = width * height * 3 / 1e6
    window_mb = frame_mb * n_frames
    print(f"  Frame size : {frame_mb:.1f} MB  |  Window buffer: {window_mb:.1f} MB\n")

    # ── build engine & writer ────────────────────────────────────────────────
    engine = TemporalBlurEngine(
        n_frames=n_frames,
        direction=direction,
        ksize=ksize,
        workers=workers,
    )

    writer = BatchWriter(
        path=str(output_path),
        fps=fps,
        width=width,
        height=height,
        codec=args.codec,
        batch_size=batch_size,
    )

    # ── ring buffer: holds the last n_frames for the sliding window ──────────
    ring: Deque[np.ndarray] = deque(maxlen=n_frames)

    # ── main loop ────────────────────────────────────────────────────────────
    t_start       = time.perf_counter()
    frame_counter = 0
    raw_batch: List[np.ndarray] = []

    pbar = tqdm(
        total=total_frames if total_frames > 0 else None,
        desc="Processing",
        unit="frame",
        dynamic_ncols=True,
        colour="cyan",
    )

    batch_pbar = tqdm(
        desc="  Batches ",
        unit="batch",
        dynamic_ncols=True,
        colour="green",
        leave=False,
    )

    try:
        for raw_frame in producer:
            raw_batch.append(raw_frame)

            if len(raw_batch) >= batch_size:
                blurred_batch = engine.process_batch(raw_batch, ring)
                for bf in blurred_batch:
                    writer.write(bf)
                frame_counter += len(raw_batch)
                pbar.update(len(raw_batch))
                batch_pbar.update(1)
                raw_batch.clear()

        # Flush leftover frames
        if raw_batch:
            blurred_batch = engine.process_batch(raw_batch, ring)
            for bf in blurred_batch:
                writer.write(bf)
            frame_counter += len(raw_batch)
            pbar.update(len(raw_batch))
            batch_pbar.update(1)
            raw_batch.clear()

    except KeyboardInterrupt:
        print("\n[interrupted] Finalising partial output…")

    finally:
        pbar.close()
        batch_pbar.close()
        writer.close()
        engine.shutdown()

    # ── summary ──────────────────────────────────────────────────────────────
    elapsed   = time.perf_counter() - t_start
    out_size  = output_path.stat().st_size / 1e6 if output_path.exists() else 0
    avg_fps   = frame_counter / elapsed if elapsed > 0 else 0

    print(f"\n{'─'*60}")
    print(f"  Frames written : {frame_counter}")
    print(f"  Elapsed        : {elapsed:.1f}s  ({avg_fps:.1f} frames/sec)")
    print(f"  Output size    : {out_size:.1f} MB")
    print(f"  Saved to       : {output_path}")
    print(f"{'─'*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="High-quality motion blur for MP4/MKV video files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input",   required=True, help="Source video (MP4/MKV)")
    p.add_argument("--output",  default=None,  help="Output MP4 path")
    p.add_argument("--frames",  type=int, default=DEFAULT_TEMPORAL_FRAMES,
                   help=f"Temporal window size (odd). Default: {DEFAULT_TEMPORAL_FRAMES}")
    p.add_argument("--dir",     default="none",
                   choices=["none", "horizontal", "vertical", "both"],
                   help="Directional blur axis. Default: none")
    p.add_argument("--ksize",   type=int, default=DEFAULT_KSIZE,
                   help=f"Directional kernel size (odd). Default: {DEFAULT_KSIZE}")
    p.add_argument("--batch",   type=int, default=DEFAULT_BATCH_SIZE,
                   help=f"Frames per write batch. Default: {DEFAULT_BATCH_SIZE}")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                   help=f"Thread pool size. Default: {DEFAULT_WORKERS}")
    p.add_argument("--codec",   default="mp4v", choices=["mp4v", "XVID"],
                   help="FourCC codec. Default: mp4v")
    return p


if __name__ == "__main__":
    run_pipeline(build_parser().parse_args())
    