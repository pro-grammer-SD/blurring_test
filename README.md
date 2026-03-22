# 🎬 motion_blur.py

High-performance cinematic motion blur for MP4 / MKV video files — runs entirely on CPU, no GPU required.

---

## ✨ Features

- **Temporal frame averaging** — triangle-weighted sliding window for smooth, ghost-free blur
- **Directional blur** — optional horizontal, vertical, or both-axis kernel on top of temporal blur
- **Threaded 3-stage pipeline** — reader → compute pool → writer run concurrently; CPU never waits on disk
- **Vectorised NumPy** — full blend computed in a single `tensordot` call; no Python pixel loops
- **Batch writes** — frames are flushed to disk in bulk, never one-at-a-time
- **Memory-bounded** — safe for 1080p and 4K at any video length
- **Windows-safe codecs** — `mp4v` / `XVID`; no external DLL needed
- **Dual progress bars** — per-frame and per-batch via `tqdm`

---

## 📦 Requirements

| Package | Version |
|---|---|
| Python | ≥ 3.9 |
| opencv-python-headless | ≥ 4.5 |
| numpy | ≥ 1.21 |
| tqdm | ≥ 4.60 |

```bash
pip install -r requirements.txt
```

> **Note:** `opencv-python-headless` is used intentionally — it skips Qt/GTK GUI bindings for a lighter install. Swap to `opencv-python` only if you need `cv2.imshow()` for debugging.

---

## 🚀 Quick Start

```bash
# Minimal — defaults to 7-frame temporal blur
python motion_blur.py --input video.mp4

# Output is saved as video_blurred.mp4 in the same directory
```

---

## 🎛️ CLI Reference

```
python motion_blur.py [OPTIONS]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--input` | path | *required* | Source video file (MP4 or MKV) |
| `--output` | path | `<input>_blurred.mp4` | Destination MP4 path |
| `--frames` | int | `7` | Temporal window size (must be odd, ≥ 3) |
| `--dir` | str | `none` | Directional blur: `none` \| `horizontal` \| `vertical` \| `both` |
| `--ksize` | int | `15` | Directional kernel size in pixels (must be odd) |
| `--batch` | int | `32` | Frames per write-batch |
| `--workers` | int | CPU count (max 8) | Thread pool size |
| `--codec` | str | `mp4v` | FourCC codec: `mp4v` \| `XVID` |

---

## 🎬 Usage Examples

**Subtle blur — general purpose:**
```bash
python motion_blur.py --input clip.mp4 --frames 5
```

**Sports / racing — strong horizontal smear:**
```bash
python motion_blur.py --input race.mp4 --frames 9 --dir horizontal --ksize 25
```

**Heavy cinematic blur — music videos:**
```bash
python motion_blur.py --input clip.mp4 --frames 15 --output cinematic.mp4
```

**Both axes — dreamy/abstract look:**
```bash
python motion_blur.py --input clip.mp4 --frames 7 --dir both --ksize 21
```

**4K — memory-conservative run:**
```bash
python motion_blur.py --input 4k_video.mp4 --batch 8 --workers 2
```

**Custom output path + XVID codec:**
```bash
python motion_blur.py --input video.mkv --output out.mp4 --codec XVID
```

---

## ⚙️ How It Works

```
┌─────────────────────────────────────────────────────────┐
│                   3-Stage Pipeline                       │
│                                                          │
│  [FrameProducer]  →  [TemporalBlurEngine]  →  [BatchWriter] │
│  background thread    ThreadPoolExecutor    background thread│
│   (disk reads)          (CPU compute)        (disk writes)  │
└─────────────────────────────────────────────────────────┘
```

All three stages run concurrently. The compute stage never waits on I/O, and I/O never waits on compute.

### Temporal Blur (triangle weighting)

Rather than a flat average, each window uses a triangle weight profile:

```
frames = 7

weight:  1  2  3  4  3  2  1   (normalised)
           ↑ centre frame (current)
```

The centre frame has the highest influence, which prevents the "double exposure ghosting" artefact common with flat-average motion blur.

### Directional Blur

A 1D motion kernel is convolved over the blended frame using `cv2.filter2D` (vectorised C++):

- `horizontal` → left-right smear (ideal for fast lateral movement)
- `vertical` → up-down smear (ideal for falling/jumping footage)
- `both` → applied sequentially on both axes (dreamy/abstract look)

---

## 💾 Memory Usage Guide

A single 1080p frame ≈ 6 MB. A single 4K frame ≈ 24 MB.

| Resolution | `--frames` | `--batch` | Peak RAM (approx.) |
|---|---|---|---|
| 1080p | 7 | 32 | ~235 MB |
| 1080p | 15 | 32 | ~435 MB |
| 4K | 7 | 32 | ~935 MB |
| 4K | 7 | 8 | ~390 MB |
| 4K | 15 | 8 | ~550 MB |

For 4K on memory-constrained machines, use `--batch 8 --workers 2`.

---

## 🪟 Windows Notes

- Both `mp4v` and `XVID` are built into OpenCV — no external codec DLL required.
- `mp4v` (default) produces `.mp4` with broad player compatibility.
- `XVID` is an alternative if `mp4v` fails to open the writer on your system:
  ```bash
  python motion_blur.py --input video.mp4 --codec XVID
  ```

---

## 📁 File Structure

```
.
├── motion_blur.py     # main script
├── requirements.txt   # pip dependencies
├── .gitignore         # .gitignore file
└── README.md          # project info
```

---

## 📄 License

MIT — use freely, modify freely.
