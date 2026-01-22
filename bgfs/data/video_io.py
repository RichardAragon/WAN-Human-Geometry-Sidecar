from __future__ import annotations

from pathlib import Path


def read_video_rgb(path, max_frames=None):
    """Read a video into RGB frames.

    Returns:
      frames: uint8 (T,H,W,3)
      fps: float (best effort; may be 0 if unknown)

    Notes:
      - imageio v3 plugin names differ across installs; do NOT hard-require plugin="ffmpeg".
      - Prefer automatic backend selection; fall back to imageio.v2 reader if needed.
    """
    import numpy as np
    import imageio
    import imageio.v3 as iio

    path = Path(path)

    # Try imageio v3 without forcing plugin name
    try:
        meta = iio.immeta(str(path))
        fps = float(meta.get("fps", 0.0) or 0.0)
        frames = iio.imread(str(path))  # (T,H,W,3) or (H,W,3)
        if frames.ndim == 3:
            frames = frames[None, ...]
        if max_frames is not None:
            frames = frames[:max_frames]
        return frames.astype("uint8"), fps
    except Exception:
        pass

    # Fallback: imageio v2 reader (often backed by imageio-ffmpeg)
    try:
        rdr = imageio.get_reader(str(path))
        meta = rdr.get_meta_data() or {}
        fps = float(meta.get("fps", 0.0) or 0.0)

        frames_list = []
        for idx, frame in enumerate(rdr):
            frames_list.append(frame)
            if max_frames is not None and (idx + 1) >= max_frames:
                break
        rdr.close()

        if not frames_list:
            raise RuntimeError("No frames decoded.")

        frames = np.stack(frames_list, axis=0).astype("uint8")
        return frames, fps
    except Exception as e:
        raise RuntimeError(f"Failed to read video {path}: {e}")

