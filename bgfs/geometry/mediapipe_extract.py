from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LandmarkResult:
    """Per-frame landmarks from MediaPipe (normalized coordinates)."""

    pose: Optional[np.ndarray]  # (33,3) x,y,z
    pose_vis: Optional[np.ndarray]  # (33,)
    left_hand: Optional[np.ndarray]  # (21,3)
    left_hand_vis: Optional[np.ndarray]  # (21,)
    right_hand: Optional[np.ndarray]  # (21,3)
    right_hand_vis: Optional[np.ndarray]  # (21,)
    face: Optional[np.ndarray]  # (478,3) (may vary by mp version)
    face_vis: Optional[np.ndarray]  # (N,)


def _to_arr(lms) -> np.ndarray:
    return np.array([[lm.x, lm.y, getattr(lm, "z", 0.0)] for lm in lms], dtype=np.float32)


def _to_vis(lms) -> np.ndarray:
    # visibility exists for pose; for hands/face we approximate with 1.0
    out = []
    for lm in lms:
        out.append(float(getattr(lm, "visibility", 1.0)))
    return np.array(out, dtype=np.float32)


def extract_landmarks_holistic(frames_rgb: np.ndarray) -> list[LandmarkResult]:
    """Extract landmarks using MediaPipe Holistic.

    Args:
        frames_rgb: uint8 (T,H,W,3), RGB order.

    Returns:
        list of LandmarkResult of length T.

    Notes:
        MediaPipe is optional. Install with:
          pip install 'bgfs[mediapipe]'
    """
    try:
        import mediapipe as mp  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "MediaPipe is not installed. Install it with: pip install 'bgfs[mediapipe]'"
        ) from e

    mp_holistic = mp.solutions.holistic

    results: list[LandmarkResult] = []
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        for frame in frames_rgb:
            r = holistic.process(frame)

            pose = pose_vis = None
            if r.pose_landmarks is not None:
                pose = _to_arr(r.pose_landmarks.landmark)
                pose_vis = _to_vis(r.pose_landmarks.landmark)

            lh = lh_vis = None
            if r.left_hand_landmarks is not None:
                lh = _to_arr(r.left_hand_landmarks.landmark)
                lh_vis = _to_vis(r.left_hand_landmarks.landmark)

            rh = rh_vis = None
            if r.right_hand_landmarks is not None:
                rh = _to_arr(r.right_hand_landmarks.landmark)
                rh_vis = _to_vis(r.right_hand_landmarks.landmark)

            face = face_vis = None
            if r.face_landmarks is not None:
                face = _to_arr(r.face_landmarks.landmark)
                face_vis = _to_vis(r.face_landmarks.landmark)

            results.append(
                LandmarkResult(
                    pose=pose,
                    pose_vis=pose_vis,
                    left_hand=lh,
                    left_hand_vis=lh_vis,
                    right_hand=rh,
                    right_hand_vis=rh_vis,
                    face=face,
                    face_vis=face_vis,
                )
            )

    return results
