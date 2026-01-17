from __future__ import annotations

from typing import Dict, List, Tuple

import torch


# MediaPipe hand landmark indices
# 0 wrist
# 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky
HAND_CHAINS = [
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
]

# MediaPipe Pose edges (subset; enough for local ROI constraints)
POSE_EDGES = [
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (23, 25), (25, 27),  # left leg
    (24, 26), (26, 28),  # right leg
    (11, 12), (23, 24), (11, 23), (12, 24),
]


def _safe_norm(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.sqrt(torch.sum(v * v, dim=-1) + eps)


def bone_lengths(points_xy: torch.Tensor, edges: List[Tuple[int, int]]) -> torch.Tensor:
    """Compute bone lengths.

    Args:
        points_xy: (...,K,2)
        edges: list of (i,j)

    Returns:
        (...,E)
    """
    a = points_xy[..., [i for i, _ in edges], :]
    b = points_xy[..., [j for _, j in edges], :]
    return _safe_norm(b - a)


def length_consistency_loss(points_xy: torch.Tensor, edges: List[Tuple[int, int]]) -> torch.Tensor:
    """Penalize length variance across time within a clip.

    points_xy: (T,K,2)
    """
    L = bone_lengths(points_xy, edges)  # (T,E)
    mean = L.mean(dim=0)
    # normalized variance; avoids preferring tiny bones
    return (((L - mean) / (mean.detach() + 1e-6)) ** 2).mean()


def angle(p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """Angle at p1 formed by p0-p1-p2 in radians."""
    v1 = p0 - p1
    v2 = p2 - p1
    v1n = v1 / (_safe_norm(v1)[..., None] + 1e-6)
    v2n = v2 / (_safe_norm(v2)[..., None] + 1e-6)
    dot = (v1n * v2n).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.acos(dot)


def joint_limit_loss_hand(points_xy: torch.Tensor) -> torch.Tensor:
    """Soft joint-limit penalty for MediaPipe hand landmarks.

    points_xy: (T,21,2)

    We constrain flexion angles at finger joints; very loose default limits.
    """
    # For each finger (excluding wrist), penalize extreme angles (too straight or too folded)
    # Limits are broad; the goal is to prevent impossible bends and flipped joints.
    min_ang = 0.15  # ~9 degrees
    max_ang = 3.05  # ~175 degrees

    penalties: List[torch.Tensor] = []
    for chain in HAND_CHAINS:
        # joints at indices chain[1:-1]
        for i in range(1, len(chain) - 1):
            p0 = points_xy[:, chain[i - 1]]
            p1 = points_xy[:, chain[i]]
            p2 = points_xy[:, chain[i + 1]]
            a = angle(p0, p1, p2)
            penalties.append(torch.relu(min_ang - a) ** 2)
            penalties.append(torch.relu(a - max_ang) ** 2)
    return torch.stack(penalties, dim=0).mean()


def temporal_smoothness_loss(points_xy: torch.Tensor, order: int = 2) -> torch.Tensor:
    """Velocity/acceleration smoothness.

    points_xy: (T,K,2)
    order: 1=velocity, 2=acceleration
    """
    if points_xy.shape[0] < 3:
        return points_xy.new_tensor(0.0)
    if order == 1:
        v = points_xy[1:] - points_xy[:-1]
        return (v * v).mean()
    if order == 2:
        a = points_xy[2:] - 2 * points_xy[1:-1] + points_xy[:-2]
        return (a * a).mean()
    raise ValueError("order must be 1 or 2")


def region_losses(region: str, points_xy: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Return a dict of standard losses for a region.

    points_xy is (T,K,2) for that region.
    """
    out: Dict[str, torch.Tensor] = {}
    if region == "hands":
        # Create edges from chains
        edges = []
        for chain in HAND_CHAINS:
            edges += [(chain[i], chain[i + 1]) for i in range(len(chain) - 1)]
        out["len_consistency"] = length_consistency_loss(points_xy, edges)
        out["joint_limits"] = joint_limit_loss_hand(points_xy)
        out["temp_vel"] = temporal_smoothness_loss(points_xy, order=1)
        out["temp_acc"] = temporal_smoothness_loss(points_xy, order=2)
    elif region in ("upper_body", "full_body", "feet", "face"):
        # Minimal defaults; you can extend per region in future.
        # For now: temporal smoothness only.
        out["temp_vel"] = temporal_smoothness_loss(points_xy, order=1)
        out["temp_acc"] = temporal_smoothness_loss(points_xy, order=2)
    else:
        out["temp_acc"] = temporal_smoothness_loss(points_xy, order=2)
    return out
