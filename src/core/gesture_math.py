import math
from typing import List, Tuple, Union
import numpy as np


def normalize_landmarks(
    landmarks: Union[List[List[float]], np.ndarray]
) -> np.ndarray:
    """
    Normalizes 21 3D hand landmarks to achieve translation and scale invariance.
    
    1. Translation invariance: Wrist (landmark 0) is centered at (0, 0, 0).
    2. Scale invariance: All coordinates are divided by the Euclidean distance 
       between Wrist (0) and Middle Finger MCP (9).
    
    Args:
        landmarks: Shape (21, 3) or iterable of 21 (x, y, z) points.
        
    Returns:
        np.ndarray of shape (21, 3) with normalized coordinates.
    """
    points = np.array(landmarks, dtype=np.float32)
    if points.shape != (21, 3):
        raise ValueError(f"Expected shape (21, 3), got {points.shape}")

    # Shift wrist to origin (0, 0, 0)
    wrist = points[0].copy()
    translated = points - wrist

    # Compute scale factor: distance between Wrist (0) and Middle MCP (9)
    # Using translated[9] directly since translated[0] is (0, 0, 0)
    scale = np.linalg.norm(translated[9])
    if scale < 1e-6:
        scale = 1.0  # Avoid division by zero

    normalized = translated / scale
    return normalized


def euclidean_distance_3d(p1: Union[List[float], np.ndarray], p2: Union[List[float], np.ndarray]) -> float:
    """Calculates 3D Euclidean distance between two points."""
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def get_finger_states(landmarks: Union[List[List[float]], np.ndarray]) -> List[bool]:
    """
    Determines whether each of the 5 fingers is OPEN (extended) or CLOSED (folded).
    Uses distance-to-wrist and palm-anchor heuristics, making it invariant to hand rotation.
    
    Hand Landmark Indices:
      Thumb:  1 (CMC), 2 (MCP), 3 (IP), 4 (TIP)
      Index:  5 (MCP), 6 (PIP), 7 (DIP), 8 (TIP)
      Middle: 9 (MCP), 10 (PIP), 11 (DIP), 12 (TIP)
      Ring:   13 (MCP), 14 (PIP), 15 (DIP), 16 (TIP)
      Pinky:  17 (MCP), 18 (PIP), 19 (DIP), 20 (TIP)
      
    Returns:
        List of 5 booleans: [thumb_open, index_open, middle_open, ring_open, pinky_open]
    """
    pts = np.array(landmarks, dtype=np.float32)

    states = []

    # 1. Thumb: Check distance between Thumb TIP (4) and Pinky MCP (17) vs Thumb IP (3) and Pinky MCP (17)
    # When thumb folds across the palm, landmark 4 moves closer to landmark 17.
    dist_thumb_tip_pinky_mcp = euclidean_distance_3d(pts[4], pts[17])
    dist_thumb_ip_pinky_mcp = euclidean_distance_3d(pts[3], pts[17])
    thumb_open = dist_thumb_tip_pinky_mcp > dist_thumb_ip_pinky_mcp
    states.append(thumb_open)

    # 2. Other 4 fingers (Index, Middle, Ring, Pinky):
    # TIP distance from wrist (0) vs PIP distance from wrist (0)
    # When a finger curls into a fist, its TIP is drawn closer to the palm/wrist than its PIP joint.
    finger_tips_pips = [
        (8, 6),   # Index
        (12, 10), # Middle
        (16, 14), # Ring
        (20, 18), # Pinky
    ]

    for tip_idx, pip_idx in finger_tips_pips:
        dist_tip_wrist = euclidean_distance_3d(pts[tip_idx], pts[0])
        dist_pip_wrist = euclidean_distance_3d(pts[pip_idx], pts[0])
        is_open = dist_tip_wrist > (dist_pip_wrist * 1.05)
        states.append(is_open)

    return states


def compute_gesture_similarity(
    live_norm_landmarks: np.ndarray,
    template_norm_landmarks: np.ndarray,
    live_finger_states: List[bool],
    template_finger_states: List[bool],
) -> Tuple[float, float, bool]:
    """
    Computes matching metrics between live normalized landmarks and a template.
    
    Returns:
        Tuple:
          - distance: Mean Euclidean distance across all 21 normalized landmarks (lower is better, 0.0 is perfect match)
          - confidence: Match score between 0.0 and 1.0 (1.0 is identical)
          - finger_states_match: Boolean indicating if all 5 finger states match exactly
    """
    # 1. Finger state check
    finger_states_match = (live_finger_states == template_finger_states)

    # 2. Average point-wise Euclidean distance across the 21 normalized landmarks
    diffs = np.linalg.norm(live_norm_landmarks - template_norm_landmarks, axis=1)
    mean_dist = float(np.mean(diffs))

    # 3. Confidence mapping:
    # A mean_dist of 0.0 -> 1.0 (100%)
    # A mean_dist of 0.5+ -> ~0.0%
    confidence = max(0.0, min(1.0, 1.0 - (mean_dist / 0.5)))

    # Penalize confidence if finger states don't match
    if not finger_states_match:
        mismatch_count = sum(1 for a, b in zip(live_finger_states, template_finger_states) if a != b)
        confidence = max(0.0, confidence * (1.0 - 0.25 * mismatch_count))

    return mean_dist, confidence, finger_states_match
