import os
import urllib.request
from typing import List, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np

from config import (
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MODEL_PATH,
    MODEL_URL,
)

# Standard 21 Hand Landmark Connections in MediaPipe
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]


def ensure_model_file(model_path: str = MODEL_PATH, model_url: str = MODEL_URL) -> str:
    """Ensures the MediaPipe hand_landmarker model task file exists, downloading it if needed."""
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print(f"[INFO] Downloading hand landmarker model to {model_path}...")
        urllib.request.urlretrieve(model_url, model_path)
        print("[INFO] Model downloaded successfully.")
    return model_path


class HandTracker:
    """
    Modern Hand Tracker using Google MediaPipe Tasks API (HandLandmarker).
    Provides cross-platform support across modern Python (3.10+), macOS, Windows, and Linux.
    Falls back gracefully to legacy solutions.hands if present in legacy environments.
    """

    def __init__(
        self,
        max_num_hands: int = MAX_NUM_HANDS,
        min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE,
    ):
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.use_tasks_api = not hasattr(mp, "solutions")

        if self.use_tasks_api:
            # Modern MediaPipe Tasks API
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            model_file = ensure_model_file()
            base_options = python.BaseOptions(model_asset_path=model_file)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_tracking_confidence,
                running_mode=vision.RunningMode.IMAGE,
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            self.legacy_hands = None
        else:
            # Legacy mp.solutions API fallback
            self.detector = None
            self.legacy_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )

    def draw_skeleton(self, frame: np.ndarray, landmarks_3d: List[List[float]]) -> None:
        """Renders high-visibility custom skeleton bones and joint points on the OpenCV frame."""
        h, w, _ = frame.shape
        pixel_pts = []

        for pt in landmarks_3d:
            px = int(pt[0] * w)
            py = int(pt[1] * h)
            pixel_pts.append((px, py))

        # Draw bones / connections (Cyan)
        for start_idx, end_idx in HAND_CONNECTIONS:
            p1 = pixel_pts[start_idx]
            p2 = pixel_pts[end_idx]
            cv2.line(frame, p1, p2, (255, 200, 0), 2, cv2.LINE_AA)

        # Draw joints (Green fill with dark border)
        for i, pt in enumerate(pixel_pts):
            radius = 5 if i in (4, 8, 12, 16, 20) else 4  # Fingertips slightly larger
            color = (0, 255, 128) if i in (4, 8, 12, 16, 20) else (0, 220, 255)
            cv2.circle(frame, pt, radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, radius + 1, (20, 20, 20), 1, cv2.LINE_AA)

    def process_frame(
        self, frame: np.ndarray, draw: bool = True
    ) -> Tuple[np.ndarray, Optional[List[List[float]]]]:
        """
        Processes a BGR OpenCV frame and returns (annotated_frame, landmarks_or_None).
        landmarks_3d has 21 points with coordinates [x, y, z].
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks_3d: Optional[List[List[float]]] = None

        if self.use_tasks_api:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_res = self.detector.detect(mp_img)

            if detection_res.hand_landmarks:
                first_hand = detection_res.hand_landmarks[0]
                landmarks_3d = [[lm.x, lm.y, lm.z] for lm in first_hand]
                if draw:
                    self.draw_skeleton(frame, landmarks_3d)
        else:
            rgb_frame.flags.writeable = False
            results = self.legacy_hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                first_hand = results.multi_hand_landmarks[0]
                landmarks_3d = [[lm.x, lm.y, lm.z] for lm in first_hand.landmark]
                if draw:
                    self.draw_skeleton(frame, landmarks_3d)

        return frame, landmarks_3d

    def close(self) -> None:
        if self.detector:
            self.detector.close()
        if self.legacy_hands:
            self.legacy_hands.close()
