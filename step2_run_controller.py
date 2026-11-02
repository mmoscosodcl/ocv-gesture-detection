#!/usr/bin/env python3
"""
Step 2: Real-time Gesture Detection & Action Controller
Monitors webcam stream, identifies saved gestures from gestures.json,
and emulates cross-platform keyboard / mouse events via PyAutoGUI with cooldown debouncing.
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import (
    CAMERA_INDEX,
    DEFAULT_GESTURES_FILE,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    IS_MACOS,
)
from src.core.action_dispatcher import ActionDispatcher
from src.core.gesture_math import compute_gesture_similarity, get_finger_states, normalize_landmarks
from src.core.gesture_storage import GestureStorage
from src.core.hand_tracker import HandTracker
from src.ui.hud import HUD


def parse_args():
    parser = argparse.ArgumentParser(description="Step 2: Real-time Gesture Action Controller.")
    parser.add_argument("--file", default=DEFAULT_GESTURES_FILE, help="Path to gestures.json")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="Webcam device index")
    parser.add_argument("--dry-run", action="store_true", help="Log actions without sending real keystrokes/clicks")
    parser.add_argument("--threshold", type=float, default=0.78, help="Confidence threshold (0.0 to 1.0) [default: 0.78]")
    return parser.parse_args()


def main():
    args = parse_args()
    storage = GestureStorage(args.file)
    gestures = storage.load_all()

    if not gestures:
        print(f"[WARNING] No gestures found in '{args.file}'.")
        print("          Run 'python step1_define_gesture.py' first to define gestures.")
        sys.exit(1)

    print(f"\n[INFO] Loaded {len(gestures)} gestures from '{args.file}':")
    for g in gestures:
        print(f"  * {g.name:15} -> {g.action.type:10} ({g.action.value}) [Cooldown: {g.cooldown}s]")

    if IS_MACOS and not args.dry_run:
        print("\n[NOTE for macOS]: If keypresses do not trigger in target apps, ensure your Terminal/IDE")
        print("                  has permission in: System Settings -> Privacy & Security -> Accessibility\n")

    # Initialize Hardware and Processing Engine
    dispatcher = ActionDispatcher(dry_run=args.dry_run)
    tracker = HandTracker()

    print(f"[INFO] Opening webcam index {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera at index {args.camera}.")
        sys.exit(1)

    prev_frame_time = time.time()
    last_triggered_name = ""
    last_action_desc = ""
    last_cooldown_remaining = 0.0
    last_cooldown_total = 1.0

    print("\n[READY] Running Gesture Controller. Press 'q' or 'ESC' in the video window to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Camera stream interrupted.")
                break

            # Mirror horizontal for natural webcam interaction
            frame = cv2.flip(frame, 1)

            # Process hand detection & skeleton
            annotated, raw_landmarks = tracker.process_frame(frame, draw=True)

            # Calculate FPS
            now = time.time()
            fps = 1.0 / (now - prev_frame_time) if (now - prev_frame_time) > 0 else 30.0
            prev_frame_time = now

            best_gesture = None
            best_confidence = 0.0
            best_dist = 999.0
            closest_gesture = None
            closest_conf = 0.0
            live_fingers_str = ""

            if raw_landmarks is not None:
                live_norm = normalize_landmarks(raw_landmarks)
                live_fingers = get_finger_states(raw_landmarks)
                live_fingers_str = "".join(["1" if f else "0" for f in live_fingers])

                for gesture in gestures:
                    dist, conf, states_match = compute_gesture_similarity(
                        live_norm,
                        gesture.get_landmarks_array(),
                        live_fingers,
                        gesture.finger_states,
                    )

                    if conf > closest_conf:
                        closest_conf = conf
                        closest_gesture = gesture

                    # Determine effective threshold: if CLI --threshold was customized, prioritize it
                    effective_thresh = args.threshold if args.threshold != 0.78 else (gesture.similarity_threshold or 0.72)

                    # Trigger condition: finger states match OR very high geometric similarity
                    is_match = (states_match and conf >= effective_thresh) or (conf >= effective_thresh + 0.05)

                    if is_match and conf > best_confidence:
                        best_confidence = conf
                        best_dist = dist
                        best_gesture = gesture

                if best_gesture is not None:
                    # Attempt dispatch with cooldown check
                    executed = dispatcher.dispatch(best_gesture)
                    if executed:
                        last_triggered_name = best_gesture.name
                        last_action_desc = f"{best_gesture.name} -> {best_gesture.action.type}:{best_gesture.action.value}"
                        print(f"[TRIGGER] {last_action_desc} (Confidence: {int(best_confidence * 100)}%)")

                    last_cooldown_remaining = dispatcher.get_remaining_cooldown(
                        best_gesture.name, best_gesture.cooldown
                    )
                    last_cooldown_total = best_gesture.cooldown
                else:
                    last_cooldown_remaining = 0.0

            # Determine subtitle / candidate status for HUD
            subtitle = "Press 'q' to quit | Real-time Gesture Controller"
            if raw_landmarks is not None:
                if best_gesture is not None:
                    subtitle = f"Matched: {best_gesture.name} ({int(best_confidence * 100)}%) | Fingers: [{live_fingers_str}]"
                elif closest_gesture is not None and closest_conf >= 0.50:
                    needed = int((closest_gesture.similarity_threshold or 0.72) * 100)
                    subtitle = f"Detecting: {closest_gesture.name} ({int(closest_conf * 100)}% / need {needed}%) | Fingers: [{live_fingers_str}]"
                else:
                    subtitle = f"Hand visible | Fingers: [{live_fingers_str}] | Align gesture pose"

            # Render Status Card HUD
            HUD.draw_banner(
                annotated,
                "Gesture Action Controller",
                subtitle,
                bg_color=(25, 30, 35),
            )
            HUD.draw_status_card(
                annotated,
                gesture_name=best_gesture.name if best_gesture else (f"~{closest_gesture.name}" if closest_gesture and closest_conf >= 0.55 else ""),
                confidence=best_confidence if best_gesture else closest_conf,
                action_desc=last_action_desc,
                cooldown_remaining=last_cooldown_remaining,
                cooldown_total=last_cooldown_total,
                fps=fps,
            )

            cv2.imshow("Step 2 - Gesture Controller", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or ESC
                print("[INFO] Exiting Gesture Controller.")
                break

    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
