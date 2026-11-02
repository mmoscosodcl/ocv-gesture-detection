#!/usr/bin/env python3
"""
Step 1: Gesture Definition & Calibration Workflow
Captures hand landmarks from webcam, normalizes them, and associates them
with user-defined keyboard or mouse actions in gestures.json.
"""

import argparse
import os
import sys
import time
from typing import List
import cv2
import numpy as np

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import (
    CAMERA_INDEX,
    DEFAULT_COOLDOWN_SECONDS,
    DEFAULT_GESTURES_FILE,
    FRAME_HEIGHT,
    FRAME_WIDTH,
)
from src.core.gesture_math import get_finger_states, normalize_landmarks
from src.core.gesture_storage import Gesture, GestureAction, GestureStorage
from src.core.hand_tracker import HandTracker
from src.ui.hud import HUD


def parse_args():
    parser = argparse.ArgumentParser(description="Step 1: Define hand gestures and map them to actions.")
    parser.add_argument("--file", default=DEFAULT_GESTURES_FILE, help="Path to gestures.json")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="Webcam device index")
    parser.add_argument("--list", action="store_true", help="List all saved gestures and exit")
    parser.add_argument("--delete", type=str, default=None, help="Delete a gesture by name and exit")
    parser.add_argument("--name", type=str, default=None, help="Name of gesture to define")
    parser.add_argument("--action-type", choices=["key", "hotkey", "mouse_click", "mouse_scroll"], default=None)
    parser.add_argument("--action-value", type=str, default=None, help="Key name, hotkey (comma-separated), or mouse button")
    parser.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN_SECONDS, help="Cooldown in seconds")
    return parser.parse_args()


def prompt_action_details():
    """Interactive CLI prompts if arguments were not provided via CLI flags."""
    print("\n--- Define New Gesture ---")
    name = input("Enter gesture name (e.g. peace, thumbs_up, fist, open_hand): ").strip()
    while not name:
        name = input("Gesture name cannot be empty. Enter name: ").strip()

    print("\nSelect Action Type:")
    print("  1) key          - Single keypress (e.g. space, enter, up, down, f5)")
    print("  2) hotkey       - Multi-key combo (e.g. ctrl,c or command,space)")
    print("  3) mouse_click  - Mouse button click (left, right, double_left)")
    print("  4) mouse_scroll - Scroll page (positive=up, negative=down)")
    choice = input("Enter choice (1-4) [default: 1]: ").strip() or "1"

    type_map = {"1": "key", "2": "hotkey", "3": "mouse_click", "4": "mouse_scroll"}
    action_type = type_map.get(choice, "key")

    if action_type == "key":
        action_val = input("Enter key to emulate (e.g. space, enter, left, right, escape) [space]: ").strip() or "space"
    elif action_type == "hotkey":
        raw = input("Enter comma-separated keys (e.g. ctrl,c or command,space): ").strip()
        action_val = [k.strip() for k in raw.split(",") if k.strip()]
    elif action_type == "mouse_click":
        action_val = input("Enter click type (left, right, double_left) [left]: ").strip() or "left"
    elif action_type == "mouse_scroll":
        raw = input("Enter scroll amount (e.g. 5 for up, -5 for down) [5]: ").strip() or "5"
        action_val = int(raw)

    cd_input = input(f"Enter cooldown in seconds [{DEFAULT_COOLDOWN_SECONDS}]: ").strip()
    cooldown = float(cd_input) if cd_input else DEFAULT_COOLDOWN_SECONDS

    return name, action_type, action_val, cooldown


def record_gesture_samples(cap, tracker: HandTracker, sample_count: int = 20) -> tuple:
    """
    Renders camera feed with a countdown, then records sample_count frames
    to compute an averaged, noise-free normalized landmark template.
    """
    countdown_duration = 3.0
    start_time = time.time()
    recorded_norm_landmarks = []
    recorded_finger_states = []

    print("\n[INFO] Look at the camera window. Get your hand pose ready!")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame from camera.")
            return None, None

        frame = cv2.flip(frame, 1)  # Mirror view
        annotated, raw_lms = tracker.process_frame(frame, draw=True)

        elapsed = time.time() - start_time
        remaining = countdown_duration - elapsed

        if remaining > 0:
            # Countdown phase
            banner_text = f"HOLD YOUR GESTURE! Starting in {remaining:.1f}s..."
            HUD.draw_banner(annotated, banner_text, "Keep hand in frame and steady", bg_color=(20, 80, 160))
        else:
            # Recording phase
            if raw_lms is not None:
                norm_lm = normalize_landmarks(raw_lms)
                f_states = get_finger_states(raw_lms)
                recorded_norm_landmarks.append(norm_lm)
                recorded_finger_states.append(f_states)

            progress_pct = int((len(recorded_norm_landmarks) / sample_count) * 100)
            banner_text = f"RECORDING SAMPLES: {len(recorded_norm_landmarks)}/{sample_count} ({progress_pct}%)"
            HUD.draw_banner(annotated, banner_text, "Keep hand steady...", bg_color=(0, 130, 40))

            if len(recorded_norm_landmarks) >= sample_count:
                # Finished recording
                break

        cv2.imshow("Step 1 - Define Gesture", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Recording aborted by user.")
            return None, None

    # Compute averaged landmark template
    avg_landmarks = np.mean(np.array(recorded_norm_landmarks), axis=0).tolist()

    # Majority vote on finger states across sampled frames
    arr_states = np.array(recorded_finger_states)
    majority_states = (np.mean(arr_states, axis=0) >= 0.5).tolist()

    return avg_landmarks, majority_states


def main():
    args = parse_args()
    storage = GestureStorage(args.file)

    # 1. Handle --list
    if args.list:
        gestures = storage.load_all()
        print(f"\nSaved gestures in {args.file} ({len(gestures)} total):")
        for g in gestures:
            fingers_desc = "".join(["1" if s else "0" for s in g.finger_states])
            print(f"  * {g.name:15} | Action: {g.action.type:12} -> {str(g.action.value):15} | Cooldown: {g.cooldown}s | Fingers: [{fingers_desc}]")
        return

    # 2. Handle --delete
    if args.delete:
        if storage.delete(args.delete):
            print(f"[SUCCESS] Gesture '{args.delete}' deleted from {args.file}")
        else:
            print(f"[NOT FOUND] Gesture '{args.delete}' was not found.")
        return

    # 3. Interactive or argument-based definition
    if args.name and args.action_type and args.action_value is not None:
        name = args.name
        action_type = args.action_type
        if action_type == "hotkey" and isinstance(args.action_value, str):
            action_val = [k.strip() for k in args.action_value.split(",") if k.strip()]
        elif action_type == "mouse_scroll":
            action_val = int(args.action_value)
        else:
            action_val = args.action_value
        cooldown = args.cooldown
    else:
        name, action_type, action_val, cooldown = prompt_action_details()

    # Initialize Camera and Hand Tracker
    print(f"\n[INFO] Opening webcam index {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera at index {args.camera}. Check webcam connections.")
        sys.exit(1)

    tracker = HandTracker()

    try:
        avg_landmarks, finger_states = record_gesture_samples(cap, tracker, sample_count=25)
        if avg_landmarks is None:
            print("[CANCELLED] Gesture was not saved.")
            return

        gesture = Gesture(
            name=name,
            finger_states=finger_states,
            normalized_landmarks=avg_landmarks,
            action=GestureAction(type=action_type, value=action_val),
            cooldown=cooldown,
            description=f"Action: {action_type} -> {action_val}",
        )

        storage.add_or_update(gesture)
        print(f"\n[SUCCESS] Gesture '{name}' saved to {args.file}!")
        print(f"          Action:   {action_type} -> {action_val}")
        print(f"          Cooldown: {cooldown}s")
        finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        active_fingers = [finger_names[i] for i, s in enumerate(finger_states) if s]
        print(f"          Extended fingers detected: {', '.join(active_fingers) if active_fingers else 'None (Fist)'}")

    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
