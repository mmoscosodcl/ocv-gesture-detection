import os
import sys
import tempfile
import time
import numpy as np
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.action_dispatcher import ActionDispatcher
from src.core.gesture_math import (
    compute_gesture_similarity,
    euclidean_distance_3d,
    get_finger_states,
    normalize_landmarks,
)
from src.core.gesture_storage import Gesture, GestureAction, GestureStorage


def test_normalize_landmarks_translation_and_scale():
    # Synthetic 21 points
    raw = np.zeros((21, 3), dtype=np.float32)
    # Put wrist at (10, 20, 30)
    raw[0] = [10.0, 20.0, 30.0]
    # Put Middle MCP (landmark 9) at (10, 20, 35) -> distance = 5.0
    raw[9] = [10.0, 20.0, 35.0]
    # Put Index TIP (landmark 8) at (10, 25, 30) -> distance from wrist = 5.0
    raw[8] = [10.0, 25.0, 30.0]

    norm = normalize_landmarks(raw)

    # 1. Wrist must be translated to (0, 0, 0)
    np.testing.assert_allclose(norm[0], [0.0, 0.0, 0.0], atol=1e-6)

    # 2. Distance from Wrist to Middle MCP (9) must be normalized to exactly 1.0
    dist_mcp = np.linalg.norm(norm[9])
    assert pytest.approx(dist_mcp, 1e-5) == 1.0

    # 3. Shape check
    assert norm.shape == (21, 3)


def test_normalize_landmarks_invalid_shape():
    with pytest.raises(ValueError):
        normalize_landmarks(np.zeros((10, 3)))


def test_finger_states_heuristic():
    # Build a synthetic hand
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = [0.0, 0.0, 0.0]  # Wrist

    # Extended index finger: PIP closer to wrist, TIP farther
    pts[6] = [0.0, 0.4, 0.0]  # Index PIP
    pts[8] = [0.0, 0.8, 0.0]  # Index TIP

    # Folded middle finger: TIP curled back close to wrist
    pts[10] = [0.1, 0.4, 0.0] # Middle PIP
    pts[12] = [0.1, 0.2, 0.0] # Middle TIP

    states = get_finger_states(pts)
    # states = [thumb, index, middle, ring, pinky]
    assert states[1] is True   # Index is extended
    assert states[2] is False  # Middle is folded


def test_gesture_similarity_exact_match():
    landmarks = np.random.rand(21, 3).astype(np.float32)
    norm = normalize_landmarks(landmarks)
    finger_states = [False, True, True, False, False]

    dist, conf, match = compute_gesture_similarity(
        live_norm_landmarks=norm,
        template_norm_landmarks=norm,
        live_finger_states=finger_states,
        template_finger_states=finger_states,
    )

    assert dist == 0.0
    assert conf == 1.0
    assert match is True


def test_gesture_similarity_mismatch():
    lms1 = np.zeros((21, 3), dtype=np.float32)
    lms1[9] = [0.0, 1.0, 0.0]

    lms2 = np.ones((21, 3), dtype=np.float32)
    lms2[9] = [0.0, 2.0, 0.0]

    norm1 = normalize_landmarks(lms1)
    norm2 = normalize_landmarks(lms2)

    dist, conf, match = compute_gesture_similarity(
        live_norm_landmarks=norm1,
        template_norm_landmarks=norm2,
        live_finger_states=[True, True, True, True, True],
        template_finger_states=[False, False, False, False, False],
    )

    assert dist > 0.0
    assert conf < 0.5
    assert match is False


def test_gesture_storage_roundtrip():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        temp_path = tf.name

    try:
        storage = GestureStorage(temp_path)
        assert storage.load_all() == []

        test_gesture = Gesture(
            name="v_sign",
            finger_states=[False, True, True, False, False],
            normalized_landmarks=np.zeros((21, 3)).tolist(),
            action=GestureAction(type="key", value="space"),
            cooldown=1.5,
            similarity_threshold=0.85,
        )

        storage.add_or_update(test_gesture)
        loaded = storage.load_all()
        assert len(loaded) == 1
        assert loaded[0].name == "v_sign"
        assert loaded[0].action.type == "key"
        assert loaded[0].action.value == "space"
        assert loaded[0].cooldown == 1.5

        # Delete test
        deleted = storage.delete("v_sign")
        assert deleted is True
        assert len(storage.load_all()) == 0
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_action_dispatcher_cooldown():
    dispatcher = ActionDispatcher(dry_run=True)
    gesture = Gesture(
        name="test_snap",
        finger_states=[True, True, False, False, False],
        normalized_landmarks=np.zeros((21, 3)).tolist(),
        action=GestureAction(type="key", value="enter"),
        cooldown=0.2,  # 200ms
    )

    # 1st dispatch -> Should succeed
    assert dispatcher.dispatch(gesture) is True

    # Immediate 2nd dispatch -> Should fail due to cooldown
    assert dispatcher.dispatch(gesture) is False
    assert dispatcher.get_remaining_cooldown(gesture.name, gesture.cooldown) > 0.0

    # Wait past cooldown
    time.sleep(0.22)
    assert dispatcher.dispatch(gesture) is True


def test_hand_tracker_process_frame():
    from src.core.hand_tracker import HandTracker
    tracker = HandTracker()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated, landmarks = tracker.process_frame(frame, draw=True)
    assert annotated.shape == (480, 640, 3)
    # Blank black frame should yield no hand
    assert landmarks is None
    tracker.close()
