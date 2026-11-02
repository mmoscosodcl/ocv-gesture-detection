import os
import platform
import sys

# Ensure writable matplotlib cache dir
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

# Platform detection
SYSTEM_OS = platform.system()
IS_MACOS = SYSTEM_OS == "Darwin"
IS_WINDOWS = SYSTEM_OS == "Windows"
IS_LINUX = SYSTEM_OS == "Linux"

# Camera Settings
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_TARGET = 30

# MediaPipe Hand Tracking Settings
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.6
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# Gesture Matching Settings
# Euclidean distance threshold between normalized 21-landmark vectors (lower = stricter)
DEFAULT_MAX_DISTANCE_THRESHOLD = 0.35
# Cosine similarity threshold (higher = stricter, 1.0 = identical direction)
DEFAULT_MIN_COSINE_SIMILARITY = 0.88

# Action Emulation Settings
DEFAULT_COOLDOWN_SECONDS = 1.0
PYAUTOGUI_PAUSE = 0.05
PYAUTOGUI_FAILSAFE = True

# Storage
DEFAULT_GESTURES_FILE = os.path.join(os.path.dirname(__file__), "gestures.json")
