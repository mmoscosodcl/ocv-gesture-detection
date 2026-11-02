# Gesture Detection & Input Emulation PoC 🥷

Real-time hand gesture recognition and action emulation Proof of Concept built with **OpenCV**, **MediaPipe**, and **PyAutoGUI**. Designed with a modular 2-step workflow and architected for cross-platform execution across **macOS**, **Windows**, and **Linux (PC)**.

---

## 🎯 Architecture & Two-Step Workflow

1. **Step 1: Gesture Definition & Calibration Workflow (`step1_define_gesture.py`)**
   - Live visual feedback with countdown.
   - Normalizes 21 3D hand landmarks (translation-invariant to the wrist, scale-invariant to hand size).
   - Detects 5-finger extension states (rotation-invariant).
   - Averages 25 frames for steady template profiling.
   - Maps gesture to **Keypress**, **Hotkey combo**, **Mouse click**, or **Mouse scroll**.
   - Saves profiles into [`gestures.json`](file:///Users/manuel/development/opencv-gestoru-detection/gestures.json).

2. **Step 2: Real-Time Detection & Action Controller (`step2_run_controller.py`)**
   - Live video processing with minimal latency.
   - Evaluates similarity against stored gesture templates.
   - Enforces a configurable debounce **cooldown** (default: 1.0s) to prevent unwanted keystroke spam.
   - Dispatches cross-platform input simulation via PyAutoGUI.
   - Renders a real-time HUD with FPS, detected gesture name, match confidence bar, and cooldown status.

---

## 🚀 Quick Start

### 1. Requirements & Setup

This project uses Python 3.10+ (recommended: Python 3.10).

```bash
# Create and activate virtual environment
python3.10 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Step 1 - Define Your Gestures

Run the interactive definition tool:
```bash
python step1_define_gesture.py
```
* Follow the on-screen prompts: enter a name (e.g. `peace`, `fist`, `thumbs_up`), choose action type (`key`, `hotkey`, `mouse_click`), and hold your hand in front of the camera.
* The script averages the recorded frames and writes the template directly to `gestures.json`.

**Utility commands:**
```bash
# List all saved gestures
python step1_define_gesture.py --list

# Delete an existing gesture
python step1_define_gesture.py --delete peace_sign

# Non-interactive recording with arguments
python step1_define_gesture.py --name peace --action-type key --action-value space --cooldown 1.0
```

### 3. Step 2 - Run the Controller

Start the live controller:
```bash
# Standard mode (triggers real keyboard/mouse inputs)
python step2_run_controller.py

# Dry-run mode (tests detection & displays HUD without sending inputs)
python step2_run_controller.py --dry-run

# Specify custom camera index or confidence threshold (e.g. 0.70)
python step2_run_controller.py --camera 0 --threshold 0.70
```

Press **`q`** or **`ESC`** in the OpenCV camera window to stop the controller.

---

## 🧪 Real-World Scenario Testing (Enter / Return Key)

### Option A: Interactive Browser Test Bed
Open our pre-built HTML test page:
```bash
open test_scenario.html
```
1. Position your browser on the left half of the screen and the OpenCV camera window on the right.
2. Click into the text box inside `test_scenario.html`.
3. Perform the `justgo` gesture to your camera.
4. The page will immediately register the Enter key, animate in green, submit text, and increment the counter.

### Option B: Quick OS Permission & Keystroke Check
To verify that PyAutoGUI has macOS Accessibility permissions to send Enter keypresses to active apps:
```bash
python test_enter_key.py
```
Click any text area (Notes, Browser, Slack) within 3 seconds, and the script will send Enter automatically.

---

## 🖥️ Cross-Platform Deployment (macOS vs PC)

### macOS Setup Note
On macOS, operating system security requires granting **Accessibility permissions** for PyAutoGUI to simulate mouse clicks and keystrokes:
1. Open **System Settings** -> **Privacy & Security** -> **Accessibility**.
2. Enable your terminal emulator (e.g. `Terminal`, `iTerm2`) or IDE (e.g. `Visual Studio Code`).
3. If permissions are missing, `step2_run_controller.py` prints an explicit troubleshooting hint.

### Windows (PC) Setup
- No additional OS permissions required.
- PyAutoGUI automatically translates keys: shortcuts mapped with `command` on Mac automatically map to `ctrl` on Windows.
- Simply run:
  ```cmd
  python -m venv .venv
  .venv\Scripts\activate
  pip install -r requirements.txt
  python step2_run_controller.py
  ```

### Linux (PC) Setup
- On X11 Linux distributions, install the required native dependencies if not already present:
  ```bash
  sudo apt-get install -y python3-tk scrot python3-dev
  ```

---

## 🧪 Automated Testing

Run the test suite with pytest:
```bash
pytest tests/test_gesture_logic.py -v
```

Tests cover:
- Landmark normalization (scale and translation invariance).
- Rotation-invariant finger state heuristics.
- Cosine and Euclidean similarity scoring.
- JSON profile serialization roundtrip.
- Action dispatcher debounce timing & cooldown logic.

---

## 📂 Project Structure

```
opencv-gestoru-detection/
├── config.py                 # Cross-platform config & environment detection
├── requirements.txt          # Python dependencies
├── gestures.json             # Stored gesture templates & action mappings
├── README.md                 # Project documentation
├── src/
│   ├── core/
│   │   ├── hand_tracker.py       # MediaPipe Hands wrapper
│   │   ├── gesture_math.py       # Landmark normalization & finger state heuristics
│   │   ├── gesture_storage.py    # JSON persistence manager
│   │   └── action_dispatcher.py  # PyAutoGUI input simulator with debounce cooldown
│   └── ui/
│       └── hud.py                # OpenCV real-time visual HUD
├── step1_define_gesture.py   # Step 1: Definition & calibration workflow
├── step2_run_controller.py   # Step 2: Live detection & action emulation
└── tests/
    └── test_gesture_logic.py # Unit tests for core algorithms
```
