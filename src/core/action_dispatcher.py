import logging
import platform
import time
from typing import Any, Dict, Optional
import pyautogui

from config import (
    DEFAULT_COOLDOWN_SECONDS,
    IS_LINUX,
    IS_MACOS,
    IS_WINDOWS,
    PYAUTOGUI_FAILSAFE,
    PYAUTOGUI_PAUSE,
)
from src.core.gesture_storage import Gesture, GestureAction

# Configure PyAutoGUI defaults
pyautogui.PAUSE = PYAUTOGUI_PAUSE
pyautogui.FAILSAFE = PYAUTOGUI_FAILSAFE

logger = logging.getLogger(__name__)


class ActionDispatcher:
    """
    Executes cross-platform keyboard and mouse actions triggered by gestures.
    Maintains cooldown timestamps per gesture to prevent rapid repeated triggering.
    """

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.last_trigger_times: Dict[str, float] = {}

    def get_remaining_cooldown(self, gesture_name: str, cooldown: float) -> float:
        """Returns seconds remaining before the gesture can trigger again. 0.0 if ready."""
        last_time = self.last_trigger_times.get(gesture_name, 0.0)
        elapsed = time.time() - last_time
        remaining = cooldown - elapsed
        return max(0.0, remaining)

    def can_trigger(self, gesture_name: str, cooldown: float) -> bool:
        """Checks if enough time has elapsed since the last trigger."""
        return self.get_remaining_cooldown(gesture_name, cooldown) <= 0.0

    def normalize_key_for_platform(self, key_name: str) -> str:
        """
        Adapts common key names across OS boundaries (e.g. Windows/Linux vs macOS).
        """
        k = key_name.lower().strip()
        if IS_MACOS:
            if k in ("win", "super", "windows"):
                return "command"
            if k == "enter":
                return "return"  # On macOS, main keyboard Enter is 'return' (code 36)
        else:  # Windows / Linux
            if k in ("command", "cmd"):
                return "ctrl"
            if k == "return":
                return "enter"
        return k

    def dispatch(self, gesture: Gesture) -> bool:
        """
        Dispatches the action associated with the gesture if not in cooldown.
        
        Returns:
            True if action was executed, False if skipped due to cooldown.
        """
        cooldown = gesture.cooldown if gesture.cooldown > 0 else DEFAULT_COOLDOWN_SECONDS
        if not self.can_trigger(gesture.name, cooldown):
            return False

        action = gesture.action
        now = time.time()
        self.last_trigger_times[gesture.name] = now

        if self.dry_run:
            print(f"[DRY-RUN] Action triggered: {gesture.name} -> {action.type}: {action.value}")
            return True

        try:
            self._execute_action(action)
            return True
        except Exception as e:
            print(f"[ERROR] Failed to execute action for '{gesture.name}': {e}")
            if IS_MACOS:
                print(
                    "[HINT] On macOS, ensure your Terminal / VS Code has Accessibility permissions:\n"
                    "       System Settings -> Privacy & Security -> Accessibility"
                )
            return False

    def _execute_action(self, action: GestureAction) -> None:
        action_type = action.type.lower()
        val = action.value

        if action_type == "key":
            key = self.normalize_key_for_platform(str(val))
            pyautogui.press(key)

        elif action_type == "hotkey":
            if isinstance(val, (list, tuple)):
                keys = [self.normalize_key_for_platform(str(k)) for k in val]
                pyautogui.hotkey(*keys)
            elif isinstance(val, str):
                # Split comma or plus separated hotkeys
                parts = [self.normalize_key_for_platform(k.strip()) for k in val.replace("+", ",").split(",")]
                pyautogui.hotkey(*parts)

        elif action_type == "mouse_click":
            btn = str(val).lower()
            if btn == "double_left":
                pyautogui.doubleClick(button="left")
            elif btn in ("left", "right", "middle"):
                pyautogui.click(button=btn)
            else:
                pyautogui.click(button="left")

        elif action_type == "mouse_scroll":
            clicks = int(val)
            pyautogui.scroll(clicks)

        else:
            raise ValueError(f"Unsupported action type: {action_type}")
