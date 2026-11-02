import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class GestureAction:
    """Represents an emulated keyboard or mouse action."""
    # "key" | "hotkey" | "mouse_click" | "mouse_scroll"
    type: str
    # string (e.g. "space", "enter"), list of strings for hotkey (e.g. ["ctrl", "c"]), or mouse button ("left", "right")
    value: Any

    def validate(self) -> None:
        valid_types = {"key", "hotkey", "mouse_click", "mouse_scroll"}
        if self.type not in valid_types:
            raise ValueError(f"Invalid action type '{self.type}'. Must be one of {valid_types}")
        if self.type == "hotkey" and not isinstance(self.value, (list, tuple)):
            raise ValueError("Hotkey action value must be a list of keys, e.g. ['ctrl', 'c']")


@dataclass
class Gesture:
    """Represents a defined gesture template with normalized landmarks and mapped action."""
    name: str
    finger_states: List[bool]
    normalized_landmarks: List[List[float]]
    action: GestureAction
    cooldown: float = 1.0
    similarity_threshold: float = 0.80
    description: str = ""

    def get_landmarks_array(self) -> np.ndarray:
        return np.array(self.normalized_landmarks, dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "finger_states": self.finger_states,
            "normalized_landmarks": self.normalized_landmarks,
            "action": asdict(self.action),
            "cooldown": self.cooldown,
            "similarity_threshold": self.similarity_threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Gesture":
        action_data = data.get("action", {})
        action = GestureAction(
            type=action_data.get("type", "key"),
            value=action_data.get("value", "space"),
        )
        return cls(
            name=data["name"],
            finger_states=data.get("finger_states", [False] * 5),
            normalized_landmarks=data.get("normalized_landmarks", []),
            action=action,
            cooldown=float(data.get("cooldown", 1.0)),
            similarity_threshold=float(data.get("similarity_threshold", 0.80)),
            description=data.get("description", ""),
        )


class GestureStorage:
    """Manages loading and saving gesture templates to JSON file."""

    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_all(self) -> List[Gesture]:
        if not os.path.exists(self.filepath):
            return []
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                payload = json.load(f)
                return [Gesture.from_dict(g) for g in payload.get("gestures", [])]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load gestures from {self.filepath}: {e}")
            return []

    def save_all(self, gestures: List[Gesture]) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.filepath)), exist_ok=True)
        payload = {
            "version": "1.0",
            "gestures": [g.to_dict() for g in gestures],
        }
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def add_or_update(self, gesture: Gesture) -> None:
        gestures = self.load_all()
        updated = False
        for i, g in enumerate(gestures):
            if g.name.lower() == gesture.name.lower():
                gestures[i] = gesture
                updated = True
                break
        if not updated:
            gestures.append(gesture)
        self.save_all(gestures)

    def delete(self, name: str) -> bool:
        gestures = self.load_all()
        new_list = [g for g in gestures if g.name.lower() != name.lower()]
        if len(new_list) < len(gestures):
            self.save_all(new_list)
            return True
        return False
