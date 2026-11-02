from typing import Optional
import cv2
import numpy as np


class HUD:
    """Renders real-time visual feedback and statistics over OpenCV video frames."""

    @staticmethod
    def draw_banner(
        frame: np.ndarray,
        title: str,
        subtitle: Optional[str] = None,
        bg_color: tuple = (30, 30, 30),
        alpha: float = 0.65,
    ) -> None:
        """Draws a modern translucent top banner."""
        h, w, _ = frame.shape
        overlay = frame.copy()
        banner_height = 85 if subtitle else 55

        cv2.rectangle(overlay, (0, 0), (w, banner_height), bg_color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cv2.putText(
            frame,
            title,
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if subtitle:
            cv2.putText(
                frame,
                subtitle,
                (16, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

    @staticmethod
    def draw_status_card(
        frame: np.ndarray,
        gesture_name: str,
        confidence: float,
        action_desc: str,
        cooldown_remaining: float,
        cooldown_total: float,
        fps: float,
    ) -> None:
        """Renders the controller status HUD at the bottom of the screen."""
        h, w, _ = frame.shape
        overlay = frame.copy()
        card_h = 110
        y1 = h - card_h

        # Semi-transparent bottom card
        cv2.rectangle(overlay, (0, y1), (w, h), (20, 24, 28), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # FPS indicator (top right)
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (w - 110, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 200),
            2,
            cv2.LINE_AA,
        )

        # Detected gesture info
        if gesture_name:
            color = (0, 255, 120) if confidence >= 0.75 else (0, 190, 255)
            cv2.putText(
                frame,
                f"Gesture: {gesture_name.upper()}",
                (16, y1 + 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2,
                cv2.LINE_AA,
            )

            # Confidence bar
            conf_bar_w = 160
            bar_x = 280
            bar_y = y1 + 18
            filled_w = int(conf_bar_w * max(0.0, min(1.0, confidence)))
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + conf_bar_w, bar_y + 14), (60, 60, 60), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + 14), color, -1)
            cv2.putText(
                frame,
                f"{int(confidence * 100)}%",
                (bar_x + conf_bar_w + 10, bar_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame,
                "Gesture: (None detected)",
                (16, y1 + 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (140, 140, 140),
                1,
                cv2.LINE_AA,
            )

        # Action and Cooldown
        if action_desc:
            cv2.putText(
                frame,
                f"Action: {action_desc}",
                (16, y1 + 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Cooldown progress bar
        if cooldown_remaining > 0.0 and cooldown_total > 0:
            ratio = max(0.0, min(1.0, cooldown_remaining / cooldown_total))
            cd_bar_w = 200
            cd_x = 16
            cd_y = y1 + 82
            cv2.rectangle(frame, (cd_x, cd_y), (cd_x + cd_bar_w, cd_y + 8), (50, 50, 50), -1)
            cv2.rectangle(frame, (cd_x, cd_y), (cd_x + int(cd_bar_w * ratio), cd_y + 8), (0, 140, 255), -1)
            cv2.putText(
                frame,
                f"Cooldown {cooldown_remaining:.1f}s",
                (cd_x + cd_bar_w + 10, cd_y + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame,
                "Ready to trigger",
                (16, y1 + 92),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 220, 120),
                1,
                cv2.LINE_AA,
            )
