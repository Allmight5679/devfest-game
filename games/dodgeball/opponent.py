import random

from .ball import Ball
from .difficulty import get_duration


class Opponent:

    def spawn_ball(self, frame, points, score):
        h, w = frame.shape[:2]

        # Default target: center of the screen.
        target_x = w // 2
        target_y = h // 2

        # Aim toward the player's chest.
        if points is not None:
            left_shoulder = points[11]
            right_shoulder = points[12]

            if (
                left_shoulder is not None
                and right_shoulder is not None
            ):
                target_x = (
                    left_shoulder[0]
                    + right_shoulder[0]
                ) / 2

                shoulder_y = (
                    left_shoulder[1]
                    + right_shoulder[1]
                ) / 2

                target_y = shoulder_y + h * 0.1

        # Keep the target inside the screen.
        target_x = max(0, min(target_x, w - 1))
        target_y = max(0, min(target_y, h - 1))

        # Choose a random throw origin.
        spawn_positions = [
            (int(w * 0.05), int(h * 0.20)),  # Top left
            (int(w * 0.95), int(h * 0.20)),  # Top right
            (int(w * 0.05), int(h * 0.45)),  # Middle left
            (int(w * 0.95), int(h * 0.45)),  # Middle right
        ]

        start_x, start_y = random.choice(spawn_positions)

        return Ball(
            start_x,
            start_y,
            target_x,
            target_y,
            duration=get_duration(score),
        )