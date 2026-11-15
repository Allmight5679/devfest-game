import cv2
import math


TARGET_RADIUS = 45

TARGET_COLOR = (255, 200, 50)
CENTER_COLOR = (255, 255, 255)

GREEN = (80, 220, 80)
YELLOW = (0, 220, 255)
RED = (60, 60, 240)


class Target:

    def __init__(
        self,
        x,
        y,
        duration=3.0,
        radius=TARGET_RADIUS,
    ):
        self.x = int(x)
        self.y = int(y)

        self.radius = radius

        self.duration = duration
        self.elapsed = 0.0

        self.finished = False
        self.hit = False


    def update(self, delta_time):
        if self.finished:
            return

        self.elapsed = min(
            self.elapsed + delta_time,
            self.duration,
        )

        # Target has expired.
        if self.elapsed >= self.duration:
            self.finished = True


    def get_remaining_time(self):
        return max(
            0.0,
            self.duration - self.elapsed,
        )


    def get_remaining_ratio(self):
        return (
            self.get_remaining_time()
            / self.duration
        )


    def register_hit(self):
        if self.finished:
            return False

        self.hit = True
        self.finished = True

        return True


    def get_color(self):
        remaining = self.get_remaining_ratio()

        if remaining > 0.5:
            return GREEN

        elif remaining > 0.25:
            return YELLOW

        else:
            return RED


    def draw(self, frame):
        if self.finished:
            return frame

        center = (self.x, self.y)

        color = self.get_color()

        remaining = self.get_remaining_ratio()

        # Draw the outer target circle.
        cv2.circle(
            frame,
            center,
            self.radius,
            TARGET_COLOR,
            2,
            cv2.LINE_AA,
        )

        # Draw the shrinking countdown ring.
        countdown_radius = max(
            1,
            int(self.radius * remaining),
        )

        cv2.circle(
            frame,
            center,
            countdown_radius,
            color,
            3,
            cv2.LINE_AA,
        )

        # Draw the center point.
        cv2.circle(
            frame,
            center,
            4,
            CENTER_COLOR,
            -1,
            cv2.LINE_AA,
        )

        return frame