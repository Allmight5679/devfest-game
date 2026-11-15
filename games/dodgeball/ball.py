import cv2


START_RADIUS = 5
END_RADIUS = 25

BALL_COLOR = (0, 0, 220)
BORDER_COLOR = (0, 0, 130)
HIGHLIGHT_COLOR = (100, 100, 255)


class Ball:

    def __init__(
        self,
        start_x,
        start_y,
        target_x,
        target_y,
        duration=3.0,
        target_name="chest",
        source="Local",
    ):
        self.start_x = start_x
        self.start_y = start_y

        self.target_x = target_x
        self.target_y = target_y

        self.target_name = target_name
        self.source = source

        self.duration = duration
        self.elapsed = 0.0
        self.progress = 0.0

        self.x = start_x
        self.y = start_y

        self.radius = START_RADIUS
        self.finished = False


    def update(self, delta_time):
        if self.finished:
            return

        self.elapsed = min(
            self.elapsed + delta_time,
            self.duration,
        )

        self.progress = min(
            self.elapsed / self.duration,
            1.0,
        )

        t = self.progress

        # Calculate the current position.
        self.x = (
            self.start_x
            + (self.target_x - self.start_x) * t
        )

        self.y = (
            self.start_y
            + (self.target_y - self.start_y) * t
        )

        # Nonlinear growth to simulate depth.
        self.radius = int(
            START_RADIUS
            + (END_RADIUS - START_RADIUS) * (t ** 1.8)
        )

        if self.progress >= 1.0:
            self.finished = True


    def draw(self, frame):
        center = (
            int(self.x),
            int(self.y),
        )

        # Main ball.
        cv2.circle(
            frame,
            center,
            self.radius,
            BALL_COLOR,
            -1,
            cv2.LINE_AA,
        )

        # Dark outline.
        cv2.circle(
            frame,
            center,
            self.radius,
            BORDER_COLOR,
            2,
            cv2.LINE_AA,
        )

        # Highlight.
        highlight_center = (
            int(self.x - self.radius * 0.3),
            int(self.y - self.radius * 0.3),
        )

        cv2.circle(
            frame,
            highlight_center,
            max(1, int(self.radius * 0.25)),
            HIGHLIGHT_COLOR,
            -1,
            cv2.LINE_AA,
        )

        return frame