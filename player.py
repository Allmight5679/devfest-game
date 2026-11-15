import cv2
import numpy as np
import time


LOCK_DURATION = 3.0
TRACKING_GRACE_PERIOD = 1.00

MIN_SHOULDER_WIDTH = 0.08
MAX_POSITION_CHANGE = 1.5
MIN_SIZE_RATIO = 0.55
MAX_SIZE_RATIO = 1.8

BOX_COLOR = (0, 255, 255)
LOCKED_COLOR = (0, 255, 0)
LOST_COLOR = (0, 0, 255)


class PlayerTracker:

    def __init__(self):
        self.reset()


    def reset(self):
        self.locked = False

        self.countdown_start = None

        self.center = None
        self.shoulder_width = None
        self.original_shoulder_width = None

        self.tracking_valid = False

        self.last_valid_time = None
        self.in_grace_period = False


    def get_starting_box(self, frame):
        h, w = frame.shape[:2]

        x1 = int(w * 0.25)
        y1 = int(h * 0.08)

        x2 = int(w * 0.75)
        y2 = int(h * 0.95)

        return x1, y1, x2, y2


    def get_player_info(self, points):
        if points is None:
            return None

        left_shoulder = points[11]
        right_shoulder = points[12]

        if left_shoulder is None or right_shoulder is None:
            return None

        left = np.array(left_shoulder)
        right = np.array(right_shoulder)

        center = (left + right) / 2

        shoulder_width = np.linalg.norm(left - right)

        if shoulder_width <= 0:
            return None

        return center, shoulder_width


    def inside_starting_box(self, frame, points):
        if points is None:
            return False

        x1, y1, x2, y2 = self.get_starting_box(frame)

        # Nose and both shoulders must be inside.
        required_indices = [0, 11, 12]

        for index in required_indices:
            point = points[index]

            if point is None:
                return False

            x, y = point

            if not (x1 <= x <= x2 and y1 <= y <= y2):
                return False

        return True


    def handle_tracking_loss(self):
        """
        Preserve the player lock during temporary
        tracking failures.
        """

        self.tracking_valid = False
        self.in_grace_period = False

        if not self.locked:
            self.countdown_start = None
            return False

        if self.last_valid_time is None:
            return False

        elapsed = (
            time.monotonic() - self.last_valid_time
        )

        if elapsed <= TRACKING_GRACE_PERIOD:
            self.in_grace_period = True

        # Do not accept an unreliable collision mask.
        return False


    def update(self, frame, points, mask):
        self.tracking_valid = False
        self.in_grace_period = False

        info = self.get_player_info(points)

        if (
            info is None
            or mask is None
            or not np.any(mask)
        ):
            return self.handle_tracking_loss()

        center, shoulder_width = info

        h, w = frame.shape[:2]

        if shoulder_width < w * MIN_SHOULDER_WIDTH:
            return self.handle_tracking_loss()

        # ---------------------------------
        # PLAYER SELECTION
        # ---------------------------------

        if not self.locked:

            if not self.inside_starting_box(frame, points):
                self.countdown_start = None
                return False

            if self.countdown_start is None:
                self.countdown_start = time.monotonic()

            elapsed = (
                time.monotonic() - self.countdown_start
            )

            if elapsed >= LOCK_DURATION:
                self.locked = True

                self.center = center
                self.shoulder_width = shoulder_width
                self.original_shoulder_width = shoulder_width

                self.last_valid_time = time.monotonic()

                self.tracking_valid = True

                return True

            return False

        # ---------------------------------
        # LOCKED PLAYER TRACKING
        # ---------------------------------

        position_change = np.linalg.norm(
            center - self.center
        )

        size_ratio = (
            shoulder_width / self.original_shoulder_width
        )

        # Reject sudden changes in player position.
        if position_change > (
            self.shoulder_width * MAX_POSITION_CHANGE
        ):
            return self.handle_tracking_loss()

        # Reject people whose apparent size is
        # significantly different from the original player.
        if not MIN_SIZE_RATIO <= size_ratio <= MAX_SIZE_RATIO:
            return self.handle_tracking_loss()

        # Update the expected position of the player.
        self.center = center

        self.shoulder_width = (
            0.8 * self.shoulder_width
            + 0.2 * shoulder_width
        )

        # Record the most recent valid detection.
        self.last_valid_time = time.monotonic()

        self.tracking_valid = True
        self.in_grace_period = False

        return True


    def draw_status(self, frame):
        output = frame.copy()

        if not self.locked:
            x1, y1, x2, y2 = self.get_starting_box(frame)

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                BOX_COLOR,
                3,
            )

            message = "Stand inside the box"

            if self.countdown_start is not None:
                elapsed = (
                    time.monotonic() - self.countdown_start
                )

                remaining = max(
                    1,
                    int(np.ceil(LOCK_DURATION - elapsed)),
                )

                message = f"Locking player: {remaining}"

            color = BOX_COLOR

        elif self.tracking_valid:
            message = "PLAYER LOCKED"
            color = LOCKED_COLOR

        elif self.in_grace_period:
            message = "PLAYER LOCKED - RECOVERING"
            color = BOX_COLOR

        else:
            message = "TRACKING LOST - PAUSED"
            color = LOST_COLOR

        cv2.putText(
            output,
            message,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA,
        )

        return output