from collections import deque


MAX_HISTORY = 15


class TargetHistory:

    def __init__(self):
        self.reset()


    def reset(self):
        self.history = deque(
            maxlen=MAX_HISTORY
        )

        self.current_target = None
        self.target_count = 0


    def get_player_position(self, points):
        if points is None:
            return None

        left = points[11]
        right = points[12]

        if left is None or right is None:
            return None

        center_x = (left[0] + right[0]) / 2
        center_y = (left[1] + right[1]) / 2

        return center_x, center_y


    def normalize_position(self, x, y, frame):
        h, w = frame.shape[:2]

        return [
            round(float(x / w), 3),
            round(float(y / h), 3),
        ]


    def get_region(self, target, frame, points):
        center = self.get_player_position(points)

        if center is None:
            return "unknown"

        h, w = frame.shape[:2]

        center_x, center_y = center

        # Left or right relative to the player.
        if target.x < center_x:
            horizontal = "left"
        else:
            horizontal = "right"

        # Upper, middle or lower relative
        # to shoulder height.
        if target.y < center_y - h * 0.10:
            vertical = "upper"

        elif target.y > center_y + h * 0.10:
            vertical = "lower"

        else:
            vertical = "middle"

        return f"{vertical}_{horizontal}"


    def start_target(
        self,
        target,
        frame,
        points,
        score,
        combo,
    ):
        self.target_count += 1

        center = self.get_player_position(points)

        player_position = None

        if center is not None:
            player_position = self.normalize_position(
                center[0],
                center[1],
                frame,
            )

        self.current_target = {
            "target_number": self.target_count,

            "position": self.normalize_position(
                target.x,
                target.y,
                frame,
            ),

            "region": self.get_region(
                target,
                frame,
                points,
            ),

            "player_position": player_position,

            "duration": target.duration,

            "score": score,
            "combo": combo,

            "source": getattr(
                target,
                "source",
                "Local",
            ),

            "result": None,
            "reaction_time": None,
            "reaction_ratio": None,
        }


    def finish_target(self, target, hit):

        if self.current_target is None:
            return

        if hit:
            self.current_target["result"] = "hit"

            self.current_target["reaction_time"] = round(
                target.elapsed,
                3,
            )

            # Reaction time relative to the
            # available target lifetime.
            self.current_target["reaction_ratio"] = round(
                target.elapsed / target.duration,
                3,
            )

        else:
            self.current_target["result"] = "miss"

        self.history.append(
            self.current_target
        )

        self.current_target = None


    def get_history(self):
        return list(self.history)