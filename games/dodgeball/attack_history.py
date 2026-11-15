from collections import deque
import math


MAX_HISTORY = 10
MAX_MOVEMENT_SAMPLES = 30
MOVEMENT_SAMPLE_INTERVAL = 0.1
MOVEMENT_THRESHOLD = 0.03


class AttackHistory:

    def __init__(self):
        self.reset()


    def reset(self):
        self.history = deque(maxlen=MAX_HISTORY)
        self.current_attack = None
        self.last_sample_time = 0.0
        self.throw_count = 0


    def normalize_position(self, x, y, frame):
        h, w = frame.shape[:2]

        return [
            round(float(x / w), 3),
            round(float(y / h), 3),
        ]


    def get_player_position(self, frame, points):
        if points is None:
            return None

        left_shoulder = points[11]
        right_shoulder = points[12]

        if (
            left_shoulder is None
            or right_shoulder is None
        ):
            return None

        center_x = (
            left_shoulder[0]
            + right_shoulder[0]
        ) / 2

        center_y = (
            left_shoulder[1]
            + right_shoulder[1]
        ) / 2

        return self.normalize_position(
            center_x,
            center_y,
            frame,
        )


    def start_attack(self, ball, frame, points, score):
        self.throw_count += 1

        player_position = self.get_player_position(
            frame,
            points,
        )

        self.current_attack = {
            "throw": self.throw_count,
            "score": score,

            "source": ball.source,
            "target_name": ball.target_name,

            "origin": self.normalize_position(
                ball.start_x,
                ball.start_y,
                frame,
            ),

            "target": self.normalize_position(
                ball.target_x,
                ball.target_y,
                frame,
            ),

            "duration": ball.duration,
            "player_start": player_position,
            "player_end": None,
            "movement_samples": [],
            "dodge_direction": None,
            "movement_distance": None,
            "result": None,
        }

        self.last_sample_time = 0.0


    def record_movement(self, ball, frame, points):
        if self.current_attack is None:
            return

        if (
            ball.elapsed - self.last_sample_time
            < MOVEMENT_SAMPLE_INTERVAL
        ):
            return

        position = self.get_player_position(
            frame,
            points,
        )

        if position is None:
            return

        samples = self.current_attack["movement_samples"]

        if len(samples) >= MAX_MOVEMENT_SAMPLES:
            return

        samples.append({
            "time": round(ball.elapsed, 2),
            "position": position,
        })

        self.last_sample_time = ball.elapsed


    def get_dodge_direction(self, start, end):
        if start is None or end is None:
            return "unknown"

        dx = end[0] - start[0]
        dy = end[1] - start[1]

        if (
            abs(dx) < MOVEMENT_THRESHOLD
            and abs(dy) < MOVEMENT_THRESHOLD
        ):
            return "stationary"

        if abs(dx) >= abs(dy):

            if dx > 0:
                return "right"

            return "left"

        else:

            if dy > 0:
                return "down"

            return "up"


    def finish_attack(self, frame, points, hit):
        if self.current_attack is None or hit is None:
            return

        end_position = self.get_player_position(
            frame,
            points,
        )

        start_position = self.current_attack["player_start"]

        self.current_attack["player_end"] = end_position

        self.current_attack["dodge_direction"] = (
            self.get_dodge_direction(
                start_position,
                end_position,
            )
        )

        if (
            start_position is not None
            and end_position is not None
        ):
            distance = math.dist(
                start_position,
                end_position,
            )

            self.current_attack["movement_distance"] = round(
                distance,
                3,
            )

        self.current_attack["result"] = (
            "hit" if hit else "dodged"
        )

        self.history.append(self.current_attack)

        self.current_attack = None


    def get_history(self):
        return list(self.history)