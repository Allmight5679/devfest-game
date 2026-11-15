import random
import json
import time
import math

from collections import deque
from concurrent.futures import ThreadPoolExecutor

from client.gemini_client import GeminiClient

from .ball import Ball
from .difficulty import get_duration


API_COOLDOWN = 3.0
THROWS_PER_REQUEST = 3

MAX_X_OFFSET = 0.15
MAX_Y_OFFSET = 0.10

MIN_SPAWN_DISTANCE = 0.15
MIN_HISTORY = 1

SPAWN_POSITIONS = {
    "top_left": (0.05, 0.20),
    "top_right": (0.95, 0.20),
    "middle_left": (0.05, 0.45),
    "middle_right": (0.95, 0.45),
    "bottom_left": (0.05, 0.85),
    "bottom_right": (0.95, 0.85),
}

DEFAULT_ORIGINS = [
    "top_left",
    "top_right",
    "middle_left",
    "middle_right",
]


class Opponent:

    def __init__(self):
        self.client = None

        try:
            self.client = GeminiClient()

        except Exception as error:
            print(
                f"Gemini unavailable: "
                f"{type(error).__name__}"
            )

        # Run Gemini requests outside the game loop.
        self.executor = ThreadPoolExecutor(
            max_workers=1
        )

        self.future = None
        self.session_id = 0

        self.last_request_time = float("-inf")

        self.attack_queue = deque()

        self.strategy = "Collecting player movement data"

        self.last_attack_source = "Local"


    def reset(self):
        # Invalidate responses from the previous game.
        self.session_id += 1

        self.attack_queue.clear()

        if self.future is not None:
            if self.future.cancel():
                self.future = None

        self.strategy = "Collecting player movement data"
        self.last_attack_source = "Local"


    def close(self):
        self.executor.shutdown(
            wait=False,
            cancel_futures=True,
        )


    def get_chest_position(self, frame, points):
        h, w = frame.shape[:2]

        target_x = w // 2
        target_y = h // 2

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

        return target_x, target_y


    def summarize_history(self, history):
        summary = []

        for attack in history[-10:]:
            summary.append({
                "throw": attack["throw"],
                "origin": attack["origin"],
                "target": attack["target"],
                "player_start": attack["player_start"],
                "player_end": attack["player_end"],
                "duration": attack["duration"],
                "dodge_direction": attack["dodge_direction"],
                "movement_distance": attack["movement_distance"],
                "result": attack["result"],
            })

        return summary


    def build_prompt(self, history, score):
        summary = self.summarize_history(history)

        return f"""
You are the strategic opponent in a
motion-controlled dodgeball game.

Analyze the player's recent attack history
and predict how they might dodge future balls.

Current player score: {score}

Recent attack history:
{json.dumps(summary)}

All coordinates are normalized between 0 and 1.

Positive X means right.
Negative X means left.
Positive Y means down.
Negative Y means up.

The history records the player's shoulder center.
Each attack is aimed relative to the player's
chest position at the moment of launch.

Plan exactly {THROWS_PER_REQUEST} upcoming throws.

Allowed launch origins:
{", ".join(SPAWN_POSITIONS.keys())}

Target offset restrictions:
- X must be between -0.15 and 0.15.
- Y must be between -0.10 and 0.10.

A negative X offset anticipates a leftward dodge.
A positive X offset anticipates a rightward dodge.

Try to exploit repeated movement patterns.
Vary the launch origins and target offsets.
Do not make every throw identical.

You do not control ball size, ball speed,
travel duration, lives, or collision detection.

Return only valid JSON using this format:

{{
    "strategy": "Short explanation",
    "throws": [
        {{
            "origin": "top_right",
            "target_offset_x": -0.08,
            "target_offset_y": 0.0
        }},
        {{
            "origin": "middle_left",
            "target_offset_x": 0.05,
            "target_offset_y": 0.0
        }},
        {{
            "origin": "bottom_right",
            "target_offset_x": -0.10,
            "target_offset_y": 0.02
        }}
    ]
}}
"""


    def request_strategy(self, history, score):
        if self.client is None:
            return

        # Wait until we have some gameplay history.
        if len(history) < MIN_HISTORY:
            return

        # Only allow one outstanding request.
        if self.future is not None:
            return

        # Keep using the existing queue.
        if len(self.attack_queue) > 1:
            return

        current_time = time.monotonic()

        # Enforce the three-second cooldown.
        if (
            current_time - self.last_request_time
            < API_COOLDOWN
        ):
            return

        usage = self.client.get_usage()

        if usage["remaining_requests"] <= 0:
            return

        prompt = self.build_prompt(
            history,
            score,
        )

        self.last_request_time = current_time

        session_id = self.session_id

        self.future = self.executor.submit(
            self.generate_strategy,
            prompt,
            session_id,
        )

        print("[Opponent] Requesting Gemini strategy...")


    def generate_strategy(self, prompt, session_id):
        response = self.client.generate(prompt)

        return session_id, response


    def validate_attack(self, attack):
        if not isinstance(attack, dict):
            return None

        origin = attack.get("origin")

        if origin not in SPAWN_POSITIONS:
            return None

        try:
            offset_x = float(
                attack["target_offset_x"]
            )

            offset_y = float(
                attack["target_offset_y"]
            )

        except (KeyError, TypeError, ValueError):
            return None

        # Reject invalid numerical values.
        if not (
            math.isfinite(offset_x)
            and math.isfinite(offset_y)
        ):
            return None

        # Reject excessively large offsets.
        if abs(offset_x) > MAX_X_OFFSET:
            return None

        if abs(offset_y) > MAX_Y_OFFSET:
            return None

        return {
            "origin": origin,
            "target_offset_x": offset_x,
            "target_offset_y": offset_y,
        }


    def poll_strategy(self):
        # Do not block the webcam while waiting.
        if self.future is None:
            return

        if not self.future.done():
            return

        future = self.future
        self.future = None

        try:
            session_id, response = future.result()

            # Ignore responses from a previous game.
            if session_id != self.session_id:
                return

            if not response:
                return

            response = response.strip()

            # Remove optional Markdown code fences.
            if response.startswith("```"):
                lines = response.splitlines()
                response = "\n".join(lines[1:])

                if response.endswith("```"):
                    response = response[:-3]

            data = json.loads(response.strip())

            throws = data.get("throws", [])

            if not isinstance(throws, list):
                return

            validated = []

            for attack in throws[:THROWS_PER_REQUEST]:
                attack = self.validate_attack(attack)

                if attack is not None:
                    validated.append(attack)

            if not validated:
                print("[Opponent] Invalid Gemini strategy.")
                return

            self.attack_queue.extend(validated)

            self.strategy = str(
                data.get("strategy", "Adaptive attack")
            )[:200]

            print(
                f"[Opponent] Strategy: {self.strategy}"
            )

            print(
                f"[Opponent] Added {len(validated)} throws. "
                f"Total queued: {len(self.attack_queue)}"
            )

        except Exception as error:
            print(
                f"[Opponent] Strategy error: "
                f"{type(error).__name__}"
            )


    def get_local_attack(self):
        # Original random opponent.
        return {
            "origin": random.choice(DEFAULT_ORIGINS),
            "target_offset_x": 0.0,
            "target_offset_y": 0.0,
        }


    def update_strategy(self, history, score):
        self.poll_strategy()

        self.request_strategy(
            history,
            score,
        )


    def get_spawn_position(
        self,
        origin,
        target_x,
        target_y,
        frame,
    ):
        h, w = frame.shape[:2]

        normalized_target = (
            target_x / w,
            target_y / h,
        )

        spawn = SPAWN_POSITIONS[origin]

        distance = math.dist(
            spawn,
            normalized_target,
        )

        # Prevent balls from spawning too close
        # to their intended impact location.
        if distance < MIN_SPAWN_DISTANCE:

            valid_positions = [
                position
                for position in SPAWN_POSITIONS.values()
                if math.dist(
                    position,
                    normalized_target,
                ) >= MIN_SPAWN_DISTANCE
            ]

            if valid_positions:
                spawn = random.choice(valid_positions)

        return (
            int(spawn[0] * w),
            int(spawn[1] * h),
        )


    def spawn_ball(self, frame, points, score):
        h, w = frame.shape[:2]

        # Check whether Gemini has returned a plan.
        self.poll_strategy()

        if self.attack_queue:
            attack = self.attack_queue.popleft()
            self.last_attack_source = "Gemini"

        else:
            attack = self.get_local_attack()
            self.last_attack_source = "Local"

        # Calculate the current chest position.
        target_x, target_y = self.get_chest_position(
            frame,
            points,
        )

        # Apply Gemini's predicted dodge offset.
        target_x += (
            attack["target_offset_x"] * w
        )

        target_y += (
            attack["target_offset_y"] * h
        )

        # Keep the target inside the camera frame.
        target_x = max(
            0,
            min(target_x, w - 1),
        )

        target_y = max(
            0,
            min(target_y, h - 1),
        )

        start_x, start_y = self.get_spawn_position(
            attack["origin"],
            target_x,
            target_y,
            frame,
        )

        print(
            f"[Opponent: {self.last_attack_source}] "
            f"Origin: {attack['origin']} | "
            f"Offset X: {attack['target_offset_x']:.2f} | "
            f"Offset Y: {attack['target_offset_y']:.2f} | "
            f"Queued: {len(self.attack_queue)}"
        )

        # Gemini has no control over duration or radius.
        return Ball(
            start_x,
            start_y,
            target_x,
            target_y,
            duration=get_duration(score),
        )