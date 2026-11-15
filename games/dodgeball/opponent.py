import random
import json
import time
import math

from collections import deque
from concurrent.futures import ThreadPoolExecutor

from client.gemini_client import GeminiClient

from .ball import Ball
from .difficulty import get_duration

from .targeting import (
    TARGETS,
    get_available_targets,
    choose_random_target,
    get_target_position,
)


API_COOLDOWN = 3.0
THROWS_PER_REQUEST = 3

MAX_X_OFFSET = 0.15
MAX_Y_OFFSET = 0.10

MIN_SPAWN_DISTANCE = 0.15
MIN_HISTORY = 3

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

        # Keep API requests outside the game loop.
        self.executor = ThreadPoolExecutor(
            max_workers=1
        )

        self.future = None
        self.session_id = 0

        self.last_request_time = float("-inf")

        self.attack_queue = deque()

        # Track the initial calibration throws.
        self.calibration_targets = []
        self.throw_count = 0

        self.strategy = "Collecting player movement data"
        self.last_attack_source = "Local"


    def reset(self):
        # Invalidate responses from previous games.
        self.session_id += 1

        self.attack_queue.clear()

        self.calibration_targets.clear()
        self.throw_count = 0

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


    def summarize_history(self, history):
        summary = []

        for attack in history[-10:]:
            summary.append({
                "throw": attack["throw"],
                "source": attack["source"],
                "target_name": attack["target_name"],
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


    def build_prompt(
        self,
        history,
        score,
        available_targets,
    ):
        summary = self.summarize_history(history)

        return f"""
You are the strategic opponent in a
motion-controlled dodgeball game.

Your goal is to analyze the player's previous
movements and predict future dodges.

Current player score: {score}

Recent attack history:
{json.dumps(summary)}

The first three attacks are randomly generated
to observe the player's natural reactions.

Use these observations and subsequent attack
outcomes to improve your strategy.

All absolute coordinates are normalized
between 0 and 1.

X increases toward the right of the screen.
Y increases toward the bottom of the screen.

The player's recorded position is their
shoulder center.

Available body targets:
{", ".join(available_targets)}

Body target meanings:
- head: player's face
- chest: player's upper torso
- left: screen-left shoulder
- right: screen-right shoulder

Plan exactly {THROWS_PER_REQUEST} upcoming throws.

Allowed launch origins:
{", ".join(SPAWN_POSITIONS.keys())}

Each throw must specify a visible body target.

Target offsets are relative to the selected
body part at the moment the ball is launched.

Target offset restrictions:
- X must be between -0.15 and 0.15.
- Y must be between -0.10 and 0.10.

Negative X anticipates movement to the left.
Positive X anticipates movement to the right.
Negative Y anticipates upward movement.
Positive Y anticipates downward movement.

Analyze which body targets cause particular
dodge directions.

Vary the body targets and launch origins.
Try to predict the player's repeated patterns.
Avoid making all three attacks identical.

You do not control ball size, ball speed,
travel duration, lives, or collision detection.

Keep the strategy explanation under 25 words.

Return only valid JSON using this format:

{{
    "strategy": "Short explanation",
    "throws": [
        {{
            "origin": "top_right",
            "target": "head",
            "target_offset_x": -0.05,
            "target_offset_y": 0.0
        }},
        {{
            "origin": "middle_left",
            "target": "right",
            "target_offset_x": 0.03,
            "target_offset_y": 0.0
        }},
        {{
            "origin": "bottom_right",
            "target": "chest",
            "target_offset_x": 0.10,
            "target_offset_y": 0.02
        }}
    ]
}}
"""


    def request_strategy(
        self,
        history,
        score,
        available_targets,
    ):
        if self.client is None:
            return

        # Wait until three attacks are completed.
        if len(history) < MIN_HISTORY:
            return

        if not available_targets:
            return

        # Only allow one outstanding request.
        if self.future is not None:
            return

        # Keep executing the existing queue.
        if len(self.attack_queue) > 1:
            return

        current_time = time.monotonic()

        # Enforce cooldown.
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
            available_targets,
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
        target = attack.get("target")

        if not isinstance(origin, str):
            return None

        if origin not in SPAWN_POSITIONS:
            return None

        if target not in TARGETS:
            return None

        try:
            offset_x = float(
                attack["target_offset_x"]
            )

            offset_y = float(
                attack["target_offset_y"]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
        ):
            return None

        if not (
            math.isfinite(offset_x)
            and math.isfinite(offset_y)
        ):
            return None

        if abs(offset_x) > MAX_X_OFFSET:
            return None

        if abs(offset_y) > MAX_Y_OFFSET:
            return None

        return {
            "origin": origin,
            "target": target,
            "target_offset_x": offset_x,
            "target_offset_y": offset_y,
        }


    def poll_strategy(self):
        if self.future is None:
            return

        if not self.future.done():
            return

        future = self.future
        self.future = None

        try:
            session_id, response = future.result()

            # Ignore responses from previous games.
            if session_id != self.session_id:
                return

            if not response:
                return

            response = response.strip()

            # Remove optional Markdown fences.
            if response.startswith("```"):
                lines = response.splitlines()
                response = "\n".join(lines[1:])

                if response.endswith("```"):
                    response = response[:-3]

            data = json.loads(response.strip())

            if not isinstance(data, dict):
                return

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

            # Preserve our queue fix.
            # Append instead of discarding old attacks.
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


    def get_local_attack(self, frame, points):
        excluded = []

        # Avoid repeating targets during the
        # initial three calibration throws.
        if self.throw_count < MIN_HISTORY:
            excluded = self.calibration_targets

        target = choose_random_target(
            frame,
            points,
            excluded=excluded,
        )

        return {
            "origin": random.choice(DEFAULT_ORIGINS),
            "target": target,
            "target_offset_x": 0.0,
            "target_offset_y": 0.0,
        }


    def update_strategy(
        self,
        history,
        score,
        frame,
        points,
    ):
        self.poll_strategy()

        available_targets = get_available_targets(
            frame,
            points,
        )

        self.request_strategy(
            history,
            score,
            available_targets,
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

        # Avoid spawning too close to the target.
        if distance < MIN_SPAWN_DISTANCE:

            valid_origins = [
                name
                for name, position
                in SPAWN_POSITIONS.items()
                if math.dist(
                    position,
                    normalized_target,
                ) >= MIN_SPAWN_DISTANCE
            ]

            if valid_origins:
                origin = random.choice(valid_origins)
                spawn = SPAWN_POSITIONS[origin]

        return (
            int(spawn[0] * w),
            int(spawn[1] * h),
            origin,
        )


    def spawn_ball(self, frame, points, score):
        h, w = frame.shape[:2]

        self.poll_strategy()

        # First three throws must be local.
        if self.throw_count < MIN_HISTORY:

            attack = self.get_local_attack(
                frame,
                points,
            )

            self.last_attack_source = "Calibration"

        elif self.attack_queue:

            attack = self.attack_queue.popleft()

            self.last_attack_source = "Gemini"

        else:

            attack = self.get_local_attack(
                frame,
                points,
            )

            self.last_attack_source = "Local"

        # Resolve the requested body target.
        target_name, position = get_target_position(
            frame,
            points,
            attack["target"],
        )

        target_x, target_y = position

        # Only apply predictive offsets if the
        # requested body target is still visible.
        if target_name == attack["target"]:

            target_x += (
                attack["target_offset_x"] * w
            )

            target_y += (
                attack["target_offset_y"] * h
            )

        else:

            print(
                f"[Opponent] Target unavailable: "
                f"{attack['target']} -> {target_name}"
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

        # Record the actual calibration target.
        if self.throw_count < MIN_HISTORY:
            self.calibration_targets.append(
                target_name
            )

        self.throw_count += 1

        start_x, start_y, actual_origin = (
            self.get_spawn_position(
                attack["origin"],
                target_x,
                target_y,
                frame,
            )
        )

        print(
            f"[Opponent: {self.last_attack_source}] "
            f"Throw: {self.throw_count} | "
            f"Origin: {actual_origin} | "
            f"Target: {target_name} | "
            f"Queued: {len(self.attack_queue)}"
        )

        # Ball speed and size remain local.
        return Ball(
            start_x,
            start_y,
            target_x,
            target_y,
            duration=get_duration(score),
            target_name=target_name,
            source=self.last_attack_source,
        )