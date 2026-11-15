import random
import math
import json
import time

from collections import deque
from concurrent.futures import ThreadPoolExecutor

from client.gemini_client import GeminiClient

from .target import Target, TARGET_RADIUS
from .collision import is_valid_spawn
from .difficulty import get_duration


MAX_SPAWN_ATTEMPTS = 100
MAX_CANDIDATES = 24

REACH_FACTOR = 0.95
FALLBACK_ARM_LENGTH_RATIO = 1.5

HUD_HEIGHT_RATIO = 0.18

MIN_HISTORY = 5
API_COOLDOWN = 3.0
TARGETS_PER_REQUEST = 3

REGIONS = [
    "upper_left",
    "upper_right",
    "middle_left",
    "middle_right",
    "lower_left",
    "lower_right",
]

ARM_LANDMARKS = [
    (11, 13, 15),
    (12, 14, 16),
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

        # Run Gemini requests without blocking gameplay.
        self.executor = ThreadPoolExecutor(
            max_workers=1
        )

        self.future = None
        self.session_id = 0

        self.last_request_time = float("-inf")

        self.target_queue = deque()

        self.strategy = "Collecting reaction data"


    def reset(self):
        self.session_id += 1

        self.target_queue.clear()

        self.strategy = "Collecting reaction data"

        if self.future is not None:
            if self.future.cancel():
                self.future = None


    def close(self):
        self.executor.shutdown(
            wait=False,
            cancel_futures=True,
        )


    # ------------------------------------------
    # PLAYER REACH ESTIMATION
    # ------------------------------------------

    def get_shoulder_width(self, points):
        if points is None:
            return None

        left = points[11]
        right = points[12]

        if left is None or right is None:
            return None

        return math.dist(left, right)


    def get_reachable_areas(self, points):
        if points is None:
            return []

        shoulder_width = self.get_shoulder_width(
            points
        )

        if shoulder_width is None:
            return []

        areas = []

        for shoulder_id, elbow_id, wrist_id in ARM_LANDMARKS:

            shoulder = points[shoulder_id]
            elbow = points[elbow_id]
            wrist = points[wrist_id]

            if shoulder is None:
                continue

            if elbow is not None and wrist is not None:

                upper_arm = math.dist(
                    shoulder,
                    elbow,
                )

                lower_arm = math.dist(
                    elbow,
                    wrist,
                )

                arm_length = upper_arm + lower_arm

            else:

                arm_length = (
                    shoulder_width
                    * FALLBACK_ARM_LENGTH_RATIO
                )

            reach = arm_length * REACH_FACTOR

            areas.append({
                "center": shoulder,
                "radius": reach,
            })

        return areas


    def is_reachable(self, x, y, areas):
        for area in areas:

            distance = math.dist(
                (x, y),
                area["center"],
            )

            if distance <= area["radius"]:
                return True

        return False


    # ------------------------------------------
    # TARGET REGIONS
    # ------------------------------------------

    def get_region(self, x, y, frame, points):
        h, w = frame.shape[:2]

        left = points[11]
        right = points[12]

        if left is None or right is None:
            return "unknown"

        center_x = (left[0] + right[0]) / 2
        center_y = (left[1] + right[1]) / 2

        if x < center_x:
            horizontal = "left"

        else:
            horizontal = "right"

        if y < center_y - h * 0.10:
            vertical = "upper"

        elif y > center_y + h * 0.10:
            vertical = "lower"

        else:
            vertical = "middle"

        return f"{vertical}_{horizontal}"


    # ------------------------------------------
    # VALID TARGET GENERATION
    # ------------------------------------------

    def generate_candidates(
        self,
        frame,
        points,
        mask,
    ):
        h, w = frame.shape[:2]

        areas = self.get_reachable_areas(
            points
        )

        if not areas:
            return []

        candidates = []

        for _ in range(MAX_SPAWN_ATTEMPTS):

            area = random.choice(areas)

            center_x, center_y = area["center"]

            reach = area["radius"]

            angle = random.uniform(
                0,
                2 * math.pi,
            )

            distance = (
                reach * math.sqrt(random.random())
            )

            x = int(
                center_x
                + distance * math.cos(angle)
            )

            y = int(
                center_y
                + distance * math.sin(angle)
            )

            # Avoid overlapping the HUD.
            if y - TARGET_RADIUS < h * HUD_HEIGHT_RATIO:
                continue

            # Ensure the target can be reached.
            if not self.is_reachable(
                x,
                y,
                areas,
            ):
                continue

            # Ensure the target does not overlap
            # the player's body.
            if not is_valid_spawn(
                mask,
                x,
                y,
                TARGET_RADIUS,
            ):
                continue

            region = self.get_region(
                x,
                y,
                frame,
                points,
            )

            candidates.append({
                "x": x,
                "y": y,
                "region": region,
            })

            if len(candidates) >= MAX_CANDIDATES:
                break

        return candidates


    # ------------------------------------------
    # GEMINI PROMPT
    # ------------------------------------------

    def build_prompt(
        self,
        history,
        score,
        combo,
        available_regions,
    ):

        history_json = json.dumps(
            history[-15:],
            indent=2,
        )

        regions_json = json.dumps(
            available_regions
        )

        return f"""
You are the adaptive opponent in ReflexAI,
a real-time motion-controlled reflex game.

The player must touch appearing targets
using either hand before the targets expire.

Your objective is to provide a challenging
but fair experience by learning the player's
reaction patterns.

Current score: {score}
Current combo: {combo}

Currently available target regions:
{regions_json}

Recent target history:
{history_json}

Each history entry contains:
- Target position in normalized screen coordinates.
- Target region relative to the player's shoulders.
- Player position.
- Target lifetime in seconds.
- Whether the target was hit or missed.
- Reaction time for successful hits.
- Reaction ratio relative to the available time.

A higher reaction ratio indicates the player
needed more of the available time.

Analyze:
1. Regions where the player frequently misses.
2. Regions with slower reaction times.
3. Whether switching between opposite sides
   makes the player slower.
4. Repeated patterns in successful touches.
5. Whether the player is maintaining a long combo.

Rules:
- Choose exactly three upcoming target regions.
- Only choose from the currently available regions.
- Challenge weaknesses without repeating one
  region excessively.
- Balance difficult targets with variety.
- Do not assume an unobserved region is difficult.
- Do not control target lifetime or size.
- Do not change scoring or combo multipliers.

Return only valid JSON in this format:

{{
    "strategy": "Brief explanation under 25 words",
    "targets": [
        "upper_left",
        "upper_right",
        "middle_left"
    ]
}}

The example regions are illustrative.
Use only regions that are currently available.
"""


    # ------------------------------------------
    # ASYNCHRONOUS GEMINI REQUESTS
    # ------------------------------------------

    def request_strategy(
        self,
        history,
        score,
        combo,
        available_regions,
    ):

        if self.client is None:
            return

        if len(history) < MIN_HISTORY:
            return

        if not available_regions:
            return

        if self.future is not None:
            return

        # Request more targets before the queue
        # becomes completely empty.
        if len(self.target_queue) > 1:
            return

        current_time = time.monotonic()

        if (
            current_time - self.last_request_time
            < API_COOLDOWN
        ):
            return

        if (
            self.client.get_usage()["remaining_requests"]
            <= 0
        ):
            return

        prompt = self.build_prompt(
            history,
            score,
            combo,
            available_regions,
        )

        self.last_request_time = current_time

        self.future = self.executor.submit(
            self.generate_strategy,
            prompt,
            self.session_id,
            available_regions,
        )

        print(
            "[ReflexAI] Requesting Gemini strategy..."
        )


    def generate_strategy(
        self,
        prompt,
        session_id,
        available_regions,
    ):

        response = self.client.generate(prompt)

        return (
            session_id,
            response,
            available_regions,
        )


    def poll_strategy(self):

        if self.future is None:
            return

        if not self.future.done():
            return

        future = self.future

        self.future = None

        try:

            (
                session_id,
                response,
                available_regions,
            ) = future.result()

            # Ignore responses from previous games.
            if session_id != self.session_id:
                return

            if not response:
                return

            response = response.strip()

            # Remove Markdown code fences if necessary.
            if response.startswith("```"):

                lines = response.splitlines()

                response = "\n".join(lines[1:])

                if response.endswith("```"):
                    response = response[:-3]

            data = json.loads(response.strip())

            if not isinstance(data, dict):
                return

            targets = data.get("targets", [])

            if not isinstance(targets, list):
                return

            validated_targets = []

            for region in targets[:TARGETS_PER_REQUEST]:

                if (
                    isinstance(region, str)
                    and region in REGIONS
                    and region in available_regions
                ):
                    validated_targets.append(region)

            if not validated_targets:
                print(
                    "[ReflexAI] Invalid Gemini strategy."
                )

                return

            self.target_queue.extend(
                validated_targets
            )

            self.strategy = str(
                data.get(
                    "strategy",
                    "Adaptive targeting",
                )
            )[:200]

            print(
                f"[ReflexAI] Strategy: "
                f"{self.strategy}"
            )

            print(
                f"[ReflexAI] Queued targets: "
                f"{list(self.target_queue)}"
            )

        except Exception as error:

            print(
                f"[ReflexAI] Strategy error: "
                f"{type(error).__name__}"
            )


    # ------------------------------------------
    # TARGET SELECTION
    # ------------------------------------------

    def spawn_target(
        self,
        frame,
        points,
        mask,
        score,
        history=None,
        combo=0,
    ):

        if history is None:
            history = []

        # Generate positions that are currently valid.
        candidates = self.generate_candidates(
            frame,
            points,
            mask,
        )

        if not candidates:
            return None

        available_regions = sorted({
            candidate["region"]
            for candidate in candidates
            if candidate["region"] in REGIONS
        })

        # Check for completed Gemini responses.
        self.poll_strategy()

        # Request a new strategy if needed.
        self.request_strategy(
            history,
            score,
            combo,
            available_regions,
        )

        candidate = None

        source = "Local"

        # First five targets collect initial data.
        if len(history) < MIN_HISTORY:

            source = "Calibration"

        else:

            # Find a valid position matching
            # Gemini's requested region.
            while self.target_queue:

                region = self.target_queue.popleft()

                matching_candidates = [
                    item
                    for item in candidates
                    if item["region"] == region
                ]

                if matching_candidates:

                    candidate = random.choice(
                        matching_candidates
                    )

                    source = "Gemini"

                    break

                print(
                    f"[ReflexAI] Region unavailable: "
                    f"{region}"
                )

        # Random fallback.
        if candidate is None:

            candidate = random.choice(
                candidates
            )

        # Target lifetime remains locally controlled.
        target = Target(
            candidate["x"],
            candidate["y"],
            duration=get_duration(score),
        )

        # Store where this target came from.
        target.source = source

        print(
            f"[ReflexAI: {source}] "
            f"Region: {candidate['region']} | "
            f"Position: ({target.x}, {target.y}) | "
            f"Duration: {target.duration}s | "
            f"Queued: {len(self.target_queue)}"
        )

        return target