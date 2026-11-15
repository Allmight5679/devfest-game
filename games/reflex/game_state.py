import math
import json

from .difficulty import (
    DEFAULT_GAME_DURATION,
    GAME_DURATION_OPTIONS,
)

from .collision import check_hand_collision
from .target_history import TargetHistory


MESSAGE_DURATION = 0.6

DOUBLE_POINTS_COMBO = 3
TRIPLE_POINTS_COMBO = 8

GREEN = (80, 220, 80)
RED = (60, 60, 240)
BLUE = (255, 170, 70)
GOLD = (0, 215, 255)
WHITE = (255, 255, 255)

DEBUG_HISTORY = False


class GameState:

    def __init__(self, duration=DEFAULT_GAME_DURATION):

        if duration not in GAME_DURATION_OPTIONS:
            raise ValueError("Invalid game duration.")

        self.duration = duration

        self.target_history = TargetHistory()

        self.reset()


    def reset(self):
        self.target = None

        self.score = 0
        self.combo = 0
        self.best_combo = 0

        self.hits = 0
        self.misses = 0

        self.elapsed = 0.0

        self.started = False
        self.game_over = False

        # Store temporary hit and miss animations.
        self.effects = []

        # Clear reaction history.
        self.target_history.reset()


    def get_remaining_time(self):
        return max(
            0.0,
            self.duration - self.elapsed,
        )


    def get_display_time(self):
        remaining = math.ceil(
            self.get_remaining_time()
        )

        minutes = remaining // 60
        seconds = remaining % 60

        return f"{minutes:02d}:{seconds:02d}"


    def get_accuracy(self):
        total = self.hits + self.misses

        if total == 0:
            return 0.0

        return self.hits / total * 100


    def get_multiplier(self):
        if self.combo >= TRIPLE_POINTS_COMBO:
            return 3

        elif self.combo >= DOUBLE_POINTS_COMBO:
            return 2

        else:
            return 1


    def add_effect(
        self,
        current_time,
        points_earned,
        hit,
    ):
        if self.target is None:
            return

        if not hit:
            color = RED

        elif points_earned == 3:
            color = GOLD

        elif points_earned == 2:
            color = BLUE

        else:
            color = GREEN

        self.effects.append({
            "x": self.target.x,
            "y": self.target.y,
            "radius": self.target.radius,
            "time": current_time,
            "points": points_earned,
            "combo": self.combo,
            "color": color,
            "hit": hit,
        })


    def update_effects(self, current_time):
        # Remove expired animations.
        self.effects = [
            effect
            for effect in self.effects
            if current_time - effect["time"]
            < MESSAGE_DURATION
        ]


    def print_latest_history(self):
        if not DEBUG_HISTORY:
            return

        history = self.target_history.get_history()

        if history:
            print("\n[Target History]")

            print(
                json.dumps(
                    history[-1],
                    indent=4,
                )
            )


    def register_hit(self, current_time):
        # Calculate points using the existing combo.
        points_earned = self.get_multiplier()

        self.score += points_earned
        self.hits += 1
        self.combo += 1

        self.best_combo = max(
            self.best_combo,
            self.combo,
        )

        # Save the target's position for animation.
        self.add_effect(
            current_time,
            points_earned,
            hit=True,
        )

        # Record the successful touch and reaction time.
        if self.target is not None:
            self.target_history.finish_target(
                self.target,
                hit=True,
            )

            self.print_latest_history()

        print(
            f"HIT! "
            f"+{points_earned} | "
            f"Score: {self.score} | "
            f"Combo: {self.combo} | "
            f"Multiplier: {self.get_multiplier()}x | "
            f"Time: {self.get_display_time()}"
        )

        self.target = None


    def register_miss(self, current_time):
        self.misses += 1
        self.combo = 0

        self.add_effect(
            current_time,
            points_earned=0,
            hit=False,
        )

        # Record the missed target.
        if self.target is not None:
            self.target_history.finish_target(
                self.target,
                hit=False,
            )

            self.print_latest_history()

        print(
            f"MISSED! "
            f"Score: {self.score} | "
            f"Time: {self.get_display_time()}"
        )

        self.target = None


    def update(
        self,
        opponent,
        frame,
        points,
        hands,
        mask,
        valid_player,
        delta_time,
        current_time,
    ):
        # Update visual effects independently.
        self.update_effects(current_time)

        if self.game_over:
            return

        # Pause gameplay if tracking is unavailable.
        if not valid_player:
            return

        # Start the timer once the player is locked.
        if not self.started:
            self.started = True
            delta_time = 0.0

        # Update the overall game timer.
        self.elapsed = min(
            self.elapsed + delta_time,
            self.duration,
        )

        # End the game when time runs out.
        if self.elapsed >= self.duration:
            self.game_over = True
            self.target = None

            # Discard any unfinished target.
            self.target_history.current_target = None

            print("GAME OVER")
            print(f"Final score: {self.score}")
            print(f"Total hits: {self.hits}")
            print(f"Total misses: {self.misses}")
            print(f"Best combo: {self.best_combo}")
            print(f"Accuracy: {self.get_accuracy():.1f}%")

            return

        # Spawn a new target using reaction history
        # and Gemini's adaptive strategy.
        if self.target is None:

            self.target = opponent.spawn_target(
                frame,
                points,
                mask,
                self.score,
                history=self.target_history.get_history(),
                combo=self.combo,
            )

            # Begin collecting data for the new target.
            if self.target is not None:
                self.target_history.start_target(
                    self.target,
                    frame,
                    points,
                    self.score,
                    self.combo,
                )

            return

        # Update the current target's lifetime.
        self.target.update(delta_time)

        # Register an expired target.
        if self.target.finished:
            self.register_miss(current_time)
            return

        # Check whether either hand touched it.
        if check_hand_collision(
            self.target,
            hands,
        ):
            if self.target.register_hit():
                self.register_hit(current_time)