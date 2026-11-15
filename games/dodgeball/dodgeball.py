import cv2
import time

import camera
import pose
import player

from .opponent import Opponent
from .game_state import GameState
from .hud import draw_interface


WINDOW_NAME = "DodgeAI"


def run():
    cap = camera.create_camera()

    detector = None
    opponent = None

    try:
        detector = pose.create_detector()

        tracker = player.PlayerTracker()
        opponent = Opponent()
        state = GameState()

        previous_time = time.monotonic()

        while True:

            # Capture the camera frame.
            frame = camera.get_frame(cap)

            if frame is None:
                print("Camera frame could not be read.")
                break

            # Detect the player's pose and mask.
            output, points, mask = pose.process_frame(
                detector,
                frame,
                show_mask=False,
                show_skeleton=False,
            )

            # Update player tracking.
            valid_player = tracker.update(
                frame,
                points,
                mask,
            )

            # Calculate frame timing.
            current_time = time.monotonic()

            delta_time = (
                current_time - previous_time
            )

            previous_time = current_time

            # Update gameplay and opponent strategy.
            state.update_ball(
                opponent,
                frame,
                points,
                valid_player,
                tracker.in_grace_period,
                delta_time,
            )

            # Draw the active ball.
            if (
                tracker.locked
                and state.ball is not None
            ):
                state.ball.draw(output)

            # Resolve collision and record outcome.
            state.resolve_collision(
                frame,
                points,
                mask,
                valid_player,
                current_time,
            )

            # Draw the updated game interface.
            output = draw_interface(
                output,
                state,
                tracker,
                current_time,
            )

            cv2.imshow(
                WINDOW_NAME,
                output,
            )

            key = cv2.waitKey(1) & 0xFF

            # Quit the game.
            if key == ord("q"):
                break

            # Restart the game.
            if key == ord("r"):

                tracker.reset()
                state.reset()
                opponent.reset()

                previous_time = time.monotonic()

    finally:
        cap.release()

        if detector is not None:
            detector.close()

        if opponent is not None:
            opponent.close()

        cv2.destroyAllWindows()