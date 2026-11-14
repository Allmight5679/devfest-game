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

    try:
        detector = pose.create_detector()

        tracker = player.PlayerTracker()
        opponent = Opponent()
        state = GameState()

        previous_time = time.monotonic()

        while True:

            # Capture the current camera frame.
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

            # Calculate the time between frames.
            current_time = time.monotonic()
            delta_time = current_time - previous_time
            previous_time = current_time

            # Draw the tracked player.
            if valid_player:
                output = pose.draw_segmentation(
                    output,
                    mask,
                )

                output = pose.draw_pose(
                    output,
                    points,
                )

            # Update the current ball.
            state.update_ball(
                opponent,
                frame,
                points,
                valid_player,
                tracker.in_grace_period,
                delta_time,
            )

            # Draw the ball, including at impact.
            if tracker.locked and state.ball is not None:
                state.ball.draw(output)

            # Resolve collisions.
            state.resolve_collision(
                mask,
                valid_player,
                current_time,
            )

            # Display player tracking status.
            output = tracker.draw_status(output)

            # Draw the game interface.
            output = draw_interface(
                output,
                state,
                current_time,
            )

            # Display the current frame.
            cv2.imshow(WINDOW_NAME, output)

            key = cv2.waitKey(1) & 0xFF

            # Quit the game.
            if key == ord("q"):
                break

            # Restart the game.
            if key == ord("r"):
                tracker.reset()
                state.reset()

                previous_time = time.monotonic()

    finally:
        cap.release()

        if detector is not None:
            detector.close()

        cv2.destroyAllWindows()