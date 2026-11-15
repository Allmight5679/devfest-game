import cv2
import time

import camera
import pose
import player

from . import hand
from .opponent import Opponent
from .game_state import GameState
from .hud import draw_interface
from .difficulty import DEFAULT_GAME_DURATION


WINDOW_NAME = "ReflexAI"

DEBUG_HANDS = False


def run(duration=DEFAULT_GAME_DURATION):
    cap = camera.create_camera()

    detector = None
    hand_detector = None
    opponent = None

    debug_hands = DEBUG_HANDS

    try:
        # Initialize both computer vision models.
        detector = pose.create_detector()
        hand_detector = hand.create_detector()

        # Initialize player tracking and Gemini opponent.
        tracker = player.PlayerTracker()
        opponent = Opponent()

        state = GameState(duration=duration)

        previous_time = time.monotonic()
        previous_valid = False

        while True:

            # Capture the current camera frame.
            frame = camera.get_frame(cap)

            if frame is None:
                print("Camera frame could not be read.")
                break

            # Detect body pose and segmentation.
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

            # Detect hands using the dedicated model.
            hands = []

            if valid_player and not state.game_over:

                hands = hand.detect_hands(
                    hand_detector,
                    frame,
                )

            # Calculate frame timing.
            current_time = time.monotonic()

            delta_time = (
                current_time - previous_time
            )

            previous_time = current_time

            # Prevent tracking interruptions from
            # advancing the game timer.
            if not valid_player or not previous_valid:
                delta_time = 0.0

            previous_valid = valid_player

            # Update gameplay and adaptive targeting.
            state.update(
                opponent,
                frame,
                points,
                hands,
                mask,
                valid_player,
                delta_time,
                current_time,
            )

            # Optional hand tracking visualization.
            if debug_hands:
                output = hand.draw_hands(
                    output,
                    hands,
                )

            # Draw the active target.
            if state.target is not None:
                state.target.draw(output)

            # Draw the gameplay interface.
            output = draw_interface(
                output,
                state,
                tracker,
                current_time,
            )

            # Display the current frame.
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

                # Clear Gemini's previous strategy
                # and queued target regions.
                opponent.reset()

                previous_time = time.monotonic()
                previous_valid = False

            # Toggle hand tracking visualization.
            if key == ord("h"):
                debug_hands = not debug_hands

    finally:
        cap.release()

        if detector is not None:
            detector.close()

        if hand_detector is not None:
            hand_detector.close()

        # Shut down the Gemini request executor.
        if opponent is not None:
            opponent.close()

        cv2.destroyAllWindows()