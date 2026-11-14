import cv2
import time
import random

import camera
import pose
import player

from .ball import Ball
from .collision import check_collision


STARTING_LIVES = 3
MESSAGE_DURATION = 1.0

WINDOW_NAME = "DodgeAI"

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)


# Determine ball travel duration based on score.
def get_duration(score):
    if score < 5:
        return 3.0
    elif score < 10:
        return 2.5
    elif score < 20:
        return 2.0
    elif score < 30:
        return 1.5
    elif score < 50:
        return 1.2
    else:
        return 0.8


def spawn_ball(frame, points, score):
    h, w = frame.shape[:2]

    # Default target: center of the screen.
    target_x = w // 2
    target_y = h // 2

    # Aim toward the player's chest.
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

    # Keep the target inside the screen.
    target_x = max(0, min(target_x, w - 1))
    target_y = max(0, min(target_y, h - 1))

    # Choose a random throw origin.
    spawn_positions = [
        (int(w * 0.05), int(h * 0.20)),  # Top left
        (int(w * 0.95), int(h * 0.20)),  # Top right
        (int(w * 0.05), int(h * 0.45)),  # Middle left
        (int(w * 0.95), int(h * 0.45)),  # Middle right
    ]

    start_x, start_y = random.choice(spawn_positions)

    return Ball(
        start_x,
        start_y,
        target_x,
        target_y,
        duration=get_duration(score),
    )


def draw_hud(frame, score, lives):
    h, w = frame.shape[:2]

    # Display the score.
    cv2.putText(
        frame,
        f"Score: {score}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        WHITE,
        2,
        cv2.LINE_AA,
    )

    # Display the remaining lives.
    cv2.putText(
        frame,
        f"Lives: {lives}",
        (20, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        RED,
        2,
        cv2.LINE_AA,
    )

    # Display the relative ball speed.
    cv2.putText(
        frame,
        f"Speed: {3.0 / get_duration(score):.1f}x",
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        WHITE,
        2,
        cv2.LINE_AA,
    )

    # Display controls.
    cv2.putText(
        frame,
        "R: Restart | Q: Quit",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        WHITE,
        2,
        cv2.LINE_AA,
    )

    return frame


def draw_centered_text(frame, message, y, color, scale=1.5):
    h, w = frame.shape[:2]

    text_size, _ = cv2.getTextSize(
        message,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        3,
    )

    x = (w - text_size[0]) // 2

    cv2.putText(
        frame,
        message,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        3,
        cv2.LINE_AA,
    )

    return frame


def run():
    cap = camera.create_camera()
    detector = None

    try:
        detector = pose.create_detector()
        tracker = player.PlayerTracker()

        ball = None

        score = 0
        lives = STARTING_LIVES
        game_over = False

        message = ""
        message_color = WHITE
        message_time = 0.0

        previous_time = time.monotonic()

        while True:
            frame = camera.get_frame(cap)

            if frame is None:
                print("Camera frame could not be read.")
                break

            output, points, mask = pose.process_frame(
                detector,
                frame,
                show_mask=False,
                show_skeleton=False,
            )

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

            # Only update gameplay if the game is active.
            if not game_over:

                if valid_player:

                    # Spawn a ball using the current score.
                    if ball is None:
                        ball = spawn_ball(
                            frame,
                            points,
                            score,
                        )

                    else:
                        ball.update(delta_time)

                elif tracker.in_grace_period:

                    # Continue animation during brief
                    # tracking interruptions.
                    if ball is not None:
                        ball.update(delta_time)

            # Draw the current ball, including at impact.
            if tracker.locked and ball is not None:
                ball.draw(output)

            # Check collision only at impact.
            if (
                not game_over
                and valid_player
                and ball is not None
                and ball.finished
            ):
                result = check_collision(ball, mask)

                if result is True:

                    lives -= 1

                    message = "HIT!"
                    message_color = RED
                    message_time = current_time

                    print(
                        f"HIT! Score: {score}, Lives: {lives}"
                    )

                    ball = None

                elif result is False:

                    score += 1

                    message = "DODGED!"
                    message_color = GREEN
                    message_time = current_time

                    print(
                        f"DODGED! Score: {score}, Lives: {lives}"
                    )

                    ball = None

                # If result is None, wait for valid tracking.

                if lives <= 0:
                    game_over = True

            # Display player tracking status.
            output = tracker.draw_status(output)

            # Display score, lives, and speed.
            output = draw_hud(
                output,
                score,
                lives,
            )

            h, w = output.shape[:2]

            # Display the temporary impact message.
            if (
                message
                and current_time - message_time < MESSAGE_DURATION
                and not game_over
            ):
                output = draw_centered_text(
                    output,
                    message,
                    int(h * 0.20),
                    message_color,
                )

            # Display the game-over screen.
            if game_over:

                overlay = output.copy()
                overlay[:] = (0, 0, 0)

                output = cv2.addWeighted(
                    output,
                    0.4,
                    overlay,
                    0.6,
                    0,
                )

                output = draw_centered_text(
                    output,
                    "GAME OVER",
                    int(h * 0.40),
                    RED,
                    scale=2.0,
                )

                output = draw_centered_text(
                    output,
                    f"Final Score: {score}",
                    int(h * 0.55),
                    WHITE,
                    scale=1.2,
                )

                output = draw_centered_text(
                    output,
                    "Press R to Restart",
                    int(h * 0.70),
                    WHITE,
                    scale=1.0,
                )

            cv2.imshow(WINDOW_NAME, output)

            key = cv2.waitKey(1) & 0xFF

            # Quit the game.
            if key == ord("q"):
                break

            # Restart the game.
            if key == ord("r"):
                tracker.reset()

                ball = None
                score = 0
                lives = STARTING_LIVES
                game_over = False

                message = ""
                message_time = 0.0

                previous_time = time.monotonic()

    finally:
        cap.release()

        if detector is not None:
            detector.close()

        cv2.destroyAllWindows()