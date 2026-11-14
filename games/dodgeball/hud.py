import cv2

from .difficulty import get_duration


WHITE = (255, 255, 255)
RED = (0, 0, 255)


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


def draw_centered_text(
    frame,
    message,
    y,
    color,
    scale=1.5,
):
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


def draw_message(frame, state, current_time):
    h, w = frame.shape[:2]

    # Display the temporary impact message.
    if state.message_active(current_time):
        frame = draw_centered_text(
            frame,
            state.message,
            int(h * 0.20),
            state.message_color,
        )

    return frame


def draw_game_over(frame, score):
    h, w = frame.shape[:2]

    # Darken the background.
    overlay = frame.copy()
    overlay[:] = (0, 0, 0)

    frame = cv2.addWeighted(
        frame,
        0.4,
        overlay,
        0.6,
        0,
    )

    # Display game-over title.
    frame = draw_centered_text(
        frame,
        "GAME OVER",
        int(h * 0.40),
        RED,
        scale=2.0,
    )

    # Display final score.
    frame = draw_centered_text(
        frame,
        f"Final Score: {score}",
        int(h * 0.55),
        WHITE,
        scale=1.2,
    )

    # Display restart instructions.
    frame = draw_centered_text(
        frame,
        "Press R to Restart",
        int(h * 0.70),
        WHITE,
        scale=1.0,
    )

    return frame


def draw_interface(frame, state, current_time):
    # Display score, lives and speed.
    frame = draw_hud(
        frame,
        state.score,
        state.lives,
    )

    # Display HIT or DODGED.
    frame = draw_message(
        frame,
        state,
        current_time,
    )

    # Display game-over screen.
    if state.game_over:
        frame = draw_game_over(
            frame,
            state.score,
        )

    return frame