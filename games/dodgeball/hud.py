import cv2
import numpy as np
import time

from .difficulty import get_duration


WHITE = (255, 255, 255)
RED = (60, 60, 240)
GREEN = (80, 220, 80)
YELLOW = (0, 220, 255)

DARK = (20, 20, 25)
GRAY = (90, 90, 90)
LIGHT_GRAY = (190, 190, 190)

PANEL_OPACITY = 0.75
FLASH_DURATION = 0.25

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_panel(frame, x1, y1, x2, y2):
    h, w = frame.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return frame

    region = frame[y1:y2, x1:x2]

    overlay = np.full_like(region, DARK)

    frame[y1:y2, x1:x2] = cv2.addWeighted(
        region,
        1.0 - PANEL_OPACITY,
        overlay,
        PANEL_OPACITY,
        0,
    )

    return frame


def draw_text(
    frame,
    message,
    x,
    y,
    scale=1.0,
    color=WHITE,
    thickness=2,
):
    cv2.putText(
        frame,
        message,
        (int(x), int(y)),
        FONT,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )

    return frame


def draw_centered_text(
    frame,
    message,
    y,
    color=WHITE,
    scale=1.5,
    thickness=3,
):
    h, w = frame.shape[:2]

    text_size, _ = cv2.getTextSize(
        message,
        FONT,
        scale,
        thickness,
    )

    x = (w - text_size[0]) // 2

    return draw_text(
        frame,
        message,
        x,
        y,
        scale,
        color,
        thickness,
    )


def draw_heart(frame, x, y, size, active):
    color = RED if active else GRAY

    radius = max(1, int(size * 0.25))

    # Two rounded upper sections.
    cv2.circle(
        frame,
        (x - radius, y),
        radius,
        color,
        -1,
        cv2.LINE_AA,
    )

    cv2.circle(
        frame,
        (x + radius, y),
        radius,
        color,
        -1,
        cv2.LINE_AA,
    )

    # Lower pointed section.
    triangle = np.array([
        [x - 2 * radius, y],
        [x + 2 * radius, y],
        [x, y + int(size * 0.65)],
    ], dtype=np.int32)

    cv2.fillConvexPoly(
        frame,
        triangle,
        color,
        cv2.LINE_AA,
    )

    return frame


def draw_score(frame, score):
    h, w = frame.shape[:2]

    panel_width = int(w * 0.24)
    panel_height = int(h * 0.15)

    draw_panel(
        frame,
        15,
        15,
        15 + panel_width,
        15 + panel_height,
    )

    draw_text(
        frame,
        "SCORE",
        30,
        15 + int(panel_height * 0.30),
        scale=max(0.4, h / 1100),
        color=LIGHT_GRAY,
        thickness=1,
    )

    draw_text(
        frame,
        f"{score:03d}",
        28,
        15 + int(panel_height * 0.82),
        scale=h / 380,
        thickness=max(2, int(h / 240)),
    )

    return frame


def draw_lives(frame, lives):
    h, w = frame.shape[:2]

    panel_width = int(w * 0.24)
    panel_height = int(h * 0.15)

    panel_x = w - panel_width - 15

    draw_panel(
        frame,
        panel_x,
        15,
        w - 15,
        15 + panel_height,
    )

    draw_text(
        frame,
        "LIVES",
        panel_x + 15,
        15 + int(panel_height * 0.30),
        scale=max(0.4, h / 1100),
        color=LIGHT_GRAY,
        thickness=1,
    )

    heart_size = min(
        int(panel_height * 0.40),
        int(panel_width * 0.23),
    )

    heart_y = 15 + int(panel_height * 0.60)

    for i in range(3):

        heart_x = (
            panel_x
            + int(panel_width * (i + 0.5) / 3)
        )

        draw_heart(
            frame,
            heart_x,
            heart_y,
            heart_size,
            active=i < lives,
        )

    return frame


def draw_title(frame, score):
    h, w = frame.shape[:2]

    panel_width = int(w * 0.32)
    panel_height = int(h * 0.12)

    center_x = w // 2

    draw_panel(
        frame,
        center_x - panel_width // 2,
        15,
        center_x + panel_width // 2,
        15 + panel_height,
    )

    draw_centered_text(
        frame,
        "DODGE AI",
        15 + int(panel_height * 0.45),
        WHITE,
        scale=min(h / 650, w / 1000),
        thickness=2,
    )

    speed = 3.0 / get_duration(score)

    draw_centered_text(
        frame,
        f"SPEED {speed:.1f}x",
        15 + int(panel_height * 0.82),
        LIGHT_GRAY,
        scale=max(0.35, min(h / 1100, w / 1600)),
        thickness=1,
    )

    return frame


def get_tracking_status(tracker):
    if not tracker.locked:

        if tracker.countdown_start is not None:

            elapsed = (
                time.monotonic()
                - tracker.countdown_start
            )

            remaining = max(
                1,
                int(np.ceil(3.0 - elapsed)),
            )

            return f"LOCKING: {remaining}", YELLOW

        return "STAND INSIDE THE BOX", YELLOW

    if tracker.tracking_valid:
        return "PLAYER LOCKED", GREEN

    if tracker.in_grace_period:
        return "RECOVERING TRACKING", YELLOW

    return "TRACKING LOST - PAUSED", RED


def draw_status_bar(frame, tracker):
    h, w = frame.shape[:2]

    bar_height = max(40, int(h * 0.055))

    y = h - bar_height - 15

    draw_panel(
        frame,
        15,
        y,
        w - 15,
        h - 15,
    )

    status, color = get_tracking_status(tracker)

    center_y = y + bar_height // 2

    # Tracking indicator.
    cv2.circle(
        frame,
        (30, center_y),
        5,
        color,
        -1,
        cv2.LINE_AA,
    )

    scale = max(
        0.3,
        min(h / 1400, w / 1600),
    )

    draw_text(
        frame,
        status,
        43,
        center_y + 5,
        scale=scale,
        color=color,
        thickness=1,
    )

    controls = "R: Restart | Q: Quit"

    text_size, _ = cv2.getTextSize(
        controls,
        FONT,
        scale,
        1,
    )

    draw_text(
        frame,
        controls,
        w - text_size[0] - 30,
        center_y + 5,
        scale=scale,
        color=LIGHT_GRAY,
        thickness=1,
    )

    return frame


def draw_starting_box(frame, tracker):
    if tracker.locked:
        return frame

    x1, y1, x2, y2 = tracker.get_starting_box(
        frame
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        YELLOW,
        3,
        cv2.LINE_AA,
    )

    return frame


def draw_impact_effect(frame, state, current_time):
    if not state.message:
        return frame

    elapsed = current_time - state.message_time

    if elapsed >= FLASH_DURATION:
        return frame

    h, w = frame.shape[:2]

    color = (
        RED if state.message == "HIT!"
        else GREEN
    )

    # Gradually fade the border.
    opacity = (
        1.0 - elapsed / FLASH_DURATION
    )

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (w - 1, h - 1),
        color,
        max(8, int(h * 0.025)),
    )

    return cv2.addWeighted(
        overlay,
        opacity,
        frame,
        1.0 - opacity,
        0,
    )


def draw_message(frame, state, current_time):
    if not state.message_active(current_time):
        return frame

    h, w = frame.shape[:2]

    hit = state.message == "HIT!"

    color = RED if hit else GREEN

    subtitle = "-1 LIFE" if hit else "+1 POINT"

    draw_centered_text(
        frame,
        state.message,
        int(h * 0.25),
        color,
        scale=h / 400,
        thickness=3,
    )

    draw_centered_text(
        frame,
        subtitle,
        int(h * 0.30),
        WHITE,
        scale=h / 850,
        thickness=2,
    )

    return frame


def draw_game_over(frame, score):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    overlay[:] = (0, 0, 0)

    frame = cv2.addWeighted(
        frame,
        0.4,
        overlay,
        0.6,
        0,
    )

    frame = draw_centered_text(
        frame,
        "GAME OVER",
        int(h * 0.40),
        RED,
        scale=2.0,
    )

    frame = draw_centered_text(
        frame,
        f"Final Score: {score}",
        int(h * 0.55),
        WHITE,
        scale=1.2,
    )

    frame = draw_centered_text(
        frame,
        "Press R to Restart",
        int(h * 0.70),
        WHITE,
        scale=1.0,
    )

    return frame


def draw_interface(
        frame,
        state,
        tracker,
        current_time,
    ):
        # Draw the initial player locking box.
        frame = draw_starting_box(
            frame,
            tracker,
        )

        # Display brief hit/dodge effects.
        frame = draw_impact_effect(
            frame,
            state,
            current_time,
        )

        # Display gameplay information.
        frame = draw_score(frame, state.score)

        frame = draw_lives(frame, state.lives)

        frame = draw_title(frame, state.score)

        # Hide the bottom bar once the player is locked.
        if not tracker.locked:
            frame = draw_status_bar(frame, tracker)

        # Display a warning if tracking is lost.
        elif not tracker.tracking_valid:
            if not tracker.in_grace_period:
                h, w = frame.shape[:2]

                frame = draw_centered_text(
                    frame,
                    "TRACKING LOST - PAUSED",
                    int(h * 0.50),
                    YELLOW,
                    scale=h / 650,
                    thickness=2,
                )

        # Display hit/dodge notifications.
        frame = draw_message(
            frame,
            state,
            current_time,
        )

        # Keep the existing game-over screen.
        if state.game_over:
            frame = draw_game_over(
                frame,
                state.score,
            )

        return frame