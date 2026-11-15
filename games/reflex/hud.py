import cv2
import time
import math

from .game_state import MESSAGE_DURATION


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (160, 160, 160)
YELLOW = (0, 220, 255)
RED = (60, 60, 240)
GREEN = (80, 220, 80)
BLUE = (255, 170, 70)
GOLD = (0, 215, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_text(
    frame,
    message,
    x,
    y,
    scale=1.0,
    color=WHITE,
    thickness=2,
):
    position = (int(x), int(y))

    # Black outline for readability.
    cv2.putText(
        frame,
        message,
        position,
        FONT,
        scale,
        BLACK,
        thickness + 3,
        cv2.LINE_AA,
    )

    # Main text.
    cv2.putText(
        frame,
        message,
        position,
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
    x,
    y,
    scale=1.0,
    color=WHITE,
    thickness=2,
):
    h, w = frame.shape[:2]

    text_size, _ = cv2.getTextSize(
        message,
        FONT,
        scale,
        thickness,
    )

    text_width = text_size[0]
    text_height = text_size[1]

    # Keep floating text inside the screen.
    x = max(
        5,
        min(
            int(x - text_width / 2),
            w - text_width - 5,
        ),
    )

    y = max(
        text_height + 5,
        min(int(y), h - 10),
    )

    return draw_text(
        frame,
        message,
        x,
        y,
        scale,
        color,
        thickness,
    )


def draw_score(frame, state):
    h, w = frame.shape[:2]

    draw_text(
        frame,
        "SCORE",
        20,
        int(h * 0.035),
        scale=max(0.35, h / 1500),
        color=WHITE,
        thickness=1,
    )

    draw_text(
        frame,
        f"{state.score:05d}",
        20,
        int(h * 0.085),
        scale=h / 650,
        thickness=max(2, int(h / 350)),
    )

    return frame


def draw_timer(frame, state):
    h, w = frame.shape[:2]

    remaining = state.get_remaining_time()

    color = RED if remaining <= 10 else WHITE

    # Small timer at the top center.
    draw_centered_text(
        frame,
        state.get_display_time(),
        w // 2,
        int(h * 0.045),
        scale=max(0.35, h / 1100),
        color=color,
        thickness=2,
    )

    # Thin countdown progress bar.
    bar_width = int(w * 0.17)
    bar_height = max(2, int(h * 0.004))

    x = (w - bar_width) // 2
    y = int(h * 0.058)

    progress = remaining / state.duration

    cv2.rectangle(
        frame,
        (x, y),
        (x + bar_width, y + bar_height),
        GRAY,
        -1,
    )

    cv2.rectangle(
        frame,
        (x, y),
        (x + int(bar_width * progress), y + bar_height),
        color,
        -1,
    )

    return frame


def draw_effect(frame, effect, current_time):
    h, w = frame.shape[:2]

    elapsed = current_time - effect["time"]

    progress = max(
        0.0,
        min(elapsed / MESSAGE_DURATION, 1.0),
    )

    if progress >= 1.0:
        return frame

    opacity = 1.0 - progress

    x = effect["x"]
    y = effect["y"]

    color = effect["color"]

    overlay = frame.copy()

    # Expand the ring as the animation fades.
    radius = int(
        effect["radius"] * (1.0 + progress * 0.7)
    )

    cv2.circle(
        overlay,
        (x, y),
        radius,
        color,
        2,
        cv2.LINE_AA,
    )

    # Float the text upward.
    float_distance = int(
        h * 0.045 * progress
    )

    text_y = y - effect["radius"] - float_distance

    scale = max(
        0.4,
        h / 1000,
    )

    if effect["hit"]:

        # Show current combo after three hits.
        if effect["combo"] >= 3:

            combo_color = color

            if effect["combo"] >= 8:
                combo_color = GOLD

            elif effect["combo"] >= 3:
                combo_color = BLUE

            draw_centered_text(
                overlay,
                f"{effect['combo']} COMBO!",
                x,
                text_y - int(h * 0.035),
                scale=scale,
                color=combo_color,
                thickness=2,
            )

        # Show points earned.
        draw_centered_text(
            overlay,
            f"+{effect['points']}",
            x,
            text_y,
            scale=scale * 1.3,
            color=color,
            thickness=2,
        )

    else:

        draw_centered_text(
            overlay,
            "MISS",
            x,
            text_y,
            scale=scale,
            color=RED,
            thickness=2,
        )

    # Fade the complete animation.
    return cv2.addWeighted(
        overlay,
        opacity,
        frame,
        1.0 - opacity,
        0,
    )


def draw_tracking_status(frame, tracker):
    h, w = frame.shape[:2]

    if not tracker.locked:

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

        message = "STAND INSIDE THE BOX"

        if tracker.countdown_start is not None:

            elapsed = (
                time.monotonic()
                - tracker.countdown_start
            )

            remaining = max(
                1,
                math.ceil(3.0 - elapsed),
            )

            message = f"LOCKING: {remaining}"

        draw_centered_text(
            frame,
            message,
            w // 2,
            int(h * 0.92),
            scale=h / 1000,
            color=YELLOW,
            thickness=2,
        )

    elif not tracker.tracking_valid:

        if not tracker.in_grace_period:

            draw_centered_text(
                frame,
                "TRACKING LOST - PAUSED",
                w // 2,
                h // 2,
                scale=h / 1000,
                color=YELLOW,
                thickness=2,
            )

    return frame


def draw_game_over(frame, state):
    h, w = frame.shape[:2]

    # Temporary game-over screen.
    frame = cv2.convertScaleAbs(
        frame,
        alpha=0.4,
        beta=0,
    )

    draw_centered_text(
        frame,
        "TIME'S UP!",
        w // 2,
        int(h * 0.35),
        scale=h / 450,
        thickness=3,
    )

    draw_centered_text(
        frame,
        f"Final Score: {state.score}",
        w // 2,
        int(h * 0.48),
        scale=h / 700,
        thickness=2,
    )

    draw_centered_text(
        frame,
        f"Best Combo: {state.best_combo}",
        w // 2,
        int(h * 0.57),
        scale=h / 1000,
        color=GOLD,
    )

    draw_centered_text(
        frame,
        f"Accuracy: {state.get_accuracy():.1f}%",
        w // 2,
        int(h * 0.64),
        scale=h / 1000,
    )

    draw_centered_text(
        frame,
        "R: Restart | Q: Quit",
        w // 2,
        int(h * 0.78),
        scale=h / 1200,
    )

    return frame


def draw_interface(
    frame,
    state,
    tracker,
    current_time,
):
    # Draw temporary hit and combo animations.
    for effect in state.effects:
        frame = draw_effect(
            frame,
            effect,
            current_time,
        )

    # Draw permanent gameplay information.
    frame = draw_score(frame, state)
    frame = draw_timer(frame, state)

    # Display tracking instructions when needed.
    if not state.game_over:
        frame = draw_tracking_status(
            frame,
            tracker,
        )

    # Display the temporary game-over screen.
    if state.game_over:
        frame = draw_game_over(
            frame,
            state,
        )

    return frame