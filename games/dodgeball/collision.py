import cv2
import numpy as np


MIN_OVERLAP = 0.15
EROSION_KERNEL_SIZE = 7


def check_collision(ball, mask):
    # Do not check collisions before impact.
    if not ball.finished:
        return None

    # Do not register a hit or dodge without valid tracking.
    if mask is None or not np.any(mask):
        return None

    # Shrink the player's mask slightly to forgive
    # inaccurate detection around body boundaries.
    kernel = np.ones(
        (EROSION_KERNEL_SIZE, EROSION_KERNEL_SIZE),
        dtype=np.uint8,
    )

    player_mask = cv2.erode(
        mask.astype(np.uint8),
        kernel,
        iterations=1,
    )

    # Create an empty mask for the ball.
    ball_mask = np.zeros_like(
        player_mask,
        dtype=np.uint8,
    )

    # Draw the ball onto its mask.
    cv2.circle(
        ball_mask,
        (int(ball.x), int(ball.y)),
        ball.radius,
        1,
        -1,
    )

    # Calculate the total area of the ball,
    # including any portion outside the camera frame.
    full_ball_mask = np.zeros(
        (ball.radius * 2 + 1, ball.radius * 2 + 1),
        dtype=np.uint8,
    )

    cv2.circle(
        full_ball_mask,
        (ball.radius, ball.radius),
        ball.radius,
        1,
        -1,
    )

    ball_area = np.count_nonzero(full_ball_mask)

    # Find where the ball and player overlap.
    overlap = np.logical_and(
        ball_mask,
        player_mask,
    )

    overlap_area = np.count_nonzero(overlap)

    # Calculate the percentage of the ball
    # overlapping with the player.
    overlap_ratio = overlap_area / ball_area

    # Register a hit if overlap is at least 15%.
    return bool(overlap_ratio >= MIN_OVERLAP)