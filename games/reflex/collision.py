import cv2
import numpy as np
import math


BODY_CLEARANCE = 15
HAND_COLLISION_PADDING = 5


def is_valid_spawn(mask, x, y, radius):
    if mask is None or not np.any(mask):
        return False

    h, w = mask.shape[:2]

    clearance_radius = radius + BODY_CLEARANCE

    x = int(x)
    y = int(y)

    # Ensure the entire target fits inside the frame.
    if (
        x - clearance_radius < 0
        or x + clearance_radius >= w
        or y - clearance_radius < 0
        or y + clearance_radius >= h
    ):
        return False

    target_mask = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    cv2.circle(
        target_mask,
        (x, y),
        clearance_radius,
        1,
        -1,
    )

    overlap = np.logical_and(
        target_mask,
        mask,
    )

    return not bool(np.any(overlap))


def check_hand_collision(target, hands):
    if target.finished or not hands:
        return False

    collision_radius = (
        target.radius + HAND_COLLISION_PADDING
    )

    # Check all 21 landmarks on each detected hand.
    for hand_points in hands:

        for point in hand_points:

            if point is None:
                continue

            distance = math.dist(
                (target.x, target.y),
                point,
            )

            if distance <= collision_radius:
                return True

    return False