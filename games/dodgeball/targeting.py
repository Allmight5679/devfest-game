import random


TARGETS = [
    "head",
    "chest",
    "left",
    "right",
]


def get_target_positions(frame, points):
    h, w = frame.shape[:2]

    targets = {}

    if points is None:
        return targets

    # Head: nose landmark.
    if points[0] is not None:
        targets["head"] = points[0]

    left_shoulder = points[11]
    right_shoulder = points[12]

    # Chest and side targets.
    if (
        left_shoulder is not None
        and right_shoulder is not None
    ):
        shoulder_x = (
            left_shoulder[0]
            + right_shoulder[0]
        ) / 2

        shoulder_y = (
            left_shoulder[1]
            + right_shoulder[1]
        ) / 2

        chest_y = shoulder_y + h * 0.1

        targets["chest"] = (
            shoulder_x,
            chest_y,
        )

        # Use screen-relative left and right.
        shoulders = sorted(
            [left_shoulder, right_shoulder],
            key=lambda point: point[0],
        )

        targets["left"] = shoulders[0]
        targets["right"] = shoulders[1]

    # Remove targets outside the camera frame.
    return {
        name: position
        for name, position in targets.items()
        if (
            0 <= position[0] < w
            and 0 <= position[1] < h
        )
    }


def get_available_targets(frame, points):
    targets = get_target_positions(
        frame,
        points,
    )

    return list(targets.keys())


def choose_random_target(
    frame,
    points,
    excluded=None,
):
    targets = get_target_positions(
        frame,
        points,
    )

    if excluded is None:
        excluded = []

    available = [
        name
        for name in targets
        if name not in excluded
    ]

    if not available:
        available = list(targets.keys())

    if not available:
        return "center"

    return random.choice(available)


def get_target_position(frame, points, target):
    h, w = frame.shape[:2]

    targets = get_target_positions(
        frame,
        points,
    )

    # Use the requested target if visible.
    if target in targets:
        return target, targets[target]

    # Otherwise, choose a visible alternative.
    if targets:
        fallback = random.choice(
            list(targets.keys())
        )

        return fallback, targets[fallback]

    # Final fallback: center of the screen.
    return "center", (w // 2, h // 2)