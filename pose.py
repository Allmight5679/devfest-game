import cv2
import mediapipe as mp
import numpy as np
import time

from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = str(
    Path(__file__).resolve().parent / "pose_landmarker_full.task"
)

BODY_COLOR = (255, 0, 0)
POINT_COLOR = (0, 0, 255)
MASK_COLOR = (0, 255, 0)

LINE_THICKNESS = 3
MIN_VISIBILITY = 0.5
MASK_THRESHOLD = 0.5
MASK_OPACITY = 0.3


CONNECTIONS = [
    # Face
    (1, 2),
    (2, 3),
    (4, 5),
    (5, 6),
    (9, 10),

    # Arms
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),

    # Hands
    (15, 17),
    (15, 19),
    (15, 21),
    (16, 18),
    (16, 20),
    (16, 22),

    # Torso
    (11, 12),
    (11, 23),
    (12, 24),
    (23, 24),

    # Legs
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),

    # Feet
    (27, 29),
    (29, 31),
    (28, 30),
    (30, 32),
]


def create_detector():
    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.7,
        min_pose_presence_confidence=0.7,
        min_tracking_confidence=0.7,
        output_segmentation_masks=True,
    )

    return vision.PoseLandmarker.create_from_options(
        options
    )


def filter_segmentation(mask, points):
    """
    Remove disconnected segmentation regions that
    do not belong to the detected player.
    """

    if mask is None or points is None:
        return None

    h, w = mask.shape

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )

    # No foreground regions were detected.
    if num_labels <= 1:
        return None

    # Important landmarks for identifying the player.
    anchor_indices = [
        0,   # Nose
        11,  # Left shoulder
        12,  # Right shoulder
        23,  # Left hip
        24,  # Right hip
    ]

    votes = {}

    for index in anchor_indices:
        point = points[index]

        if point is None:
            continue

        x = int(point[0])
        y = int(point[1])

        if not (0 <= x < w and 0 <= y < h):
            continue

        label = labels[y, x]

        # Label 0 represents the background.
        if label == 0:
            continue

        votes[label] = votes.get(label, 0) + 1

    # Reject the mask if no region matches the player.
    if not votes:
        return None

    # Select the region containing the most anchors.
    # If tied, select the larger region.
    player_label = max(
        votes,
        key=lambda label: (
            votes[label],
            stats[label, cv2.CC_STAT_AREA],
        ),
    )

    # Keep only the selected player's region.
    filtered_mask = (
        labels == player_label
    ).astype(np.uint8)

    return filtered_mask


def detect_pose(detector, frame):
    """
    Detect body landmarks and obtain the player's
    filtered segmentation mask.
    """

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame,
    )

    timestamp_ms = time.monotonic_ns() // 1_000_000

    result = detector.detect_for_video(
        mp_image,
        timestamp_ms,
    )

    if not result.pose_landmarks:
        return None, None

    landmarks = result.pose_landmarks[0]

    h, w = frame.shape[:2]

    points = []

    # Extract the 33 pose landmarks.
    for landmark in landmarks:
        if (
            landmark.visibility < MIN_VISIBILITY
            or not 0.0 <= landmark.x < 1.0
            or not 0.0 <= landmark.y < 1.0
        ):
            points.append(None)

        else:
            x = float(landmark.x * w)
            y = float(landmark.y * h)

            points.append((x, y))

    # Extract the segmentation mask.
    mask = None

    if result.segmentation_masks:
        raw_mask = (
            result.segmentation_masks[0]
            .numpy_view()
            .copy()
        )

        raw_mask = raw_mask.squeeze()

        # Ensure the mask matches the camera dimensions.
        if raw_mask.shape != (h, w):
            raw_mask = cv2.resize(
                raw_mask,
                (w, h),
                interpolation=cv2.INTER_LINEAR,
            )

        # Convert confidence values into a binary mask.
        mask = (
            raw_mask >= MASK_THRESHOLD
        ).astype(np.uint8)

        # Remove disconnected background regions.
        mask = filter_segmentation(
            mask,
            points,
        )

    return points, mask


def draw_segmentation(frame, mask):
    """
    Draw a transparent overlay representing
    the player's collision region.
    """

    if mask is None:
        return frame

    output = frame.copy()

    # Create a colored overlay.
    overlay = np.zeros_like(frame)
    overlay[:] = MASK_COLOR

    # Identify pixels belonging to the player.
    body_pixels = mask == 1

    # Blend the overlay with the camera image.
    output[body_pixels] = (
        frame[body_pixels] * (1 - MASK_OPACITY)
        + overlay[body_pixels] * MASK_OPACITY
    ).astype(np.uint8)

    return output


def draw_point(frame, point, radius=6):
    if point is None:
        return

    x, y = point

    cv2.circle(
        frame,
        (int(x), int(y)),
        radius,
        POINT_COLOR,
        -1,
    )


def draw_pose(frame, points):
    """
    Draw the player's skeleton and visible landmarks.
    """

    if points is None:
        return frame

    output = frame.copy()

    for start, end in CONNECTIONS:
        point1 = points[start]
        point2 = points[end]

        if point1 is None or point2 is None:
            continue

        cv2.line(
            output,
            (int(point1[0]), int(point1[1])),
            (int(point2[0]), int(point2[1])),
            BODY_COLOR,
            LINE_THICKNESS,
        )

    for point in points:
        draw_point(output, point)

    return output


def process_frame(
    detector,
    frame,
    show_mask=True,
    show_skeleton=True,
):
    """
    Process a camera frame and return:

    output: Annotated camera frame.
    points: List of 33 pose landmarks.
    mask: Filtered player collision mask.
    """

    points, mask = detect_pose(
        detector,
        frame,
    )

    output = frame.copy()

    if show_mask:
        output = draw_segmentation(
            output,
            mask,
        )

    if show_skeleton:
        output = draw_pose(
            output,
            points,
        )

    return output, points, mask