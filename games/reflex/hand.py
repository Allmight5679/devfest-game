import cv2
import mediapipe as mp
import time

from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = str(
    Path(__file__).resolve().parents[2]
    / "hand_landmarker.task"
)


def create_detector():
    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return vision.HandLandmarker.create_from_options(
        options
    )


def detect_hands(detector, frame):
    h, w = frame.shape[:2]

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

    hands = []

    for landmarks in result.hand_landmarks:

        points = []

        for landmark in landmarks:

            if (
                0 <= landmark.x < 1
                and 0 <= landmark.y < 1
            ):
                x = landmark.x * w
                y = landmark.y * h

                points.append((x, y))

            else:
                points.append(None)

        hands.append(points)

    return hands


def draw_hands(frame, hands):
    output = frame.copy()

    for points in hands:

        for point in points:

            if point is None:
                continue

            x, y = point

            cv2.circle(
                output,
                (int(x), int(y)),
                5,
                (0, 255, 0),
                -1,
                cv2.LINE_AA,
            )

    return output