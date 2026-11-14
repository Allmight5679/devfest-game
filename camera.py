import cv2
import pose
import player


def create_camera(camera_index=1):
    cap = cv2.VideoCapture(
        camera_index,
        cv2.CAP_AVFOUNDATION,
    )

    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Could not open camera.")

    return cap


def get_frame(cap):
    success, frame = cap.read()

    if not success:
        return None

    # Mirror the camera feed for natural gameplay.
    frame = cv2.flip(frame, 1)

    return frame


def main():
    cap = create_camera()
    detector = None

    try:
        detector = pose.create_detector()

        tracker = player.PlayerTracker()

        while True:
            frame = get_frame(cap)

            if frame is None:
                print("Camera frame could not be read.")
                break

            output, points, mask = pose.process_frame(
                detector,
                frame,
                show_mask=False,
                show_skeleton=False,
            )

            # Check whether the player is locked and valid.
            valid_player = tracker.update(
                frame,
                points,
                mask,
            )

            # Only draw the accepted player.
            if valid_player:
                output = pose.draw_segmentation(
                    output,
                    mask,
                )

                output = pose.draw_pose(
                    output,
                    points,
                )

            # Draw the starting box or tracking status.
            output = tracker.draw_status(output)

            cv2.imshow("DevFest Pose Detection", output)

            key = cv2.waitKey(1) & 0xFF

            # Quit the application.
            if key == ord("q"):
                break

            # Reset player locking.
            if key == ord("r"):
                tracker.reset()

    finally:
        cap.release()

        if detector is not None:
            detector.close()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()