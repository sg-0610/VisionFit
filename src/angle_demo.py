# src/angle_demo.py
# Goal: Open webcam, run MediaPipe BlazePose live, compute elbow/knee angles,
# and display them ONLY when the required joints are confidently detected.
# Press "q" to quit.

import cv2
import mediapipe as mp

from angles_utils import angle_degrees, landmark_to_point2d

# MediaPipe Pose landmark indices
LEFT_SHOULDER = 11
LEFT_ELBOW = 13
LEFT_WRIST = 15

LEFT_HIP = 23
LEFT_KNEE = 25
LEFT_ANKLE = 27

# Visibility threshold: increase to be stricter (e.g., 0.6 or 0.7)
VIS_THRESH = 0.5


def is_visible(lm) -> bool:
    """
    Returns True if the landmark is confidently visible.
    MediaPipe Pose landmarks usually have .visibility in [0..1].
    """
    return hasattr(lm, "visibility") and lm.visibility is not None and lm.visibility >= VIS_THRESH

def resize_to_fit(img, max_w=1280, max_h=720):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)  # never upscale
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h))


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌ Webcam could not open. Try index 0/1/2 and permissions.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,540)

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        print("✅ Angle demo running. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Could not read frame from webcam.")
                break

            # Convert BGR (OpenCV) -> RGB (MediaPipe)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = pose.process(frame_rgb)
            frame_rgb.flags.writeable = True

            annotated = frame.copy()

            if results.pose_landmarks:
                # Draw pose landmarks/skeleton
                mp_drawing.draw_landmarks(
                    annotated,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                )

                lm = results.pose_landmarks.landmark  # list of 33 landmarks

                # ----- LEFT ELBOW ANGLE -----
                if is_visible(lm[LEFT_SHOULDER]) and is_visible(lm[LEFT_ELBOW]) and is_visible(lm[LEFT_WRIST]):
                    elbow = angle_degrees(
                        landmark_to_point2d(lm[LEFT_SHOULDER]),
                        landmark_to_point2d(lm[LEFT_ELBOW]),
                        landmark_to_point2d(lm[LEFT_WRIST]),
                    )
                    elbow_text = f"Left Elbow: {elbow:.1f} deg"
                else:
                    elbow_text = "Left Elbow: N/A"

                # ----- LEFT KNEE ANGLE -----
                if is_visible(lm[LEFT_HIP]) and is_visible(lm[LEFT_KNEE]) and is_visible(lm[LEFT_ANKLE]):
                    knee = angle_degrees(
                        landmark_to_point2d(lm[LEFT_HIP]),
                        landmark_to_point2d(lm[LEFT_KNEE]),
                        landmark_to_point2d(lm[LEFT_ANKLE]),
                    )
                    knee_text = f"Left Knee: {knee:.1f} deg"
                else:
                    knee_text = "Left Knee: N/A"

                # Put text on screen
                cv2.putText(
                    annotated,
                    elbow_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    knee_text,
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # Optional: show raw visibility values (useful for debugging)
                # cv2.putText(annotated, f"Elbow vis: {lm[LEFT_ELBOW].visibility:.2f}", (20, 120),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            else:
                # No pose detected at all
                cv2.putText(
                    annotated,
                    "No pose detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            display = resize_to_fit(annotated, max_w=1280, max_h=720)
            cv2.imshow("VisionFit - Angle Demo (press 'q')", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Closed webcam and window.")


if __name__ == "__main__":
    main()