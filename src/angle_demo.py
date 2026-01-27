# src/angle_demo.py
# Goal: The goal of this program is to open the webcam, run MediaPipe BlazePose live, compute elbow/knee angles,
# and display them only when the required joints are confidently detected.
# Press "q" to quit.

import cv2
import mediapipe as mp

from angles_utils import angle_degrees, landmark_to_point2d
from squat_rules import evaluate_squat

# These. are the MediaPipe Pose landmark indices
LEFT_SHOULDER = 11
LEFT_ELBOW = 13
LEFT_WRIST = 15

LEFT_HIP = 23
LEFT_KNEE = 25
LEFT_ANKLE = 27

# These are the  Visibility threshold: These can be increased like 0.6 or 0.7 to be more stricter
VIS_THRESH = 0.5

def draw_panel(img, x, y, w, h, alpha=0.55):
    """
    Drawing a semi-transparent dark panel so text stays readable on any background.
    """
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)  # black box
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def put_text_with_outline(img, text, org, font_scale=0.8, color=(255, 255, 255), thickness=2):
    """
    White text with a dark outline  this will help in readability.
    """
    x, y = org
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def feedback_color(label: str):
    """
    color coded sqaut rules
    """
    l = label.lower()
    if "good" in l:
        return (0, 200, 0)      # green
    if "deep" in l:
        return (0, 165, 255)    # amber/orange
    if "too" in l:
        return (0, 0, 255)      # red
    return (200, 200, 200)      # grey

def is_visible(lm) -> bool:
    """
    True is returned when body landmarks are confidently visible and MediaPipe pose landmarsk usually have .visibility in [0,1].
    """
    return hasattr(lm, "visibility") and lm.visibility is not None and lm.visibility >= VIS_THRESH


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("The Webcam could not open. Try index 0/1/2 and permissions.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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

        print("Angle demo is running. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame from the webcam could not be read")
                break

            # Convert BGR (OpenCV) to RGB (MediaPipe)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = pose.process(frame_rgb)
            frame_rgb.flags.writeable = True

            annotated = frame.copy()

            # --- HUD layout ---
            panel_x, panel_y = 20, 20
            panel_w, panel_h = 620, 190
            draw_panel(annotated, panel_x, panel_y, panel_w, panel_h, alpha=0.50)

            # Title
            put_text_with_outline(annotated, "VISIONFIT  •  Live Form Check", (panel_x + 15, panel_y + 30), 0.85)


            if results.pose_landmarks:
                # Draw pose landmarks / skeleton
                mp_drawing.draw_landmarks(
                    annotated,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                )

                lm = results.pose_landmarks.landmark  # list of 33 landmarks

                # Default text values
                elbow_text = "Left Elbow: Not detected"
                knee_text = "Left Knee: Not detected"

                # Compute elbow angle if visible
                if is_visible(lm[LEFT_SHOULDER]) and is_visible(lm[LEFT_ELBOW]) and is_visible(lm[LEFT_WRIST]):
                    elbow = angle_degrees(
                        landmark_to_point2d(lm[LEFT_SHOULDER]),
                        landmark_to_point2d(lm[LEFT_ELBOW]),
                        landmark_to_point2d(lm[LEFT_WRIST]),
                    )
                    elbow_text = f"Left Elbow: {elbow:.1f} deg"

                # Compute knee angle if visible
                knee = None
                if is_visible(lm[LEFT_HIP]) and is_visible(lm[LEFT_KNEE]) and is_visible(lm[LEFT_ANKLE]):
                    knee = angle_degrees(
                        landmark_to_point2d(lm[LEFT_HIP]),
                        landmark_to_point2d(lm[LEFT_KNEE]),
                        landmark_to_point2d(lm[LEFT_ANKLE]),
                    )
                    knee_text = f"Left Knee: {knee:.1f} deg"

                # Compute squat feedback ONLY if knee exists
                feedback = None
                if knee is not None:
                    feedback = evaluate_squat(knee)

                # --- HUD draw (only ONCE) ---
                put_text_with_outline(
                    annotated,
                    elbow_text,
                    (panel_x + 15, panel_y + 80),
                    0.80,
                    (180, 180, 180) if "Not detected" in elbow_text else (255, 255, 255),
                )

                put_text_with_outline(
                    annotated,
                    knee_text,
                    (panel_x + 15, panel_y + 120),
                    0.80,
                    (180, 180, 180) if "Not detected" in knee_text else (255, 255, 255),
                )

                if feedback is None:
                    put_text_with_outline(
                        annotated,
                        "Squat: Not detected",
                        (panel_x + 15, panel_y + 160),
                        0.80,
                        (180, 180, 180),
                    )
                else:
                    c = feedback_color(feedback.label)
                    put_text_with_outline(
                        annotated,
                        f"Squat: {feedback.label}",
                        (panel_x + 15, panel_y + 160),
                        0.80,
                        c,
                    )
                    put_text_with_outline(
                        annotated,
                        feedback.detail,
                        (panel_x + 250, panel_y + 160),
                        0.70,
                        (255, 255, 255),
                    )

            else:
                # No pose detected at all (HUD only once)
                put_text_with_outline(
                    annotated,
                    "No pose detected",
                    (panel_x + 15, panel_y + 80),
                    0.80,
                    (180, 180, 180),
                )
                put_text_with_outline(
                    annotated,
                    "Left Elbow: Not detected",
                    (panel_x + 15, panel_y + 120),
                    0.80,
                    (180, 180, 180),
                )
                put_text_with_outline(
                    annotated,
                    "Left Knee: Not detected",
                    (panel_x + 15, panel_y + 160),
                    0.80,
                    (180, 180, 180),
                )  

            cv2.imshow("VisionFit - Angle Demo (press 'q')", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Closed webcam and window.")


if __name__ == "__main__":
    main()