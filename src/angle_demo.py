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

def knee_color(knee_angle:float):
    #This returns BGR colour for Open CV
    if knee_angle is None or knee_angle != knee_angle: #NaN
        return (0,0,255) #colour red
    if 110<=knee_angle <=160:
        return(0,255,0) #Colour: Green
    return(0,165,255) #Colour amber


def put_text_with_outline(img, text, org, font_scale=0.8, color=(255, 255, 255), thickness=2):
    """
    White text with a dark outline  this will help in readability.
    """
    x, y = org
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def feedback_color(label: str):
    """
    color coded squat rules
    """
    l = label.lower()
    if "good" in l:
        return (0, 200, 0)      # Green if the posture is good
    if "deep" in l:
        return (0, 165, 255)    # Amber/ orange if you are doing it too deep
    if "too" in l:
        return (0, 0, 255)      # If very high it turns into red
                                #Grey if nothing is detected
    return (200, 200, 200)

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

            # This is the HUD layout
            panel_x, panel_y = 20, 20
            panel_w, panel_h = 620, 190
            draw_panel(annotated, panel_x, panel_y, panel_w, panel_h, alpha=0.50)

            # The Title
            put_text_with_outline(annotated, "VISIONFIT  •  Live Form Check", (panel_x + 15, panel_y + 30), 0.85)


            if results.pose_landmarks:
                # Drawing pose landmarks / skeleton
                mp_drawing.draw_landmarks(
                    annotated,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                )

                lm = results.pose_landmarks.landmark  # list of 33 landmarks

                # These are the default text values
                elbow_text = "Left Elbow: Not detected"
                knee_text = "Left Knee: Not detected"

                # Computing the elbow angle if they are visible
                if is_visible(lm[LEFT_SHOULDER]) and is_visible(lm[LEFT_ELBOW]) and is_visible(lm[LEFT_WRIST]):
                    elbow = angle_degrees(
                        landmark_to_point2d(lm[LEFT_SHOULDER]),
                        landmark_to_point2d(lm[LEFT_ELBOW]),
                        landmark_to_point2d(lm[LEFT_WRIST]),
                    )
                    elbow_text = f"Left Elbow: {elbow:.1f} deg"

                # Computing the knee angle if it is visible
                knee = None
                if is_visible(lm[LEFT_HIP]) and is_visible(lm[LEFT_KNEE]) and is_visible(lm[LEFT_ANKLE]):
                    knee = angle_degrees(
                        landmark_to_point2d(lm[LEFT_HIP]),
                        landmark_to_point2d(lm[LEFT_KNEE]),
                        landmark_to_point2d(lm[LEFT_ANKLE]),
                    )
                    knee_text = f"Left Knee: {knee:.1f} deg"
                
                h,w, _=annotated.shape
                knee_lm=lm[LEFT_KNEE]
                knee_px=(int(knee_lm.x * w), int(knee_lm.y *h))

                cv2.circle(annotated, knee_px, 10, knee_color(knee), -1)

                # Computing the squat feedback only if it exists
                feedback = None
                if knee is not None:
                    feedback = evaluate_squat(knee)

                # Drawing informational UI elements like text, panels, indicators on top of the live video feed
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
                # This code is for when no pose is detected
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