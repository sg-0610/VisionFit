# src/blazepose_live.py
# Goal: Check if webcam is working, run MediaPipe BlazePose and draw skeletal lanmarks
# Press "q" to quit.

import cv2
import mediapipe as mp


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError(
            " If the Webcam could not be opened. Try a different camera index (0/1/2) "
            "or grant OS-level camera permissions to Python or VS Code."
        )

    # Requesting a resolution some webcam's may ignore this
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mp_pose = mp.solutions.pose #contans the pose detector
    mp_drawing = mp.solutions.drawing_utils # contains helper functions to draw landmarks and skeleton lines
    mp_styles = mp.solutions.drawing_styles

    # This shows how BlazePose Works
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        print("BlazePose is running, press 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("If it fails the loop stops")
                break

            # MediaPipe expects RGB input and OpenCV reads in BGR so coversion is neccessary
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Improve performance by marking image as not writeable during inference
            frame_rgb.flags.writeable = False
            results = pose.process(frame_rgb) #This is where BlazPose actually runs and the output goes into results
            frame_rgb.flags.writeable = True

            # If model detected a person then pose landmarks would exist
            annotated = frame.copy()
            if results.pose_landmarks:
                #This draws dots for joints and lines for skeleton connections (POSE_CONNECTIONS)
                mp_drawing.draw_landmarks(
                    annotated,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(), 
                )

            cv2.imshow("VisionFit - BlazePose Live (press 'q' to quit)", annotated) #Shows webcam live stream

            if cv2.waitKey(1) & 0xFF == ord("q"): #If 'q; is presses exit loop
                break
#Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Close the webcam and window")


if __name__ == "__main__":
    main()