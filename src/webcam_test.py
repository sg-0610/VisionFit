# src/webcam_test.py
# The Goal Goal of the test: Check if the webcam is working before moving forward with the project
# "q" to quit

import cv2

def main():
    cap = cv2.VideoCapture(0)

    # If there are multiple cameras try 1 or 2
    if not cap.isOpened():
        raise RuntimeError(
            "Webcam could not be opened, can be due to incorrect camera index (try 0, 1, 2) or missing OS-level camera perimsion for Pyhton/VS Code"
        )

    # Requesting webcam resolutions, in some cases web cams ignore this
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Webacm is opened, press 'q' to quit")

    try:
        while True:
            ret, frame = cap.read() #Shape: (height, width, 3), #This represents a RGB image 
            if not ret:
                print("Failed to read frame from webcam.") # If fails, exit the loop
                break

            cv2.imshow("VisionFit - Webcam Test (press 'q' to quit)", frame) #For visual confirmation that the webcam is working and films are processed in real time

            # Wait 1 miliseciond to see if key is pressed ( 1ms keeps the latency low)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Cleanup: Closed webcam and window.")


if __name__ == "__main__":
    main()