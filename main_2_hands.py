import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Variables for zoom control
prev_distance = None
zoom_speed = 2  # Adjust as necessary for smoother zooming

# Variables for panning control
is_dragging = False

# Function to check if hand is closed
def is_hand_closed(hand_landmarks):
    # Thumb tip is landmark 4, index fingertip is landmark 8
    # Compare y-coordinates to check if hand is closed (thumb is below index fingertip)
    return hand_landmarks.landmark[4].y > hand_landmarks.landmark[8].y

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    hand_centers = []

    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks) == 2:
            # Zoom gesture
            hand_centers = [np.mean([[landmark.x for landmark in hand_landmarks.landmark], 
                                     [landmark.y for landmark in hand_landmarks.landmark]], axis=1) 
                            for hand_landmarks in results.multi_hand_landmarks]
            distance = np.linalg.norm(hand_centers[0] - hand_centers[1])
            if prev_distance is not None:
                zoom_change = distance - prev_distance
                if zoom_change > 0:
                    pyautogui.scroll(zoom_speed)  # Zoom in
                elif zoom_change < 0:
                    pyautogui.scroll(-zoom_speed)  # Zoom out
            prev_distance = distance
        elif len(results.multi_hand_landmarks) == 1:
            # Panning gesture
            hand_landmarks = results.multi_hand_landmarks[0]
            hand_center = np.mean([[landmark.x for landmark in hand_landmarks.landmark], 
                                   [landmark.y for landmark in hand_landmarks.landmark]], axis=1) * [image.shape[1], image.shape[0]]
            
            if not is_hand_closed(hand_landmarks):
                if not is_dragging:
                    pyautogui.mouseDown(hand_center[0], hand_center[1])
                    is_dragging = True
                else:
                    pyautogui.moveTo(hand_center[0], hand_center[1])
            else:
                if is_dragging:
                    pyautogui.mouseUp()
                    is_dragging = False
                pyautogui.moveTo(hand_center[0], hand_center[1])
    else:
        if is_dragging:
            pyautogui.mouseUp()
            is_dragging = False
        prev_distance = None

    cv2.imshow('Virtual Map Navigation', image)
    if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()