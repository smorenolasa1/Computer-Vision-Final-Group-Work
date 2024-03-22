import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Function to calculate distance between thumb tip and index tip
def pinch_distance(hand_landmarks, image_width, image_height):
    thumb_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width,
                          hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height])
    index_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width,
                          hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height])
    return np.linalg.norm(thumb_tip - index_tip), thumb_tip, index_tip

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

baseline_distance = None  # Initial distance between thumb and index finger
zoom_sensitivity = 0.1  # Adjust for more/less sensitivity

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the image and process it with MediaPipe
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            image_height, image_width, _ = image.shape
            distance, _, _ = pinch_distance(hand_landmarks, image_width, image_height)
            
            if baseline_distance is None:
                baseline_distance = distance  # Set the baseline distance
            
            distance_change = distance - baseline_distance
            zoom_factor = distance_change * zoom_sensitivity  # Calculate the zoom factor based on distance change
            pyautogui.scroll(int(zoom_factor))  # Apply the zoom factor

    else:
        baseline_distance = None  # Reset baseline distance if no hand is detected

    # Show the image
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC key pressed
        break

cap.release()
cv2.destroyAllWindows()
