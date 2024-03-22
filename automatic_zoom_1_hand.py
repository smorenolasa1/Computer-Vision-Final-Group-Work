import cv2
import mediapipe as mp
import pyautogui
import math
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math

# Initialize mediapipe hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Function to calculate distance between two points
def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

# Define all your functions here before the while loop
def is_pinching(index_tip, thumb_tip):
    pinch_threshold = 0.05  # Adjust based on your calibration needs
    distance = calculate_distance(index_tip, thumb_tip)
    return distance < pinch_threshold

def move_mouse_based_on_hand_position(index_tip_1, index_tip_2):
    screen_width, screen_height = pyautogui.size()
    x = (index_tip_1.x + index_tip_2.x) / 2 * screen_width
    y = (index_tip_1.y + index_tip_2.y) / 2 * screen_height
    pyautogui.moveTo(x, y)

def zoom_based_on_distance_change(current_distance, previous_distance):
    distance_change_threshold = 0.09  # Adjust based on sensitivity preference
    zoom_amount = 200  # Adjust based on how much you want to zoom per action
    
    if current_distance - previous_distance > distance_change_threshold:
        pyautogui.scroll(zoom_amount)  # Zoom in
    elif previous_distance - current_distance > distance_change_threshold:
        pyautogui.scroll(-zoom_amount)  # Zoom out

def reset_actions():
    global dragging, zooming, prev_distance
    if dragging:
        dragging = False
        pyautogui.mouseUp()
    zooming = False
    prev_distance = None

def handle_dragging(hand1, hand2):
    global dragging, prev_pos
    # Get index finger tip and thumb tip landmarks for both hands
    index_tip_1 = hand1.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_tip_1 = hand1.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip_2 = hand2.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_tip_2 = hand2.landmark[mp_hands.HandLandmark.THUMB_TIP]

    # Check if both hands are pinching
    if is_pinching(index_tip_1, thumb_tip_1) and is_pinching(index_tip_2, thumb_tip_2):
        if not dragging:
            dragging = True
            pyautogui.mouseDown()
        else:
            # Calculate and apply the movement
            move_mouse_based_on_hand_position(index_tip_1, index_tip_2)
    else:
        if dragging:
            dragging = False
            pyautogui.mouseUp()

def pinch_distance(hand_landmarks, image_width, image_height):
    thumb_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width,
                          hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height])
    index_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width,
                          hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height])
    return np.linalg.norm(thumb_tip - index_tip), thumb_tip, index_tip

# Modify the handle_zooming function to use pinch_distance
def handle_zooming(hand, image_width, image_height):
    global baseline_distance, zoom_sensitivity
    distance, _, _ = pinch_distance(hand, image_width, image_height)
    if baseline_distance is None:
        baseline_distance = distance  # Set the baseline distance

    distance_change = distance - baseline_distance
    zoom_factor = distance_change * zoom_sensitivity  # Calculate the zoom factor based on distance change
    pyautogui.scroll(int(zoom_factor))  # Apply the zoom factor

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize variables
baseline_distance = None
zoom_sensitivity = 0.1

dragging = False
zooming = False
prev_distance = None  # For zooming

# Main loop
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally and convert the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # Process the image and find hands
    results = hands.process(image)

    # Convert the image color back so it can be displayed
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Get the dimensions of the image for zooming calculations
    image_height, image_width, _ = image.shape

    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks) >= 2:
            # Handle dragging with two hands
            hand1 = results.multi_hand_landmarks[0]
            hand2 = results.multi_hand_landmarks[1]
            handle_dragging(hand1, hand2)
        elif len(results.multi_hand_landmarks) == 1:
            # Modified handle zooming with one hand to include image dimensions
            hand_landmarks = results.multi_hand_landmarks[0]
            handle_zooming(hand_landmarks, image_width, image_height)
    else:
        baseline_distance = None  # Reset baseline distance if no hand is detected
        reset_actions()

    # Draw hand landmarks on the image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the resulting image
    cv2.imshow('MediaPipe Hands', image)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(5) & 0xFF == 113:
        break

# Release the resources
hands.close()
cap.release()
cv2.destroyAllWindows()