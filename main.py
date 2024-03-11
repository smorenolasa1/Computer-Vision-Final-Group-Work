# our Libraries
import cv2
import mediapipe as mp
import pyautogui

pyautogui.FAILSAFE = False

# capture the video from the webcam
cap = cv2.VideoCapture(0)

# object for the hand detection
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size() # screen size
index_y = 0

# for the video capture
while True:
    
    # to get the frame
    _, frame = cap.read()
    frame = cv2.flip(frame, 1) # flip the frame, on y axis
    frame_height, frame_width, _ = frame.shape
    
    # convert the frame to RGB, better for the hand detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks 
    if hands: # if we detected the hand
        for hand in hands: # loop to draw all marks
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark # get the landmarks
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x*frame_width)
                y = int(landmark.y*frame_height)
                if id == 8: # the index fingre
                    cv2.circle(img=frame, center=(x,y), radius=10, color=(0, 255, 255))
                    index_x = screen_width/frame_width*x # based on our screen size
                    index_y = screen_height/frame_height*y

                if id == 4:# Thumb
                    cv2.circle(img=frame, center=(x,y), radius=10, color=(0, 255, 255))
                    thumb_x = screen_width/frame_width*x
                    thumb_y = screen_height/frame_height*y
                    print('outside', abs(index_y - thumb_y))
                    # if thunb and index xlose means click
                    if abs(index_y - thumb_y) < 20:
                        pyautogui.click() #
                        pyautogui.sleep(1)
                    elif abs(index_y - thumb_y) < 100:
                        pyautogui.moveTo(index_x, index_y)
    # display the frame
    cv2.imshow('Map Navigator', frame)
    cv2.waitKey(1)