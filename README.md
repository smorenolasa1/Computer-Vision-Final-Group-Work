# Map Navigation using Virtual Gestures

## Introduction
This project enables users to navigate through digital maps (like Google Maps) using virtual hand gestures. Leveraging a webcam, the application detects specific hand movements to perform map navigation actions such as dragging and zooming in/out.

## Prerequisites
Ensure Python is installed on your system before running this script. This code has been tested with Python 3.8+.

## Installation
To run this application, you'll need to install several Python libraries including OpenCV, MediaPipe, PyAutoGUI, and NumPy. You can install these dependencies by running the following command in your terminal or command prompt:

```bash
pip3 install opencv-python
pip3 install mediapipe
pip3 install pyautogui
pip3 install numpy
```
## How It Works
The application uses the webcam to detect hand gestures for interactive map navigation:

Dragging: Pinch with both hands and move to drag the map.
Zooming: Open one hand to zoom in and close it to zoom out. The zoom sensitivity can be adjusted for a faster or slower zoom effect.

## Customization
You can adjust the following parameters in the script according to your needs:

- pinch_threshold: Sensitivity of the pinch gesture for dragging.
- zoom_sensitivity: Determines how much the zoom changes with hand movement.
  
## Troubleshooting
- If the map navigation feels too sensitive or not sensitive enough, adjust the pinch_threshold and zoom_sensitivity parameters as described above.
- Make sure your webcam is properly connected and recognized by your system.
- Run the script in an environment with good lighting to improve hand gesture detection.

## Acknowledgements
This project utilizes MediaPipe for hand tracking and PyAutoGUI for simulating mouse events based on hand gestures.
