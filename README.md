Drone Gesture Control using MediaPipe & CNN

This project implements a real-time hand gesture recognition system for drone control using MediaPipe Vision Landmark System, OpenCV, and a Convolutional Neural Network (CNN).
The system detects hand gestures from a webcam, classifies them, and maps them to drone movement commands.

Features:

1.Real-time hand detection using MediaPipe Hands
2.Gesture classification using a CNN model
3.Train–test data split with evaluation on test data
4.Live webcam prediction on real-world gestures
5.Robust to small hand movements
6.Easily extendable to real drone or simulator control

Technologies Used

1.Python 3.10.9
2.MediaPipe (Hand landmark detection)
3.OpenCV (Webcam & image processing)
4.TensorFlow Keras (CNN model)
5.NumPy
6.Scikit-learn (evaluation metrics: accuracy)

Supported Gestures:
Gesture	Drone Command
Fist	                STOP
Palm (5 fingers)	    START
One Finger	          MOVE FORWARD
Two Fingers	          MOVE BACKWARD
Thumbs Up	            MOVE UP
Three Fingers	        MOVE DOWN
Four Fingers	        HOVER
Thumb+Index(pointer)  FLIP
