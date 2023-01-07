import cv2
import mediapipe as mp
import numpy as np

mPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose = mPose.Pose()

# Open the video file
cap = cv2.VideoCapture('18.mp4')

drawspec1 = mpDraw.DrawingSpec(thickness=8,circle_radius=8,color=(0,0,255))
drawspec2 = mpDraw.DrawingSpec(thickness=8,circle_radius=8,color=(0,255,0))
# Set the width and height of the output video
output_width = 800
output_height = 700

# Loop through the frames of the input video
while True:
    # Read a frame from the input video
    success, img = cap.read()

    h, w, c = img.shape
    imgBlank = np.zeros([h, w, c])
    imgBlank.fill(255)

    results = pose.process(img)
    mpDraw.draw_landmarks(img, results.pose_landmarks,mPose.POSE_CONNECTIONS,drawspec1,drawspec2)
    mpDraw.draw_landmarks(imgBlank, results.pose_landmarks, mPose.POSE_CONNECTIONS, drawspec1, drawspec2)

    # If the frame was not read successfully, break the loop
    if not success:
        break

    # Resize the frame
    img = cv2.resize(img, (output_width, output_height))
    imgBlank = cv2.resize(imgBlank, (output_width, output_height))

    # Show the frame
    cv2.imshow('poseDetection', img)

    cv2.imshow('ExtractedPose',imgBlank)

    # Wait for 1 millisecond
    cv2.waitKey(1)

# Release the VideoCapture object
cap.release()
