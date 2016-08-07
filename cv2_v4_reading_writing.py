import cv2
import numpy as np
import os
import math
#from matplotlib import pyplot as plt
#%matplotlib inline
import cv2
print cv2.__version__

webcam = cv2.VideoCapture(0)
#ret, frame = webcam.read()
#print ret
#webcam.release()

# Open a new thread to manage the external cv2 interaction
cv2.startWindowThread()

# Create a window holder to show you image in
cv2.namedWindow("PyData Tutorial", cv2.WINDOW_NORMAL)


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('./video_rod.avi',fourcc, 20.0, (640,480))


while webcam.isOpened():
    ret, frame = webcam.read()
    video.write(frame)
    # write/append to video object
    cv2.imshow('PyData Tutorial',frame)
    if cv2.waitKey(40) & 0xFF == 27:
        break
# release both video objects created
webcam.release()
video.release()
cv2.destroyAllWindows()

#open in VLC
os.system("vlc .video_rod.avi")
