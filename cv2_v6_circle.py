import cv2
import numpy as np
import os
import math
#from matplotlib import pyplot as plt
#%matplotlib inline
import cv2
print cv2.__version__


webcam = cv2.VideoCapture(0)
cv2.namedWindow("PyData Tutorial", cv2.WINDOW_AUTOSIZE)

while webcam.isOpened():
    
    _, frame = webcam.read()
    mask = np.zeros_like(frame)
    height, width, _ = frame.shape
    
    cv2.circle(mask, (width / 2, height / 2), 200, (255, 255, 255), -1)
    frame = np.bitwise_and(frame, mask)
    
    cv2.imshow('PyData Tutorial', frame)
    if cv2.waitKey(40) & 0xFF == 27:
        break
        
# release both video objects created
webcam.release()
cv2.destroyAllWindows()

