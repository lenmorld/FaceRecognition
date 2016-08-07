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
message = ""

while webcam.isOpened():
    
    _, frame = webcam.read()

    #draw a rectangle in the middle of window
    cv2.rectangle(frame, (100, 100), (530, 400), (150, 150, 0), 3)

    #put typed text in top of rectangle
    cv2.putText(frame, message, (95, 95), cv2.FONT_HERSHEY_SIMPLEX, .7, 
                (150, 150, 0), 2)
    
    cv2.imshow('PyData Tutorial',frame)
    key = cv2.waitKey(100) & 0xFF
    if key not in [255, 27]:
        message += chr(key)
    elif key == 27:
        break
        
# release both video objects created
webcam.release()
cv2.destroyAllWindows()
