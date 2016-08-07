#draw circle with an mouse

import cv2
import numpy as np
import os
import math
#from matplotlib import pyplot as plt
#%matplotlib inline
import cv2
print cv2.__version__


# mouse callback function
def draw_circle(event,x,y,flags,param):
    global x_in, y_in
    if event == cv2.EVENT_LBUTTONDOWN:
        x_in = x 
        y_in = y
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.circle(frame, (int((x + x_in)) / 2, int((y + y_in)/2)), 
                   int(math.sqrt((y - y_in) ** 2 + (x - x_in) ** 2) / 2), (150, 150, 0), -1)
        
cv2.namedWindow('PyData Tutorial')

cv2.setMouseCallback('PyData Tutorial', draw_circle)

webcam = cv2.VideoCapture(0)
_, frame = webcam.read()
webcam.release()

while True:
    cv2.imshow('PyData Tutorial',frame)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()
