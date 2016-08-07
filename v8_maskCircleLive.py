#mask to uncover image with circle

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
        #circle is white
        cv2.circle(mask, (int((x + x_in)) / 2, int((y + y_in)/2)), 
                   int(math.sqrt((y - y_in) ** 2 + (x - x_in) ** 2) / 2), 
                   (255, 255, 255), -1)
        
cv2.namedWindow('PyData Tutorial')
cv2.setMouseCallback('PyData Tutorial', draw_circle)

webcam = cv2.VideoCapture(0)
_, frame = webcam.read()

#mask is black
mask = np.zeros_like(frame)

while True:
    _, frame = webcam.read()

    # black mask && white circles
    frame = np.bitwise_and(frame, mask)
    cv2.imshow('PyData Tutorial', frame)
    if cv2.waitKey(40) & 0xFF == 27:
        break
webcam.release()
cv2.destroyAllWindows()
