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


while True:

    _, frame = webcam.read()
    cv2.imshow("PyData Tutorial", frame)


    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()
webcam.release()


'''
cv2.imshow("PyData Tutorial", frame)
 
# Press any key to close external window
cv2.waitKey()   
cv2.destroyAllWindows()
'''

