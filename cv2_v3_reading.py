import cv2
import numpy as np
import os
import math
#from matplotlib import pyplot as plt
#%matplotlib inline
import cv2
print cv2.__version__


video = cv2.VideoCapture("./video_rod.avi")
#webcam = cv2.VideoCapture(0)

# Create a window holder to show you image in
cv2.namedWindow("PyData Tutorial", cv2.WINDOW_NORMAL)
cv2.resizeWindow("PyData Tutorial", 850, 480)





while video.isOpened():

    ret, frame = video.read()

    #close program when video is done
    if not ret:
        break

    cv2.imshow("PyData Tutorial", frame)

    if cv2.waitKey(40) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()



'''
cv2.imshow("PyData Tutorial", frame)
 
# Press any key to close external window
cv2.waitKey()   
cv2.destroyAllWindows()
'''

