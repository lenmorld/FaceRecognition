import cv2
import numpy as np
import os
import math

#take a picture
webcam = cv2.VideoCapture(0)
_, frame = webcam.read()
cv2.waitKey(1000)
webcam.release()


#run detector
detector = cv2.CascadeClassifier("xml/frontal_face.xml")

scale_factor = 1.2
min_neighbors = 5
min_size = (30, 30)
biggest_only = True
flags = cv2.CASCADE_FIND_BIGGEST_OBJECT | \
            cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else \
            cv2.CASCADE_SCALE_IMAGE

faces_coord = detector.detectMultiScale(frame,
                                        scaleFactor=scale_factor,
                                        minNeighbors=min_neighbors,
                                        minSize=min_size,
                                        flags=flags)

###### show window ################
# Open a new thread to manage the external cv2 interaction
cv2.startWindowThread()

# Create a window holder to show you image in
cv2.namedWindow("PyData Tutorial", cv2.WINDOW_NORMAL)
cv2.imshow("PyData Tutorial", frame)
 
# Press any key to close external window
cv2.waitKey()   
cv2.destroyAllWindows()
######################################



print "Type: " + str(type(faces_coord))
print faces_coord
print "Number of faces detected: " + str(len(faces_coord))



