#draw a rectangle around detected face

import cv2
import numpy as np
import os
import math

'''
#take a picture
webcam = cv2.VideoCapture(0)
_, frame = webcam.read()
cv2.waitKey(1000)
webcam.release()
'''


###### show window ################
# Open a new thread to manage the external cv2 interaction
cv2.startWindowThread()

# Create a window holder to show you image in
cv2.namedWindow("PyData Tutorial", cv2.WINDOW_NORMAL)



class FaceDetector(object):
    def __init__(self, xml_path):
        self.classifier = cv2.CascadeClassifier(xml_path)

    def detect(self, image, biggest_only=True):
        scale_factor = 1.2
        min_neighbors = 5
        min_size = (30, 30)
        biggest_only = True
        flags = cv2.CASCADE_FIND_BIGGEST_OBJECT | \
                    cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else \
                    cv2.CASCADE_SCALE_IMAGE
        faces_coord = self.classifier.detectMultiScale(image,
                                                       scaleFactor=scale_factor,
                                                       minNeighbors=min_neighbors,
                                                       minSize=min_size,
                                                       flags=flags)

        return faces_coord



class VideoCamera(object):
    def __init__(self, index=0):
        self.video = cv2.VideoCapture(index)
        self.index = index
        print self.video.isOpened()

    def __del__(self):
        self.video.release()
    
    def get_frame(self, in_grayscale=False):
        _, frame = self.video.read()
        if in_grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame



################ main ###############

webcam = VideoCamera()
detector = FaceDetector("xml/frontal_face.xml")


try:
    while True:
        frame = webcam.get_frame()
        faces_coord = detector.detect(frame)
        for (x, y, w, h) in faces_coord:
            cv2.rectangle(frame, (x, y), (x + w, y + h), 
                          (150, 150, 0), 8)
        #plt_show(frame)
        cv2.imshow("PyData Tutorial", frame)

        print "Type: " + str(type(faces_coord))
        print faces_coord
        print "Number of faces detected: " + str(len(faces_coord))

        #clear_output(wait = True)

        #PRESS ESC repeatedly to end
        if cv2.waitKey(40) & 0xFF == 27:
            break
except KeyboardInterrupt:
     print "Live Video Interrupted"


# release both video objects created
webcam.release()
# Press any key to close external window
cv2.waitKey()
cv2.destroyAllWindows()
######################################

