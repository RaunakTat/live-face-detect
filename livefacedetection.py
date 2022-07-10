#LIVE FACE and EYEs RECOGNITION

import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

#scaleFactor and minNeighbors are the tuning parameters
while True:
    ret,frame = cap.read()
    faces = face_cascade.detectMultiScale(frame,scaleFactor = 1.1,minNeighbors = 9)
    eyes = eye_cascade.detectMultiScale(frame,scaleFactor = 1.1,minNeighbors = 18)
    
    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),1)
        cv2.imshow('FACE RECOGNITION',frame)

    for x,y,w,h in eyes:
        img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),1)
        cv2.imshow('FACE RECOGNITION',frame)

    #press enter to stop video
    if cv2.waitKey(1) == 13:
        break

cap.release()
#It releases the port number
cv2.destroyAllWindows()
    
