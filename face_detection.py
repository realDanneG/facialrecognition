#Imports
import numpy as np
import cv2 as cv

#Read in the cascades
face_cascade=cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade=cv.CascadeClassifier('haarcascade_eye.xml')

#Camera
camera=cv.VideoCapture(0)

#Show image while True
while True:
    ret,img = camera.read()
    #Make grayscale
    gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #Detect faces
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        #Draw rectangle around face
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        #Used for detecting eyes
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=img[y:y+h,x:x+w]
        #Detect eyes
        eyes=eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            #draw rectangle around eyes
            cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    #Show the image
    cv.imshow('Face',img)
    #Break if esc is pressed
    key=cv.waitKey(30)&0xff
    if key == 27:
        break
#Stop using camera
camera.release()
#Break all windows
cv.destroyAllWindows()