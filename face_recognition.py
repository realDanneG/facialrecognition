#Imports
import numpy as np
import cv2 as cv

#Cascade
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

#Init camera and recognizer
camera = cv.VideoCapture(0,cv.CAP_DSHOW)
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read("trainingdata.yml")

#Set id to 0
id=0

#Set font parameters
font=cv.FONT_HERSHEY_SIMPLEX
fontscale=1
fontcolor=(255,255,255)

#Programloop
while 1:
    #Get image from camera
    ret, img = camera.read()
    #Make grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #Find faces
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    #For faces
    for (x,y,w,h) in faces:
        #Put rectangle around face
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #Make recognizer predict who it is
        id,conf=recognizer.predict(gray[y:y+h,x:x+w])
        #Check who it is
        if(id==1):
            id="You"
        #TODO: Add more id:s for each person you want to recognize
        else:
            id="Unknown"
        #Add text under face
        cv.putText(img,str(id),(x,y+h),font,fontscale,fontcolor)
    #Show the result
    cv.imshow('img',img)
    #Press esc to break
    key=cv.waitKey(30)&0xff
    if key == 27:
        break
#Release camera and break windows
camera.release()
cv.destroyAllWindows()