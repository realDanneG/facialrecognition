#Imports
import numpy as np
import cv2 as cv

#Read face cascade
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

#Camera to take images with
camera = cv.VideoCapture(0,cv.CAP_DSHOW)

#What id will face have
id = input('enter user id')

sampleN=0
#Create dataset
while 1:
    ret, img = camera.read()
    #make grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #Find face
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        #Add one to sampleN. We only want 20 pictures each time
        sampleN=sampleN+1
        #Create path string
        picture_path="dataset\\User."+str(id)+ "." +str(sampleN)+ ".jpg"
        #Save the image to the path we made.
        cv.imwrite(picture_path, gray[y:y+h, x:x+w])
        #Draw rectangle around face
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        cv.waitKey(100)
    #Show image/camerafeed
    cv.imshow('img',img)

    cv.waitKey(1)

    if sampleN > 20:
        break

#Release camera and destroy all imagewindows
camera.release()
cv.destroyAllWindows()