#Imports
import os
import numpy as np
import cv2 as cv
from PIL import Image # For face recognition we will the the LBPH Face Recognizer 

#Create new recognizer
recognizer = cv.face.LBPHFaceRecognizer_create()

#Path to dataset
# TODO: replace with your path
path="\\dataset"

def getImagesWithID(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]   

    #Init empty arrays for faces and IDS
    faces = []
    IDs = []

    for imagePath in imagePaths:      
        # Read the image and convert to grayscale
        facesImg = Image.open(imagePath).convert('L')
        faceNP = np.array(facesImg, 'uint8')

        # Get the label of the image
        ID= int(os.path.split(imagePath)[-1].split(".")[1])

        #Detect the face in the image
        faces.append(faceNP)

        IDs.append(ID)
        #Show image used
        cv.imshow("Adding faces for traning",faceNP)

        cv.waitKey(10)

    return np.array(IDs), faces
#Get id and face from the function we created
Ids,faces = getImagesWithID(path)

#Train the reconizer
recognizer.train(faces,Ids)

#Save into yml file
recognizer.save("trainingdata.yml")

#Close down all windows
cv.destroyAllWindows()