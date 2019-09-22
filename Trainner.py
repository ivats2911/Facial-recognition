import cv2
import os
import numpy as np 
from PIL import Image

recognizer = cv2.createLBPHFaceRecognizer()
detector= cv2.CascadeClassifier("lbpcascade_frontalface_improved.xml");

def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    Faces=[]
    IDs=[]
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage,'uint8')
        ID = int(os.path.split(imagePath)[-1].split('.')[1])
        Faces.append(imageNp)
        IDs.append(ID)
        cv2.imshow("Training",imageNp)
        cv2.waitKey(50)
    return IDs,Faces

IDs,Faces = getImagesAndLabels("DataSet")
recognizer.train(Faces, np.array(IDs))
recognizer.save("Trainner/Trainner.yml")
cv2.destroyAllWindows()
