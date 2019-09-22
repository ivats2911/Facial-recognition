import cv2

detector = cv2.CascadeClassifier("lbpcascade_frontalface_improved.xml");
cam = cv2.VideoCapture(0)
recognizer = cv2.createLBPHFaceRecognizer()
recognizer.load("Trainner/Trainner.yml")

ID=0
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 2, 1)

while (True):
    ret, reco = cam.read()
    gray = cv2.cvtColor(reco,cv2.COLOR_BGR2GRAY)
    Face = detector.detectMultiScale(gray, 1.3, 7)
    for(x,y,w,h) in Face:
        cv2.rectangle(reco,(x,y),(x+w,y+h),(0,255,0),2)
        ID, conf = recognizer.predict(gray[y:y+h,x:x+w])
        if(ID==1):
            ID="MOHIT"
        elif(ID==2):
            ID="SAHIL"
        elif(ID==3):
            ID="PROJECT"
        cv2.cv.PutText(cv2.cv.fromarray(reco),str(ID), (x,y+h),font,255)
    cv2.imshow("RECOGNIZER",reco) 
    if cv2.waitKey(10) ==ord('q'):
        break
    
cam.release()
cv2.destroyAllWindows()
