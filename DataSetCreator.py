import cv2

detector=cv2.CascadeClassifier("lbpcascade_frontalface_improved.xml")
cam = cv2.VideoCapture(0)

ID=raw_input('ENTER ID - ')
Num=1

while(True):
    
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Face = detector.detectMultiScale(gray, 1.3, 6)
    for (x,y,w,h) in Face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imwrite("DataSet/User."+ID+'.'+ str(Num) + ".jpg", gray[y:y+h,x:x+w])
        Num = Num+1
        cv2.waitKey(100)
    cv2.imshow("DETECTOR",img)
    cv2.waitKey(1)
    if (Num>50):
        break


cam.release()
cv2.destroyAllWindows()
