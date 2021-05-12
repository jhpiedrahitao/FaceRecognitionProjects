#Face detection
import cv2
import numpy as np

faceCascade=cv2.CascadeClassifier("resources/haarcascades/haarcascade_frontalface_default.xml")
catFaceCascade=cv2.CascadeClassifier("resources/haarcascades/haarcascade_frontalcatface.xml")

'''
path="resources/lena.png"
img=cv2.imread(path)
imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces=faceCascade.detectMultiScale(imgGray,1.1,4)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+w),(255,0,0),2)

cv2.imshow("Result", img)
cv2.waitKey(0)
'''
cap=cv2.VideoCapture(1)
cap.set(3,640) #size
cap.set(4,480) #size
cap.set(10,100) #brighthness
print(cap.isOpened())
while(True):
    success, img = cap.read() 
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(imgGray,1.1,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+w),(255,0,0),2)
        cv2.putText(img,"Human",(x,y+25),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)
    catfaces=catFaceCascade.detectMultiScale(imgGray,1.1,4)
    for (x,y,w,h) in catfaces:
        cv2.rectangle(img,(x,y),(x+w,y+w),(0,255,0),2)
        cv2.putText(img,"Cat",(x,y+25),cv2.FONT_HERSHEY_PLAIN,2,(255,255,0),2)

    cv2.imshow("Result", img)
    if (cv2.waitKey(1)  & 0xFF == ord('q')):
        break