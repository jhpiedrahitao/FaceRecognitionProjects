import cv2
import mediapipe as mp 
import time

path = "videos/"
cap = cv2.VideoCapture(path+"3.mp4")
pTime=0

mpFaceDetection=mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection= mpFaceDetection.FaceDetection(0.7)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        for id,detection in enumerate(results.detections):
            #mpDraw.draw_detection(img,detection)
            #print(id,detection)
            #print(detection.score)
            #print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw , c = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih),int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bbox[:2], (bbox[2]+bbox[0],bbox[3]+bbox[1]), (255, 0, 255), 2)
            cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20),
                        cv2.FONT_HERSHEY_PLAIN, 2, (145, 25, 200), 2)
    cTime = time.time()
    fps=int(1/(cTime-pTime))
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (145, 255, 34), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
