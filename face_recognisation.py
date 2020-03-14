import cv2
import os
import numpy as np 
import dlib
from math import hypot
import time

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
def midpoint(p1,p2):
    return int((p1.x + p2.x)/2),int((p1.y + p2.y)/2)

font = cv2.FONT_HERSHEY_SIMPLEX

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    hor_line = cv2.line(img, left_point, right_point,(0,255,0), 1)

    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    ver_line = cv2.line(img, center_top, center_bottom,(0,255,0), 1)

    #length of the line
    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    ratio = hor_line_length/ ver_line_length, ver_line_length
    return ratio

blink = 1
TOTAL = 0
thres = 5.1

recogniser = cv2.face.LBPHFaceRecognizer_create()
recogniser.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX
eye = cv2.CascadeClassifier('haarcascade_eye.xml')
id = 0

names = ['None','Nilavya','NILAVYA','Nirmal']

cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)

minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
print("[INFO] : press q for close the camera")
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyee = detector(gray)
    for face in eyee:
        
        landmarks = predictor(gray, face)
        left_eye_ratio,_ = get_blinking_ratio([36,37,38,39,40,41], landmarks)
        right_eye_ratio, myVerti = get_blinking_ratio([42,43,44,45,46,47], landmarks)
        blinking_ratio = (left_eye_ratio+right_eye_ratio)/2
        personal_threshold = 0.67 * myVerti 
        
        if (left_eye_ratio>personal_threshold or right_eye_ratio>personal_threshold) and blink == 1:
            TOTAL += 1
            time.sleep(0.3)
        if (left_eye_ratio>personal_threshold or right_eye_ratio>personal_threshold):
            blink = 0
        else:
            blink = 1

        cv2.putText(img, "Blinks: {}".format(TOTAL), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0, 0), 2)

    eyes = eye.detectMultiScale(gray, scaleFactor=1.2,minNeighbors=5,minSize=(25,25))
    for(ex,ey,ew,eh) in eyes :
        cv2.rectangle(img, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
        cv2.putText(img,"eyes",(ex+10,ey-10),font,1,(255,0,0),2)
    
    faces = faceCascade.detectMultiScale( gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)))
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recogniser.predict(gray[y:y+h,x:x+w])

        if(confidence<100):
            id = names[id]
            confidence = " {0}%".format(round(100-confidence))
        else:
            id = "unknown"
            confidence = " {0}%".format(round(100-confidence))
        cv2.putText(img,str(id),(x-5,y+5),font,1,(255,255,255),2)
        cv2.putText(img,str(confidence),(y+5,x-5),font,1,(255,255,0),1)
    cv2.imshow('camera',img)
    k = cv2.waitKey(10) &0xff
    if k == ord('q'):
        break
print("\n [INFO] Exiting Program and cleanup stuff ")
cam.release()
cv2.destroyAllWindows()
