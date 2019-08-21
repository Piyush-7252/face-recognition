import cv2

import os
import numpy as np
import faceRecognition as fr

test_img=cv2.imread('D:\pycharm project\TestImages\piyush8.jpg')
faces_detected,gray_img=fr.faceDetection(test_img)
print("faces_detected:",faces_detected)

'''for (x,y,w,h) in faces_detected:
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,255,0),5)
resized_img=cv2.resize(test_img,(1000,700))
cv2.imshow("face.detection.tutorial",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

#faces,faceID=fr.labels_for_training_data('trainingImages')
#face_recognizer=fr.train_classifier(faces,faceID)
#face_recognizer.write('trainingData.yml')
face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.yml')#use this to load training data for subsequent runs



name={0:"Piyush",1:"Priyanka"}

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+w]
    label,confidence=face_recognizer.predict(roi_gray)
    print("confidence:",confidence)
    print("label:",label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    fr.put_text(test_img,predicted_name,x,y)

for (x,y,w,h) in faces_detected:
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,255,0),5)
resized_img=cv2.resize(test_img,(1000,700))
cv2.imshow("face.detection.tutorial",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()






