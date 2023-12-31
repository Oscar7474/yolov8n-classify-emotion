import cv2
from ultralytics import YOLO
import time
import matplotlib.pyplot as plt
cap = cv2.VideoCapture(0)
model = YOLO(r'C:\\Users\\Kuo\\Downloads\\CLS\\CLS\\last.pt')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
plt.ion()
while(True):
    ret, frame = cap.read()
    toshow = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.03, 3)
    color = ['r','c','g','y','k','b','m']
    label = ['angry','disgust','fear','happy','neutral', 'sad', 'surprise']
   # plt.axes().set_xlim(0, 1)
    for (x,y,w,h) in faces:
            toshow = cv2.rectangle(toshow,(x,y),(x+w,y+h),(255,0,0),2)
    if len(faces) == 0:
        cv2.putText(toshow, 'no face detected', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    elif len(faces) > 1:
        cv2.putText(toshow, 'multiple face detected', (10, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2, cv2.LINE_AA)
    elif len(faces) == 1:
        for (x,y,w,h) in faces:
            cropped = frame[y:y+h, x:x+w]
        results = model(cropped)
        names = results[0].names
        top1 = results[0].probs.top1
        top1conf = results[0].probs.top1conf
        x = [1,2,3,4,5,6,7]
        h = [0,0,0,0,0,0,0]
        for i in range(5):
            h[results[0].probs.top5[i]] = float(results[0].probs.top5conf[i].item())
        plt.clf()
        plt.barh(x,h,color=color,tick_label=label,height=0.5)
        plt.draw()
        plt.pause(0.01)
        cv2.putText(toshow, f'{names[top1]} {top1conf:.2f}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('toshow', toshow)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
