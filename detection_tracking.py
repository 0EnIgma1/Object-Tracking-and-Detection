import cv2
from random import randint

tracker = cv2.legacy.TrackerCSRT_create()

video = cv2.VideoCapture('videos/baller.mp4')
if not video.isOpened():
    print("error loading video")

ok, frame = video.read()
if not ok :
    print("error loading the frame")

cascade = cv2.CascadeClassifier('cascade/fullbody.xml')
def detect():
    while True:
        ok, frame = video.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = cascade.detectMultiScale(frame_gray, minSize = (40,40))
        for (x,y,w,h) in detections:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.imshow('detection',frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            if x>0:
                print("haarcascade detection")
                return x,y,w,h
bbox = detect()

ok = tracker.init(frame, bbox)
colors = (randint(0,255), randint(0,255), randint(0,255))
while True:
    ok, frame = video.read()
    if not ok:
        break
    ok, bbox = tracker.update(frame)
    if ok:
        (x,y,w,h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x,y), (x+w,y+h), colors, 2)
    else:
        print("tracking failure")
        bbox = detect()
        tracker = cv2.legacy.TrackerCSRT_create()
        tracker.init(frame, bbox)
    cv2.imshow("tracking",frame)

    k = cv2.waitKey(1) & 0XFF
    if k == 27:
        break