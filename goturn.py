import cv2, sys, os
from random import randint
if not (os.path.isfile('goturn.caffemodel') and os.path.isfile('goturn.prototxt')):
    print("error loading network files")
    sys.exit()

tracker = cv2.TrackerGOTURN_create()

video = cv2.VideoCapture('videos/race.mp4')
if not video.isOpened():
    print("Error while loading the video")

ok, frame = video.read()
if not ok:
    print("error while loading the frame")
    sys.exit()

bbox = cv2.selectROI(frame) #region of interest, select the bounding region of the object
print(bbox)

ok = tracker.init(frame, bbox)
print(ok)

colors = (randint(0,255),randint(0,255),randint(0,255))
print(colors)

while True:
    ok, frame = video.read()
    if not ok:
        break
    ok, bbox = tracker.update(frame)
    #print(ok, bbox)
    if ok == True:
        (x,y,w,h) = [int(v) for v in bbox]
        #print(x,y,w,h)
        cv2.rectangle(frame, (x,y), (x+w,y+h), colors, 2, 1)
    else:
        cv2.putText(frame, "Tracking failure", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    #cv2.putText(frame, tracker, (100,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow('tracking', frame)
    if cv2.waitKey(1) & 0XFF == 27:
        break