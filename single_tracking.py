import cv2
import sys
from random import randint
tracker_types = ["BOOSTING","MIL","KCF","TLD","MEDIANFLOW","MOOSE","CSRT"]
print(tracker_types)
tracker_type = int(input("enter the option :"))
if tracker_type == 0:
    tracker = cv2.legacy.TrackerBoosting_create()
    print(tracker)

elif tracker_type == 1:
    tracker = cv2.legacy.TrackerMIL_create()
    print(tracker)

elif tracker_type == 2:
    tracker = cv2.legacy.TrackerKCF_create()
    print(tracker)

elif tracker_type == 3:
    tracker = cv2.legacy.TrackerTLD_create()
    print(tracker)

elif tracker_type == 4:
    tracker = cv2.legacy.TrackerMedianFlow_create()
    print(tracker)
elif tracker_type == 5:
    tracker = cv2.legacy.TrackerMOSSE_create()
    print(tracker)

elif tracker_type == 6:
    tracker = cv2.legacy.TrackerCSRT_create()
    print(tracker)

video = cv2.VideoCapture('videos/cheetah.mp4')
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

#KCF good,  CSRT man damn best tracking algo