import cv2
import sys
from random import randint
tracker_types = ["BOOSTING","MIL","KCF","TLD","MEDIANFLOW","MOOSE","CSRT"]
print(tracker_types)
tracker_type = int(input("enter the option :"))
def create_tracker_by_name(tracker_type):
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
    return tracker

video = cv2.VideoCapture('videos/race.mp4')
if not video.isOpened():
    print("Error while loading the video")
ok, frame = video.read()

bboxes = []
colors = []

while True:
    bbox = cv2.selectROI('Multitracker', frame)
    bboxes.append(bbox)
    colors.append((randint(0,255),randint(0,255),randint(0,255)))
    print("press Q to quit and start tracking")
    print("press any other key to select object")
    k = cv2.waitKey(0) & 0XFF
    if k == 113:    #Q to quit
        break

print(bboxes)
print(colors)

multi_tracker = cv2.legacy.MultiTracker_create()
for bbox in bboxes:
    multi_tracker.add(create_tracker_by_name(tracker_type), frame, bbox)

while video.isOpened():
    ok, frame = video.read()
    if not ok:
        break

    ok, boxes = multi_tracker.update(frame)

    for i, new_box in enumerate(boxes):
        (x,y,w,h) = [int(v) for v in new_box]
        cv2.rectangle(frame, (x,y), (x+w, y+h), colors[i], 2)
    
    cv2.imshow("Multitracker", frame)
    if cv2.waitKey(1) & 0XFF == 27: #esc
        break
