import cv2
import numpy as np

cap = cv2.VideoCapture('videos/walking.avi')
param_sm = dict(maxCorners = 100, qualityLevel = 0.3, minDistance = 7)
param_lk = dict(winSize = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
colors = np.random.randint(0,255, (100,3))
#print(np.shape(colors))
#print(colors)
ok, frame = cap.read()
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#cv2.imshow('gray',frame_gray)

edges = cv2.goodFeaturesToTrack(frame_gray, mask = None, **param_sm)
#print(len(edges))
mask = np.zeros_like(frame)
#print(np.shape(mask))
#print(mask)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    new_edges, status, errors = cv2.calcOpticalFlowPyrLK(frame_gray, frame_gray2, edges, None, **param_lk)
    
    news = new_edges[status ==1]
    olds = edges[status == 1]
    
    for i, (new, old) in enumerate(zip(news, olds)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a),int(b)), (int(c),int(d)), colors[i].tolist(), 2)
        frame = cv2.circle(frame, (int(a),int(b)),5, colors[i].tolist(), -1)

    img = cv2.add(frame,mask)
    cv2.imshow("optical flow sparse",mask)
    if cv2.waitKey(1) == 27:
        break

    frame_gray = frame_gray2.copy()
    edges = news.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()
    