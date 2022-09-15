import cv2
import numpy as np

cap = cv2.VideoCapture("videos/cheetah.mp4")
ok, first_frame = cap.read()
frame_gray_init = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

hsv = np.zeros_like(first_frame)
#print(np.shape(first_frame))
#print(np.shape(hsv))
#print(first_frame)
#print(hsv)
hsv[...,1] = 255
#print(hsv)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # https://docs.opencv.org/3.4.15/dc/d6b/group__video__track.html#ga5d10ebbd59fe09c5f650289ec0ece5af
    # 30 -> 10 -> 3
    flow = cv2.calcOpticalFlowFarneback(frame_gray_init, frame_gray, None, 0.5, 3, 15, 3, 5, 1.1, 0)
    magnitude, angle = cv2.cartToPolar(flow[...,0], flow[...,1]) # X, Y
    hsv[...,0] = angle * (180 / (np.pi / 2))
    hsv[...,2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    frame_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow("original video",frame)
    cv2.imshow('Dense optical flow', frame_rgb)
    if cv2.waitKey(1) == 27: # enter
        break

    frame_gray_init = frame_gray

cap.release()
cv2.destroyAllWindows()



















