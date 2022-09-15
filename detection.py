import cv2
image = cv2.imread('Images/football.jpg')
detector = cv2.CascadeClassifier('cascade/fullbody.xml')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow('people',image_gray)

detections = detector.detectMultiScale(image_gray)
print(detections)
print(len(detections))

for (x,y,w,h) in detections:
    cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)

cv2.imshow('detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()