import numpy as np
import cv2

def nothing(x):
	pass
cap = cv2.VideoCapture(0);

cv2.namedWindow('Tracking')

while True:
    #frame = cv2.imread('smarties.png')

    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_b = np.array([130, 60, 0])
    u_b = np.array([255,255, 255])

    mask = cv2.inRange(hsv, l_b, u_b)

    res = cv2.bitwise_and(frame, frame, mask=mask)

    peri = cv2.arcLength(frame, True)
    approx = cv2.approxPolyDP(frame, 0.04 * peri, True)
    print(approx)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
