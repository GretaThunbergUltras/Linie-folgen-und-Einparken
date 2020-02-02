import cv2
import numpy as np
import time  #
import imutils
import brickpi3
from datetime import datetime


class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)  # umfang/perimeter
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)  # approximated shape
        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)  # calculate asprect ratio of rectangle

            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
        elif len(approx) == 5:
            shape = "pentagon"
        else:
            shape = "circle"

        return shape


cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    # isolate red
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([40, 100, 100])
    upper_red = np.array([255, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    mask = cv2.blur(mask, (10,10))
    # resize and blur, threshold image
    resized = imutils.resize(mask, width=300)
    ratio = mask.shape[0] / float(resized.shape[0])
    blur = cv2.GaussianBlur(resized, (5, 5), 0)
    thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)[1]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized2 = imutils.resize(gray, width=300)
    ratio2 = gray.shape[0] / float(resized2.shape[0])
    blur2 = cv2.GaussianBlur(resized2, (5, 5), 0)
    thresh2 = cv2.threshold(blur2, 60, 255, cv2.THRESH_BINARY)[1]

    # find contours, init shape detect
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    sd = ShapeDetector()

    cnts2 = cv2.findContours(thresh2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = imutils.grab_contours(cnts2)
    sd2 = ShapeDetector()

    # loop over contours
    for c in cnts:
        # get center of contour, detect shapename
        M = cv2.moments(c)
        cx = int((M["m10"] / (M["m00"] + 1e-7)) * ratio)
        cy = int((M["m01"] / (M["m00"] + 1e-7)) * ratio)
        shape = sd.detect(c)

        # multiply contour (x,y) by resize ratio, draw contour and name on img
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")
        cv2.drawContours(mask, [c], -1, (0, 0, 0), 2)
        cv2.putText(mask, shape, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    for c in cnts2:
        # get center of contour, detect shapename
        M = cv2.moments(c)
        cx = int((M["m10"] / (M["m00"] + 1e-7)) * ratio)
        cy = int((M["m01"] / (M["m00"] + 1e-7)) * ratio)
        shape = sd2.detect(c)

        # multiply contour (x,y) by resize ratio, draw contour and name on img
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")
        cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
        cv2.putText(frame, shape, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()