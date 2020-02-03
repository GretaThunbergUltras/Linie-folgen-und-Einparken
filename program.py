import cv2
import numpy as np
import time
import imutils
import brickpi3
from datetime import datetime

speed = 20

class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)  # umfang/perimeter
        if peri > 150:  # filter out small objects
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


line_coordinates = 0, 0, 0, 0


def region_of_interest(edges2):
    height, width = edges2.shape
    # print(width)
    # print(height)
    mask2 = np.zeros_like(edges2)

    # only focus bottom half of the screen
    # and focus on central third of the screen
    polygon = np.array([[
        (width * 1 / 3, height * 2 / 3),
        (width * 2 / 3, height * 2 / 3),
        (width * 2 / 3, height),
        (width * 1 / 3, height),
    ]], np.int32)

    cv2.fillPoly(mask2, polygon, 255)
    cropped_edges2 = cv2.bitwise_and(edges2, mask2)
    return cropped_edges2


def detect_line_segments(cropped_edges2):
    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # distance precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
    min_threshold = 30  # minimal of votes
    line_segments2 = cv2.HoughLinesP(cropped_edges2, rho, angle, min_threshold,
                                     np.array([]), minLineLength=8, maxLineGap=1)
    return line_segments2


def average_slope_intercept(frame2, line_segments2):
    lane_lines = []
    if line_segments2 is None:
        return lane_lines

    height, width, _ = frame2.shape
    left_fit = []

    for line_segment in line_segments2:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            left_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame2, left_fit_average))

    return lane_lines


def make_points(frame3, line):
    height, width, _ = frame3.shape
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(y1 * 2 / 3)  # make points from middle of the frame down

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    global line_coordinates
    line_coordinates = [x1, y1, x2, y2]
    return [[x1, y1, x2, y2]]


def display_lines(frame, lines, line_color=(0, 255, 0), line_width=4):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image


def line_tracking(line_coordinates, edges):
    x1, y1, x2, y2 = line_coordinates
    if (line_coordinates != [0,0,0,0]):
        height, width = edges.shape
        print("test")
        bottom_center = width * 1 / 2
        delta = bottom_center - x1
        print(delta)
        if delta < -100:
            delta = -100
        elif delta > 100:
            delta = 100
        if delta >= -100 and delta <= 100:
            if (delta < -10):
                # print(bottom_center - x1)
                BP.set_motor_power(BP.PORT_B, speed)
                BP.set_motor_position(BP.PORT_D, -1 * delta)

            elif delta > 10:
                # print(bottom_center - x1)
                print("test3")
                BP.set_motor_power(BP.PORT_B, speed)
                BP.set_motor_position(BP.PORT_D, -1 * delta + 50)

            else:
                # print(bottom_center - x1)
                BP.set_motor_power(BP.PORT_B, speed)
                BP.set_motor_position(BP.PORT_D, 0)
    else:
        BP.set_motor_power(BP.PORT_B, 0)
        BP.set_motor_position(BP.PORT_D, 0)


BP = brickpi3.BrickPi3()

cap = cv2.VideoCapture(0)

while (True):

    current_milli_time = int(round(time.time() * 1000))
    #print(current_milli_time)
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([255, 100, 40])
    mask = cv2.inRange(hsv, lower_black, upper_black)
    edges = cv2.Canny(mask, 100, 200)
    cropped_edges = region_of_interest(edges)
    line_segments = detect_line_segments(cropped_edges)
    lane_lines2 = average_slope_intercept(frame, line_segments)
    cv2.imshow('edges', edges)
    # cv2.imshow('cropped_edges', cropped_edges)
    lane_lines_image = display_lines(frame, lane_lines2)


    # cv2.imshow('frame', frame)
    # cv2.imshow('mask', mask)

    # parkdetection start

    fwidth = frame.shape[1]
    framewidth = int(fwidth)
    fheight = frame.shape[0]
    frameheight = int(fheight)

    # isolate lower red region
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    redmask1 = cv2.inRange(hsv, lower_red, upper_red)

    # isolate upper red region
    lower_red = np.array([150, 50, 50])
    upper_red = np.array([180, 255, 255])
    redmask2 = cv2.inRange(hsv, lower_red, upper_red)

    # join both masks
    redmask = redmask1 + redmask2

    # resize and blur, threshold image
    resized = imutils.resize(redmask, width=300)
    ratio = redmask.shape[0] / float(resized.shape[0])
    redblur = cv2.GaussianBlur(resized, (5, 5), 0)
    thresh = cv2.threshold(redblur, 60, 255, cv2.THRESH_BINARY)[1]

    # find contours, init shape detect
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    sd = ShapeDetector()

    # loop over contours
    for c in cnts:
        # get center of contour, detect shapename
        M = cv2.moments(c)
        cx = int((M["m10"] / (M["m00"] + 1e-7)) * ratio)
        cy = int((M["m01"] / (M["m00"] + 1e-7)) * ratio)
        shape = sd.detect(c)

        if shape == "square" or shape == "rectangle":
            # multiply contour (x,y) by resize ratio, draw contour and name on img
            c = c.astype("float")
            c *= ratio
            c = c.astype("int")
            cv2.drawContours(lane_lines_image, [c], -1, (0, 0, 255), 2)
            cv2.putText(lane_lines_image, shape, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.line(lane_lines_image, (cx, frameheight), (cx, cy), (0, 255, 0), 4)
            line_coordinates = [cx, frameheight, cx, cy]

    # parkdetection end
    cv2.imshow("lane lines", lane_lines_image)
    cv2.imshow("redmask", redmask)
    print(line_coordinates)

    line_tracking(line_coordinates, edges)
    global line_coordinates
    line_coordinates = 0,0,0,0
    current_milli_time = int(round(time.time() * 1000)) - current_milli_time
    #print(current_milli_time)
    time.sleep(0.1)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        BP.set_motor_power(BP.PORT_B, 0)
        BP.set_motor_position(BP.PORT_D, 0)
        break

cv2.destroyAllWindows()