import cv2
import numpy as np

cap = cv2.VideoCapture(0)

def findOutline(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = -1, -1, -1, -1
    for i in contours:
        cv2.drawContours(img_contour, i, -1, (255, 0, 0), 4)
        area = cv2.contourArea(i)
        if area > 500:
            peri = cv2.arcLength(i, True)
            vertices = cv2.approxPolyDP(i, peri * 0.05, True)
            x, y, w, h = cv2.boundingRect(vertices)

    return x+w//2, y+h//2

pen_points = []

while True:
    ret, img = cap.read()
    if ret:
        img_contour = img.copy()
        # cv2.imshow('img', img)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # find the blue pen
        lower = np.array([85, 110, 86])
        upper = np.array([139, 205, 125])
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(img, img, mask=mask)
        penX, penY = findOutline(mask)
        # cv2.circle(img_contour, (penX, penY), 10, (255,0,0), cv2.FILLED)

        # pen point
        if penX!=-1 and penY != -1:
            pen_points.append([penX, penY])

        # drawing
        for point in pen_points:
            cv2.circle(img_contour, (point[0], point[1]), 10, (255,0,0), cv2.FILLED)
        
        cv2.imshow('Outline', img_contour)
    else:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break