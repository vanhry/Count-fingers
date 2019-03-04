import cv2
import numpy as np


cap = cv2.VideoCapture(0)

while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        kernel = np.ones((3, 3), np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = frame
        # apply mask
        retval, mask = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, kernel, iterations=4)
        mask = cv2.GaussianBlur(mask, (5, 5), 100)

        # find biggest contour and contour approx
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=lambda x: cv2.contourArea(x))
        max_hull = cv2.convexHull(max_contour)
        epsilon = 0.001*cv2.arcLength(max_contour,True)
        approx = cv2.approxPolyDP(max_contour, epsilon, True)
        final = cv2.drawContours(roi, [approx], -1, (0, 0, 0), 2)

        # find and draw angles
        height, width = frame.shape
        blank_image = np.zeros((height,width,3), np.uint8)+255
        contours_angle = cv2.drawContours(blank_image, [approx], -1, (0,0,0), 2)
        contours_angles = cv2.cvtColor(contours_angle, cv2.COLOR_RGB2GRAY)
        corners = cv2.goodFeaturesToTrack(contours_angles, 100, 0.3, 10)
        corners = np.int0(corners)

        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(roi, (x,y), 5, (0,255,0), -1)

        cv2.imshow('mask', frame)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

cv2.destroyAllWindows()
cap.release()
