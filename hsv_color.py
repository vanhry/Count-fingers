import numpy as np
import cv2


def nothing(x):
    pass

cv2.namedWindow('image')
cv2.createTrackbar('H_min','image', 0,255,nothing)
cv2.createTrackbar('H_max','image', 0,255,nothing)
cv2.createTrackbar('S_min','image', 0,255,nothing)
cv2.createTrackbar('S_max','image',0,255,nothing)
cv2.createTrackbar('V_min','image',0,255,nothing)
cv2.createTrackbar('V_max','image',0,255,nothing)
# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'image',0,1,nothing)


cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    # frame = cv2.imread('./memage2.jpg')
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # create trackbars for color change
    H_min = cv2.getTrackbarPos('H_min', 'image')
    S_min = cv2.getTrackbarPos('S_min', 'image')
    V_min = cv2.getTrackbarPos('V_min', 'image')
    H_max = cv2.getTrackbarPos('H_max', 'image')
    S_max = cv2.getTrackbarPos('S_max', 'image')
    V_max = cv2.getTrackbarPos('V_max', 'image')
    s = cv2.getTrackbarPos(switch,'image')

    lower = np.array([H_min, S_min, V_min])
    upper = np.array([H_max, S_max, V_max])
    mask = cv2.inRange(hsv, lower, upper)

    cv2.imshow('image', mask)
    cv2.imshow('frame', frame)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
