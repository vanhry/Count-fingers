import numpy as np
import cv2
from sklearn.cluster import DBSCAN

# function for trackbar
def nothing(x):
    pass

# Create trackbars for adjust hsv ranges
"""
cv2.namedWindow('image')
cv2.createTrackbar('H_min','image', 0,255,nothing)
cv2.createTrackbar('H_max','image', 0,255,nothing)
cv2.createTrackbar('S_min','image', 0,255,nothing)
cv2.createTrackbar('S_max','image',0,255,nothing)
cv2.createTrackbar('V_min','image',0,255,nothing)
cv2.createTrackbar('V_max','image',0,255,nothing)
"""
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    # frame = cv2.imread('./memage2.jpg')
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # get trackbars for color change
    """
    H_min = cv2.getTrackbarPos('H_min', 'image')
    S_min = cv2.getTrackbarPos('S_min', 'image')
    V_min = cv2.getTrackbarPos('V_min', 'image')
    H_max = cv2.getTrackbarPos('H_max', 'image')
    S_max = cv2.getTrackbarPos('S_max', 'image')
    V_max = cv2.getTrackbarPos('V_max', 'image')"""

    # Color ranges for human skin
    # (very sensitive from lightness and often need to adjust)
    lower = np.array([0, 0, 150])
    upper = np.array([255, 100, 255])
    mask = cv2.GaussianBlur(hsv, (5, 5), 100)
    mask = cv2.inRange(mask, lower, upper)

    # Find max contours of mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours == []:
        continue
    max_contour = max(contours, key=lambda x: cv2.contourArea(x))
    cnt = contours[0]  # delete excess dim for list
    max_hull = cv2.convexHull(max_contour)

    # Approximate area of hand
    epsilon = 0.008*cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)

    # Compute center of max_hull area
    # for throw out max hull points from bottom (not from fingers)
    center = np.mean(max_hull, axis=0, dtype=np.uint32)
    cx, cy = tuple(center[0])
    cy += 50  # little shift center to downside
    cv2.circle(frame, (int(cx), int(cy)), 10, (255, 0, 255), 1)
    final = cv2.drawContours(frame, [max_hull], -1, (255, 0, 0), 5)
    # final = cv2.drawContours(frame, approx, -1, (0,255,0),5)

    # For points on the top of fingers find nearest neighborhouds
    # join them into cluster and find mean of each
    max_hull_top = np.array([i[0] for i in max_hull if i[0][1] < center[0][1]])
    if max_hull.shape[2] == 2:
        clustering = DBSCAN(eps=14, min_samples=1).fit(max_hull_top)
        labels = clustering.labels_
        for i in np.unique(labels):
            if i >= 0 and np.count_nonzero(labels==i) >= 2:
                array = []
                for num, j in enumerate(labels):
                    if j == i:
                        array.append(max_hull_top[num])
                if array != []:
                    ax, ay = tuple(np.squeeze(np.mean(np.array(array), axis=0)))
                    cv2.circle(frame, (int(ax), int(ay)), 10, (0, 0, 255), 1)

    # Find points between fingers (in root)
    #
    for num, i in enumerate(approx):
        count = 0
        for j in max_hull:
            if np.linalg.norm(j-i) < 90 and np.linalg.norm(j-i) > 0:
                count += 1
        x, y = i.ravel()
        if count < 1 and y < cy:
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # cv2.putText(frame, str(len(np.unique(labels))),(0,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3, cv2.LINE_AA)

    cv2.imshow('image', mask)
    cv2.imshow('frame', frame)

    # Esc for exit
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
