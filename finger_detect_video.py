import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import sys


# function for trackbar
def nothing(x):
    pass

# distance between two points and vectorize function for it
distance = lambda i, j: np.linalg.norm(i-j)
vect_dist = np.vectorize(distance)


# find max and min in the sequence of point
def peakdet(v, delta, x=None):
    maxtab = []
    mintab = []
    if x is None:
        x = np.arange(len(v))
    v = np.asarray(v)
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN
    lookformax = True
    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(maxtab), np.array(mintab)


# angle between three points
def get_angle(p1, p2, p3):
    line1 = p2 - p1
    line2 = p3 - p1
    cosine = np.dot(line1, line2) / (np.linalg.norm(line1) * np.linalg.norm(line2))
    angle = np.arccos(cosine)
    # print(angle * 180 / np.pi)
    return angle * 180 / np.pi


# built angles which describes count of fingers
def find_nearest(means, roots, image):
    means = sorted(means,key=lambda x:x[0])
    roots = sorted(roots,key=lambda x:x[0])
    diff = len(means)-len(roots)
    if diff > 1:
        for i in range(diff-1):
            roots.append(roots[-1])
    means = np.array(means)
    roots = np.array(roots)
    for i in range(len(means)-1):
        angle = get_angle(roots[i],means[i],means[i+1])
        if angle > 10 and angle < 90:
            cv2.line(image, tuple(roots[i]), tuple(means[i+1]), (0,255,0),4)
            cv2.line(image, tuple(roots[i]), tuple(means[i]), (0,255,0),4)

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
    # (very sensitive from brightness  and often need to adjust)
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
    epsilon = 0.01*cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)


    # edges of max_hull
    leftmost = max_hull[max_hull[:, :, 0].argmin()][0]
    rightmost = max_hull[max_hull[:, :, 0].argmax()][0]

    # use delta like some measure of size max_hull
    # for dicrease amount of Magic Numbers
    delta = np.linalg.norm(leftmost-rightmost) // 6

    # Compute center of max_hull area
    # for throw out max hull points from bottom (not from fingers)
    center = np.mean(max_hull, axis=0, dtype=np.uint32)
    cx, cy = tuple(center[0])
    cy += (delta * 2)  # little shift center to downside
    cv2.circle(frame, (int(cx), int(cy)), 10, (255, 0, 255), 1)

    final = cv2.drawContours(frame, max_hull, -1, (255, 0, 0), 3)
    final = cv2.drawContours(frame, approx, -1, (0,0,0),5)

    # For points on the top of fingers find nearest neighborhouds
    # join them into cluster and find mean of each
    means = []
    max_hull_top = np.array([i[0] for i in max_hull if i[0][1] < cy])
    if max_hull.shape[2] == 2:
        clustering = DBSCAN(eps=14, min_samples=1).fit(max_hull_top)
        labels = clustering.labels_
        means = []
        for i in np.unique(labels):
            if i >= 0 and np.count_nonzero(labels == i) >= 2:
                array = []
                for num, j in enumerate(labels):
                    if j == i:
                        array.append(max_hull_top[num])
                if array != []:
                    ax, ay = tuple(np.squeeze(np.mean(np.array(array), axis=0)))
                    cv2.circle(frame, (int(ax), int(ay)), 10, (0, 0, 255), 1)

    # find local min (points between fingers)
    # also remove points which close to max_hull
    data = np.array(np.squeeze(approx,axis=1))[:,1]
    indexes, _ = peakdet(data,.2)
    if indexes.size > 0:
        points = np.take(np.array(np.squeeze(approx,axis=1)), indexes[:,0],axis=0)
        for i in points:
            distances = np.sqrt(np.sum(vect_dist(np.squeeze(max_hull,axis=1), i)**2, axis=1))
            if (distances > 10).all() == True:
                cv2.circle(frame, tuple(i), 10, (0,255,255),-1)

        #if list(points) and means and len(means) > len(points):
        #    find_nearest(means, points, frame)
    # cv2.putText(frame, str(len(np.unique(labels))),(0,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3, cv2.LINE_AA)

    cv2.imshow('image', mask)
    cv2.imshow('frame', frame)

    # Esc for exit
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
