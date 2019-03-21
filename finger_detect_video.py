import numpy as np
import cv2
from sklearn.cluster import DBSCAN


def distance(p1, p2):
    """
    distance between two points and vectorize function for it

    Arguments:
    p1 - first point, numpy array of int with shape (2,)
    p2 - second point, numpy array of int with shape (2,)

    Return:
    distance
    """
    return np.linalg.norm(p1-p2)


def check_for_maxhull(point, max_hull, measure):
    """
    Check if point are near to max_hull contour (not only point)

    Arguments:
    point -- numpy array of int with shape (2,) (point)
    max_hull -- numpy array with shape (n, 1, 2) (desribe contour)
    measure -- how far away from max_hull point need to be

    Returns:
    bool result of check
    """
    for index in range(len(max_hull)-1):
        if per_distance(max_hull[index],max_hull[index+1],point) < measure:
            return False
    return True


def per_distance(lp1, lp2, p):
    """
    Compute distance from two point line to one point

    Arguments:
    lp1 -- first point of line, numpy array of int with shape (2,)
    lp2 -- second point of line, numpy array of int with shape (2,)
    p -- third point outside line, numpy array of int with shape (2,)

    Returns:
    distance
    """
    return np.linalg.norm(np.cross(lp2-lp1, lp1-p))/np.linalg.norm(lp2-lp1)


def peakdet(v, delta, x=None):
    """
    Finds the local maxima and minima ("peaks") in the vector V

    Arguments:
    v -- array of shape (n,2)
    delta -- measure for define observed range for local min or max

    Returns:

    """
    maxtab = []
    mintab = []
    if x is None:
        x = np.arange(len(v))
    v = np.asarray(v)
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


def get_angle(p1, p2, p3):
    """
    Find angle between three points

    Arguments:
    p1 -- first (base) point, numpy array of int with shape (2,)
    p2 -- second point, numpy array of int with shape (2,)
    p3 -- third point , numpy array of int with shape (2,)

    Return:
    angle
    """
    line1 = p2 - p1
    line2 = p3 - p1
    cosine = np.dot(line1, line2) / (np.linalg.norm(line1) * np.linalg.norm(line2))
    angle = np.arccos(cosine)
    # print(angle * 180 / np.pi)
    return angle * 180 / np.pi


def find_nearest(means, roots, image):
    """
    Built angles between fingers and count fingers

    Arguments:
    means -- array of tuples (point on the finger's edge)
    roots -- array of tuples (point between fingers)
    image -- numpy array of shape (n, m, 3)

    Return:
    None
    """
    means = sorted(means, key=lambda x: x[0])
    roots = sorted(roots, key=lambda x: x[0])

    means = np.array(means)
    roots = np.array(roots)
    n = len(means)-len(roots)
    #if n > 1:
    #    for _ in range(n-1):
    #        roots = np.vstack([roots, roots[-1]])

    count_finger = 0
    for i in range(len(means)-1):
        if i == len(roots):
            break
        angle = get_angle(roots[count_finger], means[i], means[i+1])

        if angle > 10 and angle < 80 and distance(roots[count_finger], means[i+1]) > 70 \
            and distance(roots[count_finger], means[i]) > 70 \
            and means[i+1][1] < roots[count_finger][1] \
            and means[i][1] < roots[count_finger][1]:

            cv2.line(image, tuple(roots[count_finger]), tuple(means[i+1]), (0,255,0),4)
            cv2.line(image, tuple(roots[count_finger]), tuple(means[i]), (0,255,0),4)
            count_finger += 1

    if count_finger >= 1 and count_finger <= 5:
        cv2.putText(frame, str(count_finger+1),(0,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3, cv2.LINE_AA)
    if count_finger == 0:
        cv2.putText(frame, str(1),(0,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3, cv2.LINE_AA)



cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    # Color ranges for human skin
    # (very sensitive from brightness  and often need to adjust in file hsv_color.py)
    lower = np.array([0, 0, 140])
    upper = np.array([255, 90, 255])
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
    epsilon = 0.003*cv2.arcLength(max_contour, True)
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
    cy += (delta*2.5 )  # little shift center to downside
    # final = cv2.drawContours(frame, max_hull, -1, (255, 0, 0), 3)
    # final = cv2.drawContours(frame, approx, -1, (0,0,0),5)


    # find local min (points between fingers)
    # also remove points which close to max_hull
    data = np.array(np.squeeze(approx, axis=1))[:,1]
    indexes, _ = peakdet(data,.2)
    if indexes.size > 0:
        real_points = []
        corner_points = np.take(np.array(np.squeeze(approx, axis=1)), indexes[:,0], axis=0)
        for point in corner_points:
            if check_for_maxhull(point,max_hull,20) is True and point[1]<cy:
                cv2.circle(frame, tuple(point), 5, (0, 255, 255),-1)
                real_points.append(tuple(point))
    else:
        cv2.putText(frame, 'None', (0,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3, cv2.LINE_AA)
        continue

    # For points on the top of fingers find nearest neighborhouds
    # join them into cluster and find mean of each
    if real_points == []:
        continue
    min_root = min(real_points, key=lambda x: x[1])[1]
    means = []
    # points on the edge of point never be smallest than root_points
    max_hull_top = np.array([i[0] for i in max_hull if i[0][1] < cy])
    if max_hull.shape[2] == 2:
        clustering = DBSCAN(eps=16, min_samples=1).fit(max_hull_top)
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
                    means.append((int(ax),int(ay)))



    if list(real_points) and means and len(means) >= len(real_points):
            find_nearest(means, real_points, frame)


    cv2.imshow('image', mask)
    cv2.imshow('frame', frame)

    # Esc for exit
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break


cv2.destroyAllWindows()
cap.release()
