import numpy as np
import cv2
from sklearn.cluster import DBSCAN

# function for trackbar
def nothing(x):
    pass

# Create trackbars for adjust hsv ranges

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600,600)
cv2.createTrackbar('H_min','image', 0,255,nothing)
cv2.createTrackbar('H_max','image', 0,255,nothing)
cv2.createTrackbar('S_min','image', 0,255,nothing)
cv2.createTrackbar('S_max','image',0,255,nothing)
cv2.createTrackbar('V_min','image',0,255,nothing)
cv2.createTrackbar('V_max','image',0,255,nothing)

#cap = cv2.VideoCapture(0)

while True:
    #_, frame = cap.read()
    frame = cv2.imread('./memage_3f.jpg')
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # get trackbars for color change

    H_min = cv2.getTrackbarPos('H_min', 'image')
    S_min = cv2.getTrackbarPos('S_min', 'image')
    V_min = cv2.getTrackbarPos('V_min', 'image')
    H_max = cv2.getTrackbarPos('H_max', 'image')
    S_max = cv2.getTrackbarPos('S_max', 'image')
    V_max = cv2.getTrackbarPos('V_max', 'image')

    # Color ranges for human skin
    # (very sensitive from lightness and often need to adjust)
    lower = np.array([H_min, S_min, V_min])
    upper = np.array([H_max, S_max, V_max])
    mask = cv2.GaussianBlur(hsv, (5, 5), 100)
    mask = cv2.inRange(mask, lower, upper)

    

    # cv2.putText(frame, str(len(np.unique(labels))),(0,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3, cv2.LINE_AA)
    cv2.resizeWindow('image', 400,400)
    cv2.imshow('image', mask)
    cv2.imshow('frame', frame)

    # Esc for exit
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
#cap.release()
