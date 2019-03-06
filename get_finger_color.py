import numpy as np
import cv2


cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    # frame = cv2.imread('./memage2.jpg')
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    height, width, _ = np.shape(frame)
    frame1 = hsv[200:210,200:210]
    cv2.rectangle(hsv,(200,200),(210,210),(0,255,0))
    frame2 = hsv[220:230,220:230]
    cv2.rectangle(hsv,(220,220),(230,230),(0,255,0))
    frame3 = hsv[170:180,250:260]
    cv2.rectangle(hsv,(170,250),(180,260),(0,255,0))
    frame4 = hsv[200:210,260:270]
    cv2.rectangle(hsv,(200 ,260), (210, 270),(0,255,0))
    frame4 = hsv[240:250, 340:350]
    cv2.rectangle(hsv,(240, 340), (250, 350),(0,255,0))

    median1 = np.median(frame1, axis=0)
    median2 = np.median(frame2, axis=0)
    median3 = np.median(frame3, axis=0)
    median4 = np.median(frame4, axis=0)

    int_averages = np.array(np.median(median1), dtype=np.uint8)


    # create a new image of the same height/width as the original
    average_image = np.zeros((height, width, 3), np.uint8)
    # and fill its pixels with our average color
    average_image[:] = int_averages
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    average_image = cv2.cvtColor(average_image, cv2.COLOR_HSV2RGB)


    cv2.imshow('frame', frame)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
