import cv2
import numpy as np


framewidth = 400
frameHeight = 200
cap = cv2.VideoCapture(1)
cap.set(3,framewidth)
cap.set(4,frameHeight)

def empty(a):
    pass

cv2.namedWindow("hsv")
cv2.resizeWindow("hsv",640,240)
cv2.createTrackbar("HUE MIN","hsv",0,179,empty)
cv2.createTrackbar("HUE max","hsv",179,179,empty)
cv2.createTrackbar("sat max ","hsv",0,255,empty)
cv2.createTrackbar("sat MIN","hsv",255,255,empty)
cv2.createTrackbar("value max","hsv",0,255,empty)
cv2.createTrackbar("value  MIN","hsv",255,255,empty)


while True:
    img = cv2.imread("C:\\Users\Siam\Desktop\opencv\lamborgini.jpg")
    img = cv2.resize(img, (400, 400))
    imgHsv = cv2.cvtColor(img , cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("HUE MIN","hsv")
    h_max = cv2.getTrackbarPos("HUE max","hsv")
    s_min = cv2.getTrackbarPos("sat max ","hsv")
    s_max = cv2.getTrackbarPos("sat MIN","hsv")
    v_min = cv2.getTrackbarPos("value max","hsv")
    v_max = cv2.getTrackbarPos("value  MIN","hsv")

    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgHsv,lower,upper)
    result = cv2.bitwise_and(img ,img ,mask = mask)


    # cv2.imshow("siam",imgHsv)
    # cv2.imshow('original', img )
    # cv2.imshow("mask",mask)
    # cv2.imshow("result",result)
    mask= cv2.cvtColor(mask , cv2.COLOR_GRAY2BGR)
    h_stack = np.hstack([img , result,mask])
    cv2.imshow("hstack",h_stack)
    if(cv2.waitKey(1) & 0xFF) == ord('q'):
        break



cap.release()
cv2.destroyWindow()
