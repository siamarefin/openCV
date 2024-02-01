import cv2
import numpy as np

framewidth = 400
frameheight =  320
cap = cv2.VideoCapture(0)
cap.set(3,framewidth)
cap.set(4,frameheight)


def empty(a):
    pass
cv2.namedWindow("Parameter")
cv2.resizeWindow("Parameter",400,240)
cv2.createTrackbar("Tresshold1","Parameter",150,255,empty)
cv2.createTrackbar("Tresshold2","Parameter",255,255,empty)
cv2.createTrackbar("Area","Parameter",5000,300000,empty)





def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def getContours(img,imgcontours):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areamin = cv2.getTrackbarPos("Area","Parameter")

        if area>areamin:
            cv2.drawContours(imgcontours,cnt, - 1,(255,0,255),7)
            peri = cv2.arcLength(cnt,True)
            approx  = cv2.approxPolyDP(cnt , 0.02*peri, True)
            print(len(approx))
            x,y,w,h = cv2.boundingRect(approx)
            cv2.rectangle(imgcontours,(x,y),(x+w,y+h),(0,255,0),5)














while True:
    success, img = cap.read()
    # img = cv2.imread("C:\\Users\Siam\Desktop\opencv\card.png")
    imgcontours = img.copy()
    imgcontours = cv2.resize(imgcontours,(400,200))
    img = cv2.resize(img,(400,200))
    imgblur = cv2.GaussianBlur(img ,(7,7),1)
    imggray = cv2.cvtColor(imgblur,cv2.COLOR_BGR2GRAY)
    tresshold1 = cv2.getTrackbarPos("Tresshold1","Parameter")
    tresshold2 = cv2.getTrackbarPos("Tresshold2","Parameter")
    imgcanny = cv2.Canny(imggray, tresshold1,tresshold2)
    kernel = np.ones((5,5))
    imgdill = cv2.dilate(imgcanny,kernel,iterations=1)
    getContours(imgdill, imgcontours)
    imgstack  = stackImages(0.8,([img , imgcanny,imgdill],[imgcontours,imgcontours,imgcontours]  ))


    cv2.imshow("Result",imgstack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

