
import  cv2
import  numpy as np


circles = np.zeros((4,2),np.int32)
counter = 0


def mousePoints(event ,x , y , flags , params ):
    global  counter
    if event== cv2.EVENT_LBUTTONDOWN:
        circles[counter] = x,y
        counter = counter+ 1
        print(circles)



img = cv2.imread("C:\\Users\Siam\Desktop\opencv\opencvimg5.png")
img = cv2.resize(img,(800,600))

while True:

    if counter == 4 :
        pts1 = np.float32([[398, 75], [746, 154],[294, 569], [660, 660]])
        pts2 = np.float32([[0, 0], [400, 0],[0, 640], [400, 640]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(img, matrix, (500, 600))
        result = cv2.resize(result,(400,640))
        cv2.imshow("siam1",result)



    cv2.imshow("siam",img)
    cv2.setMouseCallback("siam", mousePoints)
    cv2.waitKey(0)
