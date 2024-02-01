import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
img = cv2.imread("C:\\Users\Siam\\Desktop\\opencv\\text1.png")
img= cv2.resize(img, (800,600))
img  = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)


# # detecting charecters
# himg, wimg  = img.shape
# boxes = pytesseract.image_to_boxes(img)
# for b in boxes.splitlines():
#     b= b.split()
#     x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
#     cv2.rectangle(img,(x,himg-y),(w,himg-h),(0,255,0),3)
#     cv2.putText(img , b[0],(x,himg - y + 25 ),  cv2.FONT_HERSHEY_COMPLEX, 1 , (0,255,255),2)


# detecting word
himg, wimg  = img.shape
boxes = pytesseract.image_to_data(img)
for x,b in enumerate(boxes.splitlines()):
    if x!=0 :
        b= b.split()
        if len(b)==12:
            x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
            # cv2.putText(img , b[0],(x,himg - y + 25 ),  cv2.FONT_HERSHEY_COMPLEX, 1 , (0,255,255),2)


print(pytesseract.image_to_string(img))
cv2.imshow("siam",img)
cv2.waitKey(0)