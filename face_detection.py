import cv2
import mediapipe as mp
import time


framewidth = 400
frameheight =  320
cap = cv2.VideoCapture(0)
cap.set(3,framewidth)
cap.set(4,frameheight)



class FaceDetector():
    def __init__(self, minDetectionCon=0.1):

        self.minDetectionCon = minDetectionCon

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)

        bboxs = []
        if self.results.detections:
            for id , detection in enumerate(self.results.detections):
                bboxc = detection.location_data.relative_bounding_box
                ih,iw,ic = img.shape
                bbox = int (bboxc.xmin*iw),int(bboxc.ymin*ih), \
                       int(bboxc.width*iw),int(bboxc.height*ih)
                bboxs.append([id,bbox,detection.score])
                cv2.rectangle(img , bbox,(0,255,0),cv2.FILLED)
                cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)


        return img , bboxs







def main():
    # cap = cv2.VideoCapture("C:\\Users\\Siam\\Desktop\\opencv\\vedio1.mp4")
    ptime = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img = cv2.resize(img, (600, 400))
        img ,bboxs = detector.findFaces(img)

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(img, f'fps:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
        cv2.imshow("siam", img)

        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
