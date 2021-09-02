import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
segmentor = SelfiSegmentation()
imgBg = cv2.imread("1.jpg")

while True:
    ret, frame = cap.read()
    frame2 = segmentor.removeBG(frame, imgBg, threshold=0.6)
    frameStacked = cvzone.stackImages([frame, frame2], 2, 1)
    cv2.imshow("Frame", frameStacked)

    if cv2.waitKey(10) == ord("q"):
        break