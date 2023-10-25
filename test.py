import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier=Classifier("D:\\DIP real time\\Model\\keras_model.h5","D:\\DIP real time\\Model\\labels.txt")

offset = 20
imgSize = 300

folder = "D:\\DIP real time\\Result"
counter = 0

labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","Saranghee"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset: y + h + offset, x - offset: x + w + offset]
        
        if imgCrop.shape[0] != 0 and imgCrop.shape[1] != 0:  # Check if imgCrop is not empty
            imgCropShape = imgCrop.shape
            aspectRatio = h / w
            
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((300 - wCal) / 2)
                imgWhite[:, wGap: wCal + wGap] = imgResize
                prediction,index = classifier.getPrediction(imgWhite)
                print(prediction, index)
                
                
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((300 - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize
                prediction,index = classifier.getPrediction(imgWhite)
            
        
        
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)
    
    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
    
    if key == ord("q"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)




















