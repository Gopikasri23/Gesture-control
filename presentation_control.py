import cv2
import os 
from cvzone.HandTrackingModule import HandDetector
from pynput.keyboard import Key,Controller
width , height = 720,720

cap = cv2.VideoCapture(0)
cap.set(3,width)
cap.set(4,height)

#keyboard
keyboard = Controller()
#variable
buttonpressed = False
buttoncounter = 0
buttondelay = 20 #30 frames
# Hand Detector
detector = HandDetector(detectionCon = 0.8 , maxHands=1)


while True:
    success ,img = cap.read()
    hands,img = detector.findHands(img)
   # print(hands)
    if hands and buttonpressed is False:
        hand = hands[0]
        no_of_fing = detector.fingersUp(hand)
        #print(no_of_fing)
        if no_of_fing == [1,1,0,0,0]:
            print("slide right")
            buttonpressed = True
            keyboard.press(Key.page_up)
            keyboard.release(Key.page_up)
           
        if no_of_fing == [0,1,1,1,1]:
            print("slide left")
            buttonpressed = True
            keyboard.press(Key.page_down)
            keyboard.release(Key.page_down)
         

    #button pressed iterations
    if buttonpressed:
        buttoncounter += 1
        if buttoncounter > buttondelay:
            buttoncounter = 0
            buttonpressed = False
     
    cv2.imshow("Image",img)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break
