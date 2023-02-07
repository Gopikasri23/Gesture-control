import cv2
import os 
from cvzone.HandTrackingModule import HandDetector

width , height = 1280,720

cap = cv2.VideoCapture(0)
cap.set(3,width)
cap.set(4,height)
folderpath = "presentation" #can set your folder path contains images here

#get the list of presentation images
pathImages = sorted(os.listdir(folderpath) ,key=len)



#variables
imgNumber = 0
hs,ws = int(120*1),(213*1) #height of small image and width of small msg. we can change it
geture_thresh = 300
buttonpressed = False
buttoncounter = 0
buttondelay = 30 #10 frames
# Hand Detector
detector = HandDetector(detectionCon = 0.8 , maxHands=1)


while True:
    success ,img = cap.read()
    pathFullImage = os.path.join(folderpath,pathImages[imgNumber])
    imgcur = cv2.imread(pathFullImage)

    hands,img = detector.findHands(img)
    #cv2.line(img,(0,geture_thresh),(width,geture_thresh),(0,255,0),10)


    if hands and buttonpressed is False:
        hand = hands[0]
        no_of_fing = detector.fingersUp(hand)
        cx, cy = hand['center']
        #print(no_of_fing)
         #if hand is at the height of the face
            # gesture 1 - left
        if no_of_fing == [1,1,0,0,0]:
            print("slide left")
            if imgNumber > 0:
                buttonpressed = True
                imgNumber -= 1

        if no_of_fing == [0,1,1,1,1]:
            print("slide right")
            if imgNumber < len(pathImages)-1:
                    buttonpressed = True
                    imgNumber += 1
    #button pressed iterations
    if buttonpressed:
        buttoncounter += 1
        if buttoncounter > buttondelay:
            buttoncounter = 0
            buttonpressed = False
    #Adding webcam image on the pictures if not need remove it
    imgsmall = cv2.resize(img, (ws, hs))
    h,w,_ = imgcur.shape
    imgcur[0:hs,w-ws:w] = imgsmall #starting and ending point of small msg

     
    cv2.imshow("Image",img)
    cv2.imshow("Slides", imgcur)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break
