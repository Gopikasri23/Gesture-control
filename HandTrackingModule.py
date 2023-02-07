import cv2
import mediapipe as mp
import time  #to calculate frame rate

cap = cv2.VideoCapture(0); #0 indicates camera 0
mpHands = mp.solutions.hands #like formalities to do
hands = mpHands.Hands() #we can write parameters in this,click to know abt it
mpDraw = mp.solutions.drawing_utils

Prev_time=0
current_time=0
while True:
    success,img = cap.read()
   
    imgRGB =  cv2.cvtColor(img , cv2.COLOR_BGR2RGB) #conerting color to rgb because hands obj only uses rgb 
    results = hands.process(imgRGB) #process frame and give us results
    #print(results.multi_hand_landmarks) 
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            for id,lm in enumerate(handlms.landmark): #id gives the id(position) of points
                #print(id,lm)
                h, w, c = img.shape
                cx , cy = int(lm.x*w),int(lm.y*h) #to convert x and y to pixels
                #print(id,cx,cy)
                if id == 3: #you can change the id here
                    cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
        mpDraw.draw_landmarks(img, handlms,mpHands.HAND_CONNECTIONS) #handlms alone gives points of single hand and handconnections gives line

    current_time = time.time()
    fps = 1/(current_time-Prev_time) #frame per second = frame rate
    Prev_time = current_time

    cv2.putText(img , str(int(fps)),(10,70),cv2.FONT_HERSHEY_SIMPLEX, 3,(255,0,255),3)
    cv2.imshow("Image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

         