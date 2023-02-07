import cv2
import mediapipe as mp
import time  
 
class handDetector():
    def __init__(self , mode=False,maxHands = 2,complexity=1,detectioncon = 0.5,trackcon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectioncon = detectioncon
        self.trackcon = trackcon
        self.complexity = complexity

        self.mpHands = mp.solutions.hands 
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.complexity,self.detectioncon,self.trackcon)
        self.mpDraw = mp.solutions.drawing_utils
 
    def findHands(self,img,draw = True):
        imgRGB =  cv2.cvtColor(img , cv2.COLOR_BGR2RGB) #conerting color to rgb because hands obj only uses rgb 
        self.results = self.hands.process(imgRGB) #process frame and give us results
        #print(self.results.multi_hand_landmarks) 
        if self.results.multi_hand_landmarks:
            if len(self.results.multi_hand_landmarks) == 2:
                cv2.putText(img, 'Both Hands', (250, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 0.9,
                        (0, 255, 0), 2)
            for handlms in self.results.multi_hand_landmarks:
                
                if draw:
                    self.mpDraw.draw_landmarks(img, handlms,self.mpHands.HAND_CONNECTIONS) #handlms alone gives points of single hand and handconnections gives line
        return img
    
    def findPosition(self,img,handno=0,draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handno]
            for id,lm in enumerate(myHand.landmark): #id gives the id(position) of points
                        #print(id,lm)
                        h, w, c = img.shape
                        cx , cy = int(lm.x*w),int(lm.y*h) #to convert x and y to pixels
                        #print(id,cx,cy)
                        lmlist.append([id, cx,cy])
                        if draw and id == 20: #you can change this value
                            cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)
        return lmlist

  
def main():
    Prev_time=0
    current_time=0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success,img = cap.read()
        img = detector.findHands(img)
        lmlist=detector.findPosition(img)
        if len(lmlist) != 0:
            print(lmlist[20])
        current_time = time.time()
        fps = 1/(current_time-Prev_time) #frame per second = frame rate
        Prev_time = current_time
        cv2.putText(img , str(int(fps)),(10,70),cv2.FONT_HERSHEY_SIMPLEX, 3,(255,0,255),3)
        cv2.imshow("Image",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
if __name__ == "__main__":
    main()
