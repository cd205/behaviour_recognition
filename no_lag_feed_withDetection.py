import cv2
import time
import multiprocessing as mp
import mediapipe 

class Camera():
    
    def __init__(self,rtsp_url):
        #load pipe for data transmittion to the process
        self.parent_conn, child_conn = mp.Pipe()
        #load process
        self.p = mp.Process(target=self.update, args=(child_conn,rtsp_url))        
        #start process
        self.p.daemon = True
        self.p.start()
        
    def end(self):
        #send closure request to process
        
        self.parent_conn.send(2)
        
    def update(self,conn,rtsp_url):
        #load cam into seperate process
        
        print("Cam Loading...")
        cap = cv2.VideoCapture(rtsp_url,cv2.CAP_FFMPEG)   
        print("Cam Loaded...")
        run = True
        
        while run:
            
            #grab frames from the buffer
            cap.grab()
            
            #recieve input data
            rec_dat = conn.recv()
            
            
            if rec_dat == 1:
                #if frame requested
                ret,frame = cap.read()
                conn.send(frame)
                
            elif rec_dat ==2:
                #if close requested
                cap.release()
                run = False
                
        print("Camera Connection Closed")        
        conn.close()
    
    def get_frame(self,resize=None):
        ###used to grab frames from the cam connection process
        
        ##[resize] param : % of size reduction or increase i.e 0.65 for 35% reduction  or 1.5 for a 50% increase
             
        #send request
        self.parent_conn.send(1)
        frame = self.parent_conn.recv()
        
        #reset request 
        self.parent_conn.send(0)
        
        #resize if needed
        if resize == None:            
            return frame
        else:
            return self.rescale_frame(frame,resize)
        
    def rescale_frame(self,frame, percent=65):
        
        return cv2.resize(frame,None,fx=percent,fy=percent) 





if __name__ == '__main__':
    mp_drawing = mediapipe.solutions.drawing_utils
    mp_holistic = mediapipe.solutions.holistic
    cam = Camera('rtsp://admin:CXNVMA@192.168.0.109/h264_stream')

    print(f"Camera is alive?: {cam.p.is_alive()}")

    
    with mp_holistic.Holistic( 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
    ) as holistic:
        while(1):
            frame = cam.get_frame(0.65)
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Make detections
            results = holistic.process(image)
            if results.pose_landmarks:
                print(
                f'Left Shoulder coordinates: ('
                f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x }, '
                f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y })'
                )

            # recolor image back to BGR for rendering
            image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            
            # 1. Draw Pose landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            cv2.imshow("Feed",image)
            
            key = cv2.waitKey(1)

            if key == 13: #13 is the Enter Key
                break

    cv2.destroyAllWindows()     

    cam.end()