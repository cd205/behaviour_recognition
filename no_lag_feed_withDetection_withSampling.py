import cv2
import time
import multiprocessing as mp
import mediapipe 
import datetime
import csv   

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
        if rtsp_url=='webcam':
            cap = cv2.VideoCapture(0) 
        else:
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


def live_camera_feed(shoulder_pos, positions, lock):
    """
    function to run and process live camera images
    """
    
    mp_drawing = mediapipe.solutions.drawing_utils
    mp_holistic = mediapipe.solutions.holistic
    #cam = Camera('rtsp://admin:CXNVMA@192.168.0.109/h264_stream')
    cam = Camera('webcam')

    print(f"Camera is alive?: {cam.p.is_alive()}")
    
    with mp_holistic.Holistic( 
    static_image_mode=False, 
    model_complexity=1, 
    smooth_landmarks=True, 
    min_detection_confidence=0.45, 
    min_tracking_confidence=0.45,

    ) as holistic:
        while(1):
            frame = cam.get_frame(0.65)
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Make detections
            results = holistic.process(image)
            if results.pose_landmarks:
                # print(
                # f'Left Shoulder coordinates: ('
                # f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x }, '
                # f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y })'
                # )
                with lock:
                    shoulder_pos.value = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x
                    positions['left_shoulder_x'] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x
                    positions['left_shoulder_y'] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y
                    positions['right_shoulder_x'] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x
                    positions['right_shoulder_y'] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y
                    

            # recolor image back to BGR for rendering
            image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            #image = cv2.flip(image, 1)

            
            # 1. Draw Pose landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            # 2. Draw Face landmarks
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=1, circle_radius=1))

            # 3. Draw Right Hand landmarks
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(0,0,255), thickness=1, circle_radius=1),
                                mp_drawing.DrawingSpec(color=(0,0,255), thickness=1, circle_radius=1))

             # 4. Draw Left Hand landmarks
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(255,0,255), thickness=1, circle_radius=1),
                                mp_drawing.DrawingSpec(color=(255,0,255), thickness=1, circle_radius=1))

            
            
            cv2.imshow("Feed",image)
            
            key = cv2.waitKey(1)

            if key == 13: #13 is the Enter Key
                break

    cv2.destroyAllWindows()     

    cam.end()



def feed_sampling(shoulder_pos, positions):
    """
    function to print values
    """
    n = 0
    while n<5:
    
        #val = np.random.randint(10,size=np.random.randint(10))
        timestamp = datetime.datetime.now().strftime('%y-%m-%d %H:%M:%S.%f')
        
        print("Time: {0} Shoulder coords (Left, right) {1} {2}, {3} {4}".format(timestamp, 
                            positions['left_shoulder_x'], positions['left_shoulder_y'],
                            positions['right_shoulder_x'], positions['right_shoulder_y']))

        fields = [timestamp, positions['left_shoulder_x'], positions['left_shoulder_y'], positions['right_shoulder_x'], positions['right_shoulder_y']]
        with open(r'shoulder_log.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
        time.sleep(2)
            
        n += 1

if __name__ == '__main__':
    with mp.Manager() as manager:
        # creating a value in server process memory
        lock = manager.Lock()
        shoulder_pos = manager.Value('i',0.0)
        positions = manager.dict()
        positions['left_shoulder_x'] = []
        positions['left_shoulder_y'] = []
        positions['right_shoulder_x'] = []
        positions['right_shoulder_y'] = []
  
  
        # creating new processes
        p1 = mp.Process(target=live_camera_feed, args=(shoulder_pos, positions, lock))
        p2 = mp.Process(target=feed_sampling, args=(shoulder_pos, positions,))
      
        p1.start()
        p2.start()
        p1.join()
        p2.join()


        # https://www.youtube.com/watch?v=-toNMaS4SeQ
        # https://github.com/niconielsen32/ComputerVision/blob/master/headPoseEstimation.py