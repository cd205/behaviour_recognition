# https://medium.com/@van.evanfebrianto/how-to-train-custom-object-detection-models-using-retinanet-aeed72f5d701

import importlib
import cv2
import time
import multiprocessing as mp
import mediapipe 
import datetime
import csv   
import detectors
importlib.reload(detectors)

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


def live_camera_feed(positions, headpose, lock):
    """
    function to run and process live camera images
    """
    
    mp_holistic = mediapipe.solutions.holistic
    mp_facemesh = mediapipe.solutions.face_mesh
    #cam = Camera('rtsp://admin:CXNVMA@192.168.0.109/h264_stream')
    cam = Camera('webcam')

    print(f"Camera is alive?: {cam.p.is_alive()}")
    
        
    print('running headpose')
    headpose = detectors.head_pose(cam, mp_facemesh, headpose, lock)
    print('run headpose')
    
    positions = detectors.shoulder_pos(cam, mp_holistic, positions, lock)
    
    # key = cv2.waitKey(1)

    # if key == 13: #13 is the Enter Key
    #     break

    cv2.destroyAllWindows()     

    cam.end()



def feed_sampling(positions, headpose):
    """
    function to print values
    """
    n = 0
    while n<10:
    
        #val = np.random.randint(10,size=np.random.randint(10))
        timestamp = datetime.datetime.now().strftime('%y-%m-%d %H:%M:%S.%f')
        
        print("Time: {0}, Shoulder coords (Left, right) {1} {2}, {3} {4}. Head direction:  {5}".format(timestamp, 
                            positions['left_shoulder_x'], positions['left_shoulder_y'],
                            positions['right_shoulder_x'], positions['right_shoulder_y'],
                            headpose['direction']))

        fields = [timestamp, positions['left_shoulder_x'], positions['left_shoulder_y'], 
                             positions['right_shoulder_x'], positions['right_shoulder_y'],
                             headpose['direction']]
        with open(r'shoulder_log.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
        time.sleep(2)
            
        n += 1

if __name__ == '__main__':
    with mp.Manager() as manager:
        # creating a value in server process memory
        lock = manager.Lock()
        positions = manager.dict()
        positions['left_shoulder_x'] = []
        positions['left_shoulder_y'] = []
        positions['right_shoulder_x'] = []
        positions['right_shoulder_y'] = []

        headpose = manager.dict()
        headpose['direction'] = []
  
  
        # creating new processes
        p1 = mp.Process(target=live_camera_feed, args=(positions, headpose, lock))
        p2 = mp.Process(target=feed_sampling, args=(positions, headpose, ))
      
        p1.start()
        p2.start()
        p1.join()
        p2.join()


        # https://www.youtube.com/watch?v=-toNMaS4SeQ
        # https://github.com/niconielsen32/ComputerVision/blob/master/headPoseEstimation.py