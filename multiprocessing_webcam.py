####################### USING LOCK
import multiprocessing
import time
import numpy as np

import importlib
import math

import cv2 as cv
import mediapipe as mp
import datetime

import utils
importlib.reload(utils)


### CONSTANTS 
mp_face_mesh = mp.solutions.face_mesh
FONTS =cv.FONT_HERSHEY_COMPLEX
closed_eyes_frame = 1

### VARIABLES
closed_eyes_counter = 0
total_blinks = 0

### landmark detection function 
def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv.circle(img, p, 2, utils.GREEN, -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks 
    return mesh_coord



###
# face bounder indices 
FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

# lips indices for Landmarks
LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 
# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

### BLINK CALCULATOR
def euclaideanDistance(point, point1):
    x, y = point 
    x1, y1 = point1
    distance = math.sqrt((x1-x)**2 + (y1-y)**2)
    return distance 

def blinkRatio(img, landmarks, right_indices, left_indices):
    
    # RIGHT_EYE
    # horizontal line
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]

    # LEFT_EYE
    # horizontal line
    lh_right = landmarks[right_indices[0]]
    lh_left = landmarks[right_indices[8]]
    # vertical line
    lv_top = landmarks[11]
    lv_bottom = landmarks[16]

    rh_distance = euclaideanDistance(rh_right, rh_left)
    rv_distance = euclaideanDistance(rv_top, rv_bottom)
    lv_distance = euclaideanDistance(lv_top, lv_bottom)
    lh_distance = euclaideanDistance(lh_right, lh_left)

    right_eye_ratio = rh_distance / rv_distance
    left_eye_ratio = lh_distance / lv_distance

    ratio = (right_eye_ratio + left_eye_ratio) / 2
    
    cv.line(img, rh_right, rh_left, utils.GREEN, 2)
    cv.line(img, rv_top, rv_bottom, utils.GREEN, 2)
    return ratio


# def feed_sampling(records):
#     """
#     function to print values
#     """
#     n = 0
#     while n<5:
#         print(records.value)
#         n+=1
#         time.sleep(1)

  
# def live_camera_feed(records, lock):
#     """
#     function to change value
#     """
#     n = 0
#     while n<5:
#         with lock:
#             val = np.random.randint(10)
#             records.value = val
#         print("Value changed to: ", val)
#         n+=1
#         time.sleep(1)

def feed_sampling(total_blinks):
    """
    function to print values
    """
    n = 0
    while n<50:
    
        #val = np.random.randint(10,size=np.random.randint(10))
        timestamp = datetime.datetime.now().strftime('%y-%m-%d %H:%M:%S.%f')
        #print(total_blinks)
        
        print("Time: {0}  B: val= {1}".format(timestamp, str(total_blinks)))
        #total_blinks = 0
     
        time.sleep(2)
            
        n += 1


def live_camera_feed(total_blinks, lock):
    """
    function to run and rocess live camera images
    """
    # variables
    
    closed_eyes_counter = 0
    #total_blinks = 0

    with mp_face_mesh.FaceMesh(
    max_num_faces=1, 
    refine_landmarks=True, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
    ) as face_mesh:
        #cap = cv.VideoCapture(0)
        cap = cv.VideoCapture('rtsp://admin:CXNVMA@192.168.0.109/h264_stream')
        print(cap)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv.flip(frame, 1)
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB) # cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            img_height, img_width = frame.shape[:2]   # This is the actual measurements of the frame size.  We'll use this to multiply by the normalised x,y coordinates from results.multi_face_landmarks
            #print(img_height, img_width)
            frame = cv.resize(frame, (640, 480))  
            
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                mesh_coords = landmarksDetection(frame, results, False)
                #print(mesh_coords[p] for p in RIGHT_EYE)
                ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
                cv.putText(frame, f'ratio {round(ratio,2)}', (100,100), FONTS, 1.0, utils.GREEN, 1)

                ## Blink counter logic
                if ratio > 5.0:
                    cv.putText(frame, 'Blink', (200,30), FONTS, 1.3, utils.RED, 2)
                    closed_eyes_counter += 1
                else:
                    if closed_eyes_counter > closed_eyes_frame:
                        with lock:
                            total_blinks.value += 1
                            print('total blinks: {}'.format(total_blinks.value))
                            closed_eyes_counter = 0
                cv.imshow('Webcam',frame)
                
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
    cap.release()
    cv.destroyAllWindows()



    
  
if __name__ == '__main__':
    with multiprocessing.Manager() as manager:
        # creating a value in server process memory
        lock = manager.Lock()
        total_blinks = manager.Value('i',0.0)
  
        # creating new processes
        p1 = multiprocessing.Process(target=live_camera_feed, args=(total_blinks, lock))
        p2 = multiprocessing.Process(target=feed_sampling, args=(total_blinks,))
      
        p1.start()
        p2.start()
        p1.join()
        p2.join()