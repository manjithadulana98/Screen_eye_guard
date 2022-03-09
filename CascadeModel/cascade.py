import numpy as np
import cv2
import urllib.request as urlreq
import matplotlib.pyplot as plt
from skimage.draw import line
from scipy.spatial import distance as dist
import time
import utils


def get_detectors(detector_name = 'haarcascade'):
    if detector_name == 'haarcascade':
        # https://github.com/Danotsonof/facial-landmark-detection
        haarcascade_face_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
        haarcascade_face = "haarcascade_frontalface_alt2.xml"
        urlreq.urlretrieve(haarcascade_face_url, haarcascade_face)
        face_detector = cv2.CascadeClassifier(haarcascade_face)
    
        haarcascade_eye_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"
        haarcascade_eye = "haarcascade_eye.xml"
        urlreq.urlretrieve(haarcascade_eye_url, haarcascade_eye)
        eye_detector = cv2.CascadeClassifier(haarcascade_eye)

        LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"
        LBFmodel = "LFBmodel.yaml"
        urlreq.urlretrieve(LBFmodel_url, LBFmodel)
        landmark_detector  = cv2.face.createFacemarkLBF() 
        landmark_detector.loadModel(LBFmodel)

        return face_detector, eye_detector, landmark_detector
    else:
        raise NotImplementedError
 
        
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def image2features(image_gray, faces, landmark_detector=None): #features can be eye images/ frp-like features
    landmark = landmark_detector.fit(image_gray, faces)[1][0][0] #landmarks of first person
    eye1_landmark= landmark[36:42]
    eye2_landmark= landmark[42:48]
    
    eye1_ratio = eye_aspect_ratio(eye1_landmark)
    eye2_ratio = eye_aspect_ratio(eye2_landmark)
    blinking_ratio = (eye1_ratio + eye2_ratio) / 2
    
    return blinking_ratio

if __name__ == '__main__':
    face_cascade, eye_cascade, landmark_detector = get_detectors(detector_name = 'haarcascade') 

    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_counter = 0
    blink_counter = 0
    previous_ratio = 100
    eye_blink_signal=[]
    start_time = time.time()
    FONTS =cv2.FONT_HERSHEY_COMPLEX

    while(True):
        frame_counter += 1
        ret, frame = cap.read()
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
            roi_color = frame[y:y+h, x:x+w]
                
            color = (255, 0, 0) #BGR 0-255 
            stroke = 2
            end_cord_x = x + w  #wdith
            end_cord_y = y + h  #height
        
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
        
    
        if len(faces)!=0:
            blinking_ratio = image2features(gray, faces, landmark_detector)
        
            blinking_ratio_1 = blinking_ratio * 100
            blinking_ratio_2 = np.round(blinking_ratio_1)
            blinking_ratio_rounded = blinking_ratio_2 / 100
            cv2.putText(frame, str(blinking_ratio_rounded), (500, 50), font, 2, (0, 200, 255),5)

            # Appending blinking ratio to a list eye_blink_signal
            eye_blink_signal.append(blinking_ratio)
            if blinking_ratio < 0.3:
                if previous_ratio > 0.3:
                    blink_counter = blink_counter + 1
    
            previous_ratio = blinking_ratio
            
            print('blinking_ratio : ', blinking_ratio)
        
        else:
            landmark= 'no face'        
    
        cv2.putText(frame, str(blink_counter), (30, 50), font, 2, (0, 0, 255),5)
        cv2.imshow('frame',frame)
    
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
        
        end_time = time.time()-start_time
        fps = frame_counter/end_time

        frame =utils.textWithBackground(frame,f'FPS: {round(fps,1)}',FONTS, 1.0, (30, 50), bgOpacity=0.9, textThickness=2)
        # writing image for thumbnail drawing shape
        # cv.imwrite(f'img/frame_{frame_counter}.png', frame)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(2)

    cap.release()
    cv2.destroyAllWindows()
    
    