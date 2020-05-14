from mtcnn import MTCNN
import scipy.misc as sci 
import cv2
import dlib
import numpy as np
import os
import random
from threading import Thread
import queue

from mask import mask 

import copy 
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
dect= 'res/shape_predictor_81_face_landmarks.dat'
Mask = mask()
detector = MTCNN()
'''
shape_predictor = dlib.shape_predictor(dect)

img = sci.imread('images/inputs/group.jpg')

faces = detector.detect_faces(img)
for face in faces:
    rect = face['box']
    face = img[rect[1]:rect[3], rect[0]:rect[2]]
    rect = dlib.rectangle(*rect)
    f = shape_predictor(img, rect)
    #print(np.matrix([[pt.x, pt.y] for pt in f.parts()]))
'''


def get_all_faces(img, faces):
    list_face = []
    list_box = []
    for face in faces:
        x1,y1,w,h = face['box']
        x2, y2 = x1+w, y1+h
        list_face.append(img[y1-20:y2+20, x1-20:x2+20])
        list_box.append([x1-20, y1-20, x2+20,y2+20])
    return list_face, list_box

def swap_face(image):
    faces = detector.detect_faces(image)
    if faces != None:
        if len(faces) > 1:
            faces, boxes = get_all_faces(image, faces)
            tmp_faces = copy.deepcopy(faces)
            tmp_id_face = list(range(0, len(faces)))
            tmp_shuffle_face = tmp_id_face[::-1]
            for face, mask in zip(tmp_id_face, tmp_shuffle_face):
                print(face, mask)
                swaped_face = Mask.apply_mask(faces[face], tmp_faces[mask], True)
                x1,y1,x2,y2 = boxes[face]
                image[y1:y2, x1:x2] = swaped_face
            return image
        else:
            return image
    return image




def run_video(src = 0):
    video = cv2.VideoCapture(src)
    while 1:
        ret, image = video.read()
        try:
            img = swap_face(image.copy())
        except:
            img = image
        cv2.imshow('Face swapping', img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            video.release()
            cv2.destroyAllWindows()

            
def show_face(src = 'images/inputs/maroon5.jpg'):
    img = sci.imread(src)
    faces = detector.detect_faces(img)
    image = swap_face(img)
    cv2.imshow('', img[:,:,::-1])
    cv2.waitKey(0)
    
class ShowStream():
    def __init__(self, src=0):
        self.video = cv2.VideoCapture(src)
        self.QImg = queue.Queue()
        self.QisM = queue.Queue()
        self.flag = False
        self.thread = Thread(target=self.stream, args=())
        self.thread.daemon = True
        self.thread.start()
        

    def stream(self):
        while True:
            ret, img = self.video.read()
            if ret:
                self.img = swap_face(img)
                self.QImg.put(self.img)
                self.flag = True

    def display(self):
        if self.flag == True:
            try:
                img = self.QImg.get()
                cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
                img = cv2.resize(img, (1280, 720))
                cv2.imshow("Tracking", img)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    self.video.release()
                    cv2.destroyAllWindows()
            except Exception as e:
                print(e.args[-1])
if __name__ == '__main__':
    src = 'images/inputs/'
    src += 'imagine.jpeg'
    run_video('images/press.mp4')
    '''
    src = 'images/inputs/group.jpg'
    img = sci.imread(src)
    faces = detector.detect_faces(img)
    faces, boxes = get_all_faces(img, faces)
    for face in faces:
        cv2.imshow('', face[:,:,::-1])
        cv2.waitKey(0)
    '''