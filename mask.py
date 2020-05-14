import cv2
import dlib
import os, sys, argparse
import numpy as np

import scipy.misc as sci 
import numpy as np 

import logging



JAW_IDX = list(np.arange(0, 17))
FACE_IDX = list(np.arange(17, 68))
MOUTH_IDX = list(np.arange(48, 61))

RIGHT_EYE_IDX = list(np.arange(36, 42))
LEFT_EYE_IDX = list(np.arange(42, 48))

NOSE_IDX = list(np.arange(27, 35))
LEFT_EYE_BROW_IDX = list(np.arange(22, 27))
RIGHT_EYE_BROW_IDX = list(np.arange(17, 22))
FOREHEAD_IDX = list(np.arange(68, 81))


MATCH_POINTS_IDX = LEFT_EYE_BROW_IDX + RIGHT_EYE_BROW_IDX + LEFT_EYE_IDX + RIGHT_EYE_IDX + NOSE_IDX + MOUTH_IDX
OVERLAY_POINTS_IDX = [
    LEFT_EYE_IDX + RIGHT_EYE_IDX + LEFT_EYE_BROW_IDX + RIGHT_EYE_BROW_IDX,
    NOSE_IDX + MOUTH_IDX,
]
FACE_ALL_POINTS_IDX = list(np.arange(0, 81))
BOUNDING_FACE_IDX = JAW_IDX + RIGHT_EYE_BROW_IDX + LEFT_EYE_BROW_IDX + FOREHEAD_IDX

class mask:
    def __init__(self, dect= 'res/shape_predictor_81_face_landmarks.dat'):
        self.BLUR_AMOUNT = 5
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(dect)
        self.init(mask = 'images/masks/dmask2.jpg')
    def init(self, mask):
        # Load mask
        self.mask = sci.imread(mask)
        self.mask_idx_triangles, self.mask_landmarks = self.analysis_new_mask(self.mask)
        
    def get_parts_mask(self, img_shape, landmarks, mode :str, blur = 5):
        '''
        Input: img_shape:list HxW of Image, landmarks: its self detected landmarks
        return: a 3 channels HxW mask : (respect to OVERLAYS_POINTS_IDX)
        '''
        # Creat empty 2D image
        img = np.zeros(img_shape, dtype = np.uint8)
        # fillCovexPoly by ConvexHull
        convexhull = None 
        if mode == 'overlay':
            convexhull = []
            for Idx in OVERLAY_POINTS_IDX:
                overlay_point2 = [landmarks[idx] for idx in Idx]
                cv2.fillConvexPoly(img, cv2.convexHull(np.asarray(overlay_point2)), 255)
        elif mode == 'face':
            convexhull = cv2.convexHull(np.asarray(landmarks))
            cv2.fillConvexPoly(img, convexhull, color = 255)
        else:
            raise ValueError(f'Mode {mode} is unsupported!')

        if blur:
            #img = (cv2.GaussianBlur(img, (blur, blur), 0))
            img = cv2.GaussianBlur(img, (blur, blur), 0)

        return img, convexhull
    
    def get_mask(self, skin, parts, blur=0):
        img = cv2.bitwise_xor(skin, parts)
        if blur:
            img = cv2.GaussianBlur(img, (blur, blur), 0)
        return img
    
    def xywh2xyxy(self, inp):
        '''
        inp: list includes: centerx, centery, width, height
        return : list : xmin, ymin, xmax, ymax
        '''
        xmin = int(inp[0] -10)
        xmax = int(inp[0] + inp[2] +10)
        ymin = int(inp[1] -10)
        ymax = int(inp[1] + inp[3] +10)
        return [xmin, ymin, xmax, ymax]
    
    def get_tm_opp(self, pts1, pts2):
        # Transformation matrix - ( Translation + Scaling + Rotation )
        # using Procuster analysis
        pts1 = np.float64(pts1)
        pts2 = np.float64(pts2)

        m1 = np.mean(pts1, axis = 0)
        m2 = np.mean(pts2, axis = 0)

        # Removing translation
        pts1 -= m1
        pts2 -= m2

        std1 = np.std(pts1)
        std2 = np.std(pts2)
        std_r = std2/std1

        # Removing scaling
        pts1 /= std1
        pts2 /= std2

        U, S, V = np.linalg.svd(np.transpose(pts1) * pts2)

        # Finding the rotation matrix
        R = np.transpose(U * V)

        return np.vstack([np.hstack((std_r * R,
            np.transpose(m2) - std_r * R * np.transpose(m1))), np.matrix([0.0, 0.0, 1.0])])
        
    def warp_image(self, img, tM, shape):
        out = np.zeros(shape, dtype=img.dtype)
        cv2.warpPerspective(img, tM, (shape[1], shape[0]), dst=out,
                            borderMode=cv2.BORDER_TRANSPARENT,
                            flags= cv2.WARP_INVERSE_MAP + cv2.WARP_POLAR_LINEAR)
        return out
    def get_facial_landmarks(self, img):
        '''
        Input: img: ndarray
        Output: landmarks
        '''
        #rects = self.face_detector.detect(img.copy(), 0)
        rects = self.face_detector(img.copy(), 0)
        if len(rects) == 0:
            #print ("No faces")
            rects = dlib.rectangle(0, 0, img.shape[1], img.shape[0])
        else:
            rects = rects[0]
        shape = self.shape_predictor(img, rects)

        return [[pt.x + 1, pt.y + 1] for pt in shape.parts()]
    
    def analysis_new_mask(self, mask):
        return self.make_indexes_triangles(mask)
    ################## NEw Delaunay Triangle ########################
    def _extract_index_nparray(self, nparray):
        index = None
        for num in nparray[0]:
            index = num
            break
        return index

    def make_indexes_triangles(self, img):
        mask_landmarks  = self.get_facial_landmarks(img)
        points = np.array(mask_landmarks, np.int32)
        convexhull = cv2.convexHull(points)
        rect = cv2.boundingRect(convexhull)
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(mask_landmarks)
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype= np.int32)
        indexes_triangles = []
        for t in triangles:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            index_pt1 = np.where((points == pt1).all(axis=1))
            index_pt1 = self._extract_index_nparray(index_pt1)

            index_pt2 = np.where((points == pt2).all(axis=1))
            index_pt2 = self._extract_index_nparray(index_pt2)

            index_pt3 = np.where((points == pt3).all(axis=1))
            index_pt3 = self._extract_index_nparray(index_pt3)

            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                triangle = [index_pt1, index_pt2, index_pt3]
                indexes_triangles.append(triangle)

        return indexes_triangles, mask_landmarks
    
    def get_every_triangle(self, img, landmarks, triangle_index):
         # Triangulation of the first face
        tr1_pt1 = landmarks[triangle_index[0]]
        tr1_pt2 = landmarks[triangle_index[1]]
        tr1_pt3 = landmarks[triangle_index[2]]
        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
        
        (x, y, w, h) = cv2.boundingRect(triangle1)
        
        cropped_triangle = img[y: y + h, x: x + w]
        cropped_tr1_mask = np.zeros((h, w), np.uint8)
        points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                           [tr1_pt2[0] - x, tr1_pt2[1] - y],
                           [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)
        cv2.fillConvexPoly(cropped_tr1_mask, points, 255.0)
        
        return cropped_triangle, cropped_tr1_mask, np.float32(points), (x, y, w, h)
    

    ################## Call this #######################
    
    def apply_mask(self, input_img, mask = None, all_parts = True):
        '''
        input_img: ndarray
        mask: ndarray or None
        return : ndarray applied mask of mask
        '''
        if mask is not None:
            mask_idx_triangles, mask_landmarks = self.analysis_new_mask(mask)
        else:
            mask, mask_idx_triangles, mask_landmarks = \
                 self.mask, self.mask_idx_triangles, self.mask_landmarks 
                 
        user_landmarks = self.get_facial_landmarks(input_img)
        
        user_facial_mask, user_convexhull\
                       = self.get_parts_mask(input_img.shape[:2], user_landmarks, 'face', blur=3)
        user_overlay_mask, _ = self.get_parts_mask(input_img.shape[:2], user_landmarks, 'overlay', blur=3)
        user_mask = self.get_mask(user_facial_mask, user_overlay_mask, blur=0)
        user_new_face = np.zeros_like(input_img, dtype= np.uint8)
        
        for triangle_index in (mask_idx_triangles):
            mask_cropped_triangle, _, mask_points, _ \
                = self.get_every_triangle(mask, mask_landmarks, triangle_index)
            _, user_cropped_mask, user_points, (x,y,w,h) \
                = self.get_every_triangle(input_img, user_landmarks, triangle_index)
            M = cv2.getAffineTransform(mask_points, user_points)
            warped_triangle = cv2.warpAffine(mask_cropped_triangle, M, (w, h))
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=user_cropped_mask)
            # Reconstructing destination face
            user_cropped_triangle = user_new_face[y: y + h, x: x + w]
            user_cropped_triangle_gray = cv2.cvtColor(user_cropped_triangle, cv2.COLOR_RGB2GRAY)
            _, mask_triangles_designed = cv2.threshold(user_cropped_triangle_gray, 1, 255, cv2.THRESH_BINARY_INV)
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)
            user_cropped_triangle = cv2.add(user_cropped_triangle, warped_triangle)
            user_new_face[y: y + h, x: x + w] = user_cropped_triangle
        
        if all_parts:
            user_mask = user_facial_mask.copy()
        
        user_mask = cv2.GaussianBlur(cv2.cvtColor(user_mask, cv2.COLOR_GRAY2RGB), (19,19), 11) / 255.0
        user_new_face = np.uint8(user_new_face * (user_mask))
        user_mask = (1.0 - user_mask)
        user_noface = np.uint8(input_img * user_mask)
        result = cv2.addWeighted(user_new_face, 0.7, user_noface, 0.66, 4)
        (x, y, w, h) = cv2.boundingRect(user_convexhull)
        center = (int((x + x + w) / 2), int((y + y + h) / 2))
        if all_parts:
            out = cv2.seamlessClone(result, input_img, user_facial_mask, center, cv2.NORMAL_CLONE)
        else:
            out = cv2.seamlessClone(result, input_img, user_facial_mask, center, cv2.MIXED_CLONE)
        return out
    ############## Swap Face##############
    
    
    
    ############## Display ###############
    def display_landmaks(self, inp):
        inp_keypoints = self.get_facial_landmarks(inp)
        img = inp.copy()
        for point in inp_keypoints:
            cv2.circle(img, (point[0,0], point[0,1]), 3,(255, 0, 0), 3)
        return img