import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
import dlib



def pre_process(img, img_width=640, img_height=480):
    """Input a raw image, process to a certain format to throw into CNN"""
    # gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # if the input image is not (640x480), resize to this shape.
    if gray.shape != (img_height, img_width):
        gray = cv2.resize(gray,(img_width, img_height),interpolation=cv2.INTER_CUBIC)
    # Do a Histogram Equalization 
    equ = cv2.equalizeHist(gray)
    # extract eyes by dlib and concatenate left and right eye images.
    eyes_img = extract_eyes_by_dlib(equ, pad_scale=0.4, resize_width=50, resize_height=30, combine=True)
    return eyes_img


def extract_eyes_by_dlib(img, pad_scale=0.4, resize_width=50, resize_height=30, combine=False):
    """Get facial landmark matrix by dlib, extract eyes from this matrix and return left and right eye image
       combined or separately.  
    """
    mat = get_landmarks(img)
    
    left_left = mat[36,0]
    left_right = mat[39,0]
    left_top = min(mat[37,1], mat[38,1])
    left_bottom = max(mat[40,1], mat[41,1])
    right_left = mat[42,0]
    right_right = mat[45,0]
    right_top = min(mat[43,1], mat[44,1])
    right_bottom = max(mat[46,1], mat[47,1])
    # left pad
    left_pad_width = int((left_right - left_left) * pad_scale)
    left_pad_height = int((left_bottom - left_top) * pad_scale)
    # right pad
    right_pad_width = int((right_right - right_left) * pad_scale)
    right_pad_height = int((right_bottom - right_top) * pad_scale)
    # left and right eye plus pads
    left_eye = img[left_top-left_pad_height : left_bottom+left_pad_height, left_left-left_pad_width : left_right+left_pad_width, :]
    right_eye = img[right_top-right_pad_height : right_bottom+right_pad_height, right_left-right_pad_width : right_right+right_pad_width, :]
    # resize
    left_eye = cv2.resize(left_eye,(resize_width, resize_height),interpolation=cv2.INTER_CUBIC)
    right_eye = cv2.resize(right_eye,(resize_width, resize_height),interpolation=cv2.INTER_CUBIC)
    if combine:
        return np.concatenate((left_eye, right_eye), axis=1)    
    return left_eye, right_eye


def get_landmarks(img):
    """Single face landmark detector. return an nparray."""
    rects = detector(img, 1)
    return np.matrix([[p.x, p.y] for p in predictor(img, rects[0]).parts()])


def main():
    predictor_path = "shape_predictor_68_face_landmarks.dat/data"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)


    example_img = io.imread("./data/Eye_chimeraToPublish/0/eyes003019.jpg")
