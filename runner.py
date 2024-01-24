from argparse import ArgumentParser

import cv2
import copy

import os
from os.path import basename, dirname, splitext
from tqdm import tqdm
from glob import glob
import sys
import numpy as np

import pickle

# opencv
from mark_detector import MarkDetector

# pose estimation
from euler_angle import EulerAngles
from pose_estimator import PoseEstimator

# scrfd
sys.path.append( '/home/john23/WorkSpace/FacePython/det_scrfd')
sys.path.append( '/home/john23/WorkSpace/FacePython/')
from scrfd import SCRFD

# centeface
sys.path.append( '/home/john23/WorkSpace/FacePython/alg_align')
from centerface import CenterFace

# blazeface 
from blazeface import BlazeFace

# PFLD
sys.path.append('/home/john23/WorkSpace/PFLD-Pytorch-Landmarks/model')
from model import PFLD, AuxiliaryNet

print("OpenCV version: {}".format(cv2.__version__))

def rot_params_rv(rvecs):
    from math import pi,atan2,asin
    R = cv2.Rodrigues(rvecs)[0]
    roll = 180*atan2(-R[2][1], R[2][2])/pi
    pitch = 180*asin(R[2][0])/pi
    yaw = 180*atan2(-R[1][0], R[0][0])/pi
    rot_params= [pitch, yaw, roll]
    return rot_params

class FaceDetections: 
    def __init__(self, fd_model, fd_model_list=['scrfd', 'centeface', 'blazeface', 'opencv']):
        assert fd_model in fd_model_list
        self.fd_model = fd_model
        if fd_model == 'opencv':
            self.mark_detector = MarkDetector() 
        elif fd_model == 'scrfd':
            self.scrfd = SCRFD( fn_model='/home/john23/WorkSpace/FacePython/det_scrfd/model/scrfd_320.onnx', thresh_score=0.7, thresh_nms=0.4, gpu_idx=0, use_onnxruntime=True) 
        elif fd_model == 'centeface':
            self.det_centerface = CenterFace('/home/john23/WorkSpace/FacePython/alg_align/model/centerface_model.onnx', thresh_score=0.7, thresh_nms=0.3, gpu_idx=0)
        elif fd_model == 'blazeface':
            import torch
            import torchvision.transforms.transforms as transforms
            self.device = torch.device("cpu")
            self.det_blazeface = BlazeFace(thresh_score=0.7, thresh_nms=0.3).to(self.device)
            self.det_blazeface.load_model('/home/john23/WorkSpace/FacePython/alg_align/model/blazeface_model.pth', \
                                            '/home/john23/WorkSpace/FacePython/alg_align/model/blazeface_anchors.npy')

    def __perprocess__(self, image):
        if self.fd_model == 'opencv':
            None
        elif self.fd_model == 'scrfd':
            #im = cv2.imread(image, cv2.IMREAD_COLOR) # in BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # in RGB
        elif self.fd_model == 'centeface':
            #im = cv2.imread(image, cv2.IMREAD_COLOR) # in BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # in RGB
        elif self.fd_model == 'blazeface':
            #im = cv2.imread(image, cv2.IMREAD_COLOR) # in BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # in RGB
        return image
    
    def inference(self, image):
        image = self.__perprocess__(image)
        if self.fd_model == 'opencv':
            facebox = self.mark_detector.extract_cnn_facebox(image)
            if facebox:
                list_bbox = np.asarray([facebox])
            else:
                list_bbox = []
        elif self.fd_model == 'scrfd':
            list_score, list_bbox, list_kps = self.scrfd.detect(image)
            self.list_kps = list_kps

        elif self.fd_model == 'centeface':
            list_bbox, list_kps = self.det_centerface.detect(image)
        elif self.fd_model == 'blazeface':
            list_bbox, list_kps = self.det_blazeface.detect(image)
        return np.asarray(list_bbox).astype(int).tolist()
    

class LandmarkDetections:
    def __init__(self, lmk_model, lmk_model_list=['scrfd', 'yin_cnn', 'PFLD'] ):
        assert lmk_model in lmk_model_list
        self.img_size = 112
        self.lmk_model = lmk_model
        if self.lmk_model == 'scrfd':
            self.scrfd = SCRFD( fn_model='/home/john23/WorkSpace/FacePython/det_scrfd/model/scrfd_320.onnx', thresh_score=0.7, thresh_nms=0.4, gpu_idx=0, use_onnxruntime=True) 
            self.eular_estimator = EulerAngles(img_shape=(self.img_size, self.img_size), landmark_format='5pts' ) 
        elif self.lmk_model == 'yin_cnn':
            self.mark_detector = MarkDetector() 
            self.eular_estimator = EulerAngles(img_shape=(self.img_size, self.img_size), landmark_format='68pts' ) 
        elif self.lmk_model == 'dlib':
            raise ValueError('dlib not ready')
        elif self.lmk_model == 'PFLD':
            import torch
            import torchvision.transforms.transforms as transforms
            self.device = torch.device("cpu")
            self.pfld = PFLD().to(self.device)
            checkpoint = torch.load( '/home/john23/WorkSpace/PFLD-Pytorch-Landmarks/checkpoint/model_weights/weights.pth76.tar', map_location=self.device)
            self.pfld.load_state_dict(checkpoint["pfld"], strict=False)
            self.pfld.eval()
            self.eular_estimator = EulerAngles(img_shape=(self.img_size, self.img_size), landmark_format='98pts' ) 

    def __perprocess__(self, face_img):
        if self.lmk_model == 'scrfd':
            None
        elif self.lmk_model == 'yin_cnn':
            None
        elif self.lmk_model == 'dlib':
            None
        elif self.lmk_model == 'PFLD':
            with torch.no_grad():
                face_img = cv2.resize(face_img, (self.img_size, self.img_size))
                face_img = transforms.ToTensor()(face_img)
                face_img = face_img.unsqueeze(0)
                face_img = face_img.to(self.device)
        return face_img

    def inference(self, face_img):
        face_img = self.__perprocess__(face_img)
        if self.lmk_model == 'scrfd':
            list_score, list_bbox, pred_landmarks = self.scrfd.detect(face_img)   
            pred_landmarks = np.squeeze(pred_landmarks)
            #list_bbox = np.sqeueeze(list_bbox)
        elif self.lmk_model == 'yin_cnn':
            pred_landmarks = self.mark_detector.detect_marks(face_img)
            pred_landmarks *= self.img_size
        elif self.lmk_model == 'dlib':
            None
        elif self.lmk_model == 'PFLD':
            with torch.no_grad():
                featrues, pred_landmarks = self.pfld(face_img)
                pred_landmarks = pred_landmarks.cpu().reshape(98,2).numpy()
                pred_landmarks = (pred_landmarks*self.img_size).astype(np.float32) 
        return pred_landmarks

    @staticmethod
    def draw_marks(image, marks, color=(255, 255, 255)):
        """Draw mark points on image"""
        for mark in marks:
            cv2.circle(image, (int(mark[0]), int(
                mark[1])), 2, color, 2, cv2.LINE_AA) 
    
    def pred_euler_angle(self, lmks):
        rvec, tvec, euler_angles, landmarks_2D = self.eular_estimator.eular_angles_from_landmarks(lmks)
        return euler_angles, landmarks_2D


def draw_marks(image, marks, color=(255, 255, 255)):
    """Draw mark points on image"""
    for mark in marks:
        cv2.circle(image, (int(mark[0]), int(mark[1])), 1, color, -1, cv2.LINE_AA) 

def load_image_bgr(fn):
    return cv2.imread(fn, cv2.IMREAD_COLOR) # in BGR

def save_jpeg( savedir, im, is_bgr = True, quality = 100):
    # grayscale or bgr, no color conversion is needed
    if len(im.shape) == 2 or is_bgr:
        cv2.imwrite(savedir, im, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    else:
        cv2.imwrite(savedir, cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality])

def resize_with_pad(image, 
                    new_shape, 
                    padding_color = (255, 255, 255)):
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image

def Get_Face_Image(bbox_list, image ):
    x1, y1, x2, y2 = bbox_list[0][:4]
    # check vaild box
    height, width = image.shape[:2]
    if x1<0: x1 = 0
    if y1<0: y1 = 0
    if x2>width-1: x2 = width-1
    if y2>height-1: y2 = height-1
    face_img = image[y1: y2, x1: x2]
    face_img_112 = resize_with_pad(face_img, (112, 112))
    face_img_112_vis = copy.deepcopy(face_img)
    return face_img_112, face_img_112_vis

def Get_Image_List(data_dir_list):
    data_list = []
    for data_dir in data_dir_list:
        for data_path in glob(f'{data_dir}/*'):
            if os.path.isdir(data_path):
                data_list.extend( glob(f'{data_path}/*') )
            else:
                if data_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    data_list.append(data_path)
    return data_list


def is_in_interval(angle, interval):
    if angle>interval[0] and angle<interval[1]:
        return True
    else:
        return False

def reduce_img_size(img, ratio=0.5):
    h = int(img.shape[0]*ratio)
    w = int(img.shape[1]*ratio)
    dim = (w,h)
    return cv2.resize(img, dim)

def _add_info(path, npimg, pyr, kps, tag, path_list, npimg_list, pyr_list, landmark2D_list, tag_list ):
    path_list.append( path )
    npimg_list.append( npimg )
    pyr_list.append( pyr )
    landmark2D_list.append( kps)
    tag_list.append( tag )


def plot_2D_curve(xxs, yys):
    for xx, yy in zip(xxs,yys):
        plt.plot( xx, yy, 'r-' )
    plt.axis([0, 6, -0.05, 0.6]) # [xmin, xmax, ymin, ymax]


