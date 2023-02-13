from argparse import ArgumentParser

import cv2
import copy

import os
from os.path import basename, dirname, splitext
from tqdm import tqdm
from glob import glob
import sys
import numpy as np
import torch
import torchvision.transforms.transforms as transforms
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
            self.scrfd = SCRFD( fn_model='/home/john23/WorkSpace/FacePython/det_scrfd/model/scrfd_320.onnx', thresh_score=0.7, thresh_nms=0.4, gpu_idx=0) 
        elif fd_model == 'centeface':
            self.det_centerface = CenterFace('/home/john23/WorkSpace/FacePython/alg_align/model/centerface_model.onnx', thresh_score=0.7, thresh_nms=0.3, gpu_idx=0)
        elif fd_model == 'blazeface':
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
    def __init__(self, lmk_model, lmk_model_list=['scrfd', 'yin_cnn', 'dlib', 'PFLD'] ):
        assert lmk_model in lmk_model_list
        self.img_size = 112
        self.lmk_model = lmk_model
        if self.lmk_model == 'scrfd':
            self.scrfd = SCRFD( fn_model='/home/john23/WorkSpace/FacePython/det_scrfd/model/scrfd_320.onnx', thresh_score=0.7, thresh_nms=0.4, gpu_idx=0) 
            self.eular_estimator = EulerAngles(img_shape=(self.img_size, self.img_size), landmark_format='5pts' ) 
        elif self.lmk_model == 'yin_cnn':
            self.mark_detector = MarkDetector() 
            self.eular_estimator = EulerAngles(img_shape=(self.img_size, self.img_size), landmark_format='68pts' ) 
        elif self.lmk_model == 'dlib':
            raise ValueError('dlib not ready')
        elif self.lmk_model == 'PFLD':
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

def Get_Image_List(data_dir):
    data_list = []
    for data_path in glob(f'{data_dir}/*'):
        if os.path.isdir(data_path):
            data_list.extend( glob(f'{data_path}/*') )
        else:
            if data_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                data_list.append()
    return data_list


def Extract_from_list(path_list, npimg_list, pyr_list,landmark2D_list, tag_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for path, npimg, pyr, kps, tag in tqdm(zip(path_list, npimg_list, pyr_list, landmark2D_list, tag_list)):
        bname = splitext(basename(path))[0]  
        img = load_image_bgr( path )
        save_jpeg(f'{output_dir}/{tag}_p{pyr[0]:.2f}_y{pyr[1]:.2f}_r{pyr[2]:.2f}-{bname}.jpg', img)
        #draw_marks(npimg, kps )
        #save_jpeg(f'{output_dir}/{tag}-p{pyr[0]:.1f}-y{pyr[1]:.1f}-{bname}.jpg', npimg)


def _add_info(path, npimg, pyr, kps, tag, path_list, npimg_list, pyr_list, landmark2D_list, tag_list ):
    path_list.append( path )
    npimg_list.append( npimg )
    pyr_list.append( pyr )
    landmark2D_list.append( kps)
    tag_list.append( tag )

#def _translate2globalscale(landmarks_2D, bbox_list[0] ):
    #landmarks_2D = landmarks_2D/112.0 * 


def Qualitycheck_POSE( Valid_dir, inValid_dir, fd_model, lmk_model, pitch_threshold=(20,-20), yaw_threshold=25, visualize=False):
    facedetector = FaceDetections(fd_model)
    landmarkdetector = LandmarkDetections(lmk_model)
    # Get img list
    valid_file_list = Get_Image_List(Valid_dir)
    invalid_file_list = Get_Image_List(inValid_dir)
    TP_count = FN_count = FP_count = TN_count = vbbox_count = ivbbox_count =  0
    multiface_count = 0
    valid_pyr_list = []
    invalid_pyr_list = []
    path_list, npimg_list, pyr_list, landmark2D_list, tag_list = [], [], [], [], []
    
    # Valid Images
    for filename in tqdm(valid_file_list[:200] ):
        image = load_image_bgr(filename)
        bbox_list = facedetector.inference(image)
        if len(bbox_list)==1:  
            if lmk_model == 'scrfd':
                landmark = landmarkdetector.inference(image)
            else:
                face_img_112, face_img_112_vis = Get_Face_Image(bbox_list, image)
                landmark = landmarkdetector.inference(face_img_112)
            if not landmark.tolist(): continue
            euler_angles_pyr, landmarks_2D = landmarkdetector.pred_euler_angle(landmark)
            valid_pyr_list.append(euler_angles_pyr)
            print( bbox_list )
            #_translate2globalscale(landmarks_2D, bbox_list[0] )
            
            if pitch_threshold[0]>euler_angles_pyr[0] and euler_angles_pyr[0]>pitch_threshold[1] and abs(euler_angles_pyr[1])<yaw_threshold:
                TP_count+=1
            else:
                FN_count+=1
                _add_info(filename, face_img_112, euler_angles_pyr, landmarks_2D, 'FN', path_list, npimg_list, pyr_list, landmark2D_list, tag_list )
            vbbox_count += 1 
        elif len(bbox_list)>1:
            multiface_count+=1
            
    # InValid Images
    for filename in tqdm(invalid_file_list[:160] ):
        image = load_image_bgr(filename)
        bbox_list = facedetector.inference(image)
        if len(bbox_list)==1:  
            if lmk_model == 'scrfd':
                landmark = landmarkdetector.inference(image)
            else:
                face_img_112, face_img_112_vis = Get_Face_Image(bbox_list, image)
                landmark = landmarkdetector.inference(face_img_112)
            if not landmark.tolist(): continue
            euler_angles_pyr, landmarks_2D = landmarkdetector.pred_euler_angle(landmark)
            invalid_pyr_list.append(euler_angles_pyr)

            if pitch_threshold[0]>euler_angles_pyr[0] and euler_angles_pyr[0]>pitch_threshold[1] and abs(euler_angles_pyr[1])<yaw_threshold:
                FP_count+=1
                _add_info(filename, face_img_112, euler_angles_pyr, landmarks_2D, 'FP', path_list, npimg_list, pyr_list, landmark2D_list, tag_list )
            else:
                TN_count+=1
                _add_info(filename, face_img_112, euler_angles_pyr, landmarks_2D, 'TN', path_list, npimg_list, pyr_list, landmark2D_list, tag_list )
            ivbbox_count += 1
        elif len(bbox_list)>1:
            multiface_count+=1
        
    with open( f'valid_pyr_list_{fd_model}_{lmk_model}.pkl', 'wb') as vf:
        pickle.dump( valid_pyr_list, vf)
    with open( f'invalid_pyr_list_{fd_model}_{lmk_model}.pkl', 'wb') as ivf:
        pickle.dump( invalid_pyr_list, ivf)

    Extract_from_list(path_list, npimg_list, pyr_list,landmark2D_list, tag_list, args.out )
    Total_amount = len(valid_file_list) + len(invalid_file_list)
    Total_amount_fb = vbbox_count + ivbbox_count
    print( 'Multiface amount:', multiface_count)
    print( f'Total amount:{Total_amount}, Total_amount_facebox:{Total_amount_fb} ' )
    print( f'vbbox_count/valid_file amount:{vbbox_count}/{len(valid_file_list)}, ivbbox_count/invalid_file amount:{ivbbox_count}/{len(invalid_file_list)} ' )
    print( f'TP_count:{TP_count}, FN_count:{FN_count}, FP_count:{FP_count}, TN_count:{TN_count}' )
    print( f'Precision:{(TP_count/(TP_count+FP_count)):.4f}, Recal:{(TP_count/(TP_count+FN_count)):.4f}')
    return 


def Threshold_Testing(fd_model,lmk_model ):
    with open( f'valid_pyr_list_{fd_model}_{lmk_model}.pkl', 'rb') as vf:
        valid_pyr_list = pickle.load(vf)
    with open( f'invalid_pyr_list_{fd_model}_{lmk_model}.pkl', 'rb') as ivf:
        invalid_pyr_list = pickle.load(ivf)
    #Total_amount = len(valid_file_list) + len(invalid_file_list)

    pitch_threshold_list = [(10,-10),(15,-15),(20,-20), (25,-25), (30,-30)]
    yaw_threshold_list = [10,15,20,25,30]

    print( f'vbbox_count:{len(valid_pyr_list)}, ivbbox_count:{len(invalid_pyr_list)}' )
    for pitch_threshold, yaw_threshold in zip(pitch_threshold_list, yaw_threshold_list):

        TP_count = FN_count = FP_count = TN_count = vbbox_count = ivbbox_count =  0
        for euler_angles_pyr in valid_pyr_list:
            if pitch_threshold[0]>euler_angles_pyr[0] and euler_angles_pyr[0]>pitch_threshold[1] and abs(euler_angles_pyr[1])<yaw_threshold:
                    TP_count+=1
            else:
                FN_count+=1
        for euler_angles_pyr in invalid_pyr_list:
            if pitch_threshold[0]>euler_angles_pyr[0] and euler_angles_pyr[0]>pitch_threshold[1] and abs(euler_angles_pyr[1])<yaw_threshold:
                FP_count+=1
            else:
                TN_count+=1
        print( f'==== Pitch_threshold{pitch_threshold}, Yaw_threshold{yaw_threshold} ====')
        print( f'TP_count:{TP_count}, FN_count:{FN_count}, FP_count:{FP_count}, TN_count:{TN_count}' )
        print( f'Precision:{(TP_count/(TP_count+FP_count)):.4f}, Recal:{(TP_count/(TP_count+FN_count)):.4f}')
        print()

def plot_2D_curve(xxs, yys):
    for xx, yy in zip(xxs,yys):
        plt.plot( xx, yy, 'r-' )
    plt.axis([0, 6, -0.05, 0.6]) # [xmin, xmax, ymin, ymax]


if __name__ == '__main__':
    fd_model_list = ['scrfd', 'centeface', 'blazeface', 'opencv']
    lmk_model_list = ['scrfd', 'yin_cnn', 'PFLD']
    
    parser = ArgumentParser()
    parser.add_argument("--img", type=str, default=None)
    parser.add_argument("--src_dir", type=str, default=None)
    parser.add_argument("-o","--out", type=str, default='') 
    parser.add_argument("--fd_model", type=str, default=None, help=fd_model_list )
    parser.add_argument("--lmk_model", type=str, default=None, help=lmk_model_list)    
    parser.add_argument("--iter", action='store_true')
    args = parser.parse_args()

    if args.iter:
        Threshold_Testing( args.fd_model, args.lmk_model)
        sys.exit(1)
    Qualitycheck_POSE( '/media/hermes/datashare/AWS-DataCollection/NORMAL', \
                    '/media/hermes/datashare/AWS-DataCollection/EXTREME_ANGLE/', \
                    args.fd_model, args.lmk_model, pitch_threshold=(20,-20), yaw_threshold=20)
    sys.exit(1)
    # Debug Seesion#
    # create save dir folder
    facedetector = FaceDetections(args.fd_model)
    landmarkdetector = LandmarkDetections(args.lmk_model)
    image = load_image_bgr(args.img)
    bbox_list = facedetector.inference(image)
    print( bbox_list )

    #facedetector = FaceDetections('centeface')
    #bbox_list = facedetector.inference(image)
    #print( bbox_list )

    #facedetector = FaceDetections('blazeface')
    #bbox_list = facedetector.inference(image)
    #print( bbox_list )

    #facedetector = FaceDetections('scrfd')
    #bbox_list = facedetector.inference(image)
    #print( bbox_list )
    if bbox_list:        
        #mark_detector.draw_marks(im, list_kps )
        #cv2.imwrite( 'marks_out.jpg', im)
        
        x1, y1, x2, y2 = bbox_list[0][:4]
        face_img = image[y1: y2, x1: x2]
        face_img_112 = resize_with_pad(face_img, (112, 112))
        face_img_112_tmp1 = copy.deepcopy(face_img_112)
        face_img_112_tmp2 = copy.deepcopy(face_img_112)
        face_img_112_tmp3 = copy.deepcopy(face_img_112)

        print('scrfd' )
        landmarkdetector = LandmarkDetections('scrfd')
        landmark = landmarkdetector.inference(face_img_112)
        #landmark = landmark[0]
        euler_angles, landmarks_2D = landmarkdetector.pred_euler_angle(landmark)
        print( 'euler_angles pyr', euler_angles )
        print( landmark.shape)
        landmarkdetector.draw_marks(face_img_112_tmp1, landmarks_2D )
        save_jpeg('mark_out_scrfd.jpg', face_img_112_tmp1)

        print('yin_cnn' )
        landmarkdetector = LandmarkDetections('yin_cnn')
        landmark = landmarkdetector.inference(face_img_112)
        euler_angles, landmarks_2D = landmarkdetector.pred_euler_angle(landmark)
        print( 'euler_angles pyr', euler_angles )
        print( landmark.shape)
        landmarkdetector.draw_marks(face_img_112_tmp3, landmarks_2D )
        save_jpeg('mark_out_yin.jpg', face_img_112_tmp3)

        print('PFLD' )
        landmarkdetector = LandmarkDetections('PFLD')
        landmark = landmarkdetector.inference(face_img_112)
        euler_angles, landmarks_2D = landmarkdetector.pred_euler_angle(landmark)
        print( 'euler_angles pyr', euler_angles )
        print( landmark.shape)
        landmarkdetector.draw_marks(face_img_112_tmp2, landmarks_2D )
        save_jpeg('mark_out_PFLD.jpg', face_img_112_tmp2)


    sys.exit(1)