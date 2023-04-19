
'''
python inference.py --img '/media/hermes/datashare/AWS-DataCollection/Consensus_NORMAL/20221121102043_04_fr_0ss.jpg'
python inference.py --real_data --vis
'''

from runner import *
from runner import _add_info


def inference_single_image(fd_model, lmk_model, filename):
    '''
    fd_model: choose one out of ['scrfd', 'centeface', 'blazeface', 'opencv']
    lmk_model: choose one out of ['scrfd', 'yin_cnn', 'PFLD']
    '''
    facedetector = FaceDetections(fd_model)
    landmarkdetector = LandmarkDetections(lmk_model)
    image = load_image_bgr(filename)
    # face detection
    bbox_list = facedetector.inference(image)
    # landmark detection
    if lmk_model=='scrfd': # scrfd output face detection and landmark detection at once
        landmark = landmarkdetector.inference(image)
    else:
        face_img_112, face_img_112_vis = Get_Face_Image(bbox_list, image)
        landmark = landmarkdetector.inference(face_img_112)
    # check if landmark is empty or not
    if not landmark.tolist():
        print( 'landmark not found.')
    else:
        euler_angles_pyr, landmarks_2D = landmarkdetector.pred_euler_angle(landmark)
        print(euler_angles_pyr)


def Qualitycheck_POSE( Valid_dir_list, inValid_dir_list, fd_model, lmk_model, pitch_threshold=(25,-25), yaw_threshold=25):
    facedetector = FaceDetections(fd_model)
    landmarkdetector = LandmarkDetections(lmk_model)

    # variables
    TP_count = FN_count = FP_count = TN_count = vbbox_count = ivbbox_count =  0
    multiface_count = 0
    valid_pyr_list = []
    invalid_pyr_list = []
    path_list, npimg_list, pyr_list, landmark2D_list, tag_list = [], [], [], [], []
    #

    # Normal Images
    for filename in tqdm(Valid_dir_list, desc='Valid_dir_list' ):
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
            #print( bbox_list )
            #_translate2globalscale(landmarks_2D, bbox_list[0] )
            
            if pitch_threshold[0]>euler_angles_pyr[0] and euler_angles_pyr[0]>pitch_threshold[1] and abs(euler_angles_pyr[1])<yaw_threshold:
                TP_count+=1
                #_add_info(filename, face_img_112, euler_angles_pyr, landmarks_2D, 'TP', path_list, npimg_list, pyr_list, landmark2D_list, tag_list )
            else:
                FN_count+=1
                _add_info(filename, face_img_112, euler_angles_pyr, landmarks_2D, 'FN', path_list, npimg_list, pyr_list, landmark2D_list, tag_list )
            vbbox_count += 1 
        elif len(bbox_list)>1:
            multiface_count+=1
            
    # Extreme angle Images
    for filename in tqdm(inValid_dir_list, desc='inValid_dir_list' ):
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

    # visualize result
    if args.vis:
        Extract_and_Save_from_list(path_list, npimg_list, pyr_list,landmark2D_list, tag_list, args.out )

    Total_amount = len(valid_file_list) + len(invalid_file_list)
    Total_amount_fb = vbbox_count + ivbbox_count
    print( 'Multiface amount:', multiface_count)
    print( f'Total amount:{Total_amount}, Total_amount_facebox:{Total_amount_fb} ' )
    print( f'vbbox_count/valid_file amount:{vbbox_count}/{len(valid_file_list)}, ivbbox_count/invalid_file amount:{ivbbox_count}/{len(invalid_file_list)} ' )
    print( f'TP_count:{TP_count}, FN_count:{FN_count}, FP_count:{FP_count}, TN_count:{TN_count}' )
    print( f'Precision:{(TP_count/(TP_count+FP_count)):.4f}, Recal:{(TP_count/(TP_count+FN_count)):.4f}')
    return 


def Extract_and_Save_from_list(path_list, npimg_list, pyr_list,landmark2D_list, tag_list, output_dir):
    print( f'Extract images to {output_dir} ...')
    os.makedirs(output_dir, exist_ok=True)
    #yaw_intervals = [(-90,-45), (-45,-30), (-30,-15), (-15,0), (0,15), (15,30), (30,45), (45,90)]
    #pitch_intervals = [(-90,-45), (-45,-30), (-30,-15), (-15,0), (0,15), (15,30), (30,45), (45,90)]
    intervals = [(-15,15), (-25,25), (-30,30), (-40,40), (-90,90)]
    progress = tqdm(total=len(path_list), desc='Extract_and_Save_from_list')
    for path, npimg, pyr, kps, tag in zip(path_list, npimg_list, pyr_list, landmark2D_list, tag_list):
        progress.update(1)
        bname = splitext(basename(path))[0]  
        img = load_image_bgr( path )
        img = reduce_img_size(img, ratio=0.3)
        for interval in intervals:
            if is_in_interval(pyr[1], interval) and is_in_interval(pyr[0], interval):
                dir_name = os.path.join(output_dir, f'yaw{interval[0]}_{interval[1]}_pitch{interval[0]}_{interval[1]}')
                if not os.path.exists( dir_name ):
                    os.makedirs( dir_name )
                save_jpeg(f'{dir_name}/{tag}_p{pyr[0]:.2f}_y{pyr[1]:.2f}_r{pyr[2]:.2f}-{bname}.jpg', img)
                break
        ''' 
        for y_interval in yaw_intervals:
            if is_in_interval(pyr[1], y_interval):
                dir_name = os.path.join(output_dir, f'yaw{y_interval[0]}_{y_interval[1]}')
                if not os.path.exists( dir_name ):
                    os.makedirs( dir_name )
                save_jpeg(f'{dir_name}/{tag}_p{pyr[0]:.2f}_y{pyr[1]:.2f}_r{pyr[2]:.2f}-{bname}.jpg', img)
        for p_interval in pitch_intervals:
            if is_in_interval(pyr[0], p_interval):
                dir_name = os.path.join(output_dir, f'pitch{p_interval[0]}_{p_interval[1]}') 
                if not os.path.exists( dir_name ):
                    os.makedirs( dir_name )
                save_jpeg(f'{dir_name}/{tag}_p{pyr[0]:.2f}_y{pyr[1]:.2f}_r{pyr[2]:.2f}-{bname}.jpg', img)
        '''
        #save_jpeg(f'{output_dir}/{tag}_p{pyr[0]:.2f}_y{pyr[1]:.2f}_r{pyr[2]:.2f}-{bname}.jpg', img)
        #draw_marks(npimg, kps )
        #save_jpeg(f'{output_dir}/{tag}-p{pyr[0]:.1f}-y{pyr[1]:.1f}-{bname}.jpg', npimg)
        


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


def extract_depth_real_data_to_list(number_of_user_id=10):
    '''
    real data source: '/media/hermes/datashare/che/dataset_depth/real/source/AccessBank'
    '''
    data_source = '/media/hermes/datashare/che/dataset_depth/real/source/AccessBank/'
    collected_data_list = []
    user_id_list = glob(f'{data_source}/*')
    for user_id in tqdm(user_id_list[:number_of_user_id]):
        session_id_list = glob(f'{user_id}/depthVerified/*')
        for session_id in session_id_list:
            collected_data_list.extend( glob(f'{session_id}/*.jpg'))
    print( 'Total number of files:',len(collected_data_list) )
    return collected_data_list




if __name__ == '__main__':
    fd_model_list = ['scrfd', 'centeface', 'blazeface', 'opencv']
    lmk_model_list = ['scrfd', 'yin_cnn', 'PFLD']
    
    parser = ArgumentParser()
    parser.add_argument("--img", type=str, default='')
    parser.add_argument("--VAL_src_dir", nargs='+',  default=[]) #  '/media/hermes/datashare/AWS-DataCollection/NORMAL'
    parser.add_argument("--INVAL_src_dir", nargs='+',  default=[])  # '/media/hermes/datashare/AWS-DataCollection/EXTREME_ANGLE/'
    parser.add_argument("-o","--out", type=str, default='inference_result')  
    parser.add_argument("--fd_model", type=str, default='blazeface', help=fd_model_list ) # ['scrfd', 'centeface', 'blazeface', 'opencv']
    parser.add_argument("--lmk_model", type=str, default='PFLD', help=lmk_model_list)  # ['scrfd', 'yin_cnn', 'PFLD']
    parser.add_argument("--iter", action='store_true')
    parser.add_argument("--vis", action='store_true')
    parser.add_argument("--real_data", action='store_true')
    args = parser.parse_args()
    number_of_user_id=250


    if args.img:
        inference_single_image( args.fd_model, args.lmk_model, filename=args.img)
        sys.exit(1)

    # tesing
    if args.iter:
        Threshold_Testing( args.fd_model, args.lmk_model)
        sys.exit(1)

    # 
    # Get img list
    valid_file_list = Get_Image_List(args.VAL_src_dir)
    invalid_file_list = Get_Image_List(args.INVAL_src_dir)
    # 
    if args.real_data:
        valid_file_list = extract_depth_real_data_to_list(number_of_user_id)

    Qualitycheck_POSE( valid_file_list,  invalid_file_list, \
                    args.fd_model, args.lmk_model, pitch_threshold=(20,-20), yaw_threshold=20)
    
    '''
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
    '''