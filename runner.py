from argparse import ArgumentParser

import cv2

from mark_detector import MarkDetector
from pose_estimator import PoseEstimator
import os
from os.path import basename, dirname
from tqdm import tqdm
from glob import glob


print("OpenCV version: {}".format(cv2.__version__))

parser = ArgumentParser()
parser.add_argument("--img", type=str, default=None)
parser.add_argument("--src_dir", type=str, default=None)
parser.add_argument("-o", type=str, default=None)                 
args = parser.parse_args()

if __name__ == '__main__':

    if not os.path.exists(args.o):
        os.makedirs(args.o, exist_ok=True)
    mark_detector = MarkDetector()
    
    for img_path in glob(f'{args.src_dir}/*'):
        print( img_path)
        #frame = cv2.imread(args.img)
        frame = cv2.imread(img_path)
        bname = basename(img_path)
        width = frame.shape[1]
        height = frame.shape[0]

        # 2. Introduce a pose estimator to solve pose.
        pose_estimator = PoseEstimator(img_size=(height, width))

        # 3. Introduce a mark detector to detect landmarks.
        

        # 4. Measure the performance with a tick meter.
        tm = cv2.TickMeter()
        print( 'shape', frame.shape)

        # Step 1: Get a face from current frame.
        facebox = mark_detector.extract_cnn_facebox(frame)

        # Any face found?
        if facebox is not None:

            # Step 2: Detect landmarks. Crop and feed the face area into the
            # mark detector.
            x1, y1, x2, y2 = facebox
            face_img = frame[y1: y2, x1: x2]

            # Run the detection.
            tm.start()
            marks = mark_detector.detect_marks(face_img)
            tm.stop()

            # Convert the locations from local face area to the global image.
            marks *= (x2 - x1)
            marks[:, 0] += x1
            marks[:, 1] += y1

            # Try pose estimation with 68 points.
            pose = pose_estimator.solve_pose_by_68_points(marks)

            # All done. The best way to show the result would be drawing the
            # pose on the frame in realtime.

            # Do you want to see the pose annotation?
            pose_estimator.draw_annotation_box(
                frame, pose[0], pose[1], color=(0, 255, 0))

            # Do you want to see the head axes?
            # pose_estimator.draw_axes(frame, pose[0], pose[1])

            # Do you want to see the marks?
            # mark_detector.draw_marks(frame, marks, color=(0, 255, 0))

            # Do you want to see the facebox?
            # mark_detector.draw_box(frame, [facebox])
            print('marks', marks.shape )
            print('pose', pose)
            face_angle = pose[1]
            pitch=face_angle[1][0]
            yaw=face_angle[0][0]
            roll=face_angle[2][0]
            cv2.imwrite(f'{args.o}/p{pitch:.3f}-y{yaw:.3f}-{bname}', frame)
        else:
            cv2.imwrite(f'{args.o}/nobox-{bname}', frame)