from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import time
import math



if __name__ == '__main__':
    dir_name = "D:/Dataset/NTHU-DDD/Training_Evaluation_Dataset/Training_Dataset/"
    subject_list = os.listdir(dir_name)
    scenario_list = ["nonsleepyCombination", "sleepyCombination", "slowBlinkWithNodding", "yawning"]
    extension =".avi"

    for subject in subject_list:
        condition_list = os.listdir(dir_name + subject)
        for condition in condition_list:
            for scenario in scenario_list:
                print(dir_name+subject+"/"+condition+"/"+scenario+extension)
                frame_num = 0
                #cap = cv2.VideoCapture(dir_name + subject + condition + scenario + extension)
                cap = cv2.VideoCapture(dir_name+subject+"/"+condition+"/"+scenario+extension)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        img_raw = frame.copy()

                        # gamma correction
                        """optical_gamma = math.log(np.mean(128) / 255) / math.log(np.mean(img_raw) / 255)
                        img_raw = img_raw.astype(np.float)
                        gamma_img = 255 * ((img_raw / 255) ** optical_gamma)
                        img_raw = img_raw.astype(np.uint8)
                        gamma_img = gamma_img.astype(np.uint8)"""

                        # Contrast normalization
                        #contrast_norm_img = cv2.normalize(img_raw, None, 0, 255, cv2.NORM_MINMAX)
                        #gamma_contrast_norm_img = cv2.normalize(gamma_img, None, 0, 255, cv2.NORM_MINMAX)


                        #cv2.imshow("face", img_raw)
                        # cv2.imshow("gamma_img", gamma_img)
                        # cv2.imshow("contrast_norm", contrast_norm_img)
                        #cv2.imshow("gamma_contrast_norm_img", gamma_contrast_norm_img)

                        # save image
                        save_dir = "D:/Dataset/NTHU-DDD-normal/Training_Evaluation_Dataset/Training_Dataset/"

                        if not os.path.exists(save_dir + subject + "/" + condition + "/" + scenario + "/" + "face/"):
                            os.makedirs(save_dir + subject + "/" + condition + "/" + scenario + "/" + "face/")
                        face_path = save_dir + subject + "/" + condition + "/" + scenario + "/" + "face/" + str(
                            frame_num) + ".jpg"

                        cv2.imwrite(face_path, img_raw)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        frame_num += 1
                    else:
                        break
                cap.release()



