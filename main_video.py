# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import sys
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append('./LPRNet')
sys.path.append('./MTCNN')
sys.path.append('./yolov5')
from LPRNet_Test import *
from yolov5.test import *
import numpy as np
import argparse
import torch
import torch.nn as nn
import time
import cv2
import glob
import shutil
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='YOLOv5 & LPR Demo')
    parser.add_argument("-video", help='video path', default='data/video/1597016617157256.mp4', type=str)
    parser.add_argument("-price", help='price pre minute', default=1.0, type=float)
    parser.add_argument("-yolo-checkpoint-path", help='yolo weights', default='yolov5/runs/exp6/weights/best.pt', type=str)
    parser.add_argument("--scale", dest='scale', help="scale the iamge", default=1, type=int)
    parser.add_argument('--mini_lp', dest='mini_lp', help="Minimum face to be detected", default=(50, 15), type=int)
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lprnet = LPRNet(class_num=len(CHARS), dropout_rate=0)
    lprnet.to(device)
    lprnet.load_state_dict(torch.load('LPRNet/saving_ckpt/lprnet_epoch_001000_model.ckpt',map_location=device)['net_state_dict'])
    lprnet.eval()
    
    STN = STNet()
    STN.to(device)
    STN.load_state_dict(torch.load('LPRNet/saving_ckpt/stn_epoch_001000_model.ckpt',map_location=device)['net_state_dict'])
    STN.eval()
    
    print("Successful to build LPR network!")
    
    cap = cv2.VideoCapture(args.video)
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver)  < 3 :
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    else :
        fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(fps)
    detect = []
    LABELS = {}
    k = 0
    while cap.isOpened():
        k += 1
        print(len(detect))
        ret, frame = cap.read()
        if ret == False:
            break
        
        image = frame

        image = cv2.resize(image,(640,640))
        image_bk = cv2.resize(image,(640,640))
        image = np.array(image)
        image = image / 255.0
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).float()
        image = image.to(device)
        model = attempt_load(args.yolo_checkpoint_path,map_location=device)
        model.eval()
        inf_out, train_out = model(image, augment=False)
        output = non_max_suppression(inf_out, conf_thres=0.1, iou_thres=0.6)
        if isinstance(output, torch.Tensor):
            output = output.cpu().numpy()
        targets = []
        for i, o in enumerate(output):
            if o is not None:
                for pred in o:
                    box = pred[:4]
                    w = (box[2] - box[0]) 
                    h = (box[3] - box[1])
                    x1 = box[0] 
                    y1 = box[1]
                    x2 = box[0] + w
                    y2 = box[1] + h
                    conf = pred[4]
                    cls = int(pred[5])

                    targets.append([i, cls, int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item()), conf.item()])

        if targets == []:
            detect.append('0')
            continue
        targets = np.array(targets)
        targets = targets[np.argsort(targets[:,-1])[::-1]]
        image_crop = image_bk[int(targets[0][3]):int(targets[0][5]),int(targets[0][2]):int(targets[0][4]),:]
        cv2.rectangle(image_bk, (int(targets[0][2]), int(targets[0][3])), (int(targets[0][4]), int(targets[0][5])),(0,0,255),2)

        try:
            im = cv2.resize(image_crop, (94, 24), interpolation=cv2.INTER_CUBIC)
        except :
            detect.append('0')
            continue
        im = (np.transpose(np.float32(im), (2, 0, 1)) - 127.5)*0.0078125
        data = torch.from_numpy(im).float().unsqueeze(0).to(device)  # torch.Size([1, 3, 24, 94]) 
        transfer = STN(data)
        preds = lprnet(transfer)
        preds = preds.cpu().detach().numpy()  # (1, 68, 18)
        labels, pred_labels = decode(preds, CHARS)
        labels = [labels[0].replace('.','')]
        # print(labels[0])
        if labels[0] not in LABELS.keys():
            LABELS[labels[0]] = 0
        else:
            LABELS[labels[0]] += 1
        detect.append('1')

    LABELS = sorted(LABELS.items(),key = lambda x:x[1],reverse = True)
    detect = np.array(detect)
    detect_start = 0
    detect_end = 0
    detect_len = len(detect)
    for i,d in enumerate(detect):
        if d == '1':
            detect_start = i
            break
    for i,d in enumerate(detect[::-1]):
        if d == '1':
            detect_end = detect_len - i
            break
    detect_time = (detect_end-detect_start)/fps
    price = args.price * detect_time
    print('Time: ', detect_time)
    print('The Plate License: ', LABELS[0][0])
    print('Place pay {:.2f}'.format(price))
    cap.release()