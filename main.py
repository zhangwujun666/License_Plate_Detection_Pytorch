#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
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

#
#John
import argparse
import json

from models.experimental import *
from utils.datasets import *
from models.common import *
#


os.environ["CUDA_VISIBLE_DEVICES"] = '0'


if __name__ == '__main__':

    #
    #John
    p, r, f1, mp, mr, map50, map, t0, t1 = 0, 0, 0, 0, 0, 0, 0, 0, 0
    #

    shutil.rmtree('result/plate_detection')
    shutil.rmtree('result/plate_recognition')
    shutil.rmtree('result/wrong')
    shutil.rmtree('result/correct')
    os.mkdir('result/plate_detection')
    os.mkdir('result/plate_recognition')
    os.mkdir('result/wrong')
    os.mkdir('result/correct')

    parser = argparse.ArgumentParser(description='YOLOv5 & LPR Demo')
    parser.add_argument("-images", help='image path', default='data/images_test/Testing236_3', type=str)
    parser.add_argument("-yolo-checkpoint-path", help='yolo weights', default='yolov5/runs/exp5/weights/best.pt', type=str)
    parser.add_argument("--scale", dest='scale', help="scale the iamge", default=1, type=int)
    parser.add_argument('--mini_lp', dest='mini_lp', help="Minimum face to be detected", default=(50, 15), type=int)
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     device = torch.device('cpu')

    lprnet = LPRNet(class_num=len(CHARS), dropout_rate=0)
    lprnet.to(device)
    lprnet.load_state_dict(torch.load('LPRNet/saving_ckpt_1/lprnet_epoch_001000_model.ckpt',map_location=device)['net_state_dict'])
    lprnet.eval()
    
    STN = STNet()
    STN.to(device)
    STN.load_state_dict(torch.load('LPRNet/saving_ckpt_1/stn_epoch_001000_model.ckpt',map_location=device)['net_state_dict'])
    STN.eval()
    
    print("Successful to build LPR network!")
    
    image_path = glob.glob(args.images+'/*')
    num_all = len(image_path)
    num_cor = 0
    for image in image_path:
        image_id = image.split('/')[-1].split('.')[0]
        # print(image_id)
        t1 = time.time()
        image = cv2.imread(image)
        width, height, _ = image.shape

        image = cv2.resize(image,(640,640))
        image_bk = cv2.resize(image,(640,640))
        image_bk_2 = cv2.resize(image,(640,640))
        image = np.array(image)
        image = image / 255.0
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).float()
        image = image.to(device)
        model = attempt_load(args.yolo_checkpoint_path,device)
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

        t2 = time.time()
        # print('plate detection time: ', t2-t1)
        if targets == []:
            continue
        targets = np.array(targets)
        targets = targets[np.argsort(targets[:,-1])[::-1]]
        image_crop = image_bk[int(targets[0][3]):int(targets[0][5]),int(targets[0][2]):int(targets[0][4]),:]
        cv2.rectangle(image_bk, (int(targets[0][2]), int(targets[0][3])), (int(targets[0][4]), int(targets[0][5])),(0,0,255),2)
        cv2.imwrite('result/plate_detection/' + image_id + '.png',cv2.resize(image_bk,(height,width)))

        im = cv2.resize(image_crop, (94, 24), interpolation=cv2.INTER_CUBIC)
        im = (np.transpose(np.float32(im), (2, 0, 1)) - 127.5)*0.0078125
        data = torch.from_numpy(im).float().unsqueeze(0).to(device)  # torch.Size([1, 3, 24, 94]) 
        transfer = STN(data)
        preds = lprnet(transfer)
        preds = preds.cpu().detach().numpy()  # (1, 68, 18)
        labels, pred_labels = decode(preds, CHARS)
        # print(pred_labels)
        labels = [labels[0].replace('.','')]
        print('True: ', image_id, '    Predict: ',labels[0])
        if image_id == labels[0]:
            num_cor += 1
            cv2.imwrite('result/correct/' + labels[0] + '.png',cv2.resize(image_bk_2,(height,width)))
        else:
            # print(image_id, labels[0])
            cv2.imwrite('result/wrong/' + labels[0] + '.png',cv2.resize(image_bk_2,(height,width)))
        t3 = time.time()
        # print('plate recongnition time: ', t3-t2)
        img = cv2ImgAddText(image_bk, labels[0], (targets[0][2], targets[0][3]-(targets[0][5]-targets[0][3])),textSize=24)
        # print(labels)
        transformed_img = convert_image(transfer)
        img = cv2.resize(img,(height,width))
        cv2.imwrite('result/plate_recognition/' + image_id + '.png',img)
    print('numbel of correct ', num_cor)
    print('numbel of all ', num_all)
    print('accuracy ', num_cor/num_all)
    
    # Information shows 
    
    # Compute statistics
#     jdict, stats, ap, ap_class = [], [], [], []
#     seen = 0
#     iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
#     nl = len(labels)
# #     tcls = labels[:,0].tolist() if nl else []  # target class
#     niou = iouv.numel()
#     stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), labels))
        
#     stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
#     if len(stats) and stats[0].any():
#         p, r, ap, f1, ap_class = ap_per_class(*stats)
#         p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
#         mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
#         nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
#     else:
#         nt = torch.zeros(1)

    # Print results
    # John 13/10/2020
    # Visualize in the Tensorborad
#     jdict, stats, ap, ap_class = [], [], [], []
#     stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
#     if len(stats) and stats[0].any():
#         p, r, ap, f1, ap_class = ap_per_class(*stats)
#         p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
#         mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
#         nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
#     else:
#         nt = torch.zeros(1)
    print('precision:', p, 'recall:', r, )
    print('information:', parser)
    print('args', args)
    print('lprnet', lprnet)
    print('STN', STN)
    print('targets', targets)
    
    
    #
    plt.figure(figsize=(10, 6))
    sns.barplot(x=['correct', 'wrong'], y=[num_cor, num_all - num_cor])
    sns.despine(bottom=True)
    # plt.ylabel('Numbel Plate Licence o')
    plt.savefig('test_result.png',dpi=300)