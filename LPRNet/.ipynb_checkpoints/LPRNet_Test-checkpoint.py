#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.getcwd())
from PIL import Image, ImageDraw, ImageFont
from model.LPRNET import LPRNet, CHARS
from model.STN import STNet
import numpy as np
import argparse
import torch
import time
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def convert_image(inp):
    # convert a Tensor to numpy image
    inp = inp.squeeze(0).cpu()
    inp = inp.detach().numpy().transpose((1,2,0))
    inp = 127.5 + inp/0.0078125
    inp = inp.astype('uint8') 

    return inp

def cv2ImgAddText(img, text, pos, textColor=(255, 0, 0), textSize=12):
    if (isinstance(img, np.ndarray)):  # detect opencv format or not
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("data/NotoSansCJK-Regular.ttc", textSize, encoding="utf-8")
    draw.text(pos, text, textColor, font=fontText)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def decode(preds, CHARS):
    # greedy decode
    pred_labels = list()
    labels = list()
    for i in range(preds.shape[0]):
        pred = preds[i, :, :]
        pred_label = list()
        for j in range(pred.shape[1]):
            pred_label.append(np.argmax(pred[:, j], axis=0))
        no_repeat_blank_label = list()
        pre_c = pred_label[0]
        for c in pred_label: # dropout repeate label and blank label
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        pred_labels.append(no_repeat_blank_label)
        
    for i, label in enumerate(pred_labels):
        lb = ""
        for i in label:
            lb += CHARS[i]
        labels.append(lb)
    
    return labels, np.array(pred_labels)  

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LPR Demo')  # 00-D-112255  151-WX-694
    parser.add_argument("-image", help='image path', default='/media/cbpm2016/D/liaolong/alpr/License_Plate_Detection_Pytorch/data/plate_test/152L393.png', type=str)
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lprnet = LPRNet(class_num=len(CHARS), dropout_rate=0)
    lprnet.to(device)
    lprnet.load_state_dict(torch.load('/media/cbpm2016/D/liaolong/alpr/License_Plate_Detection_Pytorch/LPRNet/saving_ckpt/lprnet_epoch_001200_model.ckpt')['net_state_dict'])
    lprnet.eval()
    
    STN = STNet()
    STN.to(device)
    STN.load_state_dict(torch.load('/media/cbpm2016/D/liaolong/alpr/License_Plate_Detection_Pytorch/LPRNet/saving_ckpt/stn_epoch_001200_model.ckpt')['net_state_dict'])
    STN.eval()
    
    print("Successful to build network!")
    
    since = time.time()
    image = cv2.imread(args.image)
    im = cv2.resize(image, (94, 24), interpolation=cv2.INTER_CUBIC)
    im = (np.transpose(np.float32(im), (2, 0, 1)) - 127.5)*0.0078125
    data = torch.from_numpy(im).float().unsqueeze(0).to(device)  # torch.Size([1, 3, 24, 94]) 
    transfer = STN(data)
    preds = lprnet(transfer)
    preds = preds.cpu().detach().numpy()  # (1, 68, 18)
    
    labels, pred_labels = decode(preds, CHARS)
    labels = [labels[0].replace('.','')]
    print("model inference in {:2.3f} seconds".format(time.time() - since))
            
    img = cv2ImgAddText(image, labels[0], (0, 0))
    print(labels)
    
    transformed_img = convert_image(transfer)
    cv2.imwrite('test.png',img)
    # cv2.imshow('transformed', transformed_img)
    
    # cv2.imshow("test", img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    