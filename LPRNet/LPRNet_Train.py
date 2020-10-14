#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import os
from model.LPRNET import LPRNet, CHARS
from model.STN import STNet
from data.load_data import LPRDataLoader, collate_fn
from Evaluation import eval, decode
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import argparse
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def sparse_tuple_for_ctc(T_length, lengths):
    input_lengths = []
    target_lengths = []

    for ch in lengths:
        input_lengths.append(T_length)
        target_lengths.append(ch)

    return tuple(input_lengths), tuple(target_lengths)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LPR Training')
    # parser.add_argument('--img_size', default=(94, 24), help='the image size')
    parser.add_argument('--img_size', default=(94*1, 24*1), help='the image size')
    parser.add_argument('--img_dirs_train', default="data/plate_train", help='the training images path')
    parser.add_argument('--img_dirs_val', default="./data/validation", help='the validation images path')
    parser.add_argument('--dropout_rate', default=0.5, help='dropout rate.')
    parser.add_argument('--epoch', type=int, default=10000, help='number of epoches for training')
    parser.add_argument('--batch_size', default=24*4, help='batch size')
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    lprnet = LPRNet(class_num=len(CHARS), dropout_rate=args.dropout_rate)
    lprnet.to(device)
    lprnet.load_state_dict(torch.load('LPRNet/weights/LPRNet_model_Init.pth', map_location=lambda storage, loc: storage))
    # lprnet.load_state_dict(torch.load('/media/cbpm2016/D/liaolong/alpr/License_Plate_Detection_Pytorch/saving_ckpt/lprnet_Iter_007200_model.ckpt')['net_state_dict'])
    print("LPRNet loaded")
    
    STN = STNet()
    STN.to(device)
    STN.load_state_dict(torch.load('LPRNet/weights/STN_model_Init.pth', map_location=lambda storage, loc: storage))
    # STN.load_state_dict(torch.load('/media/cbpm2016/D/liaolong/alpr/License_Plate_Detection_Pytorch/saving_ckpt/stn_Iter_007200_model.ckpt')['net_state_dict'])
    print("STN loaded")
    
    dataset = LPRDataLoader([args.img_dirs_train], args.img_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    print('training dataset loaded with length : {}'.format(len(dataset)))
    # print('validation dataset loaded with length : {}'.format(len(dataset['val'])))
    
    # define optimizer & loss
    optimizer = torch.optim.Adam([{'params': STN.parameters(), 'weight_decay': 2e-5},
                                  {'params': lprnet.parameters()}])
    # optimizer = torch.optim.SGD([{'params': STN.parameters()},
    #                               {'params': lprnet.parameters()}], lr=0.001,weight_decay=5e-4, momentum=0.9)
    ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean') # reduction: 'none' | 'mean' | 'sum'
#     ctc_loss = nn.CTCLoss()
    ## save logging and weights
    train_logging_file = 'train_logging.txt'
    validation_logging_file = 'validation_logging.txt'
    save_dir = 'LPRNet/saving_ckpt'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    start_time = time.time()
    total_iters = 0
    best_acc = 0.0
    T_length = 18 # args.lpr_max_len
    print('training kicked off..')
    print('-' * 10) 
    cur_epoch = 0
    for epoch in range(cur_epoch,args.epoch):
        # train model
        lprnet.train()
        STN.train()
        since = time.time()
        for imgs, labels, lengths in dataloader:   # img: torch.Size([2, 3, 24, 94])  # labels: torch.Size([14]) # lengths: [7, 7] (list)
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                transfer = STN(imgs)
                logits = lprnet(transfer)  # torch.Size([batch_size, CHARS length, output length ])
                log_probs = logits.permute(2, 0, 1) # for ctc loss: length of output x batch x length of chars
                log_probs = log_probs.log_softmax(2).requires_grad_()       
                input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths) # convert to tuple with length as batch_size 
                # print(log_probs.shape, labels.shape, input_lengths, target_lengths)
                loss = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
                
                loss.backward()
                optimizer.step()
                
                total_iters += 1
                # print train information
                if total_iters % 100 == 0:
                    # current training accuracy             
                    preds = logits.cpu().detach().numpy()  # (batch size, 68, 18)
                    _, pred_labels = decode(preds, CHARS)  # list of predict output
                    total = preds.shape[0]
                    start = 0
                    TP = 0
                    for i, length in enumerate(lengths):
                        label = labels[start:start+length]
                        start += length
                        pred = [_ for _ in pred_labels[i] if _ != 0]
                        pred = np.array(pred)
                        if np.array_equal(pred, label.cpu().numpy()):
                            TP += 1
                        # print(len(pred),len(label))
                    time_cur = (time.time() - since) / 100
                    since = time.time()
                    
                    for p in  optimizer.param_groups:
                        lr = p['lr']
                    print("Epoch {}/{}, Iters: {:0>6d}, loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}"
                          .format(epoch, args.epoch-1, total_iters, loss.item(), TP/total, time_cur, lr))
                    with open(train_logging_file, 'a') as f:
                        f.write("Epoch {}/{}, Iters: {:0>6d}, loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}"
                          .format(epoch, args.epoch-1, total_iters, loss.item(), TP/total, time_cur, lr)+'\n')
                    f.close()
                    
                    # save model
            if epoch % 100 == 0:

                torch.save({
                    'epoch': epoch,
                    'net_state_dict': lprnet.state_dict()},
                    os.path.join(save_dir, 'lprnet_epoch_%06d_model.ckpt' % epoch))
                
                torch.save({
                    'epoch': epoch,
                    'net_state_dict': STN.state_dict()},
                    os.path.join(save_dir, 'stn_epoch_%06d_model.ckpt' % epoch))
                                
    time_elapsed = time.time() - start_time  
    print('Finally Best Accuracy: {:.4f} in iters: {}'.format(best_acc, best_iters))
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
