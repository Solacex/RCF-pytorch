import os, sys
import numpy as np
from PIL import Image
import cv2
import shutil
import argparse
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_loader import BSDS_RCFLoader
from models import RCF
from functions import  cross_entropy_loss_RCF, SGD_caffe
from torch.utils.data import DataLoader, sampler
from utils import Logger, Averagvalue, save_checkpoint, load_vgg16pretrain
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname
from data.dataset import *
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--batch_size', default=1, type=int, metavar='BT',
                    help='batch size')
# =============== optimizer
parser.add_argument('--lr', '--learning_rate', default=1e-8, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float,
                    metavar='W', help='default weight decay')
parser.add_argument('--stepsize', default=7, type=int,
                    metavar='SS', help='learning rate step size')
parser.add_argument('--gamma', '--gm', default=0.1, type=float,
                    help='learning rate decay parameter: Gamma')
parser.add_argument('--maxepoch', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--itersize', default=10, type=int,
                    metavar='IS', help='iter size')
# =============== misc
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print_freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--gpu', default='0', type=str,
                    help='GPU ID')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--tmp', help='tmp folder', default='tmp/RCF')
# ================ dataset
parser.add_argument('--dataset', help='root folder of dataset', default='cityscapes')

args = parser.parse_args()


def main():
    args.cuda = True
#    test_loader = training_dataset()
    test_loader = get_hed_test_dataset(1)
    #with open('/home/guangrui/segmentation_DA/dataset/cityscapes_list/val.txt', 'r') as f:
#        test_list = f.readlines()
 #   test_list = [split(i.rstrip())[1] for i in test_list]
  #  assert len(test_list) == len(test_loader), "%d vs %d" % (len(test_list), len(test_loader))

    model = RCF()
    model.cuda()

    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}'"
          .format(args.resume))

    multiscale_test(model, test_loader, 
            save_dir =args.tmp)



def multiscale_test(model, test_loader, save_dir):
    model.eval()
    if not isdir(save_dir):
        os.makedirs(save_dir)
    scale = [0.5, 1, 1.5]
    for idx, batch in tqdm(enumerate(test_loader)):
        image, name = batch
        name = name[0]
        print(name)
        dir_name = name.split('/')[0]
        name = name.split('/')[1]
#        image = image[0]
 #       image = image.squeeze()
  #      image_in = image.numpy().transpose((1,2,0))
        _, _, H, W = image.shape
        multi_fuse = torch.zeros(H, W).cuda()
        for k in range(0, len(scale)):
#            im_ = cv2.resize(image_in, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
            image = F.interpolate(image, scale_factor=scale[k], mode='bilinear', align_corners=True)
#            im_ = im_.transpose((2,0,1))
            with torch.no_grad():
                results = model(image.cuda())
#                results = model(torch.unsqueeze(torch.from_numpy(im_).cuda().float(), 0))
            results = results[-1]
#            result = torch.squeeze(results[-1].detach()).cpu().numpy()
 #           fuse = cv2.resize(result, (W, H), interpolation=cv2.INTER_LINEAR)
            fuse = F.interpolate(results, size=(H,W), mode='bilinear', align_corners=True)
            multi_fuse += fuse.squeeze()
        multi_fuse = multi_fuse / len(scale)
        multi_fuse = multi_fuse.cpu().numpy()
        ### rescale trick suggested by jiangjiang
#        multi_fuse = (multi_fuse - multi_fuse.min()) / (multi_fuse.max() - multi_fuse.min())
#        result_out = Image.fromarray((multi_fuse)).convert('L')
        name = name.replace('leftImg8bit', 'gtFine_edge')
        name = name.replace('png', 'npy')
#        result_out = Image.fromarray(((1-multi_fuse) * 255).astype(np.uint8))
        print(join(save_dir,dir_name, name))
        np.save(join(save_dir,dir_name, name), multi_fuse)
#        result_out.save(join(save_dir,dir_name, name))
if __name__ == '__main__':
    main()

