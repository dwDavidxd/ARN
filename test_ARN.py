import yaml
import os
#import shutil
import numpy as np
import torch
#import torch.optim as optim
#import torch.nn as nn
#from torch.autograd import Variable, grad
#from torch.backends import cudnn
from utils.model import LoadModel

from utils.dataload import DatasetNPY_test
from torchvision import transforms
from torch.utils.data import DataLoader
import cv2
import argparse


def main(args):
    torch.manual_seed(args.seed)

    if not os.path.exists(args.path_output):
        os.makedirs(args.path_output)

    config_path = './config/adver.yaml'
    conf = yaml.load(open(config_path,'r'), Loader=yaml.FullLoader)
    img_size = conf['exp_setting']['img_size']
    img_depth = conf['exp_setting']['img_depth']
    batch_size = args.batch_size

    ae = LoadModel('autoencoder', conf['model']['autoencoder'], img_size, img_depth)
    ae = ae.cuda()
    ae.load_state_dict(torch.load(args.path_model))
    ae.eval()

    trans = transforms.ToTensor()

    img_dataset = DatasetNPY_test(npy_dirs=args.path_input, transform=trans)

    img_loader = DataLoader(img_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    trans_code = None

    cnt = 0

    for img in img_loader:
        # print(cnt)

        img = img.repeat(1, 3, 1, 1)

        img = (img * 2 - 1).cuda()

        fake_all = ae(img, insert_attrs=trans_code)[0]

        for n in range(len(img)):
            cnt += 1
            fake = ((fake_all[n].cpu().data.numpy()+1)/2).transpose(-2,-1,-3)

            fake = cv2.cvtColor(fake, cv2.COLOR_RGB2GRAY)

            # (.npy)
            name = str(cnt) + '.npy'
            path = os.path.join(args.path_output, name)
            np.save(path, fake)

            '''
            # (.png)
            name = str(cnt) + '.png'
            path = os.path.join(args.path_output, name)
            cv2.imwrite(path, fake * 255, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            '''

        print('Have processed ' + str(cnt) + ' examples-------------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Attack')
    parser.add_argument('--path_input', default='./adv_example/test/adv1')
    parser.add_argument('--path_output', default='./adv_example/processed/adv1')
    parser.add_argument('--path_model', default='./checkpoint/adver/2000.ae')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size')
    parser.add_argument('--seed', default=0)

    args = parser.parse_args()

    main(args)