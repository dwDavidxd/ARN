import torch
from torch.utils.data import Dataset, DataLoader
import re
import pickle
from PIL import Image
import os
import numpy as np


def sort_key(s):
    re_digits = re.compile(r'(\d+)')
    pieces = re_digits.split(s)
    pieces[1::2] = map(int, pieces[1::2])
    return pieces


def load_variavle(filename):
    f=open(filename,'rb')
    r=pickle.load(f)
    f.close()
    return r


class DatasetIMG(Dataset):
    def __init__(self, imgnat_dirs, imgadv1_dirs, imgadv2_dirs, transform=None):
        self.imgnat_dirs = imgnat_dirs
        self.imgadv1_dirs = imgadv1_dirs
        self.imgadv2_dirs = imgadv2_dirs
        self.img_names = self.__get_imgnames__()
        self.transform = transform

    def __get_imgnames__(self):
        tmp = []
        images_name = os.listdir(self.imgnat_dirs)
        images_name.sort(key=sort_key)
        for name in images_name:
            tmp.append(os.path.join(self.imgnat_dirs, name))
        return tmp

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        imagenat_path = self.img_names[idx]
        imagenat = Image.open(imagenat_path).convert('L')
        imageadv1_path = imagenat_path.replace(self.imgnat_dirs, self.imgadv1_dirs)
        imageadv1 = Image.open(imageadv1_path).convert('L')
        imageadv2_path = imagenat_path.replace(self.imgnat_dirs, self.imgadv2_dirs)
        imageadv2 = Image.open(imageadv2_path).convert('L')

        if self.transform:
            imagenat = self.transform(imagenat)
            imageadv1 = self.transform(imageadv1)
            imageadv2 = self.transform(imageadv2)
        return imagenat, imageadv1, imageadv2


class DatasetIMG_Label(Dataset):
    def __init__(self, imgnat_dirs, imgadv1_dirs, imgadv2_dirs, label_dirs, transform=None):
        self.imgnat_dirs = imgnat_dirs
        self.imgadv1_dirs = imgadv1_dirs
        self.imgadv2_dirs = imgadv2_dirs
        self.label_dirs = label_dirs
        self.img_names = self.__get_imgnames__()
        self.label = self.__get_label__()
        self.transform = transform

    def __get_imgnames__(self):
        tmp = []
        images_name = os.listdir(self.imgnat_dirs)
        images_name.sort(key=sort_key)
        for name in images_name:
            tmp.append(os.path.join(self.imgnat_dirs, name))
        return tmp

    def __get_label__(self):
        label = load_variavle(self.label_dirs)
        label = np.array(label)
        label = torch.from_numpy(label)
        return label

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        imagenat_path = self.img_names[idx]
        imagenat = Image.open(imagenat_path).convert('L')
        imageadv1_path = imagenat_path.replace(self.imgnat_dirs, self.imgadv1_dirs)
        imageadv1 = Image.open(imageadv1_path).convert('L')
        imageadv2_path = imagenat_path.replace(self.imgnat_dirs, self.imgadv2_dirs)
        imageadv2 = Image.open(imageadv2_path).convert('L')

        label = self.label[idx]

        if self.transform:
            imagenat = self.transform(imagenat)
            imageadv1 = self.transform(imageadv1)
            imageadv2 = self.transform(imageadv2)
        return imagenat, imageadv1, imageadv2, label


class DatasetIMG_Dual_Lable(Dataset):
    def __init__(self, imgnat_dirs, imgadv_dirs, label_dirs, transform=None):
        self.imgnat_dirs = imgnat_dirs
        self.imgadv_dirs = imgadv_dirs
        self.label_dirs = label_dirs
        self.img_names = self.__get_imgnames__()
        self.label = self.__get_label__()
        self.transform = transform

    def __get_imgnames__(self):
        tmp = []
        images_name = os.listdir(self.imgnat_dirs)
        images_name.sort(key=sort_key)
        for name in images_name:
            tmp.append(os.path.join(self.imgnat_dirs, name))
        return tmp

    def __get_label__(self):
        label = load_variavle(self.label_dirs)
        label = np.array(label)
        label = torch.from_numpy(label)
        return label

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        imagenat_path = self.img_names[idx]
        imagenat = Image.open(imagenat_path).convert('L')
        imageadv_path = imagenat_path.replace(self.imgnat_dirs, self.imgadv_dirs)
        imageadv = Image.open(imageadv_path).convert('L')

        label = self.label[idx]

        if self.transform:
            imagenat = self.transform(imagenat)
            imageadv = self.transform(imageadv)

        return imagenat, imageadv, label


class DatasetIMG_test(Dataset):
    def __init__(self, img_dirs, transform=None):
        self.img_dirs = img_dirs
        self.img_names = self.__get_imgnames__()
        self.transform = transform

    def __get_imgnames__(self):
        tmp = []
        images_name = os.listdir(self.img_dirs)
        images_name.sort(key=sort_key)
        for name in images_name:
            tmp.append(os.path.join(self.img_dirs, name))
        return tmp

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        imagenat_path = self.img_names[idx]
        image = Image.open(imagenat_path).convert('L')

        if self.transform:
            image = self.transform(image)

        return image


class DatasetNPY(Dataset):

    def __init__(self, nat_dirs, adv1_dirs, adv2_dirs, transform=None):
        self.nat_dirs = nat_dirs
        self.adv1_dirs = adv1_dirs
        self.adv2_dirs = adv2_dirs
        self.npy_names = self.__get_npynames__()
        self.transform = transform

    def __get_npynames__(self):
        tmp = []
        npy_name = os.listdir(self.nat_dirs)
        npy_name.sort(key=sort_key)
        for name in npy_name:
            tmp.append(os.path.join(self.nat_dirs, name))
        return tmp

    def __len__(self):
        return len(self.npy_names)

    def __getitem__(self, idx):
        npynat_path = self.npy_names[idx]
        npynat = np.load(npynat_path)
        npynat = npynat.astype(np.float32)

        npyadv1_path   = npynat_path.replace(self.nat_dirs, self.adv1_dirs)
        npyadv1 = np.load(npyadv1_path)
        npyadv1 = npyadv1.astype(np.float32)

        npyadv2_path   = npynat_path.replace(self.nat_dirs, self.adv2_dirs)
        npyadv2 = np.load(npyadv2_path)
        npyadv2 = npyadv2.astype(np.float32)

        if self.transform:
            npynat = self.transform(npynat)
            npyadv1 = self.transform(npyadv1)
            npyadv2 = self.transform(npyadv2)

        return npynat, npyadv1, npyadv2


class DatasetNPY_Label(Dataset):

    def __init__(self, nat_dirs, adv1_dirs, adv2_dirs, label_dirs, transform=None):
        self.nat_dirs = nat_dirs
        self.adv1_dirs = adv1_dirs
        self.adv2_dirs = adv2_dirs
        self.npy_names = self.__get_npynames__()
        self.label_dirs = label_dirs
        self.label = self.__get_label__()
        self.transform = transform

    def __get_npynames__(self):
        tmp = []
        npy_name = os.listdir(self.nat_dirs)
        npy_name.sort(key=sort_key)
        for name in npy_name:
            tmp.append(os.path.join(self.nat_dirs, name))
        return tmp

    def __get_label__(self):
        label = load_variavle(self.label_dirs)
        label = np.array(label)
        label = torch.from_numpy(label)
        return label

    def __len__(self):
        return len(self.npy_names)

    def __getitem__(self, idx):
        npynat_path = self.npy_names[idx]
        npynat = np.load(npynat_path)
        npynat = npynat.astype(np.float32)

        npyadv1_path   = npynat_path.replace(self.nat_dirs, self.adv1_dirs)
        npyadv1 = np.load(npyadv1_path)
        npyadv1 = npyadv1.astype(np.float32)

        npyadv2_path   = npynat_path.replace(self.nat_dirs, self.adv2_dirs)
        npyadv2 = np.load(npyadv2_path)
        npyadv2 = npyadv2.astype(np.float32)

        label = self.label[idx]

        if self.transform:
            npynat = self.transform(npynat)
            npyadv1 = self.transform(npyadv1)
            npyadv2 = self.transform(npyadv2)

        return npynat, npyadv1, npyadv2, label


class DatasetNPY_Dual_Label(Dataset):

    def __init__(self, nat_dirs, adv_dirs, label_dirs, transform=None):
        self.nat_dirs = nat_dirs
        self.adv_dirs = adv_dirs
        self.npy_names = self.__get_npynames__()
        self.label_dirs = label_dirs
        self.label = self.__get_label__()
        self.transform = transform

    def __get_npynames__(self):
        tmp = []
        npy_name = os.listdir(self.nat_dirs)
        npy_name.sort(key=sort_key)
        for name in npy_name:
            tmp.append(os.path.join(self.nat_dirs, name))
        return tmp

    def __get_label__(self):
        label = load_variavle(self.label_dirs)
        label = np.array(label)
        label = torch.from_numpy(label)
        return label

    def __len__(self):
        return len(self.npy_names)

    def __getitem__(self, idx):
        npynat_path = self.npy_names[idx]
        npynat = np.load(npynat_path)
        npynat = npynat.astype(np.float32)

        npyadv_path   = npynat_path.replace(self.nat_dirs, self.adv1_dirs)
        npyadv = np.load(npyadv_path)
        npyadv = npyadv.astype(np.float32)

        label = self.label[idx]

        if self.transform:
            npynat = self.transform(npynat)
            npyadv = self.transform(npyadv)

        return npynat, npyadv, label


class DatasetNPY_test(Dataset):

    def __init__(self, npy_dirs, transform=None):
        self.nat_dirs = npy_dirs
        self.npy_names = self.__get_npynames__()
        self.transform = transform

    def __get_npynames__(self):
        tmp = []
        npy_name = os.listdir(self.nat_dirs)
        npy_name.sort(key=sort_key)
        for name in npy_name:
            tmp.append(os.path.join(self.nat_dirs, name))
        return tmp

    def __len__(self):
        return len(self.npy_names)

    def __getitem__(self, idx):
        npynat_path = self.npy_names[idx]
        npynat = np.load(npynat_path)
        npynat = npynat.astype(np.float32)

        if self.transform:
            npynat = self.transform(npynat)

        return npynat


