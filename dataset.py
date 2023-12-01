# -*- coding: utf-8 -*-
import torch
import glob
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import random
import pickle
import os
from PIL import Image
from PIL import ImageFile
from PIL import ImageOps

class immune_cell_dataset(Dataset):
    def __init__(self, pkl_dir, transform, type_str='train'):
        super().__init__()
        self.pkl_dir = pkl_dir
        self.transform = transform
        self.type_str = type_str
        self.data_list = self.load_data_pkl(self.type_str)
        self.cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
        print("patch num: ", len(self.data_list))
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        [img_path, label_path]  = self.data_list[idx]
        img, gt_mask, border = self.get_data(img_path, label_path)
        mask_transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.CenterCrop(256),
            # transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

        if self.transform is not None:
            # img, gt_mask, border = self.crop(img, gt_mask, 256)
            img = self.transform(img)
            gt_mask = mask_transform(gt_mask)
            gt_mask = torch.where(gt_mask > 0.5, 1.0 ,0.0)
            border = mask_transform(border)
            border = torch.where(border > 0.5, 1.0 ,0.0)
            
        return img, gt_mask, border

    def get_data(self, img_path, label_path):
        # Default cv2 read file
        try:
            img = cv2.imread(img_path)[...,::-1].copy()
        # PIL instead when error
        except:
            print("EXCEPT")
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            img = Image.open(img_path)
            img = np.asarray(img)
            # RGB -> BGR
            img = img[:, :, [2, 1, 0]]
            img = img[...,::-1].copy()

        gt = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        gt = gt[:,:,np.newaxis]
        border = self._border_get(gt)

        return img, gt, border

    def load_data_pkl(self, type_str):
        data = []
        data_pkl = self.pkl_dir + type_str + '.pkl'
        with open(data_pkl, 'rb') as f:
            while True:
                try:
                    data.append(pickle.load(f))
                except EOFError:
                    break
        mix_data = np.concatenate(data)

        return mix_data

    def _border_get(self, mask):
        dilate_mask = cv2.dilate(mask, self.cross_kernel, iterations=2)
        erode_mask = cv2.erode(mask, self.cross_kernel, iterations=2)
        dilate_mask = np.squeeze(dilate_mask)
        erode_mask = np.squeeze(erode_mask)
        border = dilate_mask - erode_mask
        (ori_w, ori_h) = np.shape(border)
        # rescale border by vector 2
        # border = cv2.resize(border, dsize=(ori_w*2, ori_h*2), interpolation=cv2.INTER_CUBIC)
        # cv2.imwrite('border.png',border)

        return border

    def crop(img, mask, border, size):
        # padding height or width if smaller than cropping size
        print(img)
        w, h = img.size
        # print('w',w,'h',h)
        try:
            padw = size - w if w < size else 0
            padh = size - h if h < size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)
            border = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)

            # cropping
            w, h = img.size
            x = random.randint(0, w - size)
            y = random.randint(0, h - size)
            img = img.crop((x, y, x + size, y + size))
            mask = mask.crop((x, y, x + size, y + size))
            border = border.crop((x, y, x + size, y + size))

            return img, mask, border
        except:
            print('connot read')
            print('w',w,'h',h)
            pass
            size = 512
            padw = size - w if w < size else 0
            padh = size - h if h < size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)
            border = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)

            # cropping
            w, h = img.size
            x = random.randint(0, w - size)
            y = random.randint(0, h - size)
            img = img.crop((x, y, x + size, y + size))
            mask = mask.crop((x, y, x + size, y + size))
            border = border.crop((x, y, x + size, y + size))

            return img, mask, border
