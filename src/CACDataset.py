# -*- coding: utf-8 -*-
"""
Developed by: Daniel Crovo
Dataset class definition for Coronary Artery Calcificaction segmentation 

"""

from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from Utils import transform_to_hu, set_window, zero_center
from pydicom import dcmread


class CACDataset(Dataset): 
    def __init__(self, image_dir, mask_dir, transforms = None, window=[-800, 1200]): 
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transforms
        self.dicom_window = window
        self.scans = os.listdir(self.image_dir)
        self.masks = os.listdir(self.mask_dir)

    def __len__(self): 
        return len(self.scans)
    
    def __getitem__(self, idx):
        img_subfolder = self.scans[idx]
        mask_subfolder = self.masks[idx]

        img_subfolder_path = os.path.join(self.image_dir, img_subfolder)
        mask_subfolder_path = os.path.join(self.mask_dir, mask_subfolder)

        img_files = os.listdir(img_subfolder_path)
        mask_files = os.listdir(mask_subfolder_path)

        images = []
        masks = []
        for img_file, mask_file in zip(img_files, mask_files):
            img_path = os.path.join(img_subfolder_path, img_file)
            mask_path = os.path.join(mask_subfolder_path, mask_file)
            ds = dcmread(img_path)
            dicom_img = transform_to_hu(ds)
            dicom_img = set_window(dicom_img, self.dicom_window)
            dicom_img = zero_center(dicom_img)
            image = np.array(Image.fromarray(dicom_img).convert('RGB'))
            mask = np.array(Image.open(mask_path).convert('L'))
            mask[mask != 0.0] = 1.0
            images.append(image)
            masks.append(mask)

        if self.transform is not None:
            images = [self.transform(img) for img in images]
            masks = [self.transform(mask) for mask in masks]
        return images, masks


