# simple KITTI Data loading based on https://github.com/Insta360-Research-Team/DEFOM-Stereo

import torch
import torch.utils.data as data
import numpy as np
import os
from glob import glob
import os.path as osp
import cv2
from frame_utils import read_png, readDispKITTI
import torch.nn.functional as F
class KITTIDataset(data.Dataset):
    def __init__(self, root_dir, split='15', image_set='training'):
        """
        KITTI dataset loader
        Args:
            root_dir: Root directory containing KITTI dataset
            split: '12' for KITTI 2012, '15' for KITTI 2015
            image_set: 'training' or 'testing'
        """
        self.disparity_reader = readDispKITTI
        self.disparity_list = []
        self.image_list = []
        
        # Construct paths based on KITTI version
        root = os.path.join(root_dir, 'KITTI' + split)
        assert os.path.exists(root), f"KITTI {split} path {root} does not exist"

        if split == '15':
            image1_list = sorted(glob(os.path.join(root, image_set, 'image_2/*_10.png')))
            image2_list = sorted(glob(os.path.join(root, image_set, 'image_3/*_10.png')))
            disp_list = sorted(
                glob(os.path.join(root, 'training', 'disp_occ_0/*_10.png'))) if image_set == 'training' else [osp.join(
                root, 'training/disp_occ_0/000085_10.png')]*len(image1_list)
        else:  # KITTI 2012
            image1_list = sorted(glob(os.path.join(root, image_set, 'colored_0/*_10.png')))
            image2_list = sorted(glob(os.path.join(root, image_set, 'colored_1/*_10.png')))
            disp_list = sorted(
                glob(os.path.join(root, 'training', 'disp_occ/*_10.png'))) if image_set == 'training' else [osp.join(
                root, 'training/disp_occ/000085_10.png')] * len(image1_list)
                
        # Store image and disparity paths
        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [[img1, img2]]
            self.disparity_list += [[disp]]

        # Target dimensions
        self.target_width = 1248
        self.target_height = 376
            
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        """
        Returns:
            dict containing:
                - img1: Left image (C, H, W)
                - img2: Right image (C, H, W)
                - disp: Disparity map (1, H, W)
                - valid: Validity mask (1, H, W)
                - imageL_file: Path to left image
                - disp_file: Path to disparity file
        """     
        # Load disparity map
        disp = self.disparity_reader(self.disparity_list[index][0])
        if isinstance(disp, tuple):
            disp, valid = disp
        else:
            valid = disp < 1024  # KITTI specific threshold
            
        # Load images
        img1 = read_png(self.image_list[index][0])
        img2 = read_png(self.image_list[index][1])
            
        # Convert to tensors
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        disp = torch.from_numpy(disp[..., np.newaxis].copy()).permute(2, 0, 1).float()
        valid = torch.from_numpy(valid[..., np.newaxis].astype(np.bool_).copy()).permute(2, 0, 1)
        
        pad_width = max(0, self.target_width - img1.shape[2])
        pad_height = max(0, self.target_height - img1.shape[1])

        img1 = F.pad(img1, (0, pad_width, 0, pad_height), "constant", 0)
        img2 = F.pad(img2, (0, pad_width, 0, pad_height), "constant", 0)
        disp = F.pad(disp, (0, pad_width, 0, pad_height), "constant", 0)
        valid = F.pad(valid, (0, pad_width, 0, pad_height), "constant", 0)

        return {
            'img1': img1,
            'img2': img2,
            'disp': disp,
            'valid': valid,
            'imageL_file': self.image_list[index][0],
            'disp_file': self.disparity_list[index][0]
        }