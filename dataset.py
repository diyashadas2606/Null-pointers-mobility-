import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
import random

IMAGE_SIZE = (256, 256)
DATAROOT = r'D:\MAHE MOBILITY\DATA SET'

CAMERAS = [
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
]

# CLASS MAPPING
CLASS_MAP = {
    'vehicle': 1,
    'pedestrian': 2,
    'cycle': 3
}


class NuScenesDataset(Dataset):
    def __init__(self, train=True):
        self.nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=False)
        self.samples = self._collect_samples()
        self.train = train

        print(f"Total samples: {len(self.samples)} | Train={self.train}")

    
    # Collect all camera data
    def _collect_samples(self):
        data = []

        for sample in self.nusc.sample:
            for cam in CAMERAS:
                cam_token = sample['data'][cam]
                cam_data = self.nusc.get('sample_data', cam_token)

                img_path = os.path.join(DATAROOT, cam_data['filename'])
                data.append((img_path, cam_token))

        return data

    
    # Category → class id
    def _get_class_id(self, category_name):
        if 'vehicle' in category_name:
            return 1
        elif 'pedestrian' in category_name:
            return 2
        elif 'cycle' in category_name or 'bicycle' in category_name:
            return 3
        else:
            return 0  

    
    # Generate improved mask
    def _make_mask(self, cam_token):
        cam_data = self.nusc.get('sample_data', cam_token)
        sample = self.nusc.get('sample', cam_data['sample_token'])

        img_w, img_h = cam_data['width'], cam_data['height']
        mask = np.zeros((img_h, img_w), dtype=np.uint8)

        # Camera calibration
        cs_record = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        pose_record = self.nusc.get('ego_pose', cam_data['ego_pose_token'])

        intrinsic = np.array(cs_record['camera_intrinsic'])

        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            class_id = self._get_class_id(ann['category_name'])

            if class_id == 0:
                continue

            box = self.nusc.get_box(ann_token)

            #Transform to ego frame 
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)

            #Transform to camera frame 
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

            corners = box.corners()

            #DEPTH FILTER 
            if np.any(corners[2, :] <= 0):
                continue

            #Project to 2D 
            corners_2d = view_points(corners, intrinsic, normalize=True)[:2]
            corners_2d = corners_2d.T

            #Clip to image 
            corners_2d[:, 0] = np.clip(corners_2d[:, 0], 0, img_w - 1)
            corners_2d[:, 1] = np.clip(corners_2d[:, 1], 0, img_h - 1)

            corners_2d = corners_2d.astype(np.int32)

            #Draw polygon 
            cv2.fillPoly(mask, [corners_2d], class_id)

        mask = cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
        return mask

    
    # Augmentations
    
    def _augment(self, img, mask):
        # Horizontal flip
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)

        # Brightness
        if random.random() > 0.5:
            factor = 0.7 + 0.6 * random.random()
            img = np.clip(img * factor, 0, 255)

        # Gaussian noise
        if random.random() > 0.3:
            noise = np.random.normal(0, 5, img.shape)
            img = np.clip(img + noise, 0, 255)

        return img, mask

    def __len__(self):
        return len(self.samples)

    
    # MAIN
    
    def __getitem__(self, idx):
        img_path, cam_token = self.samples[idx]

       
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMAGE_SIZE)

        #MASK
        mask = self._make_mask(cam_token)

        #AUGMENT 
        if self.train:
            img, mask = self._augment(img, mask)

        #NORMALIZE (ImageNet) 
        img = img.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])

        img = (img - mean) / std

        #TO TENSOR
        img = torch.tensor(img).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.long)  # multi-class

        return img, mask
