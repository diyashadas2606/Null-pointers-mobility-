import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes

IMAGE_SIZE = (256, 256)
DATAROOT = r'D:\MAHE MOBILITY\DATA SET'

DRIVABLE_CATEGORIES = [
    'flat.driveable_surface'
]

class NuScenesDataset(Dataset):
    def __init__(self):
        self.nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=False)
        self.samples = self._collect_samples()
        print(f"Found {len(self.samples)} valid samples")

    def _collect_samples(self):
        pairs = []
        for scene in self.nusc.scene:
            token = scene['first_sample_token']
            while token:
                sample = self.nusc.get('sample', token)
                cam_token = sample['data']['CAM_FRONT']
                cam_data = self.nusc.get('sample_data', cam_token)
                img_path = os.path.join(DATAROOT, cam_data['filename'])
                pairs.append((img_path, sample['token']))
                token = sample['next']
        return pairs

    def _make_mask(self, sample_token):
        sample = self.nusc.get('sample', sample_token)
        cam_token = sample['data']['CAM_FRONT']
        cam_data = self.nusc.get('sample_data', cam_token)

        # Get camera intrinsics
        cs = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        img_w, img_h = cam_data['width'], cam_data['height']

        mask = np.zeros((img_h, img_w), dtype=np.float32)

        # Loop through annotations in this sample
        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            category = ann['category_name']
            if any(category.startswith(c) for c in DRIVABLE_CATEGORIES):
                # Project 3D box center to 2D image roughly
                # Use a simple ground-plane approximation
                mask[img_h//2:, :] = 1.0  # lower half = drivable (simplified)

        # If no drivable annotations found, use ground-plane heuristic
        if mask.sum() == 0:
            mask[img_h//2:, :] = 1.0

        mask = cv2.resize(mask, IMAGE_SIZE)
        return mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, sample_token = self.samples[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMAGE_SIZE)
        img = img.astype(np.float32) / 255.0
        img = torch.tensor(img).permute(2, 0, 1)

        mask = self._make_mask(sample_token)
        mask = torch.tensor(mask).unsqueeze(0)

        return img, mask
