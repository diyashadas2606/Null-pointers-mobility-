import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion


# CONFIG

IMAGE_SIZE = (256, 256)
DATAROOT = r'D:\MAHE MOBILITY\DATA SET'
BATCH_SIZE = 4
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CAMERAS = [
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
]


# DATASET

class NuScenesDataset(Dataset):
    def __init__(self, train=True):
        self.nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=False)
        self.samples = self._collect_samples()
        self.train = train

    def _collect_samples(self):
        data = []
        for sample in self.nusc.sample:
            for cam in CAMERAS:
                cam_token = sample['data'][cam]
                cam_data = self.nusc.get('sample_data', cam_token)
                img_path = os.path.join(DATAROOT, cam_data['filename'])
                data.append((img_path, cam_token))
        return data

    def _get_class_id(self, category):
        if 'vehicle' in category:
            return 1
        elif 'pedestrian' in category:
            return 2
        elif 'cycle' in category or 'bicycle' in category:
            return 3
        else:
            return 0

    def _make_mask(self, cam_token):
        cam_data = self.nusc.get('sample_data', cam_token)
        sample = self.nusc.get('sample', cam_data['sample_token'])

        w, h = cam_data['width'], cam_data['height']
        mask = np.zeros((h, w), dtype=np.uint8)

        cs = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        pose = self.nusc.get('ego_pose', cam_data['ego_pose_token'])

        intrinsic = np.array(cs['camera_intrinsic'])

        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            class_id = self._get_class_id(ann['category_name'])
            if class_id == 0:
                continue

            box = self.nusc.get_box(ann_token)

            # transform
            box.translate(-np.array(pose['translation']))
            box.rotate(Quaternion(pose['rotation']).inverse)

            box.translate(-np.array(cs['translation']))
            box.rotate(Quaternion(cs['rotation']).inverse)

            corners = box.corners()

            # depth filter
            if np.any(corners[2, :] <= 0):
                continue

            # project
            pts = view_points(corners, intrinsic, normalize=True)[:2].T

            # clip
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

            pts = pts.astype(np.int32)

            cv2.fillPoly(mask, [pts], class_id)

        mask = cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
        return mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cam_token = self.samples[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMAGE_SIZE)

        mask = self._make_mask(cam_token)

        # augmentation
        if self.train and np.random.rand() > 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)

        # normalize
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std

        img = torch.tensor(img).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.long)

        return img, mask


# MODEL

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + identity)

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.res = ResidualBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.res(x)
        return x, self.pool(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        self.res = ResidualBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:])
        x = torch.cat([x, skip], dim=1)
        return self.res(x)

class AdvancedUNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.e1 = EncoderBlock(3, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)

        self.b = ResidualBlock(512, 1024)

        self.d4 = DecoderBlock(1024, 512)
        self.d3 = DecoderBlock(512, 256)
        self.d2 = DecoderBlock(256, 128)
        self.d1 = DecoderBlock(128, 64)

        self.dropout = nn.Dropout2d(0.3)
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        b = self.dropout(self.b(p4))

        d4 = self.d4(b, s4)
        d3 = self.d3(d4, s3)
        d2 = self.d2(d3, s2)
        d1 = self.d1(d2, s1)

        return self.final(d1)


# LOSS

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        preds = torch.softmax(preds, dim=1)
        targets = F.one_hot(targets, num_classes=4).permute(0,3,1,2).float()

        inter = (preds * targets).sum()
        union = preds.sum() + targets.sum()

        dice = (2*inter + 1) / (union + 1)
        return 1 - dice


# IoU

def compute_iou(preds, targets, num_classes=4):
    preds = torch.argmax(preds, dim=1)
    ious = []

    for cls in range(1, num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)

        inter = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            continue

        ious.append((inter / union).item())

    return np.mean(ious) if ious else 0


# TRAIN

def train():
    dataset = NuScenesDataset(train=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = AdvancedUNet().to(DEVICE)
    ce = nn.CrossEntropyLoss()
    dice = DiceLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_iou = 0

        for imgs, masks in loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            outputs = model(imgs)

            loss = ce(outputs, masks) + dice(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_iou += compute_iou(outputs, masks)

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | IoU: {total_iou:.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("Model saved!")


# RUN
-
if __name__ == "__main__":
    train()
