#!/usr/bin/env python3
"""
=============================================================================
 STATE-OF-THE-ART PREPROCESSING PIPELINE
 Real-Time Drivable Space Segmentation — nuScenes v1.0-mini
=============================================================================

 Based on: MC_Preprocessing.pdf
 Problem : Problem Statement 2 (Hackathon)
 Target  : Google Colab (T4/L4 GPU, ~15 GB VRAM, ~16 GB RAM)
 Author  : Hackathon Preprocessing Pipeline

 PIPELINE OVERVIEW
 -----------------
 Stage 1 │ JSON Metadata Indexing  — build fast O(1) token→record dicts
 Stage 2 │ Scene-Level Splitting   — 8 train / 2 val scenes (no leakage)
 Stage 3 │ Drivable Mask Gen       — HD-map → camera-view binary masks
 Stage 4 │ Image Preprocessing     — resize 1600×900 → 704×256, scale K
 Stage 5 │ Augmentation Pipeline   — flip, scale/crop, HSDA (FFT)
 Stage 6 │ CBGS Sampling           — class-balanced grouping & sampling
 Stage 7 │ PKL Serialization       — flat offline cache for fast loading
 Stage 8 │ PyTorch Dataset/Loader  — memory-safe, AMP-ready DataLoader
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — COLAB SETUP  (run this cell first in Colab)
# ─────────────────────────────────────────────────────────────────────────────
COLAB_INSTALL_CELL = """
# ── Paste and run in a Colab cell ──────────────────────────────────────────
!pip install -q nuscenes-devkit opencv-python-headless numpy pillow
!pip install -q torch torchvision          # already present on Colab GPU runtimes
# ──────────────────────────────────────────────────────────────────────────
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import gc
import json
import math
import pickle
import random
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — GLOBAL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PreprocessingConfig:
    """
    Single source-of-truth for every tunable constant in the pipeline.
    Edit here only — all downstream code reads from this object.
    """
    # ── Paths ────────────────────────────────────────────────────────────────
    data_root: str = "/content/drive/MyDrive/Mobility_Hackathon/Extracted_data"       # Colab mount point

    output_dir: str = "/content/drive/MyDrive/Mobility_Hackathon/preprocessed"

    # ── Image dimensions ────────────────────────────────────────────────────
    # Raw sensor captures at 1600×900 (already ROI-cropped by hardware).
    # Downscale to 704×256 → 89% FLOP reduction (ref: arXiv:2312.00633).
    src_w:  int = 1600
    src_h:  int = 900
    tgt_w:  int = 704
    tgt_h:  int = 256

    # ── ImageNet normalisation ───────────────────────────────────────────────
    # Use global ImageNet stats (not dataset-specific) because all
    # backbones are pre-trained on ImageNet. Matching the input distribution
    # stabilises gradient descent and accelerates convergence.
    img_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    img_std:  Tuple[float, ...] = (0.229, 0.224, 0.225)

    # ── Official nuScenes mini splits (scene-level, never frame-level) ───────
    # Source: nuscenes-devkit/python-sdk/nuscenes/utils/splits.py
    # Splitting at scene level eliminates temporal data leakage caused by
    # adjacent 2Hz keyframes sharing background / object configurations.
    mini_train_scenes: List[str] = field(default_factory=lambda: [
        "scene-0061", "scene-0553", "scene-0655", "scene-0757",
        "scene-0796", "scene-1077", "scene-1094", "scene-1100",
    ])
    mini_val_scenes: List[str] = field(default_factory=lambda: [
        "scene-0103", "scene-0916",
    ])

    # ── Camera names ─────────────────────────────────────────────────────────
    cameras: List[str] = field(default_factory=lambda: [
        "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK",  "CAM_BACK_LEFT",  "CAM_BACK_RIGHT",
    ])

    # ── Augmentation parameters ──────────────────────────────────────────────
    flip_prob:         float = 0.5
    scale_range:       Tuple[float, float] = (0.94, 1.11)   # BEVDet paper §3.2
    hsda_prob:         float = 0.5
    hsda_alpha:        float = 0.15    # high-freq shuffle magnitude

    # ── Map projection parameters ────────────────────────────────────────────
    bev_range_m:       float = 60.0   # metres around ego vehicle
    bev_resolution_m:  float = 0.3    # metres per sample point
    map_resolution_m:  float = 0.1    # nuScenes HD-map native resolution

    # ── Serialisation ────────────────────────────────────────────────────────
    train_pkl:  str = "train_metadata.pkl"
    val_pkl:    str = "val_metadata.pkl"

    # ── DataLoader (memory-safe for Colab 16 GB RAM) ─────────────────────────
    batch_size:    int = 2       # micro-batch; use gradient accumulation ×4
    num_workers:   int = 2       # >4 triggers OOM on 16 GB RAM
    pin_memory:    bool = True


CFG = PreprocessingConfig()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — JSON METADATA INDEXING
# ─────────────────────────────────────────────────────────────────────────────

class NuScenesIndex:
    """
    Parses the v1.0-mini relational JSON tables exactly once and builds
    O(1) token → record look-up dictionaries.  All downstream code queries
    these dicts rather than re-reading JSON per iteration (avoids CPU
    bottlenecks and RAM bloat inside the PyTorch __getitem__ call).
    """

    REQUIRED_FILES = [
        "scene.json", "sample.json", "sample_data.json",
        "calibrated_sensor.json", "ego_pose.json",
        "map.json", "log.json",
    ]

    def __init__(self, cfg: PreprocessingConfig):
        self.cfg  = cfg
        self.root = Path(cfg.data_root)
        self.meta_dir = self.root / "v1.0-mini"
        self._verify_files()
        self._load_and_index()

    # ── private ──────────────────────────────────────────────────────────────

    def _verify_files(self):
        missing = [
            f for f in self.REQUIRED_FILES
            if not (self.meta_dir / f).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing metadata files in {self.meta_dir}: {missing}\n"
                "Ensure Extracted_data/v1.0-mini/ is present."
            )

    def _load(self, filename: str) -> list:
        with open(self.meta_dir / filename, "r") as fh:
            return json.load(fh)

    def _index(self, records: list, key: str = "token") -> Dict[str, dict]:
        return {r[key]: r for r in records}

    def _load_and_index(self):
        print("── Loading and indexing JSON metadata ──────────────────────")

        scenes          = self._load("scene.json")
        samples         = self._load("sample.json")
        sample_datas    = self._load("sample_data.json")
        cal_sensors     = self._load("calibrated_sensor.json")
        ego_poses       = self._load("ego_pose.json")
        maps            = self._load("map.json")
        logs            = self._load("log.json")

        # Primary token-indexed dicts
        self.scene_by_token      = self._index(scenes)
        self.sample_by_token     = self._index(samples)
        self.cal_sensor_by_token = self._index(cal_sensors)
        self.ego_pose_by_token   = self._index(ego_poses)
        self.map_by_token        = self._index(maps)
        self.log_by_token        = self._index(logs)

        # ── Diagnostic: print actual sample_data fields ──────────────────
        if sample_datas:
            print(f"   sample_data fields: {sorted(sample_datas[0].keys())}")

        # ── Build cam_data_by_sample index ───────────────────────────────
        #
        # ROOT CAUSE OF PREVIOUS FAILURES:
        #
        # Attempt 1 used: sd.get("sensor_modality") != "camera"
        #   → "sensor_modality" does not exist in sample_data.json at all.
        #     It lives in the sensor table. Every record was dropped silently.
        #
        # Attempt 2 used: sd.get("channel", "").startswith("CAM_")
        #   → "channel" also does not exist in this dataset's sample_data.json
        #     (confirmed by diagnostic output above). Still 0 records.
        #
        # DEFINITIVE FIX:
        #   The camera name IS present — embedded inside the filename path.
        #   nuScenes filenames follow this exact structure:
        #     "samples/CAM_FRONT/n008-2018-08-01-...__CAM_FRONT__....jpg"
        #                ^^^^^^^^
        #   The parent directory of the file IS the camera channel name.
        #   Extract it with Path(filename).parent.name.
        #
        #   Keyframe identification: use the is_key_frame boolean flag
        #   (which DOES exist per the diagnostic) AND cross-check that
        #   sample_token is in the known annotated sample table.
        #   Both guards together are robust across all dataset versions.

        known_sample_tokens = set(self.sample_by_token.keys())
        self.cam_data_by_sample: Dict[str, Dict[str, dict]] = {}

        cam_total = kf_cam_total = 0

        for sd in sample_datas:
            # Extract camera channel from the file path's parent directory.
            # e.g. "samples/CAM_FRONT/xxxx.jpg"  →  "CAM_FRONT"
            #      "sweeps/CAM_BACK/xxxx.jpg"     →  "CAM_BACK"
            filename = sd.get("filename", "")
            ch = Path(filename).parent.name          # e.g. "CAM_FRONT"

            if not ch.startswith("CAM_"):
                continue                             # LiDAR, RADAR, etc.
            cam_total += 1

            # Keyframe filter: annotated 2Hz samples only (not 12Hz sweeps)
            is_kf    = sd.get("is_key_frame", None)
            sam_tok  = sd.get("sample_token", "")

            # Explicit sweep guard: is_key_frame == False means sweep
            if is_kf is False:
                continue
            # Secondary guard: sample_token must be in annotated sample table
            if sam_tok not in known_sample_tokens:
                continue
            kf_cam_total += 1

            self.cam_data_by_sample.setdefault(sam_tok, {})[ch] = sd

        print(f"   Total CAM_ records : {cam_total}")
        print(f"   Keyframe CAM_ recs : {kf_cam_total}")

        # Map each scene to its log (needed for HD map lookup)
        self.scene_to_log: Dict[str, str] = {
            s["token"]: s["log_token"] for s in scenes
        }

        # Map each log to its map token
        self.log_to_map: Dict[str, str] = {
            lg["token"]: lg.get("map_token", "") for lg in logs
        }

        # Collect camera-sample pairs per scene
        self.samples_by_scene: Dict[str, List[dict]] = {}
        for sample in samples:
            st = sample["scene_token"]
            self.samples_by_scene.setdefault(st, []).append(sample)

        # Sort within each scene by timestamp (ascending)
        for st in self.samples_by_scene:
            self.samples_by_scene[st].sort(key=lambda s: s["timestamp"])

        print(f"   Scenes          : {len(scenes)}")
        print(f"   Samples         : {len(samples)}")
        print(f"   Cam sample_data : {len(self.cam_data_by_sample)} samples × ~6 cams")
        print(f"   Cal sensors     : {len(cal_sensors)}")
        print(f"   Ego poses       : {len(ego_poses)}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — SCENE-LEVEL TRAIN / VAL SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def scene_level_split(
    index: NuScenesIndex,
    cfg:   PreprocessingConfig,
) -> Tuple[List[dict], List[dict]]:
    """
    Partitions the 404 samples into train/val strictly at the scene boundary.

    WHY scene-level:
      Two adjacent 2Hz keyframes are 0.5 s apart.  They share nearly identical
      backgrounds, lighting, and object configurations.  A naïve random split
      causes information leakage → inflated mIoU that collapses on real data.
      Splitting by scene guarantees the validation set contains entirely novel
      physical environments.  (ref: arXiv:2312.06420)
    """
    print("\n── Scene-level train/val split ─────────────────────────────────")

    train_scenes_names = set(cfg.mini_train_scenes)
    val_scenes_names   = set(cfg.mini_val_scenes)

    train_records: List[dict] = []
    val_records:   List[dict] = []

    for scene_token, scene_record in index.scene_by_token.items():
        scene_name = scene_record["name"]
        samples_in_scene = index.samples_by_scene.get(scene_token, [])

        if scene_name in train_scenes_names:
            train_records.extend(samples_in_scene)
        elif scene_name in val_scenes_names:
            val_records.extend(samples_in_scene)
        else:
            # Scene not in either official split; skip (handles edge cases)
            print(f"   [WARN] Scene '{scene_name}' not in official splits — skipping.")

    print(f"   Train samples   : {len(train_records)}")
    print(f"   Val   samples   : {len(val_records)}")
    assert len(train_records) > 0 and len(val_records) > 0, \
        "Split produced empty partitions — check scene names."
    return train_records, val_records


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — COORDINATE TRANSFORMATION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def quaternion_to_rotation_matrix(q: List[float]) -> np.ndarray:
    """
    Convert a unit quaternion [w, x, y, z] to a 3×3 rotation matrix.
    Uses the standard Hamiltonian convention (nuScenes default).
    """
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    return R


def build_transform_matrix(
    translation: List[float],
    rotation_q:  List[float],
) -> np.ndarray:
    """
    Build a 4×4 homogeneous transformation matrix from translation and
    unit quaternion.  Represents: T = [R | t; 0 | 1].
    """
    R = quaternion_to_rotation_matrix(rotation_q)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = translation
    return T


def scale_intrinsic_matrix(
    K:    np.ndarray,
    sw:   float,
    sh:   float,
) -> np.ndarray:
    """
    Scale the 3×3 intrinsic camera matrix proportionally with image resize.

    When the image is downscaled by (sw, sh):
      f_x → f_x * sw,   c_x → c_x * sw
      f_y → f_y * sh,   c_y → c_y * sh

    Failing to do this severs the pixel ↔ 3D geometric relationship,
    destroying the model's ability to estimate depth.
    (ref: arXiv:2312.00633 §3.1)
    """
    K_scaled = K.copy()
    K_scaled[0, 0] *= sw   # f_x
    K_scaled[0, 2] *= sw   # c_x
    K_scaled[1, 1] *= sh   # f_y
    K_scaled[1, 2] *= sh   # c_y
    return K_scaled


def project_points_to_image(
    points_cam: np.ndarray,     # (N, 3)  in camera frame
    K:          np.ndarray,     # (3, 3)  intrinsic matrix
    img_w:      int,
    img_h:      int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projects 3D points in the camera coordinate frame to 2D pixel coordinates.

    Mathematical model (pinhole camera):
        Z · [u, v, 1]^T = K · [X, Y, Z]^T
        u = f_x * X/Z + c_x
        v = f_y * Y/Z + c_y

    Returns:
        pixels  : (M, 2)  valid integer pixel coordinates [u, v]
        valid   : (M,)    boolean mask of in-frame, front-facing points
    """
    # Keep only points in front of the camera (Z > 0)
    depth = points_cam[:, 2]
    front = depth > 0.1                          # small margin for stability

    pts = points_cam[front]
    if pts.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int32), np.array([], dtype=bool)

    # Perspective division and intrinsic projection
    uvw = (K @ pts.T)                            # (3, M)
    u = uvw[0] / uvw[2]
    v = uvw[1] / uvw[2]

    # Keep only pixels within image bounds
    in_frame = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    pixels = np.stack([u[in_frame], v[in_frame]], axis=1).astype(np.int32)
    return pixels, in_frame


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — HD MAP LOADING & DRIVABLE AREA QUERY
# ─────────────────────────────────────────────────────────────────────────────

class HDMapManager:
    """
    Manages the four nuScenes HD map PNG files.

    Each map is a grayscale raster where non-zero pixels indicate
    'semantic_prior' (a composite of drivable area, walkways, etc.).
    For drivable space segmentation we treat any non-zero pixel as
    drivable (the maps provided are already drivable-area-focused).

    Map coordinate system:
      - Origin  : bottom-left corner (in global GPS / UTM metres)
      - Scale   : CFG.map_resolution_m  metres per pixel (0.1 m/px)
      - Axis    : x → right (East),  y → up (North)  [standard GIS]

    nuScenes provides map origins for each location:
      boston-seaport      : (300, 800)
      boston-route        : (500, 1500)
      singapore-onenorth  : (200, 400)
      singapore-hollandvillage : (400, 700)
      singapore-queenstown     : (300, 700)
    (approximate values; exact values in nuscenes-devkit map_expansion)
    """

    # Approximate origins (x0, y0) in metres for each location.
    # These are the global coordinate of pixel (0, 0) = bottom-left corner.
    MAP_ORIGINS = {
        "boston-seaport":           ( 300.0,   800.0),
        "boston-route":             ( 500.0,  1500.0),
        "singapore-onenorth":       ( 200.0,   400.0),
        "singapore-hollandvillage": ( 400.0,   700.0),
        "singapore-queenstown":     ( 300.0,   700.0),
    }

    def __init__(self, cfg: PreprocessingConfig):
        self.cfg  = cfg
        self.root = Path(cfg.data_root)
        self._cache: Dict[str, np.ndarray] = {}

    def load_map(self, map_filename: str) -> np.ndarray:
        """
        Load a map PNG lazily, caching in RAM.
        Tries multiple path resolutions because map.json filenames vary:
          - Some versions store: "maps/53992ee3...png"
          - Others store just:   "53992ee3...png"
        """
        if map_filename not in self._cache:
            name_only = Path(map_filename).name
            candidates = [
                self.root / map_filename,              # as-is
                self.root / "maps" / name_only,        # bare name under maps/
                self.root / name_only,                 # bare name at root
            ]
            path = None
            for candidate in candidates:
                if candidate.exists():
                    path = candidate
                    break
            if path is None:
                raise FileNotFoundError(
                    f"HD map not found. Tried:\n"
                    + "\n".join(f"  {c}" for c in candidates)
                )
            img = np.array(Image.open(path).convert("L"), dtype=np.uint8)
            self._cache[map_filename] = img
            print(f"   Loaded map: {path.name}  shape={img.shape}")
        return self._cache[map_filename]

    def query_drivable(
        self,
        global_xy: np.ndarray,       # (N, 2)  global [x, y] in metres
        map_array: np.ndarray,        # (H, W)  uint8 map image
        map_origin: Tuple[float, float],
    ) -> np.ndarray:
        """
        Query whether each global (x, y) point lies on the drivable area.

        Coordinate mapping:
          pixel_col =  (x - x0) / res
          pixel_row = H - (y - y0) / res   ← y-axis flip (image vs GIS)

        Returns boolean array (N,).
        """
        res = self.cfg.map_resolution_m
        x0, y0 = map_origin
        H, W = map_array.shape

        col = ((global_xy[:, 0] - x0) / res).astype(np.int32)
        row = (H - (global_xy[:, 1] - y0) / res).astype(np.int32)

        # Clamp to valid range
        valid = (col >= 0) & (col < W) & (row >= 0) & (row < H)
        drivable = np.zeros(len(global_xy), dtype=bool)
        drivable[valid] = map_array[row[valid], col[valid]] > 0
        return drivable


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — DRIVABLE MASK GENERATION (MAP → CAMERA VIEW)
# ─────────────────────────────────────────────────────────────────────────────

def generate_drivable_mask(
    ego_pose_record:     dict,
    cal_sensor_record:   dict,
    map_manager:         HDMapManager,
    map_array:           np.ndarray,
    map_origin:          Tuple[float, float],
    cfg:                 PreprocessingConfig,
) -> np.ndarray:
    """
    Projects the HD drivable-area map into a single camera's pixel space,
    producing a binary segmentation mask of shape (tgt_h, tgt_w).

    Mathematical pipeline (three-stage transformation):

    Stage 1 — Global → Ego frame:
        T_ego_inv = inverse(build_transform(ego_pose.translation, ego_pose.rotation))
        P_ego = T_ego_inv @ P_global

    Stage 2 — Ego → Camera frame:
        T_cam_inv = inverse(build_transform(cal_sensor.translation, cal_sensor.rotation))
        P_cam = T_cam_inv @ P_ego

    Stage 3 — Camera → Image plane:
        [u, v, 1]^T = K_scaled / Z  ×  [X, Y, Z]^T

    Algorithm:
      1. Sample a dense BEV grid of 3D points on the ground plane (z = 0)
         in the EGO coordinate frame.
      2. Transform each point to global frame to query the HD map.
      3. Keep only drivable points.
      4. Transform drivable points to camera frame.
      5. Project to image plane → fill mask.
    """
    tgt_w, tgt_h = cfg.tgt_w, cfg.tgt_h
    bev_r  = cfg.bev_range_m
    bev_res = cfg.bev_resolution_m

    # ── Build transformation matrices ────────────────────────────────────────
    T_ego = build_transform_matrix(
        ego_pose_record["translation"],
        ego_pose_record["rotation"],
    )
    T_cam = build_transform_matrix(
        cal_sensor_record["translation"],
        cal_sensor_record["rotation"],
    )
    K_raw = np.array(cal_sensor_record["camera_intrinsic"], dtype=np.float64)  # (3,3)

    # Scale K to match the downscaled image
    sw = tgt_w / cfg.src_w
    sh = tgt_h / cfg.src_h
    K_scaled = scale_intrinsic_matrix(K_raw, sw, sh)

    # ── 1. Dense BEV grid in EGO frame (ground plane z = 0) ──────────────────
    xs = np.arange(-bev_r, bev_r, bev_res)
    ys = np.arange(-bev_r, bev_r, bev_res)
    xx, yy = np.meshgrid(xs, ys)
    N = xx.size
    # homogeneous 4-vectors: [x, y, 0, 1]
    pts_ego_h = np.stack([xx.ravel(), yy.ravel(),
                          np.zeros(N), np.ones(N)], axis=1)   # (N, 4)

    # ── 2. Transform to global frame for HD-map query ─────────────────────────
    pts_global = (T_ego @ pts_ego_h.T).T         # (N, 4)
    is_drivable = map_manager.query_drivable(
        pts_global[:, :2],
        map_array,
        map_origin,
    )

    # ── 3. Keep only drivable points → transform to camera frame ─────────────
    drivable_pts_ego = pts_ego_h[is_drivable]    # (M, 4)
    if drivable_pts_ego.shape[0] == 0:
        return np.zeros((tgt_h, tgt_w), dtype=np.uint8)

    T_cam_inv = np.linalg.inv(T_cam)
    pts_cam_h = (T_cam_inv @ drivable_pts_ego.T).T  # (M, 4)
    pts_cam   = pts_cam_h[:, :3]                     # (M, 3)

    # ── 4. Project to scaled image plane ─────────────────────────────────────
    pixels, _ = project_points_to_image(pts_cam, K_scaled, tgt_w, tgt_h)

    # ── 5. Fill binary mask ───────────────────────────────────────────────────
    mask = np.zeros((tgt_h, tgt_w), dtype=np.uint8)
    if pixels.shape[0] > 0:
        mask[pixels[:, 1], pixels[:, 0]] = 1

    # Morphological closing: fill small projection holes caused by sparse
    # BEV grid sampling (important for thin road sections)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — IMAGE PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess_image(
    file_path:   str,
    cfg:         PreprocessingConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads a camera JPEG, resizes it to (tgt_h, tgt_w), and applies
    ImageNet normalisation.

    IMPORTANT: Do NOT apply lens distortion correction here.
    The nuScenes images have already been hardware-undistorted before
    publication.  Re-applying OpenCV distortion correction would break
    the geometric relationship between pixels and the intrinsic K matrix.

    Returns:
        img_norm  : (3, tgt_h, tgt_w)  float32 CHW tensor, normalised
        img_uint8 : (tgt_h, tgt_w, 3)  uint8  HWC array for augmentation
    """
    # Read image (BGR → RGB)
    raw = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if raw is None:
        raise FileNotFoundError(f"Cannot read image: {file_path}")
    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

    # Resize: INTER_AREA is best for downsampling (anti-aliasing)
    resized = cv2.resize(raw, (cfg.tgt_w, cfg.tgt_h),
                         interpolation=cv2.INTER_AREA)

    # Normalise to [0, 1], then apply ImageNet mean/std
    img_f32 = resized.astype(np.float32) / 255.0
    mean = np.array(cfg.img_mean, dtype=np.float32)
    std  = np.array(cfg.img_std,  dtype=np.float32)
    img_norm = (img_f32 - mean) / std                  # (H, W, 3)
    img_norm = img_norm.transpose(2, 0, 1)             # CHW

    return img_norm.astype(np.float32), resized


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — AUGMENTATION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class AugmentationPipeline:
    """
    Geometric and frequency-domain augmentations for multi-view 3D perception.

    CRITICAL CONSTRAINT:
      Every spatial transformation applied to the 2D image MUST be
      geometrically mirrored in the intrinsic matrix K.  Failing to
      maintain this 2D↔3D consistency poisons the training data.
      (ref: BEVDet §3.2)

    Augmentations implemented:
      1. Random Horizontal Flip  → invert u-axis in K
      2. Random Scale + Crop     → update c_x, c_y, f_x, f_y in K
      3. HSDA (High-Frequency Shuffle Data Augmentation)
         → FFT → shuffle high-freq components → IFFT
         (ref: WACV 2025, arXiv:2412.06127)
    """

    def __init__(self, cfg: PreprocessingConfig, is_train: bool = True):
        self.cfg      = cfg
        self.is_train = is_train

    # ── Public API ───────────────────────────────────────────────────────────

    def __call__(
        self,
        img_norm:  np.ndarray,    # (3, H, W) float32 normalised
        mask:      np.ndarray,    # (H, W)    uint8   binary mask
        K:         np.ndarray,    # (3, 3)    scaled intrinsic
        img_raw:   np.ndarray,    # (H, W, 3) uint8  for HSDA
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply augmentations (training only). Returns augmented
        (img_norm, mask, K_aug).
        """
        if not self.is_train:
            return img_norm, mask, K

        K_aug = K.copy()

        # 1. Random horizontal flip
        if random.random() < self.cfg.flip_prob:
            img_norm, mask, K_aug = self._flip(img_norm, mask, K_aug)

        # 2. Random scale + crop
        img_norm, mask, K_aug = self._scale_crop(img_norm, mask, K_aug)

        # 3. HSDA — frequency-domain high-freq shuffle
        if random.random() < self.cfg.hsda_prob:
            img_norm = self._hsda(img_norm)

        return img_norm, mask, K_aug

    # ── Private methods ──────────────────────────────────────────────────────

    def _flip(
        self,
        img:  np.ndarray,
        mask: np.ndarray,
        K:    np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Random horizontal flip.

        K update: u → W - 1 - u
          Equivalently: c_x → W - 1 - c_x  (f_x sign unchanged)
        """
        W = img.shape[2]
        img_flipped  = img[:, :, ::-1].copy()
        mask_flipped = mask[:, ::-1].copy()
        K_flipped    = K.copy()
        K_flipped[0, 2] = W - 1 - K[0, 2]   # invert principal point x
        return img_flipped, mask_flipped, K_flipped

    def _scale_crop(
        self,
        img:  np.ndarray,
        mask: np.ndarray,
        K:    np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Random scale in [scale_min, scale_max] followed by random crop
        back to (tgt_h, tgt_w).

        K update after scale by factor s:
          f_x → f_x * s,  f_y → f_y * s
          c_x → c_x * s,  c_y → c_y * s

        K update after crop by (crop_x, crop_y):
          c_x → c_x - crop_x,  c_y → c_y - crop_y
          (f_x, f_y unchanged by translation)
        """
        _, H, W = img.shape
        tgt_h, tgt_w = self.cfg.tgt_h, self.cfg.tgt_w
        s_min, s_max = self.cfg.scale_range

        s = random.uniform(s_min, s_max)
        new_w = max(int(W * s), tgt_w)
        new_h = max(int(H * s), tgt_h)

        # Resize
        img_rs   = cv2.resize(img.transpose(1, 2, 0),   (new_w, new_h),
                              interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
        mask_rs  = cv2.resize(mask, (new_w, new_h),
                              interpolation=cv2.INTER_NEAREST)

        # Random crop offsets
        crop_x = random.randint(0, new_w - tgt_w)
        crop_y = random.randint(0, new_h - tgt_h)
        img_crop  = img_rs[:,   crop_y:crop_y+tgt_h, crop_x:crop_x+tgt_w].copy()
        mask_crop = mask_rs[crop_y:crop_y+tgt_h, crop_x:crop_x+tgt_w].copy()

        # Update K
        K_aug = K.copy()
        K_aug[0, 0] *= s;   K_aug[1, 1] *= s
        K_aug[0, 2] *= s;   K_aug[1, 2] *= s
        K_aug[0, 2] -= crop_x
        K_aug[1, 2] -= crop_y

        return img_crop, mask_crop, K_aug

    def _hsda(self, img: np.ndarray) -> np.ndarray:
        """
        High-Frequency Shuffle Data Augmentation (HSDA).

        Theory:
          High-frequency components (sharp edges, object boundaries,
          texture transitions) carry the critical depth and segmentation
          cues for BEV perception.  By subtly shuffling the high-frequency
          spectrum in the Fourier domain, the network is forced to rely on
          robust, invariant edge features rather than memorising low-frequency
          colour patterns or lighting conditions.
          (ref: Glisson et al., WACV 2025, arXiv:2412.06127)

        Algorithm per channel:
          1. FFT2  → shift DC to centre
          2. Define high-freq annular region (outer 50% of spectrum)
          3. Randomly shuffle phase in high-freq region by alpha * 2π
          4. Inverse FFT → clip to original range
        """
        alpha = self.cfg.hsda_alpha
        C, H, W = img.shape
        img_aug = np.empty_like(img)

        # Frequency threshold: keep low-freq (inner 50%) intact
        fh, fw = H // 2, W // 2
        r_low  = min(fh, fw) // 2          # inner radius boundary

        ys = np.arange(H) - H // 2
        xs = np.arange(W) - W // 2
        dist = np.sqrt(ys[:, None]**2 + xs[None, :]**2)
        high_freq_mask = dist > r_low      # True for high-frequency region

        for c in range(C):
            ch = img[c].astype(np.float32)
            F  = np.fft.fftshift(np.fft.fft2(ch))

            magnitude = np.abs(F)
            phase     = np.angle(F)

            # Add small random phase noise in high-freq region
            noise = np.random.uniform(-alpha * np.pi, alpha * np.pi,
                                      size=phase.shape).astype(np.float32)
            phase[high_freq_mask] += noise[high_freq_mask]

            # Reconstruct
            F_aug  = magnitude * np.exp(1j * phase)
            ch_aug = np.real(np.fft.ifft2(np.fft.ifftshift(F_aug)))

            # Clip to original range to prevent normalisation drift
            img_aug[c] = np.clip(ch_aug, img[c].min(), img[c].max())

        return img_aug


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — CLASS-BALANCED GROUPING & SAMPLING (CBGS)
# ─────────────────────────────────────────────────────────────────────────────

def compute_cbgs_weights(records: List[dict]) -> List[float]:
    """
    Compute per-sample sampling weights for Class-Balanced Grouping
    and Sampling (CBGS).

    For drivable space segmentation, the "class" is the fraction of the
    image that is drivable (drivable_fraction):
      - Very low  fraction → under-represented (rare road views)
      - Very high fraction → over-represented (clear highway shots)

    CBGS intentionally oversamples rare configurations so the loss
    landscape is not dominated by the majority class.
    (ref: BEVDet §3.3, arXiv:2203.17054)

    Implementation:
      We bin drivable_fraction into 5 discrete bins.  The weight of each
      sample is inversely proportional to the count of samples in its bin.
    """
    N_BINS = 5

    fractions = np.array(
        [r.get("drivable_fraction", 0.5) for r in records],
        dtype=np.float32,
    )

    bins      = np.floor(fractions * N_BINS).clip(0, N_BINS - 1).astype(int)
    bin_counts = np.bincount(bins, minlength=N_BINS).astype(np.float32)
    bin_counts = np.maximum(bin_counts, 1.0)            # avoid div-by-zero

    weights = [1.0 / bin_counts[bins[i]] for i in range(len(records))]
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — OFFLINE PKL SERIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def build_metadata_records(
    samples:        List[dict],
    index:          NuScenesIndex,
    map_manager:    HDMapManager,
    cfg:            PreprocessingConfig,
    is_train:       bool,
    split_name:     str,
) -> List[dict]:
    """
    Crawls the dataset exactly once and serialises per-(sample, camera)
    records into a flat list of Python dicts.

    Each record contains everything the PyTorch __getitem__ needs:
      - image_path      : absolute path to the JPEG
      - camera          : camera channel name
      - K_scaled        : (3,3) float64 intrinsic at (tgt_h × tgt_w)
      - mask            : (tgt_h, tgt_w) uint8 drivable mask
      - is_train        : bool
      - drivable_fraction : float  (for CBGS)

    Why offline:
      Dynamic JSON parsing inside DataLoader workers creates CPU
      bottlenecks and RAM bloat.  Serialising to .pkl means the
      __getitem__ only reads this compact flat dict.
      (ref: MC_Preprocessing.pdf §Colab Resource Management)
    """
    records = []
    total   = len(samples)
    print(f"\n── Building {split_name} metadata ({total} samples × ~6 cams) ─────")

    # Determine which map file to use per scene (cached across samples)
    scene_to_map_file:  Dict[str, str]                   = {}
    scene_to_map_origin: Dict[str, Tuple[float, float]]  = {}

    for i, sample in enumerate(samples):
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"   Progress: {i+1}/{total} samples …")

        scene_token = sample["scene_token"]

        # ── Resolve map for this scene ────────────────────────────────────
        if scene_token not in scene_to_map_file:
            log_token    = index.scene_to_log.get(scene_token, "")
            log_record   = index.log_by_token.get(log_token, {})
            location     = log_record.get("location", "boston-seaport")

            # Find map PNG for this location
            map_file = None
            for map_tok, map_rec in index.map_by_token.items():
                if location in map_rec.get("filename", "").lower() or \
                   location.replace("-", "").lower() in \
                   str(map_rec.get("log_tokens", [])).lower():
                    map_file = map_rec["filename"]
                    break
            # Fallback: pick first available map
            if map_file is None:
                map_file = next(iter(index.map_by_token.values()))["filename"]

            origin = map_manager.MAP_ORIGINS.get(location, (0.0, 0.0))
            scene_to_map_file[scene_token]   = map_file
            scene_to_map_origin[scene_token] = origin

        map_file   = scene_to_map_file[scene_token]
        map_origin = scene_to_map_origin[scene_token]
        map_array  = map_manager.load_map(map_file)

        # ── Process each camera ───────────────────────────────────────────
        cam_data_dict = index.cam_data_by_sample.get(sample["token"], {})
        for cam_name in cfg.cameras:
            sd = cam_data_dict.get(cam_name)
            if sd is None:
                continue                  # camera missing from this sample

            # sd["filename"] in nuScenes is always a relative path like
            # "samples/CAM_FRONT/xxxx.jpg" — join with data_root.
            # Guard against any absolute path edge case.
            raw_fname = sd["filename"]
            if os.path.isabs(raw_fname):
                img_path = raw_fname
            else:
                img_path = str(Path(cfg.data_root) / raw_fname)

            if not Path(img_path).exists():
                print(f"   [WARN] Image not found, skipping: {img_path}")
                continue

            # Calibration for this camera
            cal = index.cal_sensor_by_token[sd["calibrated_sensor_token"]]
            K_raw = np.array(cal["camera_intrinsic"], dtype=np.float64)
            sw = cfg.tgt_w / cfg.src_w
            sh = cfg.tgt_h / cfg.src_h
            K_scaled = scale_intrinsic_matrix(K_raw, sw, sh)

            # Ego pose at this sample timestamp
            ego = index.ego_pose_by_token[sd["ego_pose_token"]]

            # Generate drivable mask
            try:
                mask = generate_drivable_mask(
                    ego_pose_record=ego,
                    cal_sensor_record=cal,
                    map_manager=map_manager,
                    map_array=map_array,
                    map_origin=map_origin,
                    cfg=cfg,
                )
            except Exception as e:
                print(f"   [WARN] Mask generation failed for {cam_name} "
                      f"sample {sample['token'][:8]}…: {e}")
                mask = np.zeros((cfg.tgt_h, cfg.tgt_w), dtype=np.uint8)

            drivable_fraction = float(mask.mean())

            records.append({
                "image_path":        img_path,
                "camera":            cam_name,
                "sample_token":      sample["token"],
                "scene_token":       scene_token,
                "K_scaled":          K_scaled,
                "mask":              mask,
                "is_train":          is_train,
                "drivable_fraction": drivable_fraction,
            })

    print(f"   ✓ {len(records)} (sample, camera) records built for {split_name}.")
    return records


def save_pkl(records: List[dict], output_path: str):
    """Serialise records list to a Pickle file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as fh:
        pickle.dump(records, fh, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"   Saved: {output_path}  ({size_mb:.1f} MB)")


def load_pkl(path: str) -> List[dict]:
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 — PYTORCH DATASET & DATALOADER
# ─────────────────────────────────────────────────────────────────────────────

class DrivableSpaceDataset(Dataset):
    """
    Memory-safe PyTorch Dataset for real-time drivable space segmentation.

    At __getitem__ time:
      1. Load + preprocess image from disk
      2. Retrieve pre-computed mask and K from PKL record
      3. Apply runtime augmentations (train only)
      4. Return (image_tensor, mask_tensor) for loss computation

    Memory strategy (Colab 16 GB RAM):
      - Masks are stored as uint8 (1 byte/px) in the PKL; not float32.
      - Images are loaded on-demand, not pre-cached.
      - num_workers ≤ 2 to prevent RAM duplication across processes.
    """

    def __init__(
        self,
        pkl_path:  str,
        cfg:       PreprocessingConfig,
        is_train:  bool,
    ):
        self.cfg      = cfg
        self.is_train = is_train
        self.augment  = AugmentationPipeline(cfg, is_train=is_train)

        print(f"Loading {'train' if is_train else 'val'} PKL …", end=" ")
        self.records = load_pkl(pkl_path)
        print(f"{len(self.records)} records loaded.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]

        # ── Load + preprocess image ───────────────────────────────────────
        img_norm, img_raw = load_and_preprocess_image(rec["image_path"],
                                                      self.cfg)
        mask = rec["mask"].copy()
        K    = rec["K_scaled"].copy()

        # ── Apply augmentations ───────────────────────────────────────────
        img_norm, mask, K = self.augment(img_norm, mask, K, img_raw)

        # ── Convert to tensors ────────────────────────────────────────────
        img_tensor  = torch.from_numpy(img_norm).float()    # (3, H, W)
        mask_tensor = torch.from_numpy(mask.astype(np.int64))  # (H, W) long

        return img_tensor, mask_tensor


def build_dataloaders(
    cfg: PreprocessingConfig,
) -> Tuple[DataLoader, DataLoader]:
    """
    Constructs memory-safe DataLoaders for training and validation.

    Training loader uses WeightedRandomSampler (CBGS) to oversample
    under-represented drivable-area configurations.

    Colab-safe parameters:
      - num_workers = 2  (>4 crashes 16 GB RAM via process duplication)
      - pin_memory  = True  (zero-copy GPU transfer)
      - batch_size  = 2  (use gradient accumulation × 4 = effective batch 8)
    """
    train_path = os.path.join(cfg.output_dir, cfg.train_pkl)
    val_path   = os.path.join(cfg.output_dir, cfg.val_pkl)

    train_ds = DrivableSpaceDataset(train_path, cfg, is_train=True)
    val_ds   = DrivableSpaceDataset(val_path,   cfg, is_train=False)

    # CBGS weighted sampler for training
    cbgs_weights = compute_cbgs_weights(train_ds.records)
    sampler = WeightedRandomSampler(
        weights=cbgs_weights,
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
        persistent_workers=(cfg.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
        persistent_workers=(cfg.num_workers > 0),
    )

    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 — VALIDATION & STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def print_dataset_statistics(
    train_records: List[dict],
    val_records:   List[dict],
    cfg:           PreprocessingConfig,
):
    """Print a comprehensive summary of the preprocessed dataset."""
    print("\n" + "═"*60)
    print("  DATASET STATISTICS")
    print("═"*60)

    def summarise(records, name):
        fracs = np.array([r["drivable_fraction"] for r in records])
        cams  = {}
        for r in records:
            cams[r["camera"]] = cams.get(r["camera"], 0) + 1
        print(f"\n  [{name}]")
        print(f"    Total (sample, cam) records : {len(records)}")
        print(f"    Unique sample tokens        : "
              f"{len(set(r['sample_token'] for r in records))}")
        print(f"    Drivable fraction  mean     : {fracs.mean():.3f}")
        print(f"    Drivable fraction  std      : {fracs.std():.3f}")
        print(f"    Records per camera :")
        for cam, cnt in sorted(cams.items()):
            print(f"      {cam:<20}: {cnt}")

    summarise(train_records, "TRAIN")
    summarise(val_records,   "VAL  ")

    print(f"\n  Image size        : {cfg.src_w}×{cfg.src_h} → {cfg.tgt_w}×{cfg.tgt_h}")
    print(f"  FLOP reduction    : "
          f"{100*(1 - (cfg.tgt_w*cfg.tgt_h)/(cfg.src_w*cfg.src_h)):.0f}%")
    print(f"  Normalisation     : mean={cfg.img_mean}  std={cfg.img_std}")
    print(f"  Augmentations     : Flip(p={cfg.flip_prob}), "
          f"ScaleCrop({cfg.scale_range}), "
          f"HSDA(p={cfg.hsda_prob}, α={cfg.hsda_alpha})")
    print("═"*60 + "\n")


def verify_batch(loader: DataLoader, n_batches: int = 2):
    """
    Sanity-check the DataLoader output: shape, dtype, value range.
    Explicitly deletes tensors and calls gc.collect() to prevent
    RAM accumulation in the Colab notebook environment.
    """
    print("── DataLoader verification ──────────────────────────────────")
    for b_idx, (imgs, masks) in enumerate(loader):
        if b_idx >= n_batches:
            break
        assert imgs.dtype == torch.float32,  "Image dtype must be float32"
        assert masks.dtype == torch.int64,   "Mask dtype must be int64"
        assert imgs.ndim == 4,               "Image must be (B, C, H, W)"
        assert masks.ndim == 3,              "Mask must be (B, H, W)"
        assert masks.max() <= 1,             "Mask values must be 0 or 1"
        print(f"   Batch {b_idx}: imgs={tuple(imgs.shape)} "
              f"dtype={imgs.dtype}  "
              f"range=[{imgs.min():.2f}, {imgs.max():.2f}]  "
              f"masks={tuple(masks.shape)}  "
              f"unique_vals={masks.unique().tolist()}")
        # Explicit cleanup to avoid RAM accumulation in Colab
        del imgs, masks
        gc.collect()
    print("   ✓ DataLoader verified.\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 — MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def run_preprocessing(cfg: PreprocessingConfig = CFG):
    """
    Master function: runs the full offline preprocessing pipeline and
    writes train.pkl / val.pkl to cfg.output_dir.

    Run this ONCE in Colab before training.  The training loop only
    reads the compact .pkl files, bypassing all JSON parsing.
    """
    print("═"*60)
    print("  nuScenes Drivable Space Segmentation — Preprocessing")
    print("═"*60)

    os.makedirs(cfg.output_dir, exist_ok=True)

    # ── Stage 1: Index metadata ───────────────────────────────────────────
    index = NuScenesIndex(cfg)

    # ── Stage 2: Scene-level split ────────────────────────────────────────
    train_samples, val_samples = scene_level_split(index, cfg)

    # ── Stage 3–7: Build PKL records ─────────────────────────────────────
    map_manager = HDMapManager(cfg)

    train_records = build_metadata_records(
        train_samples, index, map_manager, cfg,
        is_train=True, split_name="TRAIN",
    )
    val_records = build_metadata_records(
        val_samples, index, map_manager, cfg,
        is_train=False, split_name="VAL",
    )

    # ── Stage 8: Statistics ───────────────────────────────────────────────
    print_dataset_statistics(train_records, val_records, cfg)

    # ── Stage 9: Serialise ────────────────────────────────────────────────
    print("── Serialising PKL files ───────────────────────────────────")
    save_pkl(train_records, os.path.join(cfg.output_dir, cfg.train_pkl))
    save_pkl(val_records,   os.path.join(cfg.output_dir, cfg.val_pkl))

    # Explicitly free the large records list to prevent RAM accumulation
    del train_records, val_records, map_manager
    gc.collect()

    # ── Stage 10: DataLoaders + verification ──────────────────────────────
    train_loader, val_loader = build_dataloaders(cfg)
    verify_batch(train_loader, n_batches=2)
    verify_batch(val_loader,   n_batches=2)

    print("✓ Preprocessing complete.  PKL files ready for training.\n")
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# COLAB USAGE GUIDE (copy-paste into Colab cells)
# ─────────────────────────────────────────────────────────────────────────────
COLAB_USAGE = """
# ════════════════════════════════════════════════════════════
#  GOOGLE COLAB USAGE GUIDE
# ════════════════════════════════════════════════════════════

# ── Cell 1: Install dependencies ────────────────────────────
!pip install -q nuscenes-devkit opencv-python-headless

# ── Cell 2: Mount Drive & set data path ─────────────────────
from google.colab import drive
drive.mount('/content/drive')

# ── Cell 3: Upload / unzip dataset ─────────────────────────
# (if dataset is zipped in Drive)
# !unzip /content/drive/MyDrive/Extracted_data.zip -d /content/

# ── Cell 4: Run preprocessing ───────────────────────────────
import sys
sys.path.insert(0, '/content')               # or wherever you uploaded this file

from nuscenes_preprocessing import PreprocessingConfig, run_preprocessing

cfg = PreprocessingConfig(
    data_root  = "/content/Extracted_data",  # adjust if needed
    output_dir = "/content/preprocessed",
)
train_loader, val_loader = run_preprocessing(cfg)

# ── Cell 5: In your training loop ───────────────────────────
import torch
from torch.cuda.amp import autocast, GradScaler

model     = YourSegmentationModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler    = GradScaler()          # AMP

ACCUM_STEPS = 4                   # effective batch = batch_size × ACCUM_STEPS

optimizer.zero_grad()
for step, (imgs, masks) in enumerate(train_loader):
    imgs  = imgs.cuda(non_blocking=True)
    masks = masks.cuda(non_blocking=True)

    with autocast():              # FP16 forward pass (halves VRAM)
        logits = model(imgs)
        loss   = criterion(logits, masks) / ACCUM_STEPS

    scaler.scale(loss).backward()

    if (step + 1) % ACCUM_STEPS == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

# ── Activation checkpointing (if OOM persists) ───────────────
from torch.utils.checkpoint import checkpoint_sequential
# Wrap your encoder:  features = checkpoint_sequential(encoder, 4, imgs)
"""

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # When run as a script, execute the full preprocessing pipeline.
    train_loader, val_loader = run_preprocessing(CFG)
