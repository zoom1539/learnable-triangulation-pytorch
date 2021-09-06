import os
import shutil
import argparse
import time
import json
from datetime import datetime
from collections import defaultdict
from itertools import islice
import pickle
import copy

import numpy as np
import cv2

import torch
from torch import nn
from torch import autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from tensorboardX import SummaryWriter

from mvn.models.triangulation import RANSACTriangulationNet, AlgebraicTriangulationNet, VolumetricTriangulationNet
from mvn.models.loss import KeypointsMSELoss, KeypointsMSESmoothLoss, KeypointsMAELoss, KeypointsL2Loss, VolumetricCELoss

from mvn.utils import img, multiview, op, vis, misc, cfg
from mvn.datasets import human36m
from mvn.datasets import utils as dataset_utils

torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True)

def collate_fn(items):
    items = list(filter(lambda x: x is not None, items))
    if len(items) == 0:
        print("All items in batch are None")
        return None

    batch = dict()
    total_n_views = min(len(item['images']) for item in items)

    indexes = np.arange(total_n_views)
    randomize_n_views = False
    if randomize_n_views:
        n_views = np.random.randint(min_n_views, min(total_n_views, max_n_views) + 1)
        indexes = np.random.choice(np.arange(total_n_views), size=n_views, replace=False)
    else:
        indexes = np.arange(total_n_views)

    batch['images'] = np.stack([np.stack([item['images'][i] for item in items], axis=0) for i in indexes], axis=0).swapaxes(0, 1)
    batch['detections'] = np.array([[item['detections'][i] for item in items] for i in indexes]).swapaxes(0, 1)
    batch['cameras'] = [[item['cameras'][i] for item in items] for i in indexes]
    batch['keypoints_3d'] = [item['keypoints_3d'] for item in items]
    # batch['cuboids'] = [item['cuboids'] for item in items]
    batch['indexes'] = [item['indexes'] for item in items]

    try:
        batch['pred_keypoints_3d'] = np.array([item['pred_keypoints_3d'] for item in items])
    except:
        pass

    return batch

def get_data(config):
    val_dataset = human36m.Human36MMultiViewDataset(
        h36m_root=config.dataset.val.h36m_root,
        pred_results_path=config.dataset.val.pred_results_path if hasattr(config.dataset.val, "pred_results_path") else None,
        train=False,
        test=True,
        image_shape=config.image_shape if hasattr(config, "image_shape") else (256, 256),
        labels_path=config.dataset.val.labels_path,
        with_damaged_actions=config.dataset.val.with_damaged_actions,
        retain_every_n_frames_in_test=config.dataset.val.retain_every_n_frames_in_test,
        scale_bbox=config.dataset.val.scale_bbox,
        kind=config.kind,
        undistort_images=config.dataset.val.undistort_images,
        ignore_cameras=config.dataset.val.ignore_cameras if hasattr(config.dataset.val, "ignore_cameras") else [],
        crop=config.dataset.val.crop if hasattr(config.dataset.val, "crop") else True,
    )

    sample = val_dataset.__getitem__(0)
    # print(sample)
    print('images num: ', len(sample['images']))

    batch = collate_fn([sample])

    images_batch, keypoints_3d_gt, _, proj_matricies_batch = dataset_utils.prepare_batch(batch, device, config)
    print('keypoints_3d_gt', keypoints_3d_gt)
    
    return images_batch, proj_matricies_batch, batch

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default='experiments/human36m/eval/human36m_vol_softmax_debug.yaml', help="Path, where config file is stored")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    config = cfg.load_config(args.config)
    device = torch.device(7)

    #
    images_batch, proj_matricies_batch, batch = get_data(config)
    print('images', images_batch)
    # print('cameras 1', np.array(batch['cameras']).shape)
    print('cameras 1', batch['cameras'][0][0].projection)
    print('cameras 2', batch['cameras'][1][0].projection)
    print('pred_keypoints_3d', batch['pred_keypoints_3d'][0])

    # 
    model = VolumetricTriangulationNet(config, device = device).to(device)
    if config.model.init_weights:
        state_dict = torch.load(config.model.checkpoint)
        for key in list(state_dict.keys()):
            new_key = key.replace("module.", "")
            state_dict[new_key] = state_dict.pop(key)
        model.load_state_dict(state_dict, strict=True)
        print("Successfully loaded pretrained weights for whole model")

    #
    model.eval()
    with torch.no_grad():
        keypoints_3d_pred, heatmaps_pred, volumes_pred, \
        confidences_pred, cuboids_pred, coord_volumes_pred, \
        base_points_pred = model(images_batch, proj_matricies_batch, batch)
    
    print('keypoints_3d_pred', keypoints_3d_pred)
    
