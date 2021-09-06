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

from mvn.models.triangulation import ModifiedVolTriNet, RANSACTriangulationNet, AlgebraicTriangulationNet, VolumetricTriangulationNet
from mvn.models.loss import KeypointsMSELoss, KeypointsMSESmoothLoss, KeypointsMAELoss, KeypointsL2Loss, VolumetricCELoss

from mvn.utils import img, multiview, op, vis, misc, cfg
from mvn.datasets import human36m
from mvn.datasets import utils as dataset_utils
from mvn.utils.img import get_square_bbox, resize_image, crop_image, normalize_image, scale_bbox


torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True)

# def collate_fn(items):
#     items = list(filter(lambda x: x is not None, items))
#     if len(items) == 0:
#         print("All items in batch are None")
#         return None

#     batch = dict()
#     total_n_views = min(len(item['images']) for item in items)

#     indexes = np.arange(total_n_views)
#     randomize_n_views = False
#     if randomize_n_views:
#         n_views = np.random.randint(min_n_views, min(total_n_views, max_n_views) + 1)
#         indexes = np.random.choice(np.arange(total_n_views), size=n_views, replace=False)
#     else:
#         indexes = np.arange(total_n_views)

#     batch['images'] = np.stack([np.stack([item['images'][i] for item in items], axis=0) for i in indexes], axis=0).swapaxes(0, 1)
#     batch['detections'] = np.array([[item['detections'][i] for item in items] for i in indexes]).swapaxes(0, 1)
#     batch['cameras'] = [[item['cameras'][i] for item in items] for i in indexes]
#     batch['keypoints_3d'] = [item['keypoints_3d'] for item in items]
#     # batch['cuboids'] = [item['cuboids'] for item in items]
#     batch['indexes'] = [item['indexes'] for item in items]

#     try:
#         batch['pred_keypoints_3d'] = np.array([item['pred_keypoints_3d'] for item in items])
#     except:
#         pass

#     return batch

# def get_data(config):
#     val_dataset = human36m.Human36MMultiViewDataset(
#         h36m_root=config.dataset.val.h36m_root,
#         pred_results_path=config.dataset.val.pred_results_path if hasattr(config.dataset.val, "pred_results_path") else None,
#         train=False,
#         test=True,
#         image_shape=config.image_shape if hasattr(config, "image_shape") else (256, 256),
#         labels_path=config.dataset.val.labels_path,
#         with_damaged_actions=config.dataset.val.with_damaged_actions,
#         retain_every_n_frames_in_test=config.dataset.val.retain_every_n_frames_in_test,
#         scale_bbox=config.dataset.val.scale_bbox,
#         kind=config.kind,
#         undistort_images=config.dataset.val.undistort_images,
#         ignore_cameras=config.dataset.val.ignore_cameras if hasattr(config.dataset.val, "ignore_cameras") else [],
#         crop=config.dataset.val.crop if hasattr(config.dataset.val, "crop") else True,
#     )

#     sample = val_dataset.__getitem__(0)
#     # print(sample)
#     print('images num: ', len(sample['images']))

#     batch = collate_fn([sample])

#     images_batch, keypoints_3d_gt, _, proj_matricies_batch = dataset_utils.prepare_batch(batch, device, config)
#     print('keypoints_3d_gt', keypoints_3d_gt)
    
#     return images_batch, proj_matricies_batch, batch

def get_data(device):
    print('1.1')

    #
    img_path1 = './data/human36m/processed/S9/Directions-1/imageSequence/55011271/img_000001.jpg'
    bbox1 = (294, 215, 700, 621)    #LTRB
    image1 = cv2.imread(img_path1)

    R1 = np.array([[ 0.931572  ,  0.3634829 , -0.00732918],
                [ 0.0681007 , -0.19426748, -0.97858185],
                [-0.35712156,  0.91112036, -0.20572759]])
    t1 = np.array([[  19.193213],
                    [ 404.2284  ],
                    [5702.169   ]])
    K1 = np.array([[1149.6757 ,    0. ,       508.84863],
                    [   0.    ,  1147.5917 ,  508.0649 ],
                    [   0.    ,     0.     ,    1.     ]])
    dist1 = np.array([-0.19421363,  0.24040854, -0.00274089, -0.00161903,  0.00681998])
    camera1 = multiview.Camera(R1, t1, K1, dist1)
    print('cameras 1 befor', camera1.projection)

    
    image1 = crop_image(image1, bbox1)
    camera1.update_after_crop(bbox1)
    image_shape_before_resize = image1.shape[:2]
    image1 = resize_image(image1, [384, 384])
    camera1.update_after_resize(image_shape_before_resize, [384, 384])
    image1 = normalize_image(image1).transpose(2,0,1)
    print('cameras 1', camera1.projection)

    print('1.2')

    #
    img_path2 = './data/human36m/processed/S9/Directions-1/imageSequence/60457274/img_000001.jpg'
    image2 = cv2.imread(img_path2)
    bbox2 = (245, 128, 775, 658)    
    R2 = np.array([[ 0.9154607 , -0.39734608,  0.0636223 ],
       [-0.04940629, -0.26789168, -0.9621814 ],
       [ 0.3993629 ,  0.8776959 , -0.2648757 ]])
    t2 = np.array([[ -69.27132],
       [ 422.1843 ],
       [4457.8936 ]])
    K2 = np.array([[1145.5114  ,   0.   ,    514.9682 ],
                    [   0.   ,   1144.7739 ,  501.88202],
                    [   0.   ,      0.   ,      1.     ]])
    dist2 = np.array([-0.19838409,  0.21832368, -0.00181336, -0.00058721, -0.00894781])
    camera2 = multiview.Camera(R2, t2, K2, dist2)
    print('cameras 2 befor', camera2.projection)

    image2 = crop_image(image2, bbox2)
    camera2.update_after_crop(bbox2)
    image_shape_before_resize = image2.shape[:2]
    image2 = resize_image(image2, [384, 384])
    camera2.update_after_resize(image_shape_before_resize, [384, 384])
    image2 = normalize_image(image2).transpose(2,0,1)
    print('cameras 2', camera2.projection)

    print('1.3')
    
    #
    images = torch.from_numpy(np.array([[image1, image2]])).float().to(device)
    print('1.4')

    cameras = np.array([[camera1], [camera2]])
    print('1.5')
    
    pred_keypoints_3d = np.array([[-231.9645 ,      1.1912533 ,  79.543144 ],
                                    [-191.73877  ,  -75.70233 ,   539.3524   ],
                                    [-174.86665 ,  -120.261795 ,  996.0705   ],
                                    [  86.050644,  -130.14496  ,  972.63446  ],
                                    [  98.70048 ,   -92.295425,   512.3088   ],
                                    [ 135.1022 ,    -30.561798,    49.469345 ],
                                    [ -44.298347 , -126.237785,   985.7758   ],
                                    [ -42.940315 ,  -96.19707 ,  1247.1335   ],
                                    [ -27.393295 ,  -98.28345,   1511.5562   ],
                                    [ -30.933146 ,  -86.36923 ,  1702.5986   ],
                                    [-742.1502   , -139.97827 ,  1401.3849   ],
                                    [-502.098   ,  -102.70539 ,  1408.9042   ],
                                    [-199.99274 ,   -71.2198  ,  1475.1078   ],
                                    [ 141.05072  ,  -95.80665 ,  1464.4432   ],
                                    [ 441.60175  ,  -89.860634 , 1395.7427   ],
                                    [ 665.36633  , -158.52461 ,  1373.0132   ],
                                    [ -30.979435,  -148.00125 ,  1606.1694   ]])
    print('1.6')

    return images, cameras, pred_keypoints_3d

if __name__ == '__main__':

    device = torch.device(7)
    print('1')
    #
    images, cameras, pred_keypoints_3d = get_data(device)
    print('2')

    print('images ', images)
    print('cameras 1', cameras[0][0].projection)
    print('cameras 2', cameras[1][0].projection)
    print('pred_keypoints_3d', pred_keypoints_3d)

    # 
    model = ModifiedVolTriNet(device = device).to(device)
    checkpoint = "./data/pretrained/human36m/human36m_vol_softmax_10-08-2019/checkpoints/0040/weights.pth"
    state_dict = torch.load(checkpoint)
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
        base_points_pred = model(images, cameras, pred_keypoints_3d)

    print('keypoints_3d_pred', keypoints_3d_pred)
    
