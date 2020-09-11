import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from enum import Enum
import random

from ulaval_6dof_object_tracking.utils.data import compute_2Dboundingbox
from ulaval_6dof_object_tracking.utils.transform import Transform
from tracking_event_6dof.utils.data import delta_transform


class OffsetDepth(object):
    """
    Source : https://github.com/lvsn/6DOF_tracking_evaluation/blob/master/ulaval_6dof_object_tracking/deeptrack/data_augmentation.py
    """

    def __call__(self, data):
        rgbA, depthA, rgbB, depthB, prior = data
        prior_T = Transform.from_parameters(*prior)
        depthA = self.normalize_depth(depthA, prior_T)
        depthB = self.normalize_depth(depthB, prior_T)
        return rgbA.astype(np.float32), depthA, rgbB.astype(np.float32), depthB, prior

    @staticmethod
    def normalize_depth(depth, pose):
        depth = depth.astype(np.float32)
        zero_mask = depth == 0
        depth += pose.matrix[2, 3] * 1000
        depth[zero_mask] = 5000
        return depth


class NormalizeFrame:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        frames = []
        for index_i, (rgb, depth) in enumerate(data):
            rgbd = np.concatenate((rgb, depth[:, :, np.newaxis]), axis=2)
            rgbd = torch.tensor(rgbd.astype(int)).permute(2, 1, 0).float()
            for index_j in range(4):
                rgbd[index_j] -= self.mean[index_i*4+index_j]
                rgbd[index_j] /= self.std[index_i*4+index_j]
            frames.append(rgbd)
        return frames


class NormalizeEvent:
    def __init__(self, max_value, timestamp=False):
        self.max_value = max_value
        self.timestamp = timestamp

    def __call__(self, data):
        if isinstance(data, list):
            data_selected = data[0]
        else:
            data_selected = data

        if self.timestamp:
            data_selected[:2] /= self.max_value
        else:
            data_selected[:] /= self.max_value
        return data


class CropBoundingBox:
    def __init__(self, size, camera, object_width, std=0, interpolate='bilinear'):
        self.std = std
        self.size = size
        self.camera = camera
        self.object_width = object_width
        self.interpolate = interpolate

    def __call__(self, data, target=None, poseA=None):
        if type(target) != type(None):
            if isinstance(target, (list, tuple)):
                poseA = target[1]
            else:
                poseA = target

        data = data.unsqueeze(0)
        self.camera.center_y = self.camera.height - self.camera.center_y
        bb = compute_2Dboundingbox(
            poseA, self.camera, self.object_width, scale=(1000, -1000, -1000))
        self.camera.center_y = self.camera.height - self.camera.center_y
        left = np.min(bb[:, 1])
        right = np.max(bb[:, 1])
        top = np.min(bb[:, 0])
        bottom = np.max(bb[:, 0])

        amp = np.random.normal(0, self.std, 1)
        direction = np.random.uniform(0, 2*np.pi)
        offset_bb = amp * np.array([np.cos(direction), np.sin(direction)])
        h = data.shape[3]
        w = data.shape[2]
        left = max(0, left + int(offset_bb[0]))
        right = min(w, right + int(offset_bb[0]))
        top = max(0, top + int(offset_bb[1]))
        bottom = min(h, bottom + int(offset_bb[1]))

        data = data[:, :, left:right, top:bottom]
        try:
            data = F.interpolate(data, size=self.size,
                                 mode=self.interpolate).squeeze(0)
        except RuntimeError:
            print("CROP OUT OF FRAME")
            device = data.device
            data = torch.zeros((data.shape[1], self.size[0], self.size[1]))
            data.to(device)

        if target:
            return data, target
        else:
            return data


class EventSpikeTensor:
    # https://arxiv.org/abs/1909.05190
    def __init__(self, size, bins=9, delta_ms=1000/33, device="cpu", noise=True,
                 mean=(262.07047, 416.94632), std=(21.8644, 39.653587)):
        self.size = size
        self.bins = bins
        self.bins_t = np.linspace(0, delta_ms*1000*1000, bins+1)

        self.device = device
        self.mean = mean
        self.std = std
        self.noise = noise

    def __call__(self, events, target=None):
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        events = events[['Polarity', 'time', 'X', 'Y']]
        events = events.to_numpy().astype(int)
        events = torch.tensor(events).to(device)
        events[:, 2] = events[:, 2]*self.size[1] + events[:, 3]

        frame = torch.zeros((self.bins, self.size[0]*self.size[1])).to(device)
        for i in range(self.bins):
            t_mask = (self.bins_t[i] < events[:, 1]) & (
                events[:, 1] < self.bins_t[i+1])
            coords = events[:, 2][t_mask]
            p_mask = events[:, 0][t_mask] > 0
            coords_p, values_p = torch.unique(
                coords[p_mask], return_counts=True)
            coords_n, values_n = torch.unique(
                coords[~p_mask], return_counts=True)
            frame[i, coords_n] += values_n
            frame[i, coords_p] -= values_p
        frame = frame.reshape((self.bins, ) + self.size).to(device)

        if self.noise:
            nb = np.random.normal(self.mean, self.std)
            for polarity in range(2):
                xs = np.random.randint(
                    0, high=frame.shape[1], size=int(nb[polarity]))
                ys = np.random.randint(
                    0, high=frame.shape[2], size=int(nb[polarity]))
                bins = np.random.randint(
                    0, high=self.bins, size=int(nb[polarity]))
                for x, y, i in zip(xs, ys, bins):
                    if polarity == 0:
                        frame[i, x, y] -= 1
                    else:
                        frame[i, x, y] += 1

        if target:
            return frame, target
        else:
            return frame
