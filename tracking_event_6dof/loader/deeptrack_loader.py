import os
import numpy as np
import pandas as pd
import torch

from tracking_event_6dof.loader.deeptrack_loader_base import DeepTrackLoaderBase
from tracking_event_6dof.loader.frame import FrameNone
from tracking_event_6dof.utils.camera import Camera
from tracking_event_6dof.utils.data import delta_transform

from ulaval_6dof_object_tracking.utils.transform import Transform


class RGBDELoader(DeepTrackLoaderBase):
    def __init__(self, root, data_transform=None, target_transform=None,
                 pre_transform=None, is_frame=True, read_data=True, event_type="raw", pose_type='numpy', ignore_frameA=False):
        self.is_frame = is_frame
        super(RGBDELoader, self).__init__(root, event_type, data_transform,
                                          target_transform, pre_transform, frame_number=1,
                                          read_data=read_data, pose_type=pose_type)
        self.ignore_frameA = ignore_frameA

    def from_index(self, index):
        if self.is_frame:
            if self.ignore_frameA:
                id, _ = self.ids[index]
                _, frameB, _, poses = self.data[id]
                rgbdB = frameB.get_rgb_depth(self.root)
                data = [None, rgbdB]
                poses = poses.get_poses(self.root, 0, self.dt_frame)
                poseA = poses[2]
                poses[2] = poses[1]
                poses[1] = poseA
            else:
                data, poses = self.load_rgbd(index)
        else:
            data, poses = self.load_events(index)
        return data, poses

    def load(self, path):
        self.camera = Camera.load_from_json(path)
        self.event_camera = Camera.load_from_json(path, filename='dvs')
        self.dt_event = 1e8/3
        self.dt_frame = 1e8/3

        self.path = path
        self.load_data()

        self.raw_events = np.load(os.path.join(path, 'fevents.npz'))
        self.raw_events = self.raw_events[self.raw_events.files[0]]
        self.raw_events = pd.DataFrame(data=self.raw_events,
                                       columns=['Timestamp', 'X', 'Y', 'Polarity'])
        self.raw_events.Timestamp *= 1000
        self.raw_events = self.raw_events[['Polarity', 'Timestamp', 'X', 'Y']]

        self.dead_pixel = [(151, 205)]
        for dead_pixel in self.dead_pixel:
            self.raw_events = self.raw_events[~(
                (self.raw_events.X == dead_pixel[0]) & (self.raw_events.Y == dead_pixel[1]))]

        self.frame_ts = np.load(os.path.join(path, 'ts_frames.npz'))
        self.frame_ts = self.frame_ts[self.frame_ts.files[0]]
        self.frame_ts *= 1000

        poses_path = os.path.join(path, 'poses.npy')
        if os.path.exists(poses_path):
            self.poses = np.load(poses_path)
        else:
            print("WARNING: Poses not found")
            self.poses = None

        self.matrix_transformation = Transform()
        np_mat = np.load(os.path.join(path, 'transfo_mat.npy'))
        np_mat[0:3, -1] /= 1000
        self.matrix_transformation.matrix = np_mat

        self.size = len(self.raw_frames)*self.frame_number - self.frame_number
        self.unload_data()

    def load_data(self):
        self.raw_frames = np.load(os.path.join(self.path, 'frames.npz'))
        self.raw_frames = self.raw_frames[self.raw_frames.files[0]]

    def unload_data(self):
        del self.raw_frames

    def _get_poses(self, indexA, indexB, transform=False):
        if self.poses is None:
            return None, None, None
        indexA = indexA
        indexB = indexB
        poseA = Transform.from_parameters(*self.poses[indexA], is_degree=True)
        poseB = Transform.from_parameters(*self.poses[indexB], is_degree=True)

        if transform:
            pose_A_flip = Transform.scale(1, -1, -1).combine(poseA)
            pose_A_flip = self.matrix_transformation.combine(
                pose_A_flip, copy=True)
            poseA = Transform().scale(1, -1, -1).combine(pose_A_flip, copy=True)

            pose_B_flip = Transform.scale(1, -1, -1).combine(poseB)
            pose_B_flip = self.matrix_transformation.combine(
                pose_B_flip, copy=True)
            poseB = Transform().scale(1, -1, -1).combine(pose_B_flip, copy=True)

        delta = delta_transform(poseA, poseB).to_parameters(
            isDegree=True).astype(np.float32)
        return delta, poseA, poseB

    def __len__(self):
        return len(self.raw_frames)-1

    def load_rgbd(self, index):
        pose = self._get_poses(index, index)[1]
        rgbd = self.raw_frames[index]
        rgbd = (rgbd[:, :, 2::-1].astype(np.uint8), rgbd[:, :, 3])
        return rgbd, pose

    def _get_events(self, index):
        start_t = self.frame_ts[index]
        end_t = self.frame_ts[index+1]
        delta = (end_t - start_t)
        end_t = start_t + delta

        events = self.raw_events[(self.raw_events.Timestamp >= start_t) & (
            self.raw_events.Timestamp < end_t)]

        events.rename(columns={'Timestamp': 'time'}, inplace=True)
        events.time -= start_t

        return events

    def load_events(self, index):
        pose = self._get_poses(index, index, transform=True)
        return self._get_events(index), pose
