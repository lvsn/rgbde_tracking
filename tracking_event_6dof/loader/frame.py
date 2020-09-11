import os
from pyquaternion import Quaternion
import pandas as pd
import numpy as np
import torch

from tracking_event_6dof.utils.data import delta_transform
from ulaval_6dof_object_tracking.utils.frame import Frame
from ulaval_6dof_object_tracking.utils.transform import Transform


class Poses:
    def __init__(self, poses, id):
        self.poses = poses
        self.id = id

    def is_on_disk(self):
        return self.poses is None

    def exists(self, path):
        return os.path.exists(os.path.join(path, self.id, 'poses.pkl'))

    def get_poses(self, path, seq_index, dt, keep_in_ram=False):
        raise NotImplementedError

    def clear(self):
        self.poses = None

    def dump(self, path):
        if not self.is_on_disk():
            if not os.path.exists(os.path.join(path, self.id)):
                os.makedirs(os.path.join(path, self.id))
            self.poses.to_pickle(os.path.join(path, self.id, 'poses.pkl'))
            self.clear()

    def load(self, path):
        self.poses = pd.read_pickle(os.path.join(path, self.id, "poses.pkl"))


class PosesNumpy(Poses):
    def get_poses(self, path, seq_index, dt, keep_in_ram=False):
        if self.is_on_disk():
            self.load(path)
        poses = self.poses

        poseA = Transform.from_matrix(poses[0])
        poseB = Transform.from_matrix(poses[1])

        delta = delta_transform(poseA, poseB).to_parameters(isDegree=True)
        delta = torch.tensor(delta).type(torch.FloatTensor)

        if not keep_in_ram:
            self.clear()
        return [delta, poseA, poseB]

    def dump(self, path):
        if not self.is_on_disk():
            if not os.path.exists(os.path.join(path, self.id)):
                os.makedirs(os.path.join(path, self.id))
            np.save(os.path.join(path, self.id, 'poses.npy'), self.poses)
            self.clear()

    def exists(self, path):
        return os.path.exists(os.path.join(path, self.id, 'poses.npy'))

    def load(self, path):
        self.poses = np.load(os.path.join(path, self.id, "poses.npy"))


class EventsRaw:
    def __init__(self, events, id):
        self.events = events
        self.id = id

    def is_on_disk(self):
        return self.events is None

    def exists(self, path):
        return os.path.exists(os.path.join(path, self.id, 'events.pkl'))

    def get_events(self, path, seq_index, dt, keep_in_ram=False):
        if self.is_on_disk():
            self.load(path)
        events = self.events

        start_t = seq_index*dt
        end_t = (seq_index+1)*dt
        events = events[(events.time >= start_t) & (events.time < end_t)]

        if not keep_in_ram:
            self.clear()
        return events

    def clear(self):
        self.events = None

    def dump(self, path):
        if not self.is_on_disk():
            if not os.path.exists(os.path.join(path, self.id)):
                os.makedirs(os.path.join(path, self.id))
            self.events.to_pickle(os.path.join(path, self.id, 'events.pkl'))
            self.clear()

    def load(self, path):
        self.events = pd.read_pickle(os.path.join(path, self.id, "events.pkl"))


class FrameNumpy(Frame):
    def __init__(self, rgb, depth, id, first=True):
        super(FrameNumpy, self).__init__(rgb, depth, id)
        self.first = first

    def dump(self, path):
        if not self.is_on_disk():
            if not os.path.exists(os.path.join(path, self.id)):
                os.makedirs(os.path.join(path, self.id))
            depth8 = self.numpy_int16_to_uint8(self.depth)
            frame = np.concatenate((self.rgb, depth8), axis=2)
            if self.first:
                np.save(os.path.join(path, self.id, 'a_frame.npy'), frame)
            else:
                np.save(os.path.join(path, self.id, 'b_frame.npy'), frame)
            self.clear_image()

    def load(self, path):
        if self.first:
            frame = np.load(os.path.join(path, self.id, 'a_frame.npy'))
        else:
            frame = np.load(os.path.join(path, self.id, 'b_frame.npy'))
        self.depth = self.numpy_uint8_to_int16(frame[:, :, 3:])
        self.rgb = frame[:, :, 0:3]

    @staticmethod
    def numpy_int16_to_uint8(depth):
        x, y = depth.shape
        out = np.ndarray((x, y, 2), dtype=np.uint8)
        out[:, :, 0] = np.right_shift(depth, 8)
        out[:, :, 1] = depth.astype(np.uint8)
        return out

    @staticmethod
    def numpy_uint8_to_int16(depth8):
        x, y, c = depth8.shape
        out = np.ndarray((x, y), dtype=np.int16)
        out[:, :] = depth8[:, :, 0]
        out = np.left_shift(out, 8)
        out[:, :] += depth8[:, :, 1]
        return out


class FrameNone(Frame):
    def __init__(self, *args, **kargs):
        pass

    def dump(self, path):
        pass
