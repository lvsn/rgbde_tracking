import os
import json
import torch
import uuid
import numpy as np

from tqdm import tqdm

from ulaval_6dof_object_tracking.utils.transform import Transform
from pytorch_toolbox.loader_base import LoaderBase

from tracking_event_6dof.utils.camera import Camera
from tracking_event_6dof.loader.frame import EventsRaw, Poses, PosesNumpy, FrameNumpy, FrameNone


class DeepTrackLoaderBase(LoaderBase):
    def __init__(self, root, event_type, data_transform=None, target_transform=None, pre_transform=None, read_data=True, frame_number=1, pose_type='numpy'):
        self.data = {}
        self.ids = []
        self.read_data = read_data
        self.set_save_type('numpy')
        self.set_event_type(event_type)
        self.set_poses_type(pose_type)
        self.frame_number = frame_number
        self.data_transform = data_transform
        self.target_transform = target_transform
        self.pre_transform = pre_transform
        super(DeepTrackLoaderBase, self).__init__(root, [], [])

    def _list_dir(self, path):
        return os.listdir(path)

    def load(self, path):
        raise NotImplementedError

    def make_dataset(self, dir):
        if self.read_data:
            try:
                self.load(dir)
            except FileNotFoundError as e:
                print("[Warning] no dataset saved at path {}".format(dir))
                print(e)
                print("Resuming...")
        return self.ids

    def from_index(self, index):
        raise RuntimeError("Not Implemented")

    def __len__(self):
        return len(self.ids)

    def set_save_type(self, frame_class):
        if frame_class == "numpy":
            self.frame_class = FrameNumpy
        elif frame_class == "none":
            self.frame_class = FrameNone
        elif frame_class == "hdf5":
            raise RuntimeError("Not Implemented")
        else:
            raise RuntimeError("Not Implemented")

    def set_event_type(self, event_type):
        if event_type == "raw":
            self.event_class = EventsRaw
        elif event_type == "frame":
            self.event_class = EventsFrame
        else:
            raise RuntimeError("Not Implemented")

    def set_poses_type(self, pose_type):
        if pose_type == "raw":
            self.pose_class = Poses
        elif pose_type == "numpy":
            self.pose_class = PosesNumpy
        else:
            raise RuntimeError("Not Implemented")

    def _new_id(self):
        return str(uuid.uuid4().int & 2**64-1)

    def add_frame_init(self, rgb, depth):
        index = self._new_id()
        frame = self.frame_class(rgb, depth, index)
        self.data[index] = [frame]
        self.ids.append(index)
        return index

    def add_pair(self, rgb, depth, poses, events, id):
        frame = self.frame_class(rgb, depth, id, first=False)
        event = self.event_class(events, id)
        poses = self.pose_class(poses, id)
        self.data[id].append(frame)
        self.data[id].append(event)
        self.data[id].append(poses)

    def __getitem__(self, index):
        data, target = self.from_index(index)

        if self.pre_transform:
            data, target = self.pre_transform(data, target)
        if self.data_transform:
            data = self.data_transform(data)
        if self.target_transform:
            target = self.target_transform(target)

        return data, target
