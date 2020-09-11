import numpy as np
import torch
import torch.nn.functional as F
import yaml
import os

from ulaval_6dof_object_tracking.deeptrack.deeptrack_net import DeepTrackNet
from ulaval_6dof_object_tracking.utils.data import normalize_scale, combine_view_transform, compute_2Dboundingbox
from ulaval_6dof_object_tracking.utils.transform import Transform

from tracking_event_6dof.network.deeptrack_net import DeepTrackNetSpike
from tracking_event_6dof.loader.data_augmentation import OffsetDepth, NormalizeEvent, CropBoundingBox, EventSpikeTensor, NormalizeFrame

class Tracker:
    def __init__(self, model, model_path, pose_origin, render, camera, channel_in=0):
        self.render = render
        self.camera = camera
        self.current_pose = pose_origin
        self.repeat = 1

        metadata_path = os.path.join(model_path, "meta.yml")
        model_path = os.path.join(model_path, "model_best.pth.tar")

        with open(metadata_path) as data_file:
            data = yaml.load(data_file)
        self.metadata = data
        self.metadata['translation_range'] = float(
            self.metadata['translation_range'])
        self.metadata['rotation_range'] = float(
            self.metadata['rotation_range'])
        image_size = self.metadata['image_size']
        # Support legacy metadata format
        self.image_size = image_size

        if model:
            if channel_in:
                model = model(
                    image_size=self.image_size[0], channel_in=channel_in)
            else:
                model = model(image_size=self.image_size[0])
            model.load(model_path)
            model.eval()
            if torch.cuda.is_available():
                model.cuda()
        self.model = model

        self.poses = [self.current_pose.to_parameters(rodrigues=False)]

    def _predict(self, frame):
        raise NotImplementedError

    def predict(self, frame):
        try:
            for _ in range(self.repeat):
                prediction = self._predict(frame)
                prediction_transform = Transform.from_parameters(
                    *prediction, is_degree=self.is_degree)
                self.current_pose = combine_view_transform(
                    self.current_pose, prediction_transform)
            self.poses.append(self.current_pose.to_parameters(rodrigues=False))
            return prediction
        except ValueError:
            return None

    def reset(self, pose):
        raise NotImplementedError


class TrackerFrame(Tracker):
    def __init__(self, model_path, pose_origin, render, camera, repeat=3):
        super().__init__(DeepTrackNet, model_path, pose_origin, render, camera)
        self.is_degree = True
        self.repeat = 3
        mean = torch.Tensor(self.metadata["mean"])
        std = torch.Tensor(self.metadata["std"])
        self.normalize_frame = NormalizeFrame(mean, std)

    def _predict(self, frame):
        if frame is None:
            return np.zeros(6)
        rgbA, depthA, points = self.render.poseA(self.current_pose)
        bb = self.render.camera.project_points(
            points, y_negative=True).astype(np.int32)
        rgbB, depthB = normalize_scale(*frame, bb, output_size=self.image_size)
        depthB[:-3, :-3] = depthB[3:, 3:]

        depthA = depthA.astype(np.float32)
        depthA = OffsetDepth.normalize_depth(depthA, self.current_pose)
        depthB = depthB.astype(np.float32)
        depthB = OffsetDepth.normalize_depth(depthB, self.current_pose)

        rgbdA, rgbdB = self.normalize_frame(
            [(rgbA, depthA), (rgbB, depthB)])

        if torch.cuda.is_available():
            rgbdA = rgbdA.cuda()
            rgbdB = rgbdB.cuda()

        prediction = self.model(rgbdA.unsqueeze(0), rgbdB.unsqueeze(0))[0]
        prediction = prediction.detach().cpu().numpy()
        prediction[:3] *= float(self.metadata['translation_range'])
        prediction[3:] *= self.metadata["rotation_range"]

        return prediction

    # TODO : Move it
    def _normalize_mean_std(self, imgs, mean, std):
        frames = []
        for index_i, (rgb, depth) in enumerate(imgs):
            rgbd = np.concatenate((rgb, depth[:, :, np.newaxis]), axis=2)
            rgbd = torch.tensor(rgbd.astype(int)).permute(2, 1, 0).float()
            for index_j in range(4):
                rgbd[index_j] -= mean[index_i*4+index_j]
                rgbd[index_j] /= std[index_i*4+index_j]
            frames.append(rgbd)
        return frames

    def reset(self, pose):
        self.current_pose = pose


class TrackerEvent(Tracker):
    def __init__(self, model_path, pose_origin, render, camera, spike=True, network=DeepTrackNetSpike):
        super().__init__(network, model_path, pose_origin, render, camera, channel_in=9)
        self.is_degree = True
        if spike:
            self.event_frame = EventSpikeTensor(
                (camera.width, camera.height), noise=False)
        else:
            raise NotImplementedError
        self.normalize_event = NormalizeEvent(self.metadata['max_value'])
        self.crop_bb = CropBoundingBox(
            self.image_size, self.camera, self.metadata['object_width'])
        self.last_frame = None

    def _predict(self, events, frame=None):
        if not np.any(frame):
            frame = self.event_frame(events)
        frame = self.crop_bb(frame, poseA=self.current_pose)
        frame = self.normalize_event(frame)

        if torch.cuda.is_available():
            frame = frame.cuda()

        prediction = self.model(frame.unsqueeze(0))[0]

        prediction = prediction.detach().cpu().numpy()
        prediction[:3] *= float(self.metadata['translation_range'])
        prediction[3:] *= self.metadata["rotation_range"]
        return prediction


class TrackerHybrid(Tracker):
    def __init__(self, model_frame_path, model_event_path, pose_origin, render, camera_event, camera_frame, transform=None):
        super().__init__(None, model_frame_path, pose_origin, render, None)

        self.transform = transform
        self.tracker_event = TrackerEvent(
            model_event_path, pose_origin, render, camera_event)
        self.tracker_frame = TrackerFrame(
            model_frame_path, pose_origin, render, camera_event)
        self.diff_pose = np.zeros(6)

        self.i = 0

    def to_event(self, pose):
        pose_f_flip = Transform.scale(1, -1, -1).combine(pose, copy=True)
        pose_e_flip = self.transform.combine(pose_f_flip, copy=True)
        pose_e = Transform().scale(1, -1, -1).combine(pose_e_flip, copy=True)
        return pose_e

    def to_frame(self, pose):
        pose_e_flip = Transform().scale(1, -1, -1).combine(pose, copy=True)
        pose_f_flip = self.transform.inverse().combine(pose_e_flip, copy=True)
        new_pose_f = Transform().scale(1, -1, -1).combine(pose_f_flip, copy=True)
        return new_pose_f

    def predict(self, event_frame, rgb_frame):
        self.tracker_event.current_pose = self.to_event(
            self.tracker_frame.current_pose)
        pose = self.tracker_event._predict(event_frame)
        self.last_event_frame = self.tracker_event.last_frame

        prediction = Transform.from_parameters(
            *pose, is_degree=self.tracker_event.is_degree)
        new_pose_e = combine_view_transform(
            self.tracker_event.current_pose, prediction)
        new_pose_f = self.to_frame(new_pose_e)

        self.tracker_frame.current_pose = new_pose_f
        self.tracker_event.current_pose = new_pose_e

        self.tracker_frame.predict(rgb_frame)

        self.current_pose = (new_pose_f, self.tracker_frame.current_pose)
        self.poses.append((self.current_pose[0].to_parameters(rodrigues=False),
                           self.current_pose[1].to_parameters(rodrigues=False)))

    def reset(self, pose):
        pose_e = self.to_event(pose)
        self.current_pose = (pose_e, pose)

        self.tracker_frame.current_pose = pose
        self.tracker_event.current_pose = pose_e
