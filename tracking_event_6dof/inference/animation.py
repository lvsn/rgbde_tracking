import os
import numpy as np
import imageio
from collections.abc import Iterable
from skimage.transform import resize
from PIL import Image, ImageDraw, ImageFont

from tracking_event_6dof.utils.data import calculate_center_pose
from ulaval_6dof_object_tracking.utils.data import normalize_scale


class Animation:
    def __init__(self, size, path, render, labels=None, fps=15, quality=10):
        self.writer = imageio.get_writer(path, fps=fps, quality=quality)
        self.size = size
        self.render = render
        self.labels = labels

    def overlay_pose(self, frame_bg, pose, color=0):
        frame, _, _ = self.render.poseB(pose)
        frame = frame.copy()
        mask = frame.sum(axis=2)[:, :] > 0
        frame[~mask, :] = frame_bg[~mask, :]

        if color == 0:
            frame[mask, 2] = frame[mask, 0]
            frame[mask, 1] = frame[mask, 0]
            frame[mask, 0] = 0
        elif color == 1:
            frame[mask, 2] = frame[mask, 0]
            frame[mask, 1] = 0
            frame[mask, 0] = frame[mask, 0]
        else:
            frame[mask, 2] = 0
            frame[mask, 0] = frame[mask, 0]
            frame[mask, 1] = frame[mask, 0]
        frame[mask] = 0.75*frame[mask] + 0.25*frame_bg[mask]
        return frame

    def _add_frame(self, poses, frame_truth, label, fail, event_bb):
        if not isinstance(poses, Iterable):
            poses = [poses]

        for color_index, pose in enumerate(poses):
            color_index = len(poses) - color_index - 1
            frame_truth = self.overlay_pose(frame_truth, pose, color_index)
        frame = frame_truth

        center_x, center_y, length, _ = \
            calculate_center_pose(pose, self.render.camera,
                                  self.render.object_width)
        delta = length//2
        left = int(max(0, center_x-delta))
        right = int(min(frame.shape[1], center_x+delta))
        top = int(max(0, center_y-delta))
        bottom = int(min(frame.shape[0], center_y+delta))
        crop_part = frame[top:bottom, left:right, :]
        crop_part = (resize(crop_part, (200, 200))*255).astype(np.uint8)
        crop_part[200-2:200] = 255
        crop_part[:, 200-2:200] = 255
        frame[0:200, 0:200] = crop_part

        if event_bb is not None:
            import cmapy
            event_bb = event_bb.sum(0)
            event_bb -= event_bb.min()
            event_bb /= event_bb.max()
            event_bb *= 255
            event_bb = event_bb.astype(np.uint8)
            event_bb = np.swapaxes(event_bb, 0, 1)
            event_bb = cmapy.colorize(event_bb, 'viridis')
            frame[:event_bb.shape[0], -event_bb.shape[1]:] = event_bb

        frame[-2:, :] = 255
        frame[:2, :] = 255
        frame[:, -2:, :] = 255
        frame[:, :2, :] = 255

        if label or fail is not None:
            pil_img = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil_img)
            if label:
                position = (frame.shape[1]//2, 10)
                draw.text(position, label, fill=(255, 255, 255))
            if fail is not None:
                font = ImageFont.truetype(
                        os.path.join(__file__, '..', 'utils', 'DejaVuSans.ttf'), size=30)
                position = (frame.shape[1]-50, -50)
                draw.text(position, str(fail), fill=(255, 255, 255), font=font)
            frame = np.array(pil_img)
        return frame

    def add_frame(self, poses, frame_truth, fail=None, event_bb=None):
        camera_height = self.render.camera.height
        camera_width = self.render.camera.width
        frame = np.zeros((camera_height*self.size[1],
                          camera_width*self.size[0], 3))
        for j in range(self.size[1]):
            for i in range(self.size[0]):
                index = j*self.size[0] + i
                if index >= len(poses):
                    break
                if isinstance(frame_truth, (list, tuple)):
                    truth = frame_truth[index]
                else:
                    truth = frame_truth

                left = i*camera_width
                right = (i+1)*camera_width
                top = j*camera_height
                bottom = (j+1)*camera_height

                if event_bb is not None:
                    current_event_bb = event_bb[index]
                else:
                    current_event_bb = None

                if fail is not None:
                    current_fail = fail[index]
                else:
                    current_fail = None

                if self.labels:
                    label = self.labels[index]
                else:
                    label = None
                frame[top:bottom, left:right, :] = \
                    self._add_frame(poses[index], truth,
                                    label, current_fail, current_event_bb)
        self.writer.append_data(frame)

    def __del__(self):
        self.writer.close()
