from tracking_event_6dof.loader.deeptrack_loader import RGBDELoader
from tracking_event_6dof.utils.render import Render
from tracking_event_6dof.inference.tracker import TrackerEvent, TrackerFrame, TrackerHybrid
from tracking_event_6dof.inference.animation import Animation
from tracking_event_6dof.inference.tracker_comparator import Comparator

import argparse
import numpy as np
import torch
import os
from tqdm import tqdm
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tracking failures')
    parser.add_argument(
        '-f', '--modelf', help="Path to save RGB network", required=True)
    parser.add_argument(
        '-e', '--modele', help="Path to save event network", required=True)
    parser.add_argument('-d', '--dataset',
                        help="Path to datasets", required=True)
    parser.add_argument('-m', '--model3d',
                        help="Path to 3D model", required=True)
    parser.add_argument('-a', '--animation',
                        help="Path to save tracking video", required=False)

    arguments = parser.parse_args()

    model_frame_path = arguments.modelf
    model_event_path = arguments.modele
    datasets_path = arguments.dataset
    animation_path = arguments.animation
    model_3d_path = arguments.model3d

    frame_failures = 0
    hybrid_failures = 0

    with open(os.path.join(model_frame_path, 'meta.yml')) as f:
        frame_size = yaml.load(f)['image_size']

    for seq_name in os.listdir(datasets_path):
        loader_frame = RGBDELoader(os.path.join(datasets_path, seq_name))
        loader_frame.load_data()
        loader_event = RGBDELoader(os.path.join(
            datasets_path, seq_name), is_frame=False)

        vpRender = Render(loader_frame.camera, image_size=frame_size,
                          model_path=model_3d_path)

        if animation_path:
            assert os.path.isdir(animation_path)
            animation = Animation((2, 1), os.path.join(animation_path, f'{seq_name}.mp4'), vpRender,
                                  ['Deeptrack (Garon et al.)', 'Ours'])
        else:
            animation = None

        initial_pose = loader_frame[0][1]
        comparator = Comparator(
            len(loader_frame), initial_pose=initial_pose, animation=animation, reset=True)

        tracker_frame = TrackerFrame(
            model_frame_path, initial_pose, vpRender, loader_frame.camera)
        tracker_hybrid = TrackerHybrid(model_frame_path, model_event_path, initial_pose,
                                       vpRender, loader_event.event_camera, loader_frame.camera, transform=loader_event.matrix_transformation)

        comparator.add_tracker_single(
            tracker_frame, loader_event, loader_frame)
        comparator.add_tracker_both(tracker_hybrid, loader_event, loader_frame)

        comparator.run()
        print(f'------------- {seq_name} -----------')
        print(f'tracker_frame: {tracker_frame.failure}')
        print(f'tracker_hybrid: {tracker_hybrid.failure}')

        frame_failures += tracker_frame.failure
        hybrid_failures += tracker_hybrid.failure
        loader_frame.unload_data()

    print(f'------------- Total tracking failures -----------')
    print(f'tracker_frame: {frame_failures}')
    print(f'tracker_hybrid: {hybrid_failures}')
