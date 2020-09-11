from tracking_event_6dof.inference.animation import Animation
from enum import Enum, auto
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from ulaval_6dof_object_tracking.utils.transform import Transform
from ulaval_6dof_object_tracking.evaluate_sequence import eval_pose_error
from collections.abc import Iterable


class TrackerType(Enum):
    SINGLE = auto()
    BOTH = auto()


class Comparator:
    def __init__(self, size, initial_pose=None, start=0, animation=None,
                 reset=False, threshold_translation=0.3, threshold_rotation=20, reset_counter=7):
        self.animation = animation
        self.trackers = []
        self.start = start
        self.size = size

        self.reset = reset
        self.threshold_rotation = threshold_rotation
        self.threshold_translation = threshold_translation
        self.counter = reset_counter

        if initial_pose:
            self.poses_truth = [initial_pose.to_parameters()]
        else:
            self.poses_truth = None

    def _add_tracker(self, tracker, loader_event, loader_frame, _type):
        tracker.loader_event = loader_event
        tracker.loader_frame = loader_frame
        tracker.type = _type
        tracker.fail_counter = 0
        tracker.failure = 0
        self.trackers.append(tracker)
        tracker.last_event_frame = None

    def add_tracker_single(self, tracker, loader_event, loader_frame):
        self._add_tracker(tracker, loader_event,
                          loader_frame, TrackerType.SINGLE)

    def add_tracker_both(self, tracker, loader_event, loader_frame):
        self._add_tracker(tracker, loader_event,
                          loader_frame, TrackerType.BOTH)

    def run(self):
        for i in tqdm(range(self.start, self.size)):
            real_frame = []
            for tracker in self.trackers:
                events_frame, _ = tracker.loader_event[i]
                rgbdB, pose_truth = tracker.loader_frame[i]

                if tracker.type == TrackerType.BOTH:
                    args = [events_frame, rgbdB]
                elif tracker.type == TrackerType.SINGLE:
                    args = [rgbdB]
                tracker.predict(*args)

                real_frame.append(rgbdB[0])

                if self.reset:
                    if isinstance(tracker.current_pose, Iterable):
                        tracker_pose = tracker.current_pose[-1]
                    else:
                        tracker_pose = tracker.current_pose
                    err_t, err_r = eval_pose_error(
                        [pose_truth.matrix], [tracker_pose.matrix])
                    if (err_t[0] > self.threshold_translation or
                            err_r[0] > self.threshold_rotation):
                        tracker.fail_counter += 1
                        if tracker.fail_counter == 7:
                            tracker.reset(pose_truth)
                            tracker.fail_counter = 0
                            tracker.failure += 1
                    else:
                        tracker.fail_counter = 0

            if self.animation:
                self.animation.add_frame(
                    [tracker.current_pose for tracker in self.trackers],
                    real_frame, fail=[
                        tracker.failure for tracker in self.trackers],
                    event_bb=[
                        tracker.last_event_frame for tracker in self.trackers],
                )

            self.poses_truth.append(pose_truth.to_parameters())
