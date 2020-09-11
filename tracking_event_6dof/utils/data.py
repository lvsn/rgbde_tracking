import numpy as np
import numpy as np
from pyquaternion import Quaternion

from ulaval_6dof_object_tracking.utils.data import combine_view_transform
from tracking_event_6dof.utils.quaternion_transform import QuaternionTransform


def compute_3D_project_points(pose, scale_size=230, scale=(1, 1, 1)):
    obj_x = pose.matrix[0, 3] * scale[0]
    obj_y = pose.matrix[1, 3] * scale[1]
    obj_z = pose.matrix[2, 3] * scale[2]
    offset = scale_size / 2
    points = np.ndarray((4, 3), dtype=np.float)
    points[0] = [obj_x - offset, obj_y - offset, obj_z]     # top left
    points[1] = [obj_x - offset, obj_y + offset, obj_z]     # top right
    points[2] = [obj_x + offset, obj_y - offset, obj_z]     # bottom left
    points[3] = [obj_x + offset, obj_y + offset, obj_z]     # bottom right
    return points


def calculate_center_pose(pose, camera, object_width):
    points = compute_3D_project_points(
        pose, object_width, scale=(1000, -1000, -1000))
    bb = camera.project_points(points, round=False)
    left = np.min(bb[:, 1])
    right = np.max(bb[:, 1])
    top = np.min(bb[:, 0])
    bottom = np.max(bb[:, 0])
    def center(a, b): return a + (b - a)/2
    return center(left, right), center(top, bottom), (right - left), (bottom - top)


def inverse_z(pose):
    pose = pose.copy()
    pose[1:3, 3] *= -1
    rotation = QuaternionTransform(Quaternion(axis=[1, 0, 0], angle=np.pi),
                                   (0, 0, 0)).to_transform()
    return combine_view_transform(pose, rotation)


def delta_transform(poseA, poseB):
    poseA = poseA.copy()
    poseB = poseB.copy()
    delta_translation = poseB[0:3, 3] - poseA[0:3, 3].copy()
    delta_rotation = poseB.rotate(transform=poseA.inverse())
    delta_rotation[0:3, 3] = delta_translation
    return delta_rotation
