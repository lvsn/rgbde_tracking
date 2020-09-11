import numpy as np
import yaml
import json
import os

from ulaval_6dof_object_tracking.utils.camera import Camera as CameraDeeptrack


class Camera(CameraDeeptrack):
    @staticmethod
    def load_from_json(path, filename='camera'):
        file = path
        if path[-5:] != ".json":
            file = os.path.join(path, "{}.json".format(filename))
        with open(file) as data_file:
            data = json.load(data_file)
        distortion = np.array([[0., 0., 0., 0., 0.]])
        try:
            distortion = np.array([data["distortion"]])
        except KeyError:
            pass
        camera = Camera((data["focalX"], data["focalY"]),
                        (data["centerX"], data["centerY"]),
                        (data["width"], data["height"]),
                        distortion)
        return camera

    @staticmethod
    def load_from_simulator(path):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        camera = data['cameras'][0]['camera']
        intrinsics = camera['intrinsics']['data']
        focals = (intrinsics[0], intrinsics[1])
        centers = (intrinsics[2], intrinsics[3])
        size = (camera['image_width'], camera['image_height'])

        distortion = np.array([[0., 0., 0., 0., 0.]])

        return Camera(focals, centers, size, distortion)

    def save(self, path, filename='camera'):
        dict = {
            "focalX": self.focal_x,
            "focalY": self.focal_y,
            "centerX": self.center_x,
            "centerY": self.center_y,
            "width": self.width,
            "height": self.height,
            "distortion": list(self.distortion[0])
        }
        with open(os.path.join(path, "{}.json".format(filename)), 'w') as data_file:
            json.dump(dict, data_file)

    def project_points(self, points, round=True, y_negative=False):
        # Note: By default center_y is in positive convention.
        # Set y_negatif to True if your y axis is in negative convention.
        center_y = self.center_y
        if y_negative:
            center_y = self.height - self.center_y
        computed_pixels = np.zeros((points.shape[0], 2))
        computed_pixels[:, 1] = points[:, 0] * \
            self.focal_x / points[:, 2] + self.center_x
        computed_pixels[:, 0] = points[:, 1] * \
            self.focal_y / points[:, 2] + self.center_y
        computed_pixels[:, 0] = points[:, 1] * \
            self.focal_y / points[:, 2] + center_y
        if round:
            computed_pixels = np.round(computed_pixels).astype(np.int)
        return computed_pixels
