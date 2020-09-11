import argparse
import os
import numpy as np

from ulaval_6dof_object_tracking.utils.transform import Transform
from ulaval_6dof_object_tracking.utils.camera import Camera
from ulaval_6dof_object_tracking.utils.data import combine_view_transform, normalize_scale, compute_2Dboundingbox
from ulaval_6dof_object_tracking.utils.model_renderer import ModelRenderer
from ulaval_6dof_object_tracking.utils.plyparser import PlyParser
from ulaval_6dof_object_tracking.utils.pointcloud import maximum_width
from tracking_event_6dof.utils.data import compute_3D_project_points


class Render:
    def __init__(self, camera, shader_path=None,
                 model_path="/dataset/3D_models/dragon", image_size=(174, 174)):

        self.camera = camera
        if shader_path == None:
            shader_path = os.path.join(os.path.dirname(__file__), "shader")

        self.image_size = image_size
        window_size = (self.camera.width, self.camera.height)

        bounding_box = 0

        widths = []

        geometry_path = os.path.join(model_path, "geometry.ply")
        self.model_3d = PlyParser(geometry_path).get_vertex()
        object_max_width = maximum_width(self.model_3d) * 1000
        with_add = bounding_box / 100 * object_max_width
        widths.append(int(object_max_width + with_add))
        widths.sort()

        self.object_width = widths[int(len(widths)/2)]
        self.ao_path = os.path.join(model_path, "ao.ply")
        self.vpRender = ModelRenderer(geometry_path, shader_path, self.camera, [
                                      window_size, image_size], backend='egl')

    def poseA(self, pose, light_direction=None, light_diffuse=None,
              ambiant_light=None):
        if os.path.exists(self.ao_path):
            self.vpRender.load_ambiant_occlusion_map(self.ao_path)

        bb = compute_2Dboundingbox(
            pose, self.camera, self.object_width, scale=(1000, 1000, -1000))
        left = np.min(bb[:, 1])
        right = np.max(bb[:, 1])
        top = np.min(bb[:, 0])
        bottom = np.max(bb[:, 0])

        self.vpRender.setup_camera(self.camera, left, right, bottom, top)

        rgbd = self.vpRender.render_image(pose, fbo_index=1,
                                          ambiant_light=ambiant_light,
                                          light_diffuse=light_diffuse,
                                          light_direction=light_direction)
        points = compute_3D_project_points(
            pose, self.object_width, scale=(1000, -1000, -1000))
        return rgbd + (points, )

    def poseB(self, pose, bb=None, points=None,
              light_direction=None, light_diffuse=None,
              ambiant_light=None):
        if os.path.exists(self.ao_path):
            self.vpRender.load_ambiant_occlusion_map(self.ao_path)

        if np.any(points):
            bb = self.camera.project_points(
                points, y_negative=True).astype(np.int32)

        self.vpRender.setup_camera(
            self.camera, 0, self.camera.width, self.camera.height, 0)
        rgb, depth = self.vpRender.render_image(pose, fbo_index=0,
                                                light_direction=light_direction, light_diffuse=light_diffuse,
                                                ambiant_light=ambiant_light)
        rgb = rgb[..., :3]
        rgbd = (rgb, depth)
        if np.any(bb):
            rgbd = normalize_scale(rgb, depth, bb, output_size=self.image_size)

        return rgbd + (bb, )
