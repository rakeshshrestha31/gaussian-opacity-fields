from pathlib import Path
import numpy as np
import math
import torch
import pyrender
import open3d as o3d
import trimesh
import cv2

# from scene import Scene
from scene.dataset_readers import sceneLoadTypeCallbacks
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.vis_utils import apply_depth_colormap
from utils.camera_utils import cameraList_from_camInfos
from utils.system_utils import searchForMaxIteration
from gaussian_renderer import render


class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "metadata.json")):
            print("Found metadata.json file, assuming multi scale Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Multi-scale"](args.source_path, args.white_background, args.eval, args.load_allres)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            raise NotImplementedError

        # if shuffle:
        #     random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
        #     # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            # self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

            if self.gaussians is not None:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                               "point_cloud",
                                                               "iteration_" + str(self.loaded_iter),
                                                               "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

class PyRenderScene:
    def __init__(self, viewpoint_camera):
        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        image_height = int(viewpoint_camera.image_height)
        image_width = int(viewpoint_camera.image_width)
        fx = image_width * 0.5 / tanfovx
        fy = image_height * 0.5 / tanfovy
        cx = image_width / 2.0
        cy = image_height / 2.0

        self.R_gl_cv = np.asarray([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0],
        ])
        self.R_cv_gl = self.R_gl_cv.T

        self.scene = pyrender.Scene(
            ambient_light=np.array([0.35]*3 + [1.0])
        )
        self.cam_node = None
        self.direc_l = None
        self.spot_l = None
        self.point_l = None

        self.intrinsic_cam = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
        self.set_camera_pose(np.eye(4))

        r = pyrender.OffscreenRenderer(
            viewport_width=image_width, viewport_height=image_height
        )

        self.r = r

    def __del__(self):
        self.r.delete()

    def render(self):
        return self.r.render(self.scene)

    def load_mesh(self, mesh_filename):
        trimesh_mesh = trimesh.load(str(mesh_filename))

        # Remove vertex colors
        if hasattr(trimesh_mesh.visual, 'vertex_colors'):
            trimesh_mesh.visual.vertex_colors = None

        # Remove face colors
        if hasattr(trimesh_mesh.visual, 'face_colors'):
            trimesh_mesh.visual.face_colors = None

        # Remove texture/material
        if hasattr(trimesh_mesh.visual, 'material'):
            trimesh_mesh.visual.material = None

        trimesh.repair.fix_normals(trimesh_mesh, multibody=True)
        pyrender_mesh = pyrender.Mesh.from_trimesh(trimesh_mesh, smooth=False)
        mesh_node = self.scene.add(pyrender_mesh)

    def set_camera_pose(self, T):
        T[:3, :3] = T[:3, :3] @ self.R_cv_gl

        attrs_to_remove = ["cam_node", "direc_l", "spot_l", "point_l"]
        for attr in attrs_to_remove:
            node = getattr(self, attr)
            if node is not None:
                self.scene.remove_node(node)

        self.cam_node = self.scene.add(self.intrinsic_cam, pose=T)

        direc_l = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
        spot_l = pyrender.SpotLight(color=np.ones(3), intensity=5.0,
                           innerConeAngle=np.pi/16*0.1, outerConeAngle=np.pi/6*0.1)
        point_l = pyrender.PointLight(color=np.ones(3), intensity=10.0)

        self.direc_l = self.scene.add(direc_l, pose=T)
        self.spot_l = self.scene.add(spot_l, pose=T)
        self.point_l = self.scene.add(point_l, pose=T)

        # light1 = pyrender.DirectionalLight(color=[1., 1., 1.], intensity=3.0)
        # scene.add(light1, pose=camera_pose)

        # light2 = pyrender.SpotLight(color=np.ones(3), intensity=2.0,
        #                             innerConeAngle=np.pi/16.0,
        #                             outerConeAngle=np.pi/6.0)
        # scene.add(light2, pose=camera_pose)


def render_set(model_path, name, iteration, views, gaussians, pipeline):
    # mesh_path = os.path.join(model_path, name, f"ours_{iteration}", "fusion", "mesh_binary_search_7.ply")
    mesh_path = os.path.join(model_path, "test", f"ours_{iteration}", "tsdf", "tsdf.ply")

    render_path = Path(model_path) / name / f"ours_{iteration}" / "mesh_renders"
    render_path.mkdir(parents=True, exist_ok=True)

    pose_mesh = o3d.geometry.TriangleMesh()

    renderer = PyRenderScene(views[0])
    renderer.load_mesh(mesh_path)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        T_wc = (view.world_view_transform.T).inverse()
        renderer.set_camera_pose(T_wc.cpu().numpy())

        # T = T_wc.cpu().numpy()
        # T[:3, :3] = T[:3, :3] @ renderer.R_cv_gl
        # pose_mesh += o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5).transform(T)

        rendered_color, _ = renderer.render()
        cv2.imwrite(str(render_path / f"{idx:05d}.png"), rendered_color)

    o3d.io.write_triangle_mesh("/tmp/pose_mesh.ply", pose_mesh)


@torch.no_grad()
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams):
    gaussians = None  # GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    # render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline)
    render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    print(f"{args.source_path=}")

    render_sets(model.extract(args), args.iteration, pipeline.extract(args))
