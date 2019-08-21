import sys
sys.path.append("..")

import argparse
import os, time, datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataio import *
from data_util import *
import util

from deep_voxels import DeepVoxels
from projection import ProjectionHelper
import transformations

class ExplorerModel():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda')
        self._init_model(opt)
        self.pose_dataset = TestDataset(pose_dir=os.path.join(opt.data_root, 'pose'))

    def _init_model(self, opt):
        input_image_dims = [opt.img_sidelength, opt.img_sidelength]
        proj_image_dims = [64, 64] # Height, width of 2d feature map used for lifting and rendering.

        # Read origin of grid, scale of each voxel, and near plane
        _, grid_barycenter, scale, near_plane, _ = \
            util.parse_intrinsics(os.path.join(opt.data_root, 'intrinsics.txt'), trgt_sidelength=input_image_dims[0])

        if near_plane == 0.0:
            near_plane = opt.near_plane

        # Read intrinsic matrix for lifting and projection
        lift_intrinsic = util.parse_intrinsics(os.path.join(opt.data_root, 'intrinsics.txt'),
                                            trgt_sidelength=proj_image_dims[0])[0]
        proj_intrinsic = lift_intrinsic

        # Set up scale and world coordinates of voxel grid
        voxel_size = (1. / opt.grid_dim) * scale
        self.grid_origin = torch.tensor(np.eye(4)).float().to(self.device).squeeze()
        self.grid_origin[:3,3] = grid_barycenter

        # Minimum and maximum depth used for rejecting voxels outside of the cmaera frustrum
        depth_min = 0.
        depth_max = opt.grid_dim * voxel_size + near_plane
        grid_dims = 3 * [opt.grid_dim]

        # Resolution of canonical viewing volume in the depth dimension, in number of voxels.
        frustrum_depth = 2 * grid_dims[-1]

        self.model = DeepVoxels(lifting_img_dims=proj_image_dims,
                        frustrum_img_dims=proj_image_dims,
                        grid_dims=grid_dims,
                        use_occlusion_net=not opt.no_occlusion_net,
                        num_grid_feats=opt.num_grid_feats,
                        nf0=opt.nf0,
                        img_sidelength=input_image_dims[0])
        self.model.to(self.device)

        # Projection module
        self.projection = ProjectionHelper(projection_intrinsic=proj_intrinsic,
                                    lifting_intrinsic=lift_intrinsic,
                                    projection_image_dims=proj_image_dims,
                                    lifting_image_dims=proj_image_dims,
                                    depth_min=depth_min,
                                    depth_max=depth_max,
                                    grid_dims=grid_dims,
                                    voxel_size=voxel_size,
                                    near_plane=near_plane,
                                    frustrum_depth=frustrum_depth,
                                    device=self.device,
                                    )

        print("*" * 100)
        print("Frustrum depth")
        print(frustrum_depth)
        print("Near plane")
        print(near_plane)
        print("Intrinsic")
        print(lift_intrinsic)
        print("Number of generator parameters:")
        util.print_network(self.model)
        print("*" * 100)

        util.custom_load(self.model, opt.checkpoint)
        self.model.eval()

    def matrix_4x4_from_raw_pose(self, pose):
        base_pose = self.pose_dataset[0]
        translation, euler = pose
        rot_mat = transformations.euler_matrix(euler[0], euler[1], euler[2])
        trans_mat = transformations.translation_matrix(translation)
        trans_mat =  torch.from_numpy(trans_mat).float()
        rot_mat = torch.from_numpy(rot_mat).float()
        out = torch.matmul(trans_mat, rot_mat)
        out = torch.matmul(base_pose, out)
        return out

    def request_image(self, pose):
        if isinstance(pose, torch.Tensor) and pose.shape == [1, 4, 4]:
            pass
        else:
            pose = self.matrix_4x4_from_raw_pose(pose)

        pose = pose.squeeze().to(self.device)
        # compute projection mapping
        proj_mapping = self.projection.compute_proj_idcs(pose.squeeze(), self.grid_origin)
        if proj_mapping is None:  # invalid sample
            raise ValueError('(invalid sample)')

        proj_ind_3d, proj_ind_2d = proj_mapping

        # Run through model
        output, depth_maps, = self.model(None,
                                    [proj_ind_3d], [proj_ind_2d],
                                    None, None,
                                    None)

        output[0] = output[0][:, :, 5:-5, 5:-5]

        output_img = np.array(output[0].squeeze().cpu().detach().numpy())
        output_img = output_img.transpose(1, 2, 0)
        output_img += 0.5

        depth_img = depth_maps[0].squeeze(0).cpu().detach().numpy()
        depth_img = depth_img.transpose(1, 2, 0)

        return output_img, depth_img
