import sys
sys.path.append("..")

import argparse
import os, time, datetime
import tkinter as tk

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataio import *
from data_util import *
import util

from deep_voxels import DeepVoxels
from projection import ProjectionHelper

import explorer_model
import explorer_view

def main():
    opt = parse_args()
    device = torch.device('cuda')

    exp_model = explorer_model.ExplorerModel(opt)


    # Create the training dataset loader
    dataset = TestDataset(pose_dir=os.path.join(opt.data_root, 'pose'))


    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    dir_name = os.path.join(datetime.datetime.now().strftime('%m_%d'),
                            datetime.datetime.now().strftime('%H-%M-%S_') +
                            '_'.join(opt.checkpoint.strip('/').split('/')[-2:]) + '_'
                            + opt.data_root.strip('/').split('/')[-1])

    traj_dir = os.path.join(opt.logging_root, 'test_traj', dir_name)
    depth_dir = os.path.join(traj_dir, 'depth')

    data_util.cond_mkdir(traj_dir)
    data_util.cond_mkdir(depth_dir)

    forward_time = 0.

    print('starting testing...')
    with torch.no_grad():
        iter = 0
        depth_imgs = []
        for trgt_pose in dataloader:
            # exec(util.embed_breakpoint())
            start = time.time()

            output_img, depth_img = exp_model.request_image(trgt_pose)

            end = time.time()
            forward_time += end - start

            print("Iter %d" % iter)

            depth_imgs.append(depth_img)
            # exec(util.embed_breakpoint())
            cv2.imwrite(os.path.join(traj_dir, "img_%05d.png" % iter), output_img.astype(np.uint8)[:, :, ::-1])

            iter += 1

        depth_imgs = np.stack(depth_imgs, axis=0)
        depth_imgs = (depth_imgs - np.amin(depth_imgs)) / (np.amax(depth_imgs) - np.amin(depth_imgs))
        depth_imgs *= 2**8 - 1
        depth_imgs = depth_imgs.round()

        for i in range(len(depth_imgs)):
            cv2.imwrite(os.path.join(depth_dir, "img_%05d.png" % i), depth_imgs[i].astype(np.uint8))

    print("Average forward pass time over %d examples is %f"%(iter, forward_time/iter))


def main_2():
    opt = parse_args()
    device = torch.device('cuda')

    exp_model = explorer_model.ExplorerModel(opt)
    root = tk.Tk()
    explorer_view.ExplorerView(opt, exp_model, root).pack(side="top", fill="both", expand=True)
    root.mainloop()



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_test', type=str, required=True,
                        help='Whether to run training or testing. Options are \"train\" or \"test\".')
    parser.add_argument('--data_root', required=True,
                        help='Path to directory that holds the object data. See dataio.py for directory structure etc..')
    parser.add_argument('--logging_root', required=True,
                        help='Path to directory where to write tensorboard logs and checkpoints.')

    parser.add_argument('--experiment_name', type=str, default='', help='(optional) Name for experiment.')
    parser.add_argument('--max_epoch', type=int, default=400, help='Maximum number of epochs to train for.')
    parser.add_argument('--lr', type=float, default=0.0004, help='Learning rate.')
    parser.add_argument('--l1_weight', type=float, default=200, help='Weight of l1 loss.')
    parser.add_argument('--sampling_pattern', type=str, default='all', required=False,
                        help='Whether to use \"all\" images or whether to skip n images (\"skip_1\" picks every 2nd image.')

    parser.add_argument('--img_sidelength', type=int, default=512,
                        help='Sidelength of generated images. Default 512. Only less than native resolution of images is recommended.')

    parser.add_argument('--no_occlusion_net', action='store_true', default=False,
                        help='Disables occlusion net and replaces it with a fully convolutional 2d net.')
    parser.add_argument('--num_trgt', type=int, default=2, required=False,
                        help='How many novel views will be generated at training time.')

    parser.add_argument('--checkpoint', default='',
                        help='Path to a checkpoint to load model weights from.')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='Start epoch')

    parser.add_argument('--grid_dim', type=int, default=32,
                        help='Grid sidelength. Default 32.')
    parser.add_argument('--num_grid_feats', type=int, default=64,
                        help='Number of features stored in each voxel.')
    parser.add_argument('--nf0', type=int, default=64,
                        help='Number of features in outermost layer of U-Net architectures.')
    parser.add_argument('--near_plane', type=float, default=np.sqrt(3)/2,
                        help='Position of the near plane.')

    opt = parser.parse_args()
    print('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))
    return opt

if __name__ == "__main__":
    main_2()
