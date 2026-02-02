from utils import geometry
from utils.misc import setup_logging, load_joint_annotations
from models.pointnet import PointNetPP
from models.mlp import MLPModel
from datasets.contact_pose_3d import ContactPose3D, eval_collate_fn
import train_test_splits
import open3d as o3d
import torch
torch.nn.Module.dump_patches = True

import torch.nn as nn
from torch.utils.data import DataLoader
from torch_scatter import scatter_mean
import os
import numpy as np
import argparse

import json
import pickle

from datetime import datetime
from utils.logger import setup_logger
from pprint import pprint

def annealed_mean(pred, bins, T=0.38):
    pred = np.exp(np.log(pred) / T)
    pred /= pred.sum(axis=1, keepdims=True)
    texture_bin_centers = (bins[:-1] + bins[1:]) / 2.0
    pred = np.sum(pred * texture_bin_centers, axis=1)
    return pred

def mode(pred, bins):
    texture_bin_centers = (bins[:-1] + bins[1:]) / 2.0
    pred = np.argmax(pred, axis=1)
    return texture_bin_centers[pred]

def show_prediction(pred, kept_joints, bins, object_name, session_name,
    models_dir='data/object_models', contactpose_dir='data/contactpose_data',
    binvoxes_dir='data/binvoxes',save_dir='data/save'):
    # create texture
    pred = np.exp(pred) / np.sum(np.exp(pred), axis=1, keepdims=True)
    pred = annealed_mean(pred, bins)
    # pred = mode(pred, bins)
    
    mesh_filename = os.path.join(models_dir, '{:s}.ply'.format(object_name))
    binvox_filename = os.path.join(binvoxes_dir,
        '{:s}_hollow.binvox'.format(object_name))
    joint_locs = load_joint_annotations(session_name, object_name)
    geometry.show_prediction(pred, mesh_filename, binvox_filename, joint_locs,
        kept_joints)

def eval(args):

    device = 'cuda'

    logger.info('load checkpoint from %s' % args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    
    logger.info('load texture weights from %s' % args.texture_weights)
    texture_weights = np.load(args.texture_weights)
    texture_weights = torch.from_numpy(texture_weights).to(dtype=torch.float)

    
    model = PointNetPP(args.n_feats, len(texture_weights), droprate=args.droprate)
    model.load_state_dict(checkpoint.state_dict())
    model.to(device=device)
    logger.info('state dict loaded')

    loss = nn.CrossEntropyLoss(weight=texture_weights, ignore_index=1)


    splits = getattr(train_test_splits, 'split_{:s}'.format(args.split))

    dataset = ContactPose3D(args.data_dir, args.contactpose_dir, args.grid_size, args.n_rotations,
                            args.joint_droprate, eval_mode=True, **splits['test'])

    dataloader = DataLoader(dataset, batch_size=args.n_rotations, pin_memory=True, num_workers=args.num_workers,
                            collate_fn=eval_collate_fn, shuffle=False)
    
    logger.info('dataset loaded')

    all_data = []
    for batch_idx, data in enumerate(dataloader):

        session_name, object_name, filename = dataset.filenames[batch_idx]
        logger.info('{:d} / {:d}: {:s} {:s}'.format(batch_idx+1, len(dataloader), session_name, object_name))

        use_idx = range(1, 5)
        for idx in use_idx:
            data[idx] = data[idx].to(device=device, non_blocking=True)
        data[2] = data[2] / args.grid_size - 0.5
        with torch.no_grad():

            pred = model(x=data[1], pos=data[2], batch=data[3])

        preds = [p.cpu().numpy() for p in torch.chunk(pred, args.n_rotations)]
        targ = data[4].cpu().numpy()
        targ = targ[:len(targ)//args.n_rotations]
        all_data.append([session_name, object_name, preds, targ])
        if args.show_object is True:
            avg_pred = np.stack(preds).mean(0)
            kept_joints = data[-1][0].cpu().numpy()
            show_prediction(avg_pred, kept_joints, dataset.texture_bins, object_name, session_name, save_dir = os.path.join(args.log_dir, 'save'))

    if args.save_preds:
        output_dir = args.log_dir
        filename = os.path.join(output_dir, 'predictions_joint_droprate={:.2f}.pkl'.format(args.joint_droprate))

        with open(filename, 'wb') as f:
            pickle.dump({
                'data': all_data,
                'checkpoint_filename': args.checkpoint,
                'joint_droprate': args.joint_droprate}, f)
        logger.info('{:s} written'.format(filename))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/mesh_prediction_data', type=str)
    parser.add_argument('--contactpose_dir', default='./data/contactpose_data', type=str)
    parser.add_argument('--split', type=str, required=True, choices=('objects', 'participants', 'overfit'))
    parser.add_argument('--checkpoint', default='./data/checkpoints/pointnet_split_objects_mesh/checkpoint_model_66_train_loss=1.442994.pth')
    parser.add_argument('--texture_weights', default='./data/texture_bin_weights.npy')
    parser.add_argument('--config', default='configs/pointnet.ini', type=str)

    parser.add_argument('--show_object', action='store_true')
    parser.add_argument('--save_preds', action='store_true')
    parser.add_argument('--output_filename_suffix', default=None)
    parser.add_argument('--log_root', default='./logs', type=str)
    parser.add_argument('--tag', default='contactpose_infer')

    # Hyperparameters
    parser.add_argument('--grid_size', type=int, default=64, help='Grid size')
    parser.add_argument('--n_rotations', type=int, default=12, help='Number of rotations')
    parser.add_argument('--max_joint_dist_cm', type=float, default=30.0, help='Maximum joint distance in cm')
    parser.add_argument('--color_sigmoid_a', type=float, default=0.05, help='Color sigmoid parameter a')
    parser.add_argument('--joint_droprate', type=float, default=0, help='Joint dropout rate')
    parser.add_argument('--droprate', type=float, default=0, help='General dropout rate')
    parser.add_argument('--lr_step_size', type=int, default=1000, help='Learning rate step size')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='Learning rate gamma')
    parser.add_argument('--uniform_texture_weights', type=str, choices=['yes', 'no'], default='no', help='Use uniform texture weights')
    parser.add_argument('--n_feats', type=int, default=22, help='Number of features')

    # Optimizer settings
    parser.add_argument('--batch_size', type=int, default=25, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=1000, help='Maximum number of epochs')
    parser.add_argument('--val_interval', type=int, default=1, help='Validation interval')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--base_lr', type=float, default=0.001, help='Base learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Optimizer momentum')

    # Miscellaneous
    parser.add_argument('--log_interval', type=int, default=100, help='Log interval')
    parser.add_argument('--shuffle', type=str, choices=['yes', 'no'], default='yes', help='Shuffle data')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')

    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.log_dir = os.path.join(args.log_root, '%s_%s'%(args.tag, stamp))
    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger(output=args.log_dir, color=True, name='contactpose_infer')

    logger.info('Configerations:')
    logger.info(''.join([f"  {arg}: {value}\n" for arg, value in vars(args).items()]))

    eval(args)
