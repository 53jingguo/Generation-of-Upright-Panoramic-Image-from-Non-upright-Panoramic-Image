
import glob
import os
from collections import OrderedDict

import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import mat73

import scipy.io

cmap = plt.cm.jet


def parse_command():
    modality_names = ['rgb', 'rgbd', 'd']

    import argparse
    parser = argparse.ArgumentParser(description='FCRN')
    parser.add_argument('--decoder', default='upproj', type=str)
    parser.add_argument('--resume',
                        default='E:\\liujingguo\\first\\img_new\\upproj\\run_4\\checkpoint.pth.tar',
                        type=str, metavar='PATH',
                        help='path to latest checkpoint (default: ./run/run_1/checkpoint-5.pth.tar)')
    parser.add_argument('--resumetest',
                        default='E:\\liujingguo\\img_new\\upproj\\run_4\\checkpoint.pth.tar',
                        type=str, metavar='PATH',
                        help='path to latest checkpoint (default: ./run/run_1/checkpoint-5.pth.tar)')
    parser.add_argument('-b', '--batch-size', default=8, type=int, help='mini-batch size (default: 4)')
    parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                        help='number of total epochs to run (default: 15)')
    parser.add_argument('--lr', '--learning-rate', default=0.0002, type=float,
                        metavar='LR', help='initial learning rate (default 0.0001)')
    parser.add_argument('--lr_patience', default=2, type=int, help='Patience of LR scheduler. '
                                                                   'See documentation of ReduceLROnPlateau.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 10)')

    # add options, be check
    parser.add_argument('--isTrain', default=True,type=bool, help='you guess')
    parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
    parser.add_argument('--load_iter', type=int, default='0',
                        help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
    parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
    parser.add_argument('--epoch', type=str, default='latest',
                        help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
    parser.add_argument('--save_latest_freq', type=int, default=1, help='frequency of saving the latest results')
    parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
    parser.add_argument('--save_epoch_freq', type=int, default=1,
                        help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
    parser.add_argument('--n_epochs_decay', type=int, default=100,
                        help='number of epochs to linearly decay learning rate to zero')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--name', type=str, default='experiment_name',
                        help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')



    #parser.add_argument('--dataset', type=str, default="nyu")
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--train_ymal', default=r"D:\******\train.yaml", type=str, help='path to test ymal')
    parser.add_argument('--val_ymal', default=r"D:\*****\val.yaml", type=str, help='path to test ymal')
    parser.add_argument('--test_ymal', default=r"D:\******\test.yaml",
                        type=str, help='path to test ymal')

    args = parser.parse_args()
    return args


def get_output_directory(args):
    if args.resume:
        return os.path.dirname(args.resume)
    else:
        save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        save_dir_root = os.path.join(save_dir_root, 'img_new', args.decoder)
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

        save_dir = os.path.join(save_dir_root, 'run_' + str(run_id))
        return save_dir

arr_LUT = np.zeros((181, 181, 16*32, 2), dtype=int)

def createLUT():
    i = -90
    max_angle = 90
    while i <= max_angle:
        j = -90
        while j <= max_angle:
            LUT_name = 'D:\*******/2022\旋转代码\旋转代码\旋转代码\ImageRotate321\LUT16\LUT_P' + str(i) + '_R' + str(j) + '_Y0.mat'

            arr_LUT[i + max_angle, j + max_angle, :] = mat73.loadmat(LUT_name)['LUT']
            j = j + 1
        i = i + 1

def checkLUT(i, j):
    return torch.from_numpy(arr_LUT[i + 90, j + 90, :]).reshape((16,32,2))

# 保存检查点
def save_checkpoint(state, epoch, output_directory):
    checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch) + '.pth.tar')
    torch.save(state, checkpoint_filename)


def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:, :, :3]  # H, W, C


def merge_into_row(input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))  # H, W, C
    depth_target_col = 255 * np.transpose(np.squeeze(depth_target.cpu().numpy()), (1, 2, 0))  # H, W, C
    depth_pred_col = 255 * np.transpose(np.squeeze(depth_pred.cpu().numpy()), (1, 2, 0))  # H, W, C

    img_merge = np.hstack([rgb, depth_target_col, depth_pred_col])

    return img_merge


def merge_into_row_with_gt(input, depth_input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))  # H, W, C
    depth_input_cpu = np.squeeze(depth_input.cpu().numpy())
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_input_cpu), np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_input_cpu), np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_input_col = colored_depthmap(depth_input_cpu, d_min, d_max)
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)

    img_merge = np.hstack([rgb, depth_input_col, depth_target_col, depth_pred_col])

    return img_merge


def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)
