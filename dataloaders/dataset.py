
import torch
import torch.utils.data
import utils
import numpy as np
from skimage import io,transform
import OpenEXR, Imath, array
import scipy.io
import numpy as np
import random
import math
import os.path as osp
import yaml
from yaml import SafeLoader
import mat73

args = utils.parse_command()
utils.createLUT()
class Dataset(torch.utils.data.Dataset):
    '''PyTorch dataset module for effiicient loading'''

    def __init__(self,
                 root_path,
                 yaml_path):
        # Set up a reader to load the panos
        self.root_path = root_path
        self.yaml = yaml_path

        # Create tuples of inputs/GT
        self.rotatedImage_list = self.make_list(self.yaml)

    def make_list(self,path):
        f1=open(path)
        test_list=[]
        data = yaml.safe_load(f1)

        for key,values in data.items():
            for value in values:
                test_list.append(key+'_'+value)
        return test_list

    def __getitem__(self, idx):
        '''Load the data'''

        relative_paths = self.rotatedImage_list[idx]

        path_file = relative_paths.split('_')[0]

        rotatedImg_name = relative_paths.lstrip(path_file)
        rotatedImg_name = rotatedImg_name.lstrip('_')

        R_name = rotatedImg_name.split('_B_')[0] + ".mat"

        pitch = R_name.split('_')[0]
        pitch = pitch.lstrip('P')
        pitch = int(pitch)
        # pitch = (float(pitch) + 90) / 180.
        roll = R_name.split('_')[1]
        roll = roll.lstrip('R')
        roll = int(roll)
        # roll = (float(roll) + 90) / 180.
        # p_and_R = [pitch, roll]
        # p_and_R = np.array(p_and_R)
        Img_name = rotatedImg_name.split('_B_')[1]
        # print(R_name)
        LUT_H = 16
        LUT_H_Half = LUT_H / 2.
        LUT_W = 32
        LUT_W_Half = LUT_W / 2.
        grid256 = torch.zeros((LUT_H, LUT_W, 2))
        grid = utils.checkLUT(pitch, roll)
        grid256[..., 1] = (grid[..., 0] - LUT_H_Half) / LUT_H_Half
        grid256[..., 0] = (grid[..., 1] - LUT_W_Half) / LUT_W_Half



        rgb_rotatedImg = self.readRGBPano(osp.join(self.root_path, "rotatedImg",path_file, rotatedImg_name))
        rgb_Img = self.readRGBPano(osp.join(self.root_path, "M3D_low",path_file, Img_name))

        pano_data = [rgb_rotatedImg, rgb_Img, grid256]

        pano_data[0] = torch.from_numpy(pano_data[0].transpose(2, 0, 1)).float()
        pano_data[1] = torch.from_numpy(pano_data[1].transpose(2, 0, 1)).float()
        pano_data[2] = pano_data[2].permute(2, 0, 1).float()

        return pano_data

    def __len__(self):
        '''Return the size of this dataset'''
        return len(self.rotatedImage_list)

    def readRGBPano(self, path):
        '''Read RGB and normalize to [0,1].'''
        rgb = io.imread(path).astype(np.float32) / 255.
        return rgb


def safe_load(stream):
    """
    Parse the first YAML document in a stream
    and produce the corresponding Python object.

    Resolve only basic YAML tags. This is known
    to be safe for untrusted input.
    """
    return yaml.load(stream, SafeLoader)