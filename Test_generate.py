import numpy as np
import scipy
import torch
import mat73
import utils
import criteria
import os
import torch.nn as nn
from PIL import Image
from skimage import io
from dataloaders import dataset
import networks
import time

args = utils.parse_command()
gpu_ids = args.gpu_ids
testdir = args.test_ymal
root_path = 'D:\***/****'
test_set = dataset.Dataset(root_path, testdir)
test_loader = torch.utils.data.DataLoader(
         test_set, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
checkpoint = torch.load('********************')
def deform_input(inp, deformation):
     _, h_old, w_old, _ = deformation.shape
     _, _, h, w = inp.shape
     if h_old != h or w_old != w:
          deformation = deformation.permute(0, 3, 1, 2)
          deformation = torch.nn.functional.interpolate(deformation, size=(h, w), mode='bilinear', align_corners=True)
          deformation = deformation.permute(0, 2, 3, 1)
     return torch.nn.functional.grid_sample(inp, deformation, align_corners=True)



netG_encoder = checkpoint['netG_encoder']
netG_decoder = checkpoint['netG_decoder']

print("=> loaded checkpoint")


del checkpoint

total_time = 0
for i, (input, Img_target, grid) in enumerate(test_loader):

   input, Img_target, grid = input.cuda(), Img_target.cuda(), grid.cuda()
   torch.cuda.synchronize()
   start = time.time()
   with torch.no_grad():
       flow = netG_encoder(input)
       input = netG_decoder(input, flow)

   torch.cuda.synchronize()
   end = time.time()
   total_time = total_time + (end - start)
print(total_time/i+1)
   # filename = 'E:\\***********\\test\\test' + str(i) + '.png'
   # img_pred = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))
   # img_pred = Image.fromarray(img_pred.astype('uint8'))
   # img_pred.save(filename)

