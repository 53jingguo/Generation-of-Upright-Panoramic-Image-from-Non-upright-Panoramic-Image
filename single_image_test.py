import numpy as np
import torch
from PIL import Image
from skimage import io

checkpoint = torch.load('D:\\project\\upright adjustment\\checkpoint.pth.tar')


netG_encoder = checkpoint['netG_encoder']
netG_decoder = checkpoint['netG_decoder']

input0 = io.imread(r'E:\*******\***\.png').astype(np.float32) / 255.
input0 = torch.from_numpy(input0.transpose(2, 0, 1)).float()
input0 = torch.unsqueeze(input0, 0)
print("=> loaded checkpoint ")

# clear memory
del checkpoint
# del model_dict
torch.cuda.empty_cache()

input0 = input0.cuda()
torch.cuda.synchronize()
with torch.no_grad():
   flow = netG_encoder(input0)
   input = netG_decoder(input0, flow)
   filename = r'E:\*****\**********\plane.png'
   img_pred = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))
   img_pred = Image.fromarray(img_pred.astype('uint8'))
   img_pred.save(filename)