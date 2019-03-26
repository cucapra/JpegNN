import os
import torch
from torchvision import datasets, models, transforms
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
from scipy import ndimage as ndi

def _smooth(image, sigma, mode, cval, multichannel):
    """Return image with each channel smoothed by the Gaussian filter."""
    smoothed = np.empty(image.shape, dtype=np.double)

    # apply Gaussian filter to all channels independently
    if multichannel:
        sigma = (sigma, )*(image.ndim - 1) + (0, )
    ndi.gaussian_filter(image, sigma, output=smoothed,
                        mode=mode, cval=cval)
    return smoothed

class Interpolate(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    def __call__(self, sample):
        cvimg = np.asarray(sample)
        cvimg=cv2.cvtColor(cvimg, cv2.COLOR_RGB2BGR) 

        resized = cv2.resize(cvimg, (self.output_size,self.output_size),interpolation=cv2.INTER_AREA )
        pilimg = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return pilimg
class Gaussian(object):
    def __init__(self,downscale=2):
        # automatically determine sigma which covers > 99% of distribution
        #self.sigma = 2 * downscale / 6.0
        self.sigma = 2 * downscale/(np.sqrt(2)*np.pi)
    def __call__(self,sample):
        gaussian=_smooth(np.array(sample), self.sigma, mode='reflect', cval=0, multichannel=True)
        #gaussian=sample.filter(ImageFilter.GaussianBlur(self.radius))
        #cvimg = cv2.cvtColor(np.array(sample), cv2.COLOR_RGB2BGR)
        #gaussian=cv2.GaussianBlur(cvimg, (5,5), 0)
        #gaussian = cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB)
        #pilimg = Image.fromarray(gaussian)
        return  Image.fromarray(gaussian.astype('uint8'), 'RGB')

class JpegLayer(torch.nn.Module):
    def __init__(self, block_size = 8):
        super(JpegLayer, self).__init__()
        self.bs = block_size
        self.dctmtx = self.__dctmtx(self.bs)

    def forward(self, input):
        #preprocessing
        rgb = input
        ycbcr = self.__rgb2ycbcr(rgb)-128/255
        
        blks = torch.split(ycbcr, self.bs, 2)
        sts = torch.stack(blks, 2)
        blks = torch.split(sts, self.bs, 4)
        sts = torch.stack(blks, 3)
        #dct 
        dcts = torch.matmul(torch.matmul(self.dctmtx,sts),self.dctmtx.t() )
        print('mean', (255*dcts).view(-1,8,8).abs().mean(dim = 0))
        print('std', ( (dcts*255).view(-1,8,8).abs().std(dim = 0)) )

            
    def __dctmtx(self, bs=8):
        dct_mtx = torch.empty(bs, bs)
        for j in range(0, bs):
            for i in range(0, bs):
                if i == 0:
                    dct_mtx[i][j] = 1/math.sqrt(bs)
                else:
                    dct_mtx[i][j] = math.sqrt(2/bs) * \
                    math.cos((math.pi*(2*j+1)*i)/(2*bs)) 
        return dct_mtx.type(torch.cuda.FloatTensor)
    def __subsample(self, im, row=2, col=2):
        permute = im.permute(1,0,2,3)
        
        means = []
        means.append(permute[0])
        for i in range(1,3):
            sts1 = torch.stack((torch.split(permute[i], row, 1)), 1)
            sts2 = torch.stack((torch.split(sts1, col, 3)), 2)
            a,b,c,d,e= sts2.shape#batch,w/r,h/c,r,c
            mean = torch.mean(sts2.view(a,b,c,-1).float(),3)
            
            means.append(mean)
             
        return means 
        
    def __rgb2ycbcr(self,rgb):
        ycbcr = rgb.permute(0,2,3,1)
        xform = torch.cuda.FloatTensor([[.299,.587,.114],
                  [-0.168735892 ,- 0.331264108, 0.5],
                  [.5,- 0.418687589, - 0.081312411]])
        ycbcr = torch.matmul(ycbcr, xform.t())
        ycbcr[:,:,:,[1,2]] += 128/255
        #put mask
        ycbcr = torch.clamp(ycbcr,0,1)
        ycbcr = ycbcr.permute(0,3,1,2)
        return ycbcr

  
model_ft = JpegLayer()
print(model_ft)

input_size = 256

data_transforms = {
    0: transforms.Compose([
        #transforms.CenterCrop(128),
        transforms.ToTensor(),
    ]),
    1: transforms.Compose([
        Gaussian(),
        Interpolate(128),
        transforms.ToTensor(),
    ]),
    2: transforms.Compose([
        #transforms.CenterCrop(128),
        transforms.ToTensor(),
    ]),
}


print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
paths = ['/data/zhijing/frequency/pyjpeg50/','/data/zhijing/frequency/pyjpeg50/','/data/zhijing/frequency/uncomp/']
image_datasets = {x: datasets.ImageFolder(paths[x], data_transforms[x]) for x in range(len(paths))}

dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1, shuffle=False, num_workers=4) for x in range(len(paths))}

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = model_ft.to(device)

visualize = True

for i in range(1,len(paths)):
    if i == 2:
        print("---------original-----------")
    for j, (data, _) in enumerate(dataloaders_dict[i]):
        if visualize:
            f1 = data[0].numpy()
            f1 = (np.transpose(f1,(1,2,0))*255).astype(np.uint8)
            im = plt.imshow(f1)
            plt.show()
        inputs = data.to(device)
        model_ft(inputs)      
        if j == 0:
            break

