import os
import torch
from torchvision import datasets, models, transforms
from PIL import Image
import numpy as np
import math
import argparse
from comp_utils import *
from huffman import *

parser = argparse.ArgumentParser(description = \
                'Test JpegLayer Compression Ratio')
parser.add_argument('--data_dir', '-d', type=str,\
    default='/data/jenna/data/', \
    help='Directory of the input data. \
    String. Default: /data/jenna/data/')

args,unparsed = parser.parse_known_args()




class JpegLayer(torch.nn.Module):
    def __init__(self, block_size = 8, quality = 75):
        super(JpegLayer, self).__init__()
        quantizeY = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]])/255
        quantizeC = np.array([
        [17,18,24,47,99,99,99,99],
        [18,21,26,66,99,99,99,99],
        [24,26,56,99,99,99,99,99],
        [47,66,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99]     
        ])/255
        quantize = np.array([quantizeY, quantizeC, quantizeC])
        self.quantize = torch.nn.Parameter(torch.cuda.FloatTensor(quantize))
        
        self.quality = self.__scale_quality(quality)
        self.bs = block_size
        self.dctmtx = self.__dctmtx(self.bs)


    def forward(self, input):
        #preprocessing
        rgb = input
        ycbcr = self.__rgb2ycbcr(rgb)-128/255
        #mean filter subsample
        samples = self.__subsample(ycbcr) 
        
        samples2 = []
        for i in range(0,3):
        #blocking
            sts1 = torch.stack((torch.split(samples[i], self.bs, 1)), 1)
            sts2 = torch.stack((torch.split(sts1, self.bs, 3)), 2)
        #dct
            dcts = torch.matmul(torch.matmul(self.dctmtx,sts2),self.dctmtx.t())
            comp = torch.round(dcts/self.quantize[i])
            blocks = comp.view(-1, self.bs, self.bs)
            samples2.append(blocks)

        return samples2
            
    def __scale_quality(self, quality=75):
        if(quality<=0):
            quality = 1
        elif(quality >= 100):
            quality = 100
        if (quality < 50):
          quality = 5000 / quality
        else:
          quality = 200 - quality * 2
      
        return quality

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

input_size = 224

data_transforms = {
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(args.data_dir, x), data_transforms[x]) for x in ['val']}

dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1, shuffle=True, num_workers=4) for x in ['val']}

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = model_ft.to(device)

def get_activation(name):
    print(name)
    def hook(model, input, output):
        result = []
        for i in range(3):
            result.append(output[i].detach())
        activation[name] = result
    return hook

activation = {}    
model_ft.register_forward_hook(get_activation('JpegLayer'))



for name, (data, _) in enumerate(dataloaders_dict['val']):
    f1 = torch.squeeze(data).cpu().data.numpy()
    f1 = (np.transpose(f1,(1,2,0))*255).astype(np.uint8)
    im = Image.fromarray(f1,'RGB')
    im.save("data/uncomp/"+str(name)+".bmp")


    output = model_ft(data.to(device))
    f2 = activation['JpegLayer']

    dc_y=np.empty(f2[0].shape[0], dtype=np.int32)
    ac_y=np.empty((f2[0].shape[0],63), dtype=np.int32)
    dc_c=np.empty(f2[1].shape[0]*2, dtype=np.int32)
    ac_c=np.empty((f2[1].shape[0]*2,63), dtype=np.int32)
    
    for i in range(3):
        for j in range(f2[i].shape[0]):
            zz = block_to_zigzag(f2[i][j])
            if i%3 == 0:
                dc_y[j] = zz[0]
                ac_y[j] = zz[1:]
            else:
                dc_c[j*(i-1)] = zz[0]
                ac_c[j*(i-1)] = zz[1:]
    H_DC_Y = HuffmanTree(np.vectorize(bits_required)(dc_y))
    H_DC_C = HuffmanTree(np.vectorize(bits_required)(dc_c))
    H_AC_Y = HuffmanTree(
        flatten(run_length_encode(ac_y[i, :])[0]
        for i in range(f2[0].shape[0])))
    H_AC_C = HuffmanTree(
        flatten(run_length_encode(ac_c[i, :])[0]
        for i in range(f2[1].shape[0]*2)))
    tables = {
    'dc_y': H_DC_Y.value_to_bitstring_table(),
    'ac_y': H_AC_Y.value_to_bitstring_table(),
    'dc_c': H_DC_C.value_to_bitstring_table(),
    'ac_c': H_AC_C.value_to_bitstring_table()}
    write_to_file("data/jpeg/"+str(name)+".jpg", dc_y, ac_y, dc_c, ac_c, [f2[0].shape[0],f2[1].shape[0],f2[1].shape[0]], tables)
