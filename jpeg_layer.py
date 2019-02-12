# coding=utf-8
import math
import torch
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch_dct as dct

class JpegLayer(torch.nn.Module):
    def __init__(self, block_size = 8, quality = 75):
        super(JpegLayer, self).__init__()
        #quantizeY = torch.Tensor(block_size, block_size)
        quantizeY = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]])/255
        #torch.nn.init.uniform(quantize, 0, 1)
        #quantizeC = torch.Tensor(block_size, block_size)
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
        print(self.dctmtx)
    def forward(self, input):
        
        #preprocessing
        rgb = 255*input
        ycbcr = self.__rgb2ycbcr(rgb)
        #mean filter subsample
        samples = self.__subsample(ycbcr) 
        
        samples2 = []
        for i in range(0,3):
        #blocking
            sts1 = torch.stack((torch.split(samples[i]-128, self.bs, 1)), 1)
            sts2 = torch.stack((torch.split(sts1, self.bs, 3)), 2)
        #dct
            dcts = torch.matmul(torch.matmul(self.dctmtx,sts2),self.dctmtx.t())
            #dcts = dct.dct_2d(sts2)
        #quantization
            comp = torch.round(dcts/
            torch.round( torch.clamp(
            torch.round( (
            torch.round(self.quantize[0]*255)
            *self.quality+50)/100)
            ,min=1,max=255) ) 
            )
            #torch.round(dcts/
            #torch.round(#torch.clamp( 
            #torch.round( (
            #torch.round(self.quantize*255)
            #*self.quality+50 ) /100)
            #,min=1, max=255)
            #) )
        
        #decompression
            nograd_quantize = \
            torch.round(torch.clamp(
            torch.round( (
            torch.round(self.quantize[0].clone()*255)
            *self.quality+50 ) /100)
            ,min=1,max=255
            ) )
            decomp =  torch.round(comp*nograd_quantize)
            idcts = torch.matmul(torch.matmul(self.dctmtx.t(), decomp), self.dctmtx)
            #idcts = dct.idct_2d(decomp)
            sts3 = torch.cat(torch.unbind(idcts, 2), 3)
            sts4 = torch.cat(torch.unbind(sts3, 1), 1)
            print('sts4',sts4.size())
            samples2.append(sts4)
        ycbcr2 = self.__upsample(samples2)
        y_pred = torch.round(self.__ycbcr2rgb(ycbcr2+128))
#        a,b,c,d = y_pred.shape
#        for i in range(a):
#            for j in range(b):
#                for k in range(c):
#                    for p in range(d):
#                        if y_pred[i][j][k][p].item()!=rgb[i][j][k][p].item():
#                            print("diff!: ", i, j, k, p,"-",\
#                            y_pred[i][j][k][p].item(),\
#                                rgb[i][j][k][p].item())
#        
        
        return y_pred/255
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
        #block_size = torch.cuda.FloatTensor(bs)
        dct_mtx = torch.empty(bs, bs)#.type(torch.cuda.FloatTensor)
        for j in range(0, bs):
            for i in range(0, bs):
                if i == 0:
                    dct_mtx[i][j] = 1/math.sqrt(bs)
                else:
                    dct_mtx[i][j] = math.sqrt(2/bs) * \
                    math.cos((math.pi*(2*j+1)*i)/(2*bs)) 
        return dct_mtx.type(torch.cuda.FloatTensor)
    def __upsample(self, samples, row=2, col=2):
        upsamples = []
        upsamples.append(samples[0])
        a,b,c = samples[1].shape #batches, w/r,h/c
        for i in range(1,3):
            upsample = torch.unsqueeze(samples[i],3)
            print(upsample.size())
            upsample = upsample.repeat(1,1,1,4)
            upsample = upsample.view(a,b,c,row,col)
            sts1 = torch.cat(torch.unbind(upsample, 2),3)
            sts2 = torch.cat(torch.unbind(sts1, 1),1)
            print(sts2.size())
            upsamples.append(sts2)
        sts3 = torch.stack(upsamples)
        return sts3.permute(1,0,2,3)
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
             
        return means # sts4.permute(1,0,2,3)
        

    def __rgb2ycbcr(self,rgb):
        ycbcr = rgb.permute(0,2,3,1)
        xform = torch.cuda.FloatTensor([[.299,.587,.114],
                  [-0.168735892 ,- 0.331264108, 0.5],
                  [.5,- 0.418687589, - 0.081312411]])
        ycbcr = torch.matmul(ycbcr, xform.t())
        ycbcr[:,:,:,[1,2]] += 128
        #put mask
        ycbcr = torch.clamp(ycbcr, min=0, max=255)
        ycbcr = ycbcr.permute(0,3,1,2)
        return ycbcr

    def __ycbcr2rgb(self,im):
        rgb = im.permute(0,2,3,1)
        xform = torch.cuda.FloatTensor([[1, 0, 1.402],
                [1, - 0.344136286, - 0.714136286], 
                [1, 1.772, 0]])

        rgb[:,:,:,[1,2]] -= 128
        rgb = torch.matmul(rgb, xform.t())
        #put mask
        rgb = torch.clamp(rgb, min=0, max=255)
        
        rgb = rgb.permute(0,3,1,2)
        return rgb


