# coding=utf-8
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch_dct as dct

class JpegLayer(torch.nn.Module):
    def __init__(self, block_size = 8):
        super(JpegLayer, self).__init__()
        quantize = torch.Tensor(block_size, block_size)
        quantize = torch.FloatTensor([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]])/255
        #torch.nn.init.uniform(quantize, 0, 1)
        self.quantize = torch.nn.Parameter(quantize)
        self.bs = block_size
        self.dctmtx = self.__dctmtx(self.bs)
    def forward(self, input):
        
        #preprocessing
        rgb = 255*input
        ycbcr = self.__rgb2ycbcr(rgb) - 128
         
        #blocking
        sts1 = torch.stack((torch.split(ycbcr, self.bs, 2)), 2)
        sts2 = torch.stack((torch.split(sts1, self.bs, 4)), 3)
        #dct
        dcts = torch.matmul(torch.matmul(self.dctmtx,sts2),self.dctmtx.t())
        #dcts = dct.dct_2d(sts2)
        #quantization
        comp = torch.round(dcts/(torch.round(self.quantize*255))  )
        #decompression
        nograd_quantize = torch.round(self.quantize.clone()*255)
        decomp = comp*nograd_quantize 
        idcts = torch.matmul(torch.matmul(self.dctmtx.t(), decomp), self.dctmtx)
        #idcts = dct.idct_2d(decomp)
        sts3 = torch.cat(torch.unbind(idcts, 3), 4)
        ycbcr2 = torch.cat(torch.unbind(sts3, 2), 2)+128
        y_pred = self.__ycbcr2rgb(ycbcr2)
        return y_pred/255

    def __dctmtx(self, bs=8):
        #block_size = torch.cuda.FloatTensor(bs)
        dct_mtx = torch.empty(bs, bs)#.type(torch.cuda.FloatTensor)
        for i in range(0, bs):
            for j in range(0, bs):
                if i == 0:
                    dct_mtx[i][j] = 1/math.sqrt(bs)
                else:
                    dct_mtx[i][j] = math.sqrt(2/bs) * \
                    math.cos((math.pi*(2*j+1)*i)/(2*bs)) 
        return dct_mtx.type(torch.cuda.FloatTensor)

    def __rgb2ycbcr(self,rgb):
        ycbcr = rgb.permute(0,2,3,1)
        xform = torch.cuda.FloatTensor([[.299,.587,.114],
                              [-.1687,-.3313,.5],
                              [.5,-.4187,-.0813]])
        ycbcr = torch.matmul(ycbcr, xform.t()) 
        ycbcr[:,:,:,[1,2]] += 128
        #put mask
        m1 = torch.lt(ycbcr, 0).float()
        m2 = torch.gt(ycbcr,255).float()
        ycbcr = ycbcr*(1-m1)*(1-m2) + m2*255

        ycbcr = ycbcr.permute(0,3,1,2)
        return ycbcr

    def __ycbcr2rgb(self,im):
        rgb = im.permute(0,2,3,1)
        xform = torch.cuda.FloatTensor([[1, 0, 1.402],
                        [1, -0.34414, -.71414], 
                        [1, 1.772, 0]])

        rgb[:,:,:,[1,2]] -= 128
        rgb = torch.matmul(rgb, xform.t())
        #put mask
        m1 = torch.lt(rgb, 0).float()
        m2 = torch.gt(rgb,255).float()
        rgb = rgb*(1-m1)*(1-m2) + m2*255
        
        rgb = rgb.permute(0,3,1,2)
        return rgb


