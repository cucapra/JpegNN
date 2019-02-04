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
        [72, 92, 95, 98, 112, 100, 103, 99]])
        self.quantize = torch.nn.Parameter(quantize)
        #self.dequantize = deepcopy 
        #self.dct_mtx = self.__dctmtx(block_size)
        #print(self.dct_mtx)
        self.bs = block_size

    def forward(self, input):
#        expand = self.quantize.unsqueeze(0).unsqueeze(0).expand_as(input)
#        y_pred = input/expand
#        return y_pred


        #preprocessing
        ycbcr = self.__rgb2ycbcr(input)
        ycbcr -= 128
        
        #blocking
        a, b, c, d = list(input.size())
        blks = torch.split(ycbcr, self.bs, 2)
        sts = torch.stack(blks, 2)
        blks = torch.split(sts, self.bs, 4)
        sts = torch.stack(blks, 3)
        #dct
        dcts = dct.dct_2d(sts)
        #quantization
        #expand = torch.tensor(input.size())/\
        #       torch.tensor([1,1,self.bs,self.bs])

        #y_pred = ycbcr/self.quantize.repeat(list(expand.int()))
        comp = (dcts/self.quantize).int()
        
        #decompression
        decomp = comp.float()*self.quantize        
        idcts = dct.idct_2d(decomp)
        sts = torch.cat(torch.unbind(idcts, 3), 4)
        sts = torch.cat(torch.unbind(sts, 2), 2)
        sts += 128
        y_pred = self.__ycbcr2rgb(sts)

        return y_pred

#    def __matmul(X, Y):
#        results = [] 
#        for i in range(X.size(0)):
#            result = torch.mm(X[i], Y)
#            results.append(result)
#        return torch.cat(results)
#
#    def __dctmtx(self, bs=8):
#        #block_size = torch.cuda.FloatTensor(bs)
#        dct_mtx = torch.empty(bs, bs, dtype=torch.float)
#        for i in range(0, bs):
#            for j in range(0, bs):
#                if i == 0:
#                    dct_mtx[i][j] = math.sqrt(bs)
#                else:
#                    dct_mtx[i][j] = math.sqrt(2/bs) * \
#                    torch.cos((math.pi*(2*j+1)*i)/(2*bs))
#                return dct_mtx

    def __rgb2ycbcr(self,input):
        xform = torch.cuda.FloatTensor([[.299,.587,.114],
                              [-.1687,-.3313,.5],
                              [.5,-.4187,-.0813]])
        ycbcr = torch.transpose(torch.transpose(input, 1, 2), 2,3)
        ycbcr = torch.matmul(ycbcr, xform.t()) 
        ycbcr[:,:,:,[1,2]] += 128
        ycbcr = torch.transpose(torch.transpose(ycbcr, 2, 3),1,2)
        return ycbcr

    def __ycbcr2rgb(self,im):
        xform = torch.cuda.FloatTensor([[1,0,1.402],
                            [1,0.34414, -.71414],
                            [1,1.772,0]])
        rgb = torch.transpose(torch.transpose(im, 1, 2), 2, 3)
        rgb[:,:,:,[1,2]] -= 128
        rgb = torch.matmul(rgb, xform.t())
        m1 = torch.lt(rgb, 0).float()
        m2 = torch.gt(rgb,255).float()
        rgb = rgb*(1-m1)*(1-m2) + m2*255
        rgb = torch.transpose(torch.transpose(rgb, 2, 3), 1, 2)

        return rgb


