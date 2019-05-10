# coding=utf-8
import re
from ast import literal_eval

import math,copy
import torch
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from scipy.fftpack import dct, idct
import gradient
from learnable_quantization import LearnableQuantization
import operator
from collections import OrderedDict
from itertools import islice

class sequential(torch.nn.Module):
    def __init__(self, *args):
        super(sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, input):
        for i, module in enumerate(self._modules.values()):
            if i==0: 
                input, output2,output3 = module(input)
            else: 
                input = module(input)
        return input, output2,output3

class JpegLayer(torch.nn.Module):
    def __init__(self, rand_qtable = False, block_size = 8, cnn_only = True,  quality = 50):
        super(JpegLayer, self).__init__()
        #if rand_qtable == False:
        #    f = open("init.txt","r")
        #    a = re.sub('\s+','',f.read())
        #    quantize = np.array(literal_eval(a))/255
        #    quantize /= 3
        #    self.quantize = torch.nn.Parameter(torch.FloatTensor(quantize.copy()), requires_grad = True)
        #    
        #else:
        #    quantize = torch.FloatTensor(3, block_size, block_size)
        #    torch.nn.init.uniform(quantize,50/255,50/255)
        #    #for x in range(2,6):
        #    #    for y in range(2,6):
        #    #        quantize[:,x,y] = 1/255
        #    #quantize[0,0,0] = 7000/255
        #    #quantize[1,0,0] = 1200/255
        #    #quantize[2,0,0] = 2000/255
        #    #quantize[:,0,1] = 100/255
        #    #quantize[:,1,0] = 100/255
        #    self.quantize = torch.nn.Parameter( quantize, requires_grad = True )
        #self.quality = self.__scale_quality(quality) \
        #                if cnn_only else 100
        self.bs = block_size
        self.dctmtx = torch.nn.Parameter(self.__dctmtx(self.bs),requires_grad = False)
        #gradients
        #self.round = gradient.RoundNoGradient
        self.clamp = gradient.ClampNoGradient
        print(self.dctmtx.is_cuda)
        self.LQ = LearnableQuantization()
        self.xform1 = torch.nn.Parameter(torch.FloatTensor([[.299,.587,.114],
                  [-0.168735892 ,- 0.331264108, 0.5],
                  [.5,- 0.418687589, - 0.081312411]]),
                  requires_grad = False)
        self.xform2 = torch.nn.Parameter(torch.FloatTensor([[1, 0, 1.402],
                [1, - 0.344136286, - 0.714136286], 
                [1, 1.772, 0]]),
                requires_grad = False)

        #mask_qtable = torch.zeros([3, block_size, block_size],dtype = torch.uint8)
        #for i in range(2, block_size-2):
        #    mask_qtable[:,i,2:block_size-2] = 1
        #print(mask_qtable)
        #self.mask_qtable = torch.nn.Parameter(mask_qtable, requires_grad = False)
        #print(self.quantize*255)
        #torch.set_printoptions(precision=8)
        #print(self.dctmtx)

    def dct2(self,a):
        return dct(dct(a,axis=0, norm='ortho'), axis = 1, norm='ortho')

    def forward(self, input):
        #preprocessing
        #rgb = torch.nn.functional.interpolate(input,size=[224,224])
        b,c,w,h = input.shape
        wpad = math.ceil(w/16)*16
        hpad = math.ceil(h/16)*16
        padding = torch.nn.ConstantPad2d((0, hpad-h,0,wpad-w),0)
        rgb = padding(input)
        ycbcr = self.__rgb2ycbcr(rgb)-128/255
        #mean filter subsample
        samples = self.__subsample(ycbcr) 
        #print(ycbcr[0].shape,samples[0].shape)
        #print(ycbcr[0][0]*255,samples[0][0]*255)
        samples2 = []
        nzeros = 0
        means = 0
        decimal = 5
        for i in range(0,3):
            #if i != 0:
            #    samples2.append(samples[i])
            #    continue
        #blocking
            sts1 = torch.stack((torch.split(samples[i], self.bs, 1)), 1)
            sts2 = torch.stack((torch.split(sts1, self.bs, 3)), 2)
            #if i==0: print(i,self.quantize[i]*255)
            #print(torch.all( (sts2+128/255)>=0) )
            #idcts = sts2*self.quantize[i]
            #idcts = self.round.apply((sts2 + 128/255)/self.quantize[i]) * self.quantize[i] - 128/255

 #       #dct
            dcts = torch.matmul(torch.matmul(self.dctmtx,sts2),self.dctmtx.t() )
            decomp, mean, nzeros = self.LQ(dcts)
 #           
 #           #dctst= self.dct2(np.array(ycbcr[0,0,0:8,0:8].cpu() ) )
 #           #print(torch.matmul(self.dctmtx,sts2)[0][0][0])
 #           #print(dctst, dcts[0][0][0])
 #           #print(np.all(dctst==np.array(dcts[0][0][0].cpu()) ) )
 #           decomp = 0
 #           comp = 0
 #           mean_dct = dcts.view(-1,8,8).mean(dim=0)
 #           std_dct = dcts.view(-1,8,8).std(dim = 0)
 #           mask = ( (dcts.view(-1,8,8).abs() ).mean(dim=0)>0).expand_as(dcts)
 #           dcts = torch.where(mask, (dcts - mean_dct)/std_dct, 0*dcts)
 #           #dcts = torch.where(self.mask_qtable[i], dcts, 0*dcts)
 #           #dcts[:,:,:,4:8,:] = 0 #let's forget about first component, don't train it
 #           #std_dct[4:8,:] = 0
 #           #mean_dct[4:8,:] = 0
 #           if i == 0: 
 #               print(i)
 #               print('mean freq', dcts.view(-1,8,8).abs().mean(dim=0))
 #               print('qtable',self.quantize[i]*255)
 #           if self.training:
 #               comp = dcts/( self.quantize[i]*self.quality/100 )*decimal
 #               round_comp = self.round.apply(comp)
 #               decomp = round_comp * ( self.quantize[i]*self.quality/100) /decimal
 #           else:#eval mode
 #              # qtable = torch.round(self.quantize[i]*255*self.quality/100 + 0.5)/255
 #               qtable = self.quantize[i]*self.quality/100 
 #               comp = dcts/qtable*decimal
 #               round_comp = self.round.apply(comp)
 #               decomp = round_comp*qtable/decimal
 #           decomp = torch.where(mask, decomp * std_dct + mean_dct, 0*decomp)
 #           #nzeros += round_comp.nonzero().size(0)
 #           #cnt += round_comp.numel()
 #           #print(nzeros/cnt)
 #           mean = comp.view(-1,8,8).abs().mean(dim = 0)
 #           #print(i,mean)
 #           #mean = self.quantize[i].sum()
            if i == 0:
                means = mean
            else:
                means += mean
 #           
            idcts = torch.matmul(torch.matmul(self.dctmtx.t(), decomp), self.dctmtx)
            sts3 = torch.cat(torch.unbind(idcts, 2), 3)
            sts4 = torch.cat(torch.unbind(sts3, 1), 1)
            samples2.append(sts4)
        #nzeros = nzeros/cnt
        ycbcr2 = self.__upsample(samples2)
        y_pred = self.__ycbcr2rgb(ycbcr2+128/255)
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
        #return y_pred
        #print(means/3)
        return y_pred[:,:,0:w,0:h], means, nzeros

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
                    
        return dct_mtx.type(torch.FloatTensor)

    def __upsample(self, samples, row=2, col=2):
        upsamples = []
        upsamples.append(samples[0])

        a,b,c = samples[1].shape #batches, w/r,h/c
        for i in range(1,3):
            upsample = torch.unsqueeze(samples[i],3)
            upsample = upsample.repeat(1,1,1,4)
            upsample = upsample.view(a,b,c,row,col)
            sts1 = torch.cat(torch.unbind(upsample, 2),3)
            sts2 = torch.cat(torch.unbind(sts1, 1),1)
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
            mean = torch.mean(sts2.view(a,b,c,-1),3)
            means.append(mean)
             
        return means 
        

    def __rgb2ycbcr(self,rgb):
        ycbcr = rgb.permute(0,2,3,1)
        ycbcr = torch.matmul(ycbcr, self.xform1.t())
        ycbcr[:,:,:,[1,2]] += 128/255
        #put mask
        #ycbcr = torch.clamp(ycbcr,0,1)
        ycbcr = ycbcr.permute(0,3,1,2)
        return ycbcr

    def __ycbcr2rgb(self,im):
        rgb = im.permute(0,2,3,1)
        rgb[:,:,:,[1,2]] -= 128/255
        rgb = torch.matmul(rgb, self.xform2.t())
        #put mask
        #rgb = self.clamp.apply(rgb,0,1)
        rgb = rgb.permute(0,3,1,2)
        return rgb


