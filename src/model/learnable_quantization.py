import torch
import torch.nn as nn
import numpy as np

class LearnableQuantization(nn.Module):
    def __init__(self, bandwidth=3, n_dev = 3,temperature = 2, noise = 0.2):
        super(LearnableQuantization, self).__init__()
        #distributions
        grid = torch.arange(2**bandwidth*-1,2**bandwidth+1).type(torch.FloatTensor) - 0.5
        grid[-1] += 1
        self.grid = nn.Parameter(grid.repeat( (8, 8, 1) ) , requires_grad = False)
        #trainable parameter
        quantize = torch.FloatTensor(8,8)
        t = 0.002/(2**bandwidth)
        alpha = t + 3*t/(2**bandwidth)
        quantize.fill_(alpha)
        self.alpha = nn.Parameter(quantize)
        self.beta = nn.Parameter(quantize.clone()*0, requires_grad = False)
        self.deviation = nn.Parameter(quantize.clone()/3)
        #hyperparameter
        self.n_dev = n_dev
        self.T = temperature
        self.noise = noise
        #layer
        self.sigmoid = nn.LogSigmoid()
        #gumbel
        self.loc = nn.Parameter(torch.tensor([0.0]),requires_grad=False)
        self.scale=nn.Parameter(torch.tensor([1.0]),requires_grad=False)

    def forward(self, x):
        #for i in range(8):
        #    for j in range(8):
        #        print(torch.max(x.view(-1,8,8)[:,i,j]))
        mean = (x.view(-1,8,8)/self.alpha).abs().mean(dim = 0)
        nzeros = 0
        grid = self.grid * self.alpha.unsqueeze(-1) + self.beta.unsqueeze(-1)
        if self.training:
            cdf = self.sigmoid( -1 * (x.unsqueeze(-1) - grid)/self.deviation.unsqueeze(-1) )
            print(cdf[0,0,0])
            pi = ( (cdf[:,:,:,:,:,1:] - cdf[:,:,:,:,:,:-1] 
                    + self.noise) 
                    / (cdf[:,:,:,:,:,-1] - cdf[:,:,:,:,:,1]
                    + self.noise*(cdf.shape[-1]-1)).unsqueeze(-1) )
            g = torch.distributions.Gumbel(self.loc,self.scale)
            u = g.expand(pi.shape).sample()
            z = ( (pi.log()+u)/self.T ).exp().permute(5,0,1,2,3,4)
            z = (z/z.sum(0)).permute(1,2,3,4,5,0)
            out = (z*grid[:,:,:-1]).sum(-1)
        else:
            
            y = self.alpha*torch.round((x-self.beta)/self.beta) + self.beta
            out =torch.min(grid[:,:,-1],torch.max(grid[:,:,0],y))
        #print(x.abs().view(-1,8,8).mean(0), out.abs().view(-1,8,8).mean(0))
        return out, mean, nzeros
