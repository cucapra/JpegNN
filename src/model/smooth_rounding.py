# import matplotlib.pyplot as plt
# import numpy as np
import math,sys,torch
import numpy as np
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# def Rd(x,q=5):
#     return round(x/q)

def smoothRd(_input,q=5., scale=100.):
    # smaller scale yields smoother outputs

    # For python or numpy:
    # base = _input//q
    # rem = _input-base*q
    # if rem>q/2:
    #     delta = rem-q
    #     offset=0.5
    # else:
    #     delta = rem
    #     offset = -0.5
    # smooth_rd = 1 / (1 + math.exp(-delta*scale))+offset # sigmoid

    # For pytorch
    # https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/symbolic_script.cpp
    # rem = _input%q

    noGradQ = q.detach().repeat(*_input.shape[:-2],1,1)
    # rem = torch.remainder(_input,q) # sign same as q
    rem = torch.fmod(_input,noGradQ) # sign same as _input
    idx = rem<0
    rem[idx] += noGradQ[idx]
    base = (_input-rem)/q
    
    delta = rem.clone()
    idx = rem>q/2
    if isinstance(q, torch.Tensor):
        delta[idx] = rem[idx]-noGradQ[idx]
    else:
        delta[idx] = rem[idx]-q

    offset = rem.clone().fill_(0.5)
    offset[~idx] = -0.5
    smooth_rd = torch.sigmoid(delta*scale/q) + offset
    # print(_input.requires_grad, q.requires_grad, base.requires_grad, rem.requires_grad, smooth_rd.requires_grad)
    return base+smooth_rd

class LearnableQuantization(nn.Module):
    def __init__(self, bandwidth=3, n_dev = 3,temperature = 1, noise = 0):
        super(LearnableQuantization, self).__init__()
        # #distributions
        # self.K = 2**bandwidth
        # grid = torch.arange(self.K/2*-1,self.K/2+1).type(torch.FloatTensor) - 0.5
        # self.grid = nn.Parameter(grid.repeat( (8, 8, 1) ) , requires_grad = False)
        # #trainable parameter
        # quantize = torch.FloatTensor(8,8)
        # t = 2/(2**bandwidth)
        # alpha = t + 3*t/(2**bandwidth)
        # quantize.fill_(alpha)
        # self.alpha = nn.Parameter(quantize)
        # self.beta = nn.Parameter(quantize.clone()*0, requires_grad = True)
        # #self.deviation = nn.Parameter(quantize.clone()/3, requires_grad = True)
        # #hyperparameter
        # self.n_dev = n_dev
        # self.T = temperature
        # self.noise = noise
        # #gumbel
        # self.loc = nn.Parameter(torch.tensor([0.0]),requires_grad=False)
        # self.scale=nn.Parameter(torch.tensor([1.0]),requires_grad=False)

        qtable = [
            [[16., 11., 10., 16., 24., 40., 51., 61],
            [12., 12., 14., 19., 26., 58., 60., 55.], 
            [14., 13., 16., 24., 40., 57., 69., 56.], 
            [14., 17., 22., 29., 51., 87., 80., 62.], 
            [18., 22., 37., 56., 68., 109., 103., 77.], 
            [24., 35., 55., 64., 81., 104., 113., 92.], 
            [49., 64., 78., 87., 103., 121., 120., 101.], 
            [72., 92., 95., 98., 112., 100., 103., 99.]],

            [[17., 18., 24., 47., 99., 99., 99., 99.], 
            [18., 21., 26., 66., 99., 99., 99., 99.], 
            [24., 26., 56., 99., 99., 99., 99., 99.], 
            [47., 66., 99., 99., 99., 99., 99., 99.], 
            [99., 99., 99., 99., 99., 99., 99., 99.], 
            [99., 99., 99., 99., 99., 99., 99., 99.], 
            [99., 99., 99., 99., 99., 99., 99., 99.], 
            [99., 99., 99., 99., 99., 99., 99., 99.]], 

            [[17., 18., 24., 47., 99., 99., 99., 99.], 
            [18., 21., 26., 66., 99., 99., 99., 99.], 
            [24., 26., 56., 99., 99., 99., 99., 99.], 
            [47., 66., 99., 99., 99., 99., 99., 99.], 
            [99., 99., 99., 99., 99., 99., 99., 99.], 
            [99., 99., 99., 99., 99., 99., 99., 99.], 
            [99., 99., 99., 99., 99., 99., 99., 99.], 
            [99., 99., 99., 99., 99., 99., 99., 99]]
        ]
        self.qtable = nn.Parameter(torch.FloatTensor(qtable), requires_grad=True)
        #resize
        resize = torch.zeros((3,8,8))
        resize[0] = torch.tensor([
        [3.4400, 2.3875, 1.9897, 1.0016, 1.0389, 0.3645, 0.3035, 0.3291],
        [2.2794, 1.0046, 0.7606, 0.5959, 0.4451, 0.4062, 0.2093, 0.2721],
        [1.6991, 0.6141, 0.4613, 0.4080, 0.2264, 0.2138, 0.2584, 0.2058],
        [0.8828, 0.5757, 0.5145, 0.2779, 0.2151, 0.1700, 0.2046, 0.1689],
        [0.6887, 0.2974, 0.3386, 0.2682, 0.1977, 0.1678, 0.1730, 0.1498],
        [0.4882, 0.3275, 0.3980, 0.2217, 0.2232, 0.1427, 0.1718, 0.1441],
        [0.6117, 0.1563, 0.1993, 0.1491, 0.2155, 0.1302, 0.0959, 0.0986],
        [0.3379, 0.1993, 0.1591, 0.1523, 0.2073, 0.1663, 0.0997, 0.0826]]) 
        resize[1] = torch.tensor([
        [0.8870, 0.2992, 0.1490, 0.1180, 0.1198, 0.0658, 0.0746, 0.0528],
        [0.4187, 0.2443, 0.1635, 0.1283, 0.0837, 0.0895, 0.1018, 0.0618],
        [0.1757, 0.1409, 0.2205, 0.1115, 0.0606, 0.0702, 0.0764, 0.0454],
        [0.1223, 0.0710, 0.0943, 0.0546, 0.0368, 0.0567, 0.0404, 0.0340],
        [0.0844, 0.0735, 0.0755, 0.0465, 0.1198, 0.0519, 0.0371, 0.0205],
        [0.0956, 0.0864, 0.0617, 0.0416, 0.0404, 0.0303, 0.0349, 0.0323],
        [0.0676, 0.0645, 0.0555, 0.0375, 0.0221, 0.0455, 0.0638, 0.0317],
        [0.0582, 0.0392, 0.0282, 0.0196, 0.0197, 0.0606, 0.0143, 0.0495]]) 
        resize[2] = torch.tensor([
        [0.6556, 0.1522, 0.0993, 0.0758, 0.0638, 0.0494, 0.0465, 0.0406],
        [0.2041, 0.0915, 0.0620, 0.0529, 0.0321, 0.0293, 0.0399, 0.0309],
        [0.0909, 0.0636, 0.0757, 0.0430, 0.0279, 0.0226, 0.0274, 0.0187],
        [0.0712, 0.0629, 0.0464, 0.0274, 0.0297, 0.0354, 0.0188, 0.0282],
        [0.0583, 0.0523, 0.0420, 0.0279, 0.0448, 0.0256, 0.0168, 0.0120],
        [0.0582, 0.0433, 0.0428, 0.0310, 0.0178, 0.0207, 0.0176, 0.0144],
        [0.0436, 0.0298, 0.0260, 0.0164, 0.0145, 0.0229, 0.0246, 0.0140],
        [0.0313, 0.0296, 0.0219, 0.0134, 0.0156, 0.0270, 0.0125, 0.0238]])  
        self.resize = nn.Parameter(resize*1.2, requires_grad = False)

    def forward(self, inp, i):
        x=inp/self.resize[i]
        # mean = 0 #(x.view(-1,8,8)).abs().mean(dim = 0)

        mean = (x.view(-1,8,8)).abs().mean(dim = 0)
        nzeros = 0

        out = self.qtable[i].detach() * smoothRd(x, self.qtable[i]) # smoothRd((x-self.beta)/self.alpha) + self.beta
        # if self.training:
        #     cdf = torch.sigmoid( -1 * (x.unsqueeze(-1) - r)/ torch.abs(self.alpha.unsqueeze(-1)/3 )  )
        #     cdf = cdf.permute(5,0,1,2,3,4)
        #     pi = ( (cdf[1:] - cdf[:-1] 
        #             #+self.noise )
        #             #)/ 
        #             #(
        #             #cdf[self.K] - cdf[0] 
        #             #+
        #             #self.noise*self.K
        #             ) 
        #             )
        #     #g = torch.distributions.Gumbel(self.loc,self.scale)
        #     u = 0 #g.expand(pi.shape).sample()
        #     #z = ( (pi.log()+u)/self.T ).exp()
        #     #z = torch.exp(torch.log(pi))
        #     z = pi
        #     #print(z.sum(0))
        #     #z = z/z.sum(0)
        #     z = z.permute(1,2,3,4,5,0)
        #     out = (z*grid[:,:,:-1]).sum(-1)
        # else:
        #     y = self.alpha*torch.round((x-self.beta)/self.alpha) + self.beta
        #     out = torch.min(grid[:,:,-1],torch.max(grid[:,:,0],y))

        #print(x.abs().view(-1,8,8).mean(0), out.abs().view(-1,8,8).mean(0))
        #print('-------info', i, '--------') 
        #print(x[0,0,0,0,0], out[0,0,0,0,0])
        #print('grid',grid[0,0,0],grid[0,0,-1])
        out = out*self.resize[i]
        
        
        return out, mean, nzeros



if __name__ == '__main__':
    # x = []
    # y = []
    # for i in range(1000):
    #     x.append(i/20-25)
    #     y.append(smoothRd(x[-1]))
    # plt.scatter(x,y)
    # plt.savefig('test2.png')
    # plt.close()

    x = torch.arange(1000)/20.-25
    # x = torch.arange(40)/4. - 10.
    # x.requires_grad = True
    q = x.clone().fill_(5.)
    q.requires_grad = True
    y = smoothRd(x, q=q, scale=100.)
    assert q.requires_grad==y.requires_grad
    plt.scatter(x.detach().cpu(), y.detach().cpu())
    plt.savefig('test.png')
    plt.close()