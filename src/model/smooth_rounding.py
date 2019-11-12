# import matplotlib.pyplot as plt
# import numpy as np
import math,torch

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
    rem = torch.fmod(_input,q) # sign same as _input
    rem[rem<0] += q[rem<0]
    # rem = torch.remainder(_input,q) # sign same as q
    base = (_input-rem)/q

    delta = rem.clone()
    if isinstance(q, torch.Tensor):
        delta[rem>q/2] = rem[rem>q/2]-q[rem>q/2]
    else:
        delta[rem>q/2] = rem[rem>q/2]-q

    offset = rem.clone().fill_(0.5)
    offset[rem<=q/2] = -0.5
    smooth_rd = torch.sigmoid(delta*scale/q) + offset

    return base+smooth_rd

if __name__ == '__main__':
    # x = []
    # y = []
    # for i in range(1000):
    #     x.append(i/20-25)
    #     y.append(smoothRd(x[-1]))
    # plt.scatter(x,y)
    # plt.savefig('test2.png')
    # plt.close()

    # x = torch.arange(1000)/20.-25
    x = torch.arange(40)/4. - 10.
    # x.requires_grad = True
    q = x.clone().fill_(5.)
    q.requires_grad = True
    y = smoothRd(x, q=q, scale=100.)
    assert q.requires_grad==y.requires_grad
    plt.scatter(x.detach().cpu(), y.detach().cpu())
    plt.savefig('test.png')
    plt.close()


