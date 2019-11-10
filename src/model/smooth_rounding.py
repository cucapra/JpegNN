# import matplotlib.pyplot as plt
# import numpy as np
import math,torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# def Rd(x,q=5):
#     return round(x/q)

def smoothRd(_input,q=5, scale=100.):
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
    rem = _input%q
    base = (_input-rem)/q

    delta = rem.clone()
    offset = torch.ones(size=rem.shape, dtype=rem.dtype)*0.5

    delta[rem>q/2] = rem[rem>q/2]-q
    offset[rem<=q/2] *= -1
    smooth_rd = 1 / (1 + torch.exp(-delta*scale/q))+offset # sigmoid

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

    x = torch.arange(1000)/20.-25
    y = smoothRd(x)
    plt.scatter(x, y)
    plt.savefig('test.png')
    plt.close()


