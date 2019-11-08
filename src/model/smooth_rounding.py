# import matplotlib.pyplot as plt
# import numpy as np
import math,torch


def Rd(x,q=5):
    return round(x/q)
def smoothRd(x,q=5):
    base = x//q
    rem = x-base*q
    scale = 100/q

    if rem>q/2:
        delta = rem-q
        offset=0.5
    else:
        delta = rem
        offset = -0.5
    differentiable_round = 1 / (1 + math.exp(-delta*scale))+offset # sigmoid
    return base+differentiable_round

# x = []
# y = []
# for i in range(1000):
#     x.append(i/20-25)
#     y.append(smoothRd(x[-1]))
# plt.scatter(x,y)
# plt.savefig('test2.png')
# plt.close()
