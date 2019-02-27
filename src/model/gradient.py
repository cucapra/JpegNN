import torch
from torch.autograd import Variable

class RoundNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        #print(x)
        return x.round()
    @staticmethod
    def backward(ctx, g):
        #print("round gradient!",g)
        return g

class ClampNoGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i, min_val, max_val):
        ctx._mask = (i.ge(min_val) * i.le(max_val))
        return i.clamp(min_val, max_val)

    @staticmethod
    def backward(ctx, grad_output):
        mask = Variable(ctx._mask.type_as(grad_output.data))
        return grad_output * mask, None, None
#    @staticmethod
#    def forward(ctx, x):
#        ctx.save_for_backward(x)
#        return x.clamp(min=0,max=1)
#    @staticmethod
#    def backward(ctx, g):
#        x, = ctx.saved_tensors
#        
#        g[x < 0] = 0
#        g[x > 1] = g
#        return g
 
class ClampNoGradient1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=1,max=255)
    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        grad_input = g.clone()
        grad_input[x < 1/255] = 1/255
        grad_input[x > 1] = 1
        return grad_input      
