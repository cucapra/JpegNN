import torch

class RoundNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
    @staticmethod
    def backward(ctx, g):
        #print("round gradient!",g)
        return g

class ClampNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=0,max=1)
    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        grad_input = g.clone()
        grad_input[x < 0] = 0
        grad_input[x > 1] = 1
        return grad_input
 
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
