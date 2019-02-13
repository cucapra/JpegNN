import torch

class RoundNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()
    @staticmethod
    def backward(ctx, g):
        return g 

class ClampNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=0,max=255)
    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        grad_input = g.clone()
        grad_input[x < 0] = 0
        grad_input[x>255] = 255
        return grad_input
