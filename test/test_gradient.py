import torch
import gradient

input = torch.rand(2,6)*10-5
#input = torch.autograd.Variable(torch.randn(2, 6), requires_grad=True)
#print(input)
x = torch.nn.Parameter(torch.arange(1,13).view(2,6).float(), requires_grad = True)
#y = gradient.RoundNoGradient.apply(input/x)*x
#gradient
print(-input/x+torch.round(input/x))

y = gradient.RoundNoGradient.apply(input/x)*(x.clone().detach())
print(-input/x)

#y = 1/x*x
#x = torch.clamp(input,0,1)
#x = gradient.ClampNoGradient.apply(input,0,1)

#x = torch.round(input)
#y = torch.add(x,1)
#y = y+1
#print(y)
y = y.sum()
y.backward()
print(x.grad)

#print(input)


